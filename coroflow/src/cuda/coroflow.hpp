#pragma once

#include "../../declarations.h"
#include "../coro.hpp"
#include "../task.hpp"
#include "utility.hpp"

namespace cf { // begin of namespace cf ===================================

// ==========================================================================
//
// Declaration of class Coroflow
//
// ==========================================================================
//


class Coroflow {

  friend void CUDART_CB _cuda_stream_callback(cudaStream_t st, cudaError_t stat, void* void_args);

  public:

    Coroflow(size_t num_threads);

    ~Coroflow();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    auto suspend();
    auto cuda_suspend(cudaStream_t st);

    void schedule();

    void wait();

    bool is_DAG();


  private:

    void _process(Task* tp);
    void _enqueue(Task* tp);


    void _invoke_coro_task(Task* tp);
    void _invoke_static_task(Task* tp);


    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    );

    std::vector<std::thread> _workers;
    std::vector<std::unique_ptr<Task>> _tasks;
    std::queue<Task*> _queue;
    std::vector<bool> _callbacks;

    std::mutex _mtx;
    std::condition_variable _cv;
    std::atomic<bool> _stop{false};
    std::atomic<size_t> _finished{0};
};

// cuda callback
void CUDART_CB _cuda_stream_callback(cudaStream_t st, cudaError_t stat, void* void_args) {
  checkCudaError(stat);

  // unpack
  auto* data = (std::pair<Coroflow*, Coro::promise_type*>*) void_args;
  Coroflow* cf = data->first;
  Coro::promise_type* prom = data->second;

  cf->_callbacks[prom->_id] = false;
  cf->_enqueue(cf->_tasks[prom->_id].get());
}
// ==========================================================================
//
// Definition of class Coroflow
//
// ==========================================================================

auto Coroflow::cuda_suspend(cudaStream_t st) {

  struct awaiter: std::suspend_always {
    cudaStream_t _st;
    std::pair<Coroflow*, Coro::promise_type*> _data;
    explicit awaiter(Coroflow* cf, cudaStream_t st): _data{cf, nullptr}, _st{st} {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) {
      _data.second = &(coro_handle.promise());
      Coroflow* cf = _data.first;
      cf->_callbacks[coro_handle.promise()._id] = true;
      cudaStreamAddCallback(_st, _cuda_stream_callback, (void*)&_data, 0);
    }
    
  };

  return awaiter{this, st};
}

auto Coroflow::suspend() {  // value from co_await
  struct awaiter: public std::suspend_always { // definition of awaitable for co_await
    explicit awaiter() noexcept {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) const noexcept {
      // TODO: add CPU callback?
    }
  };

  return awaiter{};
}

Coroflow::Coroflow(size_t num_threads) {
  _workers.reserve(num_threads);

  for(size_t t = 0; t < num_threads; ++t) {
    _workers.emplace_back([this, t]() {
        while(true) {
          Task* tp{nullptr};
          {
            std::unique_lock<std::mutex> lock(_mtx);
            //_cv.wait_for(lock, t * std::chrono::milliseconds(300), [this]{ return this->_stop.load() || (!this->_queue.empty()); });
            _cv.wait(lock, [this]{ return _stop || (!_queue.empty()); });
            if(_stop) {
              return;
            }

            tp = _queue.front();
            _queue.pop();
          }
          if(tp) {
            _process(tp);
          }
        }
      }
    );
  }
}

Coroflow::~Coroflow() {
  for(auto& w: _workers) {
    w.join();
  } 
}

void Coroflow::wait() {
  for(auto& w: _workers) {
    w.join();
  } 
  _workers.clear();
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle Coroflow::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle Coroflow::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

void Coroflow::schedule() {

  _callbacks.resize(_tasks.size(), false);

  std::vector<Task*> srcs;
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      srcs.push_back(t.get());
    }
  }

  for(auto tp: srcs) {
    _enqueue(tp);
  }
}


bool Coroflow::is_DAG() {
  std::stack<Task*> dfs;
  std::vector<bool> visited(_tasks.size(), false);
  std::vector<bool> in_recursion(_tasks.size(), false);

  for(auto& t: _tasks) {
    if(!_is_DAG(t.get(), visited, in_recursion)) {
      return false;
    }
  }

  return true;
}

bool Coroflow::_is_DAG(
  Task* tp,
  std::vector<bool>& visited,
  std::vector<bool>& in_recursion
) {
  if(!visited[tp->_id]) {
    visited[tp->_id] = true;
    in_recursion[tp->_id] = true;

    for(auto succp: tp->_succs) {
      if(!visited[succp->_id]) {
        if(!_is_DAG(succp, visited, in_recursion)) {
          return false;
        }
      }
      else if(in_recursion[succp->_id]) {
        return false;
      }
    }
  }

  in_recursion[tp->_id] = false;

  return true;
}

void Coroflow::_invoke_static_task(Task* tp) {
  std::get_if<Task::StaticTask>(&tp->_handle)->work();
  for(auto succp: tp->_succs) {
    if(succp->_join_counter.fetch_sub(1) == 1) {
      _enqueue(succp);
    }
  }

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    {
      std::scoped_lock lock(_mtx);
      _stop = true;
      _cv.notify_all();
    }
  }
}

void Coroflow::_invoke_coro_task(Task* tp) {
  auto* coro = std::get_if<Task::CoroTask>(&tp->_handle);
  if(!coro->done()) {
    coro->resume();
    if(!_callbacks[coro->coro._coro_handle.promise()._id]) {
      _enqueue(tp);
    }
  }
  else {
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _enqueue(succp);
      }
    }

    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      {
        std::scoped_lock lock(_mtx);
        _stop = true;
        _cv.notify_all();
      }
    }
  }
}

void Coroflow::_process(Task* tp) {

  switch(tp->_handle.index()) {
    case Task::STATICTASK: {
      _invoke_static_task(tp);
    }
    break;

    case Task::COROTASK: {
      _invoke_coro_task(tp);
    }
    break;
  }
}

void Coroflow::_enqueue(Task* tp) {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _queue.push(tp);
  }
  _cv.notify_one();
}

} // end of namespace cf ==============================================
