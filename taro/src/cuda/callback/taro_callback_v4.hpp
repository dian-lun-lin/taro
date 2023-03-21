#pragma once

#include "../../../declarations.h"
#include "../../coro.hpp"
#include "../../task.hpp"
#include "../../worker.hpp"
#include "../utility.hpp"
#include "../../../../3rd-party/taskflow/taskflow/taskflow.hpp"

namespace taro { // begin of namespace taro ===================================

class TaroCBV4;
  
// directly use taskflow to schedule
//
// "stream-stealing" approach
// As suggested by CUDA doc, we use cudaLaunchHostFunc rather than cudaStreamAddCallback
// cudaStreamAddcallback
// cudaStream is handled by Taro
// if none of existing streams is available, enqueue the task back to master que
//
// ==========================================================================
//
// Declaration of class TaroCBV4
//
// ==========================================================================
//


class TaroCBV4 {

  //friend void CUDART_CB _cuda_stream_callback_v4(cudaStream_t st, cudaError_t stat, void* void_args);
  friend void CUDART_CB _cuda_stream_callback_v4(void* void_args);

  struct cudaCallbackData {
    TaroCBV4* taro{nullptr};
    Coro::promise_type* prom{nullptr};
    cudaStream_t stream{nullptr};
  };


  public:

    // num_streams here does not mean anything
    // this arg is for ease of benchmarking
    TaroCBV4(size_t num_threads, size_t num_streams = 0);

    ~TaroCBV4();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    auto suspend();

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto cuda_suspend(C&&);

    void schedule();

    bool is_DAG() const;


  private:

    void _process(tf::Subflow& sublfow, Task* tp);

    void _enqueue(Task* tp);
    void _enqueue(const std::vector<Task*>& tps);

    void _invoke_coro_task(tf::Subflow& sublfow, Task* tp);
    void _invoke_static_task(tf::Subflow& sublfow, Task* tp);
    void _invoke_inner_task(tf::Subflow& sublfow, Task* tp);

    Worker* _this_worker();

    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    ) const;

    std::vector<std::unique_ptr<Task>> _tasks;

    std::vector<cudaStream_t> _streams;
    std::vector<size_t> _in_stream_tasks;

    // TODO: may change to priority queue
    std::queue<Task*> _que;

    std::mutex _qmtx;
    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;

    std::atomic<bool> _stop{false};
    std::atomic<size_t> _finished{0};
    std::atomic<int> _cbcnt{0};

    tf::Taskflow _taskflow;
    tf::Executor _executor;
};

// ==========================================================================
//
// callback
//
// ==========================================================================

// cuda callback
void CUDART_CB _cuda_stream_callback_v4(void* void_args) {
  auto* data = (TaroCBV4::cudaCallbackData*) void_args;

  auto* taro = data->taro;
  auto* prom = data->prom;
  taro->_enqueue(taro->_tasks[prom->_id].get());
  taro->_cbcnt.fetch_sub(1);
}

// ==========================================================================
//
// Definition of class TaroCBV4
//
// ==========================================================================

TaroCBV4::TaroCBV4(size_t num_threads, size_t num_streams): 
  _executor{num_threads},
  _streams{num_streams},
  _in_stream_tasks{num_streams, 0}
{
  for(auto& stream: _streams) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
}

TaroCBV4::~TaroCBV4() {
  for(auto& stream: _streams) {
    checkCudaError(cudaStreamDestroy(stream));
  }
}

void TaroCBV4::schedule() {


  _taskflow.emplace([this](tf::Subflow& subflow){
    std::vector<Task*> srcs;
    for(auto& t: _tasks) {
      if(t->_join_counter.load() == 0) {
        srcs.push_back(t.get());
      }
    }
    _enqueue(srcs);

    // TODO: use condition variable 
    // _cv.wait(lock, [this]{ return (_stop &&  _cbcnt != 0) || !_que.empty; });
    while(!_stop || _cbcnt != 0) {
      //std::cerr << "_stop: " << _stop << ", _cbcnt: " << _cbcnt << "\n";
      std::scoped_lock lock(_qmtx);
      if(!_que.empty()) {
        subflow.emplace([this, &srcs](tf::Subflow& subflow){
          do{
            Task* tp = _que.front();
            _que.pop();
            _process(subflow, tp);
          } while(!_que.empty());
        });
      }
    }
  });

  _executor.run(_taskflow).wait();
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto TaroCBV4::cuda_suspend(C&& c) {

  struct awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaCallbackData data;

    explicit awaiter(TaroCBV4* taro, C&& c): kernel{std::forward<C>(c)} {
      data.taro = taro; 
    }
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) {

      // set callback data
      data.stream = _get_stream();
      data.prom = &(coro_handle.promise());

      // enqueue the kernel to the stream
      data.taro->_cbcnt.fetch_add(1);
      {
        std::scoped_lock lock(data.taro->_kernel_mtx);
        kernel(data.stream);
        cudaLaunchHostFunc(data.stream, _cuda_stream_callback_v4, (void*)&data);
      }
    }

    private:

      cudaStream_t _get_stream() {

        // choose the best stream id
        size_t stream_id;
        {
          std::scoped_lock lock(data.taro->_stream_mtx);
          stream_id = std::distance(
            data.taro->_in_stream_tasks.begin(), 
            std::min_element(data.taro->_in_stream_tasks.begin(), data.taro->_in_stream_tasks.end())
          );
          ++data.taro->_in_stream_tasks[stream_id];
        }

        return data.taro->_streams[stream_id];
      }
  };

  return awaiter{this, std::forward<C>(c)};
}

auto TaroCBV4::suspend() {
  struct awaiter: std::suspend_always {
    TaroCBV4* _taro;
    explicit awaiter(TaroCBV4* taro) noexcept : _taro{taro} {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) const noexcept {
      auto id = coro_handle.promise()._id;
      _taro->_enqueue(_taro->_tasks[id].get());
    }
  };

  return awaiter{this};
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle TaroCBV4::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle TaroCBV4::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

void TaroCBV4::_enqueue(Task* tp) {
  {
    std::scoped_lock lock(_qmtx);
    _que.push(tp);
  }
}

void TaroCBV4::_enqueue(const std::vector<Task*>& tps) {
  {
    std::scoped_lock lock(_qmtx);
    for(auto* tp: tps) {
      _que.push(tp);
    }
  }
}

void TaroCBV4::_process(tf::Subflow& subflow, Task* tp) {

  switch(tp->_handle.index()) {
    case Task::STATICTASK: {
      _invoke_static_task(subflow, tp);
    }
    break;

    case Task::COROTASK: {
      _invoke_coro_task(subflow, tp);
    }
    break;

    default:
      assert(false);
  }
}

void TaroCBV4::_invoke_static_task(tf::Subflow& subflow, Task* tp) {
  subflow.emplace([this, tp](){
    std::get_if<Task::StaticTask>(&tp->_handle)->work();
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _enqueue(succp);
      }
      if(_finished.fetch_add(1) + 1 == _tasks.size()) {
        _stop = true;
      }
    }
  });
}

void TaroCBV4::_invoke_coro_task(tf::Subflow& subflow, Task* tp) {
  subflow.emplace([this, tp](){
    auto* coro_t = std::get_if<Task::CoroTask>(&tp->_handle);
    coro_t->resume();
    if(coro_t->coro._coro_handle.promise()._final) {
      for(auto succp: tp->_succs) {
        if(succp->_join_counter.fetch_sub(1) == 1) {
          _enqueue(succp);
        }
      }
      if(_finished.fetch_add(1) + 1 == _tasks.size()) {
        _stop = true;
      }
    }
  });
}


bool TaroCBV4::is_DAG() const {
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

bool TaroCBV4::_is_DAG(
  Task* tp,
  std::vector<bool>& visited,
  std::vector<bool>& in_recursion
) const {
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


} // end of namespace taro ==============================================
