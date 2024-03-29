#pragma once

#include <taro/declarations.hpp>
#include <taro/utility/cuda.hpp>
#include "../taskflow/taskflow/taskflow.hpp"
#include "task.hpp"
#include "coro.hpp"

namespace taro { // begin of namespace taro ===================================

class TaroTaskflow;
  
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
// Declaration of class TaroCBTaroTaskflow
//
// ==========================================================================
//


class TaroTaskflow {

  friend void CUDART_CB _cuda_stream_callback_taskflow(void* void_args);

  struct cudaCallbackData {
    TaroTaskflow* taro{nullptr};
    Coro::promise_type* prom{nullptr};
    cudaStream_t stream{nullptr};
  };


  public:

    // num_streams here does not mean anything
    // this arg is for ease of benchmarking
    TaroTaskflow(size_t num_threads, size_t num_streams = 0);

    ~TaroTaskflow();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    //auto suspend();

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto cuda_suspend(C&&);

    void schedule();

    bool is_DAG() const;

    void wait();


  private:

    void _process(const std::vector<Task*>& tps);
    void _process(Task* tp);

    void _invoke_coro_task(Task* tp);
    void _invoke_static_task(Task* tp);
    void _invoke_inner_task(Task* tp);

    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    ) const;

    std::vector<std::unique_ptr<Task>> _tasks;

    std::vector<cudaStream_t> _streams;
    std::vector<size_t> _in_stream_tasks;

    // TODO: may change to priority queue

    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;

    std::atomic<size_t> _finished{0};
    std::atomic<size_t> _cbcnt{0};

    std::promise<void> _prom;

    tf::Executor _executor;
};

// ==========================================================================
//
// callback
//
// ==========================================================================

// cuda callback
void CUDART_CB _cuda_stream_callback_taskflow(void* void_args) {
  auto* data = (TaroTaskflow::cudaCallbackData*) void_args;

  auto* taro = data->taro;
  auto* prom = data->prom;
  taro->_process(taro->_tasks[prom->_id].get());
  taro->_cbcnt.fetch_sub(1);
}

// ==========================================================================
//
// Definition of class TaroTaskflow
//
// ==========================================================================

TaroTaskflow::TaroTaskflow(size_t num_threads, size_t num_streams): 
  _executor{num_threads},
  _streams{num_streams},
  _in_stream_tasks{num_streams, 0}
{
  for(auto& stream: _streams) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
}

TaroTaskflow::~TaroTaskflow() {
  for(auto& stream: _streams) {
    checkCudaError(cudaStreamDestroy(stream));
  }
}

void TaroTaskflow::wait() {
}

void TaroTaskflow::schedule() {

  auto fu = _prom.get_future();

  std::vector<Task*> srcs;
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      srcs.push_back(t.get());
    }
  }
  _process(srcs);

  fu.get();

  while(_cbcnt.load() != 0) {}
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto TaroTaskflow::cuda_suspend(C&& c) {

  struct awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaCallbackData data;

    explicit awaiter(TaroTaskflow* taro, C&& c): kernel{std::forward<C>(c)} {
      data.taro = taro; 
    }
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) {

      // set callback data
      data.stream = data.taro->_streams[_get_stream()];
      data.prom = &(coro_handle.promise());

      // enqueue the kernel to the stream
      data.taro->_cbcnt.fetch_add(1);
      {
        std::scoped_lock lock(data.taro->_kernel_mtx);
        kernel(data.stream);
        cudaLaunchHostFunc(data.stream, _cuda_stream_callback_taskflow, (void*)&data);
      }
    }

    private:

      size_t _get_stream() {

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

        return stream_id;
      }
  };

  return awaiter{this, std::forward<C>(c)};
}

//auto TaroTaskflow::suspend() {
  //struct awaiter: std::suspend_always {
    //TaroTaskflow* _taro;
    //explicit awaiter(TaroTaskflow* taro) noexcept : _taro{taro} {}
    //void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) const noexcept {
      //auto id = coro_handle.promise()._id;
      //_taro->_enqueue(_taro->_tasks[id].get());
      //_taro->_cv.notify_one();
    //}
  //};

  //return awaiter{this};
//}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle TaroTaskflow::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle TaroTaskflow::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

void TaroTaskflow::_process(const std::vector<Task*>& tps) {

  for(auto& tp: tps) {

    switch(tp->_handle.index()) {
      case Task::STATICTASK: {
        _invoke_static_task(tp);
      }
      break;

      case Task::COROTASK: {
        _invoke_coro_task(tp);
      }
      break;

      default:
        assert(false);
    }
  }
}

void TaroTaskflow::_process(Task* tp) {

  switch(tp->_handle.index()) {
    case Task::STATICTASK: {
      _invoke_static_task(tp);
    }
    break;

    case Task::COROTASK: {
      _invoke_coro_task(tp);
    }
    break;

    default:
      assert(false);
  }
}

void TaroTaskflow::_invoke_static_task(Task* tp) {
  _executor.silent_async([this, tp](){
    std::get_if<Task::StaticTask>(&tp->_handle)->work();
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _process(succp);
      }
      if(_finished.fetch_add(1) + 1 == _tasks.size()) {
        _prom.set_value();
      }
    }
  });
}

void TaroTaskflow::_invoke_coro_task(Task* tp) {
  _executor.silent_async([this, tp](){
    auto* coro_t = std::get_if<Task::CoroTask>(&tp->_handle);
    // when this thread (i.e., t1) calls cuda_suspend and insert a callback to CUDA runtime
    // the CUDA runtime may finish cuda kernel very fast and 
    // use its own CPU thread to call the callback to enque the coroutine back
    // then, another thread (i.e., t2) may get this coroutine and performs resume()
    // However, t1 may still in resume()
    // which in turn causing data race
    // That is, two same coroutines are executed in parallel by t1 and t2
    // hence we use lock in each coro to check if a coroutine is in busy used 
    // final has similar issue as well

    bool final{false};
    {
      std::scoped_lock lock(coro_t->coro._mtx);
      coro_t->resume();
      final = coro_t->coro._coro_handle.promise()._final;
    }

    if(final) {
      for(auto succp: tp->_succs) {
        if(succp->_join_counter.fetch_sub(1) == 1) {
          _process(succp);
        }
      }
      if(_finished.fetch_add(1) + 1 == _tasks.size()) {
        _prom.set_value();
      }
    }
  });
}


bool TaroTaskflow::is_DAG() const {
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

bool TaroTaskflow::_is_DAG(
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
