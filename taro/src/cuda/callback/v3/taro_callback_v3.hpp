#pragma once

#include <taro/declarations.h>
#include "../../utility/utility.hpp"
#include "worker.hpp"
#include "task.hpp"
#include "coro.hpp"

namespace taro { // begin of namespace taro ===================================

class TaroCBV3;
  
// work-stealing alrotihm using C++20 synchronization primitives
// callback tries enque tasks back to original worker to minimize conext switch
// we don't use notifier as it cannot wake up a specific waiting thread
// we use C++ binary semaphore for each thread instead
//
// C++20 synchronization primitives we use:
// jthread
// C++ binary semaphore
// latch
// atomic.wait()
//
// As suggested by CUDA doc, we use cudaLaunchHostFunc rather than cudaStreamAddCallback
// cudaStreamAddcallback
// cudaStream is handled by Taro
// work-stealing approach
// TODO: check memory order
//
// ==========================================================================
//
// Declaration of class TaroCBV3
//
// ==========================================================================
//


class TaroCBV3 {

  //friend void CUDART_CB _cuda_stream_callback_v1(cudaStream_t st, cudaError_t stat, void* void_args);
  friend void CUDART_CB _cuda_stream_callback_v3(void* void_args);

  struct cudaStream {
    cudaStream_t st;
    size_t id;
  };

  struct cudaCallbackData {
    TaroCBV3* cf;
    Coro::promise_type* prom;
    Worker* worker;
    size_t stream_id;
  };


  public:

    TaroCBV3(size_t num_threads, size_t num_streams);

    ~TaroCBV3();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    //auto suspend();

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto cuda_suspend(C&&);

    void schedule();

    void wait();

    bool is_DAG() const;


  private:

    void _process(Worker& worker, Task* tp);

    void _enqueue(Worker& worker, Task* tp, TaskPriority p = TaskPriority::LOW);
    void _enqueue(Task* tp, TaskPriority p = TaskPriority::LOW);

    void _invoke_coro_task(Worker& worker, Task* tp);
    void _invoke_static_task(Worker& worker, Task* tp);

    Worker* _this_worker();

    void _exploit_task(Worker& worker);
    bool _explore_task(Worker& worker, const std::stop_token& stop);

    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    ) const;

    std::vector<std::jthread> _threads;
    std::vector<Worker> _workers;
    std::vector<cudaStream> _streams;
    std::vector<std::unique_ptr<Task>> _tasks;
    std::unordered_map<std::thread::id, size_t> _wids;

    std::vector<size_t> _in_stream_tasks;
    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;

    std::atomic<size_t> _finished{0};
    std::atomic<size_t> _pending_tasks{0};
    std::atomic<size_t> _cbcnt{0};
    size_t _MAX_STEALS;
    size_t _cnt;
};

// ==========================================================================
//
// callback
//
// ==========================================================================

// cuda callback
void CUDART_CB _cuda_stream_callback_v3(void* void_args) {

  // unpack
  auto* data = (TaroCBV3::cudaCallbackData*) void_args;
  auto* cf = data->cf;
  auto* prom = data->prom;
  auto* worker = data->worker;
  size_t stream_id = data->stream_id;
  
  {
    std::scoped_lock lock(cf->_stream_mtx);
    --cf->_in_stream_tasks[stream_id];
  }


  {
    // high priory queue is owned by the callback function
    // Due to CUDA runtime, we cannot guarntee whether the cuda callback function is called sequentially
    // we need a lock to atomically enqueue the task
    // note that there is no lock in enqueue functions
    
    std::scoped_lock lock(worker->_mtx);
    cf->_enqueue(*worker, cf->_tasks[prom->_id].get(), TaskPriority::HIGH);
    worker->_wait_task.release();
  }
  cf->_cbcnt.fetch_sub(1);
}

// ==========================================================================
//
// Definition of class TaroCBV3
//
// ==========================================================================

TaroCBV3::TaroCBV3(size_t num_threads, size_t num_streams): 
  _workers{num_threads}, 
  _MAX_STEALS{(num_threads + 1) << 1},
  _threads{num_threads}
{

  //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  // GPU streams
  _in_stream_tasks.resize(num_streams, 0);
  _streams.reserve(num_streams);
  for(size_t i = 0; i < num_streams; ++i) {
    _streams[i].id = i;
    cudaStreamCreateWithFlags(&(_streams[i].st), cudaStreamNonBlocking);
  }

  std::latch initial_done{num_threads};
  std::mutex wmtx;
  size_t cnt{0};

  for(size_t id = 0; id < num_threads; ++id) {
    auto& worker = _workers[id];
    worker._id = id;

    _threads[id] = std::jthread([this, id, num_threads, &worker, &cnt, &initial_done, &wmtx](const std::stop_token &stop) {

      worker._thread = &_threads[id];

      {
        std::scoped_lock lock(wmtx);
        _wids[std::this_thread::get_id()] = worker._id;
      }
      initial_done.count_down();

      // begin of task scheduling ===================================================
      do {
        worker._wait_task.acquire();

        do {
          _exploit_task(worker);

          if(!_explore_task(worker, stop)) {
            return; // stop
          }

        } while(_pending_tasks.load(std::memory_order_acquire) > 0);
      } while(true);

      // end of task scheduling =====================================================
      
    });

  }

  initial_done.wait();
}

void TaroCBV3::_exploit_task(Worker& worker) {

  _exploit_task_high:
    while(auto task = worker._que.steal(TaskPriority::HIGH)) {
      _pending_tasks.fetch_sub(1, std::memory_order_release);
      _process(worker, task.value());
    }

  _exploit_task_low:
    while(auto task = worker._que.pop(TaskPriority::LOW)) {
      _pending_tasks.fetch_sub(1, std::memory_order_release);
      _process(worker, task.value());
      if(!worker._que.empty(TaskPriority::HIGH)) {
        goto _exploit_task_high;
      }
    }

}

bool TaroCBV3::_explore_task(Worker& worker, const std::stop_token& stop) {

  size_t num_steals{0};
  size_t num_yields{0};


  do {

    // TODO: difference between round robin and random?
    for(size_t i = 1; i < _threads.size(); ++i) {
      size_t idx = (worker._id + i) % _threads.size();
      auto opt = _workers[idx]._que.steal(); // from LOW to HIGH
      if(opt) {
        _pending_tasks.fetch_sub(1, std::memory_order_release);
        _process(worker, opt.value());
        return true;
      }

      if(num_steals++ > _MAX_STEALS) {
        std::this_thread::yield();
        if(num_yields++ > 100) {
          return true;
        }
      }
      
    }

    if(!worker._que.empty(TaskPriority::HIGH)) {
      return true;
    }
  } while(!stop.stop_requested());

  return false; // stop
}

TaroCBV3::~TaroCBV3() {
  for(auto& st: _streams) {
    cudaStreamDestroy(st.st);
  }
}

void TaroCBV3::wait() {
  for(auto& t: _threads) {
    t.join();
  }

  while(_cbcnt.load() != 0) {}
}

void TaroCBV3::schedule() {

  std::vector<Task*> srcs;
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      srcs.push_back(t.get());
    }
  }

  for(auto src: srcs) {
    _enqueue(src);
    auto& worker = _workers[_cnt++ % _threads.size()];
    worker._wait_task.release();
  }
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto TaroCBV3::cuda_suspend(C&& c) {

  struct awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaCallbackData data;

    explicit awaiter(TaroCBV3* cf, C&& c): kernel{std::forward<C>(c)} {
      data.cf = cf; 
    }
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) {

      // choose the best stream id
      size_t stream_id;
      {
        std::scoped_lock lock(data.cf->_stream_mtx);
        stream_id = std::distance(
          data.cf->_in_stream_tasks.begin(), 
          std::min_element(data.cf->_in_stream_tasks.begin(), data.cf->_in_stream_tasks.end())
        );
        ++data.cf->_in_stream_tasks[stream_id];
      }

      // set callback data
      data.prom = &(coro_handle.promise());
      data.stream_id = stream_id;
      data.worker = data.cf->_this_worker();


      // enqueue the kernel to the stream
      data.cf->_cbcnt.fetch_add(1);
      {
        std::scoped_lock lock(data.cf->_kernel_mtx);
        kernel(data.cf->_streams[stream_id].st);
        cudaLaunchHostFunc(data.cf->_streams[stream_id].st, _cuda_stream_callback_v3, (void*)&data);
      }

    }
    
  };

  return awaiter{this, std::forward<C>(c)};
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle TaroCBV3::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle TaroCBV3::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

bool TaroCBV3::is_DAG() const {
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

void TaroCBV3::_enqueue(Worker& worker, Task* tp, TaskPriority p) {
  worker._que.push(tp, p);
  _pending_tasks.fetch_add(1, std::memory_order_relaxed);
}

// this enqueue is only used by main thread
void TaroCBV3::_enqueue(Task* tp, TaskPriority p) {
  auto& worker = _workers[_cnt++ % _threads.size()];
  worker._que.push(tp, p);
  _pending_tasks.fetch_add(1, std::memory_order_relaxed);
}

void TaroCBV3::_process(Worker& worker, Task* tp) {

  switch(tp->_handle.index()) {
    case Task::STATICTASK: {
      _invoke_static_task(worker, tp);
    }
    break;

    case Task::COROTASK: {
      _invoke_coro_task(worker, tp);
    }
    break;
  }
}

void TaroCBV3::_invoke_static_task(Worker& worker, Task* tp) {
  std::get_if<Task::StaticTask>(&tp->_handle)->work();
  size_t cnt{0};
  for(auto succp: tp->_succs) {
    if(succp->_join_counter.fetch_sub(1) == 1) {
      _enqueue(worker, succp);
      size_t idx = (worker._id + cnt++) % _threads.size(); // "nofity" one
      _workers[idx]._wait_task.release(); // we release the worker thread first, otherwise it may go to sleep
    }
  }

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    for(auto& w: _workers) {
      w._thread->request_stop();
      w._wait_task.release();
    }
  }
}

void TaroCBV3::_invoke_coro_task(Worker& worker, Task* tp) {
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
    size_t cnt{0};
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _enqueue(worker, succp);
        size_t idx = (worker._id + cnt++) % _threads.size(); // "nofity" one
        _workers[idx]._wait_task.release(); // we release the worker thread first, otherwise it may go to sleep
      }
    }

    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      for(auto& w: _workers) {
        w._thread->request_stop();
        w._wait_task.release();
      }
    }
  }
}

Worker* TaroCBV3::_this_worker() {
  auto it = _wids.find(std::this_thread::get_id());
  return (it == _wids.end()) ? nullptr : &_workers[it->second];
}

bool TaroCBV3::_is_DAG(
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
