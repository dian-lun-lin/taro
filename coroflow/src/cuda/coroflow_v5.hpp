#pragma once

#include "../../declarations.h"
#include "../coro.hpp"
#include "../task.hpp"
#include "../worker.hpp"
#include "utility.hpp"
#include "../../../3rd-party/taskflow/notifier.hpp"
#include "../../../3rd-party/taskflow/wsq.hpp"

namespace cf { // begin of namespace cf ===================================

class CoroflowV5;
  
// cudaStreamAddcallback
// cudaStream is handled by Coroflow
// work-stealing approach
// As suggested by CUDA doc, we use cudaLaunchHostFunc rather than cudaStreamAddCallback
// TODO: maybe embed CUDA GRAPH? (CUDA graph capturer)
//
// ==========================================================================
//
// Declaration of class CoroflowV5
//
// ==========================================================================
//


class CoroflowV5 {

  //friend void CUDART_CB _cuda_stream_callback_v5(cudaStream_t st, cudaError_t stat, void* void_args);
  friend void CUDART_CB _cuda_stream_callback_v5(void* void_args);

  struct cudaStream {
    cudaStream_t st;
    size_t id;
  };

  struct cudaCallbackData {
    CoroflowV5* cf;
    Coro::promise_type* prom;
    size_t stream_id;
    //size_t num_kernels;
  };


  public:

    CoroflowV5(size_t num_threads, size_t num_streams);

    ~CoroflowV5();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    auto suspend();

    template <typename C, std::enable_if_t<is_cuda_task_v<C>, void>* = nullptr>
    auto cuda_suspend(C&&);

    void schedule();

    void wait();

    bool is_DAG() const;


  private:

    void _process(Worker& worker, Task* tp);

    void _enqueue(Worker& worker, const std::vector<Task*>& tps);
    void _enqueue(Worker& worker, Task* tp);
    void _enqueue(Task* tp);
    void _enqueue(const std::vector<Task*>& tps);

    void _invoke_coro_task(Worker& worker, Task* tp);
    void _invoke_static_task(Worker& worker, Task* tp);

    Worker* _this_worker();

    void _exploit_task(Worker& worker);
    Task* _explore_task(Worker& worker);
    bool _wait_for_task(Worker& worker);


    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    ) const;

    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
    std::vector<cudaStream> _streams;
    std::vector<std::unique_ptr<Task>> _tasks;
    std::unordered_map<std::thread::id, size_t> _wids;

    std::vector<size_t> _in_stream_tasks;
    WorkStealingQueue<Task*> _que;

    std::mutex _qmtx;
    std::mutex _wmtx;
    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;
    std::condition_variable _wcv;

    Notifier _notifier;
    std::atomic<bool> _stop{false};
    std::atomic<size_t> _finished{0};
    size_t _MAX_STEALS;
};

// ==========================================================================
//
// callback
//
// ==========================================================================

// cuda callback
void CUDART_CB _cuda_stream_callback_v5(void* void_args) {

  // unpack
  auto* data = (CoroflowV5::cudaCallbackData*) void_args;
  auto* cf = data->cf;
  auto* prom = data->prom;
  auto stream_id = data->stream_id;
  
  {
    std::scoped_lock lock(cf->_stream_mtx);
    --cf->_in_stream_tasks[stream_id];
  }
  cf->_enqueue(cf->_tasks[prom->_id].get());
  cf->_notifier.notify(false);
}

// ==========================================================================
//
// Definition of class CoroflowV5
//
// ==========================================================================

CoroflowV5::CoroflowV5(size_t num_threads, size_t num_streams): 
  _workers{num_threads}, _notifier{num_threads}, _MAX_STEALS{(num_threads + 1) << 1} {

  //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  // GPU streams
  _in_stream_tasks.resize(num_streams, 0);
  _streams.reserve(num_streams);
  for(size_t i = 0; i < num_streams; ++i) {
    _streams[i].id = i;
    cudaStreamCreate(&(_streams[i].st));
  }

  // CPU threads
  _threads.reserve(num_threads);
  size_t cnt{0};
  for(size_t id = 0; id < num_threads; ++id) {
    _threads.emplace_back([this, id, num_threads, &cnt]() {

      auto& worker = _workers[id];
      worker._id = id;
      worker._vtm = id;
      worker._thread = &_threads[id];
      worker._waiter = &_notifier._waiters[id];

      {
        std::scoped_lock lock(_wmtx);
        _wids[std::this_thread::get_id()] = worker._id;
        if(cnt++; cnt == num_threads) {
          _wcv.notify_one();
        }
      }

      // TODO: must use 1 instead of !done
      // TODO: before we call schedule, we know there's no task in the queue
      // can we not enter into scheduling loop until we call schedule?
      while(1) {
        _exploit_task(worker);

        if(!_wait_for_task(worker)) {
          break;
        }
      }
    });

  }

  std::unique_lock<std::mutex> lock(_wmtx);
  _wcv.wait(lock, [&](){ return cnt == num_threads; });
}

// get a task from worker's own queue
void CoroflowV5::_exploit_task(Worker& worker) {
  while(auto task = worker._que.pop()) {
    _process(worker, task.value());
  }
}

// try to steal
Task* CoroflowV5::_explore_task(Worker& worker) {

  size_t num_steals{0};
  size_t num_yields{0};
  std::uniform_int_distribution<size_t> rdvtm(0, _workers.size() - 1);

  Task* task{nullptr};

  do {
    auto opt = ((worker._id == worker._vtm) ? _que.steal() : _workers[worker._vtm]._que.steal());

    if(opt) {
      task = opt.value();
      _process(worker, task);
      break;
    }

    if(num_steals++ > _MAX_STEALS) {
      std::this_thread::yield();
      if(num_yields++ > 100) {
        break;
      }
    }
    worker._vtm = rdvtm(worker._rdgen);
  } while(!_stop);

  return task;
}

bool CoroflowV5::_wait_for_task(Worker& worker) {

  Task* task{nullptr};
  explore_task:
    task = _explore_task(worker);

  // TODO: why do we need to wake up another worker to avoid starvation?
  // I thought std::this_thread::yield() already did that
  if(task) {
    _notifier.notify(false);
    return true;
  }
  
  // ======= 2PC guard =======
  _notifier.prepare_wait(worker._waiter);
  
  if(!_que.empty()) {
    _notifier.cancel_wait(worker._waiter);
    worker._vtm = worker._id; 
    goto explore_task;
  }

  if(_stop) {
    _notifier.cancel_wait(worker._waiter);
    _notifier.notify(true);
    return false;
  }

  // TODO: why do we need to use index-based scan to avoid data race?
  for(size_t vtm = 0; vtm < _workers.size(); ++vtm) {
    if(!_workers[vtm]._que.empty()) {
      _notifier.cancel_wait(worker._waiter);
      worker._vtm = vtm;
      goto explore_task;
    }
  }

  _notifier.commit_wait(worker._waiter);

  goto explore_task;
}


CoroflowV5::~CoroflowV5() {
  for(auto& st: _streams) {
    cudaStreamDestroy(st.st);
  }
}

void CoroflowV5::wait() {
  for(auto& st: _streams) {
    checkCudaError(cudaStreamSynchronize(st.st));
  }
  for(auto& t: _threads) {
    t.join();
  }
}

void CoroflowV5::schedule() {

  std::vector<Task*> srcs;
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      srcs.push_back(t.get());
    }
  }

  _enqueue(srcs);
  _notifier.notify(srcs.size());
}

template <typename C, std::enable_if_t<is_cuda_task_v<C>, void>*>
auto CoroflowV5::cuda_suspend(C&& c) {

  struct awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaCallbackData data;

    explicit awaiter(CoroflowV5* cf, C&& c): kernel{std::forward<C>(c)} {
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


      // enqueue the kernel to the stream
      {
        std::scoped_lock lock(data.cf->_kernel_mtx);
        kernel(data.cf->_streams[stream_id].st);
        cudaLaunchHostFunc(data.cf->_streams[stream_id].st, _cuda_stream_callback_v5, (void*)&data);
      }

    }
    
  };

  return awaiter{this, std::forward<C>(c)};
}

auto CoroflowV5::suspend() {
  struct awaiter: std::suspend_always {
    CoroflowV5* _cf;
    explicit awaiter(CoroflowV5* cf) noexcept : _cf{cf} {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) const noexcept {
      auto id = coro_handle.promise()._id;
      _cf->_enqueue(*(_cf->_this_worker()), _cf->_tasks[id].get());
      _cf->_notifier.notify(false);
    }
  };

  return awaiter{this};
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle CoroflowV5::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle CoroflowV5::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

bool CoroflowV5::is_DAG() const {
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

void CoroflowV5::_enqueue(Worker& worker, Task* tp) {
  worker._que.push(tp);
}

void CoroflowV5::_enqueue(Worker& worker, const std::vector<Task*>& tps) {
  for(auto* tp: tps) {
    worker._que.push(tp);
  }
}

void CoroflowV5::_enqueue(Task* tp) {
  {
    std::scoped_lock lock(_qmtx);
    _que.push(tp);
  }
}

void CoroflowV5::_enqueue(const std::vector<Task*>& tps) {
  {
    std::scoped_lock lock(_qmtx);
    for(auto* tp: tps) {
      _que.push(tp);
    }
  }
}

void CoroflowV5::_process(Worker& worker, Task* tp) {

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

void CoroflowV5::_invoke_static_task(Worker& worker, Task* tp) {
  std::get_if<Task::StaticTask>(&tp->_handle)->work();
  for(auto succp: tp->_succs) {
    if(succp->_join_counter.fetch_sub(1) == 1) {
      _enqueue(worker, succp);
      _notifier.notify(false);
    }
  }

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    _stop = true;
    _notifier.notify(true);
  }
}

void CoroflowV5::_invoke_coro_task(Worker& worker, Task* tp) {
  auto* coro = std::get_if<Task::CoroTask>(&tp->_handle);
  coro->resume();

  if(coro->coro._coro_handle.promise()._final) {
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _enqueue(worker, succp);
        _notifier.notify(false);
      }
    }

    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      // TODO: we need to check if there's no callback
      _stop = true;
      _notifier.notify(true);
    }
  }
}

Worker* CoroflowV5::_this_worker() {
  auto it = _wids.find(std::this_thread::get_id());
  return (it == _wids.end()) ? nullptr : &_workers[it->second];
}

bool CoroflowV5::_is_DAG(
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


} // end of namespace cf ==============================================
