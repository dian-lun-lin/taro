#pragma once

#include "../../declarations.h"
#include "../coro.hpp"
#include "../task.hpp"
#include "../worker.hpp"
#include "utility.hpp"
#include "../../../3rd-party/taskflow/notifier.hpp"
#include "../../../3rd-party/taskflow/wsq.hpp"

namespace taro { // begin of namespace taro ===================================

class TaroV7;
  
// cudaEvent SHOULD USE cudaDisableTiming to enable best performance
// cudaEventRecord SHOULD SPECIFY STREAM
//
// cudaStream is handled by Taro
// work-stealing approach
// As suggested by CUDA doc, we use cudaLaunchHostFunc rather than cudaStreamAddCallback
// TODO: maybe embed CUDA GRAPH? (CUDA graph capturer)
//
// once we call callback, meaning this stream is now empty -> grab another gpu task
//
// ==========================================================================
//
// Declaration of class TaroV7
//
// ==========================================================================
//


class TaroV7 {


  public:

    TaroV7(size_t num_threads, size_t num_streams);

    ~TaroV7();

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

    cudaWorker* _cuda_find_available_worker(Worker& worker);
    //void _cuda_assign_task(Worker& worker, cudaWorker* gw);
    void _cuda_exploit_task(Worker& worker, cudaWorker& gw, bool& assigned);
    void _cuda_explore_task(Worker& worker, cudaWorker& gw, bool& assigned);

    void _exploit_task(Worker& worker);
    Task* _explore_task(Worker& worker);
    bool _wait_for_task(Worker& worker);

    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    ) const;

    size_t _num_streams;
    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
    std::vector<std::unique_ptr<Task>> _tasks;
    std::unordered_map<std::thread::id, size_t> _wids;

    WorkStealingQueue<Task*> _que;
    std::vector<bool> _stream_stat;

    std::mutex _qmtx;
    std::mutex _wmtx;
    std::condition_variable _wcv;

    Notifier _notifier;
    std::atomic<bool> _stop{false};
    std::atomic<size_t> _finished{0};
    size_t _MAX_STEALS;
    size_t _CUDA_MAX_STEALS;
};

// ==========================================================================
//
// Definition of class TaroV7
//
// ==========================================================================

TaroV7::TaroV7(size_t num_threads, size_t num_streams): 
  _workers{num_threads}, _notifier{num_threads}, 
  _MAX_STEALS{(num_threads + 1) << 1}, _CUDA_MAX_STEALS{(num_threads + 1) << 1}, 
  _num_streams{num_streams} 
{


  // CPU threads
  _threads.reserve(num_threads);
  size_t cnt{0};
  for(size_t id = 0; id < num_threads; ++id) {
    _threads.emplace_back([this, id, num_threads, num_streams, &cnt]() {

      auto& worker = _workers[id];
      worker._id = id;
      worker._vtm = id;
      worker._thread = &_threads[id];
      worker._waiter = &_notifier._waiters[id];

      // evenly distribute cuda workers to workers
      for(size_t s = id; s < num_streams; s += num_threads) {
        worker._gws.emplace_back();
      }

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
void TaroV7::_exploit_task(Worker& worker) {
  while(auto task = worker._que.pop()) {
    _process(worker, task.value());
  }
}

// try to steal
Task* TaroV7::_explore_task(Worker& worker) {

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

bool TaroV7::_wait_for_task(Worker& worker) {

  Task* task{nullptr};
  bool assigned{false};

  cuda_assign_task:
    size_t num_finds{0};
    while(num_finds++ < _num_streams) {
      auto gw = _cuda_find_available_worker(worker); 
      // if there is an available cudaWorker
      // try to find a cuda task
      _cuda_exploit_task(worker, *gw, assigned);
      _cuda_explore_task(worker, *gw, assigned);
    }

  // if we assign a cuda task to a cudaWorker, it's very likely the local queue is enqueued
  if(assigned) {
    return true;
  }
    

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

  //  
  if(_cuda_all_avalaible()) {
    for(size_t vtm = 0; vtm < _workers.size(); ++vtm) {
      if(!_workers[vtm]._que.empty()) {
        _notifier.cancel_wait(worker._waiter);
        worker._vtm = vtm;
        goto explore_task;
      }
    }
  }

  _notifier.commit_wait(worker._waiter);

  goto cuda_assign_task;
}


TaroV7::~TaroV7() {
}

void TaroV7::wait() {
  for(auto& t: _threads) {
    t.join();
  }
}

void TaroV7::schedule() {

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
auto TaroV7::cuda_suspend(C&& c) {

  struct awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    // TODO: WorkStealingQueue<T> uses std::atomic<T>, which requries triviallyCopyable type
    // We need to store cuda task here and pass the pointer to queue
    std::function<void(cudaWorker&)> cuda_task; 
    TaroV7& cf;

    explicit awaiter(TaroV7* cf, C&& c): cf{*cf}, kernel{std::forward<C>(c)} {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) {
      cuda_task = [coro_handle, this](cudaWorker& gw) {
        // update cur_task
        // TODO: do we need to handle final cur_task?
        // TODO: Does decentralized WorkStealingQueue help?
        auto id = coro_handle.promise()._id;
        gw.cur_task = cf._tasks[id].get();
        kernel(gw.st);
      };

      cf._this_worker()->_gque.push(&cuda_task);
      cf._notifier.notify(false);
    }
  };
  return awaiter{this, std::forward<C>(c)};
}



// find an available cudaWorker
cudaWorker* TaroV7::_cuda_find_available_worker(Worker& worker) {
  //auto git = std::find_if(
    //worker._gws.begin(), worker._gws.end(), 
    //[](const cudaWorker& gw) { return cudaStreamQuery(gw.st) == cudaSuccess; }
  //);
  std::cerr << "find cuda worker\n";

  cudaWorker* choose{nullptr};
  for(auto& gw: worker._gws) {
    if(cudaStreamQuery(gw.st) == cudaSuccess)  {
      if(gw.cur_task != nullptr) {
        // previous coro finished, we need to enqueue the coro back
        _enqueue(worker, gw.cur_task);
        gw.cur_task = nullptr;
        _notifier.notify(false);
      }
      choose = &gw;
    }
  }
  return choose;
}

//void TaroV7::_cuda_update(Worker& worker) {
//}

void TaroV7::_cuda_exploit_task(Worker& worker, cudaWorker& gw, bool& assigned) {
  std::cerr << "cuda exploit\n";
  if(!assigned) {
    if(!worker._gque.empty()) {
      // exploit
      // get a new cuda task from queue
      auto opt = worker._gque.pop();
      auto cuda_task = *(opt.value());
      
      cuda_task(gw);
      assigned = true;
    }
  }
}

// try to steal from other workers
void TaroV7::_cuda_explore_task(Worker& worker, cudaWorker& gw, bool& assigned) {
  std::cerr << "cuda explore\n";
  if(!assigned) {
    size_t num_steals{0};
    size_t num_yields{0};
    std::uniform_int_distribution<size_t> rdvtm(0, _workers.size() - 1);
    while(true) {
      size_t vtm = rdvtm(worker._rdgen);
      if(worker._id != vtm) {
        auto opt = _workers[vtm]._gque.steal();
        if(opt) {
          auto cuda_task = *(opt.value());
          cuda_task(gw);
          assigned = true;
          break;
        }
      }
      if(num_steals++ > _CUDA_MAX_STEALS) {
        std::this_thread::yield();
        if(num_yields++ > 100) {
          break;
        }
      }
    }
  }
}

auto TaroV7::suspend() {
  struct awaiter: std::suspend_always {
    TaroV7* _cf;
    explicit awaiter(TaroV7* cf) noexcept : _cf{cf} {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) const noexcept {
      auto id = coro_handle.promise()._id;
      _cf->_enqueue(*(_cf->_this_worker()), _cf->_tasks[id].get());
      _cf->_notifier.notify(false);
    }
  };

  return awaiter{this};
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle TaroV7::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle TaroV7::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

bool TaroV7::is_DAG() const {
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

void TaroV7::_enqueue(Worker& worker, Task* tp) {
  worker._que.push(tp);
}

void TaroV7::_enqueue(Worker& worker, const std::vector<Task*>& tps) {
  for(auto* tp: tps) {
    worker._que.push(tp);
  }
}

void TaroV7::_enqueue(Task* tp) {
  {
    std::scoped_lock lock(_qmtx);
    _que.push(tp);
  }
}

void TaroV7::_enqueue(const std::vector<Task*>& tps) {
  {
    std::scoped_lock lock(_qmtx);
    for(auto* tp: tps) {
      _que.push(tp);
    }
  }
}

void TaroV7::_process(Worker& worker, Task* tp) {

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

void TaroV7::_invoke_static_task(Worker& worker, Task* tp) {
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

void TaroV7::_invoke_coro_task(Worker& worker, Task* tp) {
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

Worker* TaroV7::_this_worker() {
  auto it = _wids.find(std::this_thread::get_id());
  return (it == _wids.end()) ? nullptr : &_workers[it->second];
}

bool TaroV7::_is_DAG(
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
