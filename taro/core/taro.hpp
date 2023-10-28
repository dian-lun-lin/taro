#pragma once

#include "../declarations.hpp"
#include "worker.hpp"
#include "task.hpp"
#include "coro.hpp"



// forward deaclare for SYCL
namespace sycl {
  inline namespace _V1 {
    class queue;
  }
}


namespace taro { // begin of namespace taro ===================================

class Taro;
class Pipeline;
class cudaAwait;
class syclAwait;
class AsyncIOAwait;
class EventAwait;

template <size_t V>
class SemaphoreAwait;


// TODO: memory_order
// TODO: can we assign a kenel mutex for each stream
  
// move await_suspend to await_ready 
//
// work-stealing alrotihm using C++20 synchronization primitives
// callback tries enque tasks back to original worker to minimize conext switch
// we don't use notifier as it cannot wake up a specific waiting thread
// we use C++ atomic for each thread instead
//
// C++20 synchronization primitives we use:
// jthread
// latch
// atomic.wait()
// atomic.notify_one()
//
// As suggested by CUDA doc, we use cudaLaunchHostFunc rather than cudaStreamAddCallback
// cudaStreamAddcallback
// work-stealing approach
//
// ==========================================================================
//
// Declaration of class Taro
//
// ==========================================================================
//


class Taro {

  friend class Pipeline;
  friend class cudaAwait;
  friend class syclAwait;
  friend class EventAwait;
  friend class AsyncIOAwait;

  template <size_t V>
  friend class SemaphoreAwait;

  friend void _cuda_callback(void* void_args);
  friend void _cuda_polling(void* void_args);
  friend void _sycl_polling(void* void_args);
  friend void _async_io_consume(AsyncIOAwait& async_io);

  public:

    Taro(size_t num_threads);
    ~Taro();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    TaskHandle emplace(C&&);

    auto suspend();

    auto suspend(Task* task);

    void schedule();

    void wait();

    bool is_DAG() const;

    // Await declare
    cudaAwait cuda_await(size_t num_streams);
    syclAwait sycl_await(sycl::queue& que);
    EventAwait event_await(size_t num_events);
    template <size_t V>
    SemaphoreAwait<V> semaphore_await(size_t num_semaphores);
    AsyncIOAwait async_io_await(size_t queue_size);

    // Pattern declare
    Pipeline pipeline(size_t num_pipes, size_t num_lines, size_t num_tokens);

  private:

    void _process(Worker& worker, Task* tp);

    void _enqueue(Worker& worker, Task* tp, TaskPriority p = TaskPriority::LOW);

    void _invoke_coro_task(Worker& worker, Task* tp);
    void _invoke_static_task(Worker& worker, Task* tp);

    Worker* _this_worker();

    void _exploit_task(Worker& worker);
    bool _explore_task(Worker& worker, const std::stop_token& stop);
    void _enqueue_back(Worker& worker, size_t task_id);

    void _request_stop();

    bool _is_DAG(
      Task* tp,
      std::vector<bool>& visited,
      std::vector<bool>& in_recursion
    ) const;

    void _notify(Worker& worker);

    void _done(size_t task_id);

    void _init();

    std::vector<std::jthread> _threads;
    std::vector<Worker> _workers;
    std::vector<std::unique_ptr<Task>> _tasks;
    std::unordered_map<std::thread::id, size_t> _wids;
    std::atomic<int64_t> _pending_tasks{0}; 
    std::atomic<size_t> _finished{0};
    size_t _MAX_STEALS;
    size_t _cnt{0};
    std::atomic<size_t> _callback_polling_cnt{0};

};


// ==========================================================================
//
// Definition of class Taro
//
// ==========================================================================

inline
Taro::Taro(size_t num_threads): 
  _workers{num_threads}, 
  _MAX_STEALS{(num_threads + 1) << 1},
  _threads{num_threads}
{

  // TODO num_threads has warning (invalid conversion: unsigned long to long)
  std::latch initial_done{int(_workers.size())};
  std::mutex wmtx;

  for(size_t id = 0; id < num_threads; ++id) {
    auto* worker = &_workers[id];
    worker->_id = id;

    _threads[id] = std::jthread([this, id, worker, &initial_done, &wmtx](const std::stop_token &stop) {

      worker->_thread = &_threads[id];

      {
        std::scoped_lock lock(wmtx);
        _wids[std::this_thread::get_id()] = worker->_id;
      }
      initial_done.count_down();

      // begin of task scheduling ===================================================
      worker->_status.wait(Worker::STAT::SLEEP);

      do {
        worker->_status.store(Worker::STAT::BUSY);

        do {  
          _exploit_task(*worker);

          if(!_explore_task(*worker, stop)) {
            return;
          }
        } while(_pending_tasks.load() > 0);

        // false means a task is enqueued by callback
        if(worker->_status.exchange(Worker::STAT::SLEEP) == Worker::STAT::BUSY) {
          worker->_status.wait(Worker::STAT::SLEEP);
        }
      } while(!stop.stop_requested());

      // end of task scheduling =====================================================
      
    });

  }

  initial_done.wait();
}

inline
Taro::~Taro() {
  _request_stop();
  for(auto& t: _threads) {
    if(t.joinable()) {
      t.join();
    }
  }
}


inline
auto Taro::suspend() {
  struct awaiter: std::suspend_always {
    Taro& _taro;
    explicit awaiter(Taro& taro) noexcept : _taro{taro} {}
    
    bool await_ready() {
      // enqueue next pipe
      Worker* worker = _taro._this_worker();
      size_t task_id = worker->_work_on_task_id;
      {
        std::scoped_lock lock(worker->_mtx);
        _taro._enqueue(*worker, _taro._tasks[task_id].get(), TaskPriority::HIGH);
      }
      return false;
    }
  };

  return awaiter{*this};
}

inline
auto Taro::suspend(Task* task) {
  Worker* worker = _this_worker();
  
  struct awaiter: std::suspend_always {
    Taro& _taro;
    Worker* _worker;
    Task* _task;
    explicit awaiter(Taro& taro, Worker* worker, Task* task) noexcept : _taro{taro}, _worker{worker}, _task{task} {}
    
    bool await_ready() {
      return false;
    }
    void await_suspend(std::coroutine_handle<> coro) {
      _taro._enqueue(*_worker, _task, TaskPriority::LOW);
    }
    void await_resume() {
    }
  };

  return awaiter{*this, worker, task};
}

inline
void Taro::_exploit_task(Worker& worker) {

  _exploit_task_high:
    while(auto task = worker._que.steal(TaskPriority::HIGH)) {
      _pending_tasks.fetch_sub(1);
      _process(worker, task.value());
    }

  //_exploit_task_low:
  while(auto task = worker._que.pop(TaskPriority::LOW)) {
    _pending_tasks.fetch_sub(1);
    _process(worker, task.value());
    if(!worker._que.empty(TaskPriority::HIGH)) {
      goto _exploit_task_high;
    }
  }

}

inline
bool Taro::_explore_task(Worker& worker, const std::stop_token& stop) {

  size_t num_steals{0};
  size_t num_yields{0};


  do {

    // TODO: difference between round robin and random?
    for(size_t i = 1; i < _workers.size(); ++i) {
      size_t idx = (worker._id + i) % _workers.size();
      auto opt = _workers[idx]._que.steal(); // steal task from priority LOW to priority HIGH
      if(opt) {
        _pending_tasks.fetch_sub(1);
        _process(worker, opt.value());
        _notify(worker);
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

inline
void Taro::_enqueue_back(Worker& worker, size_t task_id) {
  {
    // The worker that is enqueued may not be this_worker
    // we need a lock to atomically enqueue the task
    // note that there is no lock in enqueue functions
    std::scoped_lock lock(worker._mtx);
    _enqueue(worker, _tasks[task_id].get(), TaskPriority::HIGH);
  }

  if(worker._status.exchange(Worker::STAT::SIGNALED) == Worker::STAT::SLEEP) {
    worker._status.notify_one();
  }
  // TODO: do we need to notify another worker if we already wake a worker up?
  _notify(worker);

  //// if we don't have counter,
  //// taro will not wait for this function to finish
  //// and may be destroyed, inducing seg fault
  //_callback_polling_cnt.fetch_sub(1);
}

inline
void Taro::wait() {
  for(auto& t: _threads) {
    t.join();
  }
  
  while(_callback_polling_cnt != 0) {}
}

inline
void Taro::_init() {
  if(_finished != 0) {
    _finished = 0;
    for(auto& t: _tasks) {
      t->_join_counter.store(t->_preds.size(), std::memory_order_relaxed);
    }
  }
}

inline
void Taro::schedule() {
  if(_tasks.empty()) {
    _request_stop();
    return;
  }

  _init();

  std::vector<Task*> srcs;
  for(auto& t: _tasks) {
    if(t->_join_counter.load(std::memory_order_relaxed) == 0 && !t->_wait_first) {
      srcs.push_back(t.get());
    }
  }

  // enqueue tasks before wake up workers to avoid data racing (i.e., worker._que.push())
  for(auto src: srcs) {
    auto& worker = _workers[_cnt++ % _workers.size()];
    _enqueue(worker, src);
  }

  for(size_t i = 0; i < std::min(srcs.size(), _workers.size()); ++i) {
    _workers[i]._status.store(Worker::STAT::SIGNALED);
    _workers[i]._status.notify_one();
  }
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle Taro::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle Taro::emplace(C&& c) {
  auto t = std::make_unique<Task>(_tasks.size(), std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  //std::get<Task::CoroTask>(t->_handle).coro._coro_handle.promise()._id = _tasks.size();
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

inline
bool Taro::is_DAG() const {
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

inline
void Taro::_enqueue(Worker& worker, Task* tp, TaskPriority p) {
  worker._que.push(tp, p);
  _pending_tasks.fetch_add(1); // make sure to add pending tasks after push
}

// this enqueue is only used by main thread
//void Taro::_enqueue(Task* tp, TaskPriority p) {
  //auto& worker = _workers[_cnt++ % _threads.size()];
  //worker._que.push(tp, p);
  //_pending_tasks.fetch_add(1, std::memory_order_relaxed);
//}

inline
void Taro::_process(Worker& worker, Task* tp) {

  worker._work_on_task_id = tp->_id;

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

inline
void Taro::_invoke_static_task(Worker& worker, Task* tp) {
  std::get_if<Task::StaticTask>(&tp->_handle)->work();
  for(auto succp: tp->_succs) {
    if(succp->_join_counter.fetch_sub(1) == 1) {
      _enqueue(worker, succp);
      _notify(worker);
    }
  }

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    _request_stop();
  }
}

inline
void Taro::_notify(Worker& worker) {
  size_t cnt{1};
  do {
    unsigned tmp = Worker::STAT::SLEEP;
    size_t idx = (worker._id + cnt++) % _workers.size();
    if(_workers[idx]._status.compare_exchange_weak(tmp, Worker::STAT::SIGNALED)) {
      // TODO: not sure if success can be memory_order_relaxed
      _workers[idx]._status.notify_one();
      return;
    }
  } while(cnt < _workers.size());

  // everyone seems to be busy
  // but busy workers may turn to sleep after we check the status
  // the worker need to wake up in case everyone go to sleep
  worker._status.store(Worker::STAT::SIGNALED);
}

inline
void Taro::_invoke_coro_task(Worker& worker, Task* tp) {
  auto* coro_t = std::get_if<Task::CoroTask>(&tp->_handle);

  coro_t->resume();

  if(coro_t->done()) {
    if(coro_t->is_handled() || coro_t->is_inner()) {
      return;
    }
    
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _enqueue(worker, succp);
        _notify(worker);
      }
    }

    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      _request_stop();
    }
  }
}

inline
void Taro::_request_stop() {
  for(auto& w: _workers) {
    w._thread->request_stop();
    w._status.store(Worker::STAT::SIGNALED);
    w._status.notify_one();
  }
}

inline
Worker* Taro::_this_worker() {
  auto it = _wids.find(std::this_thread::get_id());
  return (it == _wids.end()) ? nullptr : &_workers[it->second];
}

inline
bool Taro::_is_DAG(
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

// notify the scheduler this task is done
inline
void Taro::_done(size_t task_id) {
  Worker& worker = *_this_worker();
  auto* tp = _tasks[task_id].get();
  auto* coro_t = std::get_if<Task::CoroTask>(&tp->_handle);
  if(coro_t->is_handled()) {
    return;
  }
  
  for(auto succp: tp->_succs) {
    if(succp->_join_counter.fetch_sub(1) == 1) {
      _enqueue(worker, succp);
      _notify(worker);
    }
  }

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    _request_stop();
  }
}


} // end of namespace taro ==============================================
