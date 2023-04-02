#pragma once

namespace taro { // begin of namespace taro ===================================

class TaroCBV3;
  
// As suggested by CUDA doc, we use cudaLaunchHostFunc rather than cudaStreamAddCallback
// cudaStreamAddcallback
// cudaStream is handled by Taro
// work-stealing approach
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

    auto suspend();

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
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
    std::vector<std::mutex> _coro_mtx;

    std::vector<size_t> _in_stream_tasks;
    WorkStealingQueue<Task*> _que;

    std::mutex _qmtx;
    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;
    std::vector<std::mutex> _mtxes;

    Notifier _notifier;
    std::atomic<bool> _stop{false};
    std::atomic<size_t> _finished{0};
    std::atomic<size_t> _cbcnt{0};
    size_t _MAX_STEALS;
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

  // after enqueue, cf may finish the task and be destructed
  // in that case _notifier is destructed before we call notify in the callback
  // TODO: memory_order_acq_rel?
  switch(worker->_stage) {
    case Worker::Stage::EXPLORE:
      cf->_enqueue(*worker, cf->_tasks[prom->_id].get(), TaskPriority::LOW);
      break;
    case Worker::Stage::EXPLOIT:
      cf->_enqueue(*worker, cf->_tasks[prom->_id].get(), TaskPriority::HIGH);
      worker->_update_stage(Worker::Stage::EXPLORE);
      break;
    case Worker::Stage::SLEEP:
      cf->_enqueue(*worker, cf->_tasks[prom->_id].get(), TaskPriority::HIGH);
      worker->_cv.nofify_one();
      break;
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
  _notifier{num_threads}, 
  _MAX_STEALS{(num_threads + 1) << 1},
  _threads{num_threads},
  _mtxes{num_threads}
{

  //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  // GPU streams
  _in_stream_tasks.resize(num_streams, 0);
  _streams.reserve(num_streams);
  for(size_t i = 0; i < num_streams; ++i) {
    _streams[i].id = i;
    cudaStreamCreateWithFlags(&(_streams[i].st), cudaStreamNonBlocking);
  }

  std::mutex wmtx;
  std::condition_variable wcv;

  // CPU threads
  _threads.resize(num_threads);
  size_t cnt{0};
  for(size_t id = 0; id < num_threads; ++id) {
    auto& worker = _workers[id];
    worker._id = id;
    worker._vtm = id;
    worker._waiter = &_notifier._waiters[id];

    _threads[id] = std::thread([this, id, num_threads, &worker, &cnt, &wmtx, &wcv]() {

      worker._thread = &_threads[id];

      {
        std::scoped_lock lock(wmtx);
        _wids[std::this_thread::get_id()] = worker._id;
        if(cnt++; cnt == num_threads) {
          wcv.notify_one();
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

  std::unique_lock<std::mutex> lock(wmtx);
  wcv.wait(lock, [&](){ return cnt == num_threads; });
}

// get a task from worker's own queue
void TaroCBV3::_exploit_task(Worker& worker) {
  while(auto task = worker._que.pop()) {
    _process(worker, task.value());
  }

  worker._update_stage(Worker::Stage::EXPLORE);
}

// try to steal
Task* TaroCBV3::_explore_task(Worker& worker) {

  size_t num_steals{0};
  size_t num_yields{0};
  std::uniform_int_distribution<size_t> rdvtm(0, _workers.size() - 1);

  Task* task{nullptr};

  do {
    auto opt = ((worker._id == worker._vtm) ? _que.steal() : _workers[worker._vtm]._que.steal(TaskPriority::LOW));

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
  } while(!_stop && worker._stage.load(std::memory_order_relaxed) == Worker::Stage::EXPLORE);

  return task;
}

bool TaroCBV3::_wait_for_task(Worker& worker) {

  Task* task{nullptr};
  explore_task:
    task = _explore_task(worker);

  if(task || worker._stage.load(std::memory_order_relaxed) != Worker::Stage::EXPLORE) {
    // at least one worker needs to be waken up
    worker._vtm._cv.notify_one();
    return true;
  }
  
  
  if(!_que.empty()) {
    //_notifier.cancel_wait(worker._waiter);
    worker._vtm = worker._id; 
    goto explore_task;
  }

  if(_stop) {
    //_notifier.cancel_wait(worker._waiter);
    //_notifier.notify(true);
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
  return true;

  //goto explore_task;
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

  //for(auto& st: _streams) {
    //checkCudaError(cudaStreamSynchronize(st.st));
  //}
  //checkCudaError(cudaDeviceSynchronize());
}

void TaroCBV3::schedule() {

  std::vector<Task*> srcs;
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      srcs.push_back(t.get());
    }
  }

  _enqueue(srcs);
  _notifier.notify(srcs.size());
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

auto TaroCBV3::suspend() {
  struct awaiter: std::suspend_always {
    TaroCBV3* _cf;
    explicit awaiter(TaroCBV3* cf) noexcept : _cf{cf} {}
    void await_suspend(std::coroutine_handle<Coro::promise_type> coro_handle) const noexcept {
      auto id = coro_handle.promise()._id;
      _cf->_enqueue(*(_cf->_this_worker()), _cf->_tasks[id].get());
      _cf->_notifier.notify(false);
    }
  };

  return awaiter{this};
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

void TaroCBV3::_enqueue(Worker& worker, Task* tp) {
  {
    std::scoped_lock lock(_mtxes[worker._id]);
    worker._que.push(tp);
  }
}

void TaroCBV3::_enqueue(Worker& worker, const std::vector<Task*>& tps) {
  {
    std::scoped_lock lock(_mtxes[worker._id]);

    for(auto* tp: tps) {
      worker._que.push(tp);
    }
  }
}

void TaroCBV3::_enqueue(Task* tp) {
  {
    std::scoped_lock lock(_qmtx);
    _que.push(tp);
  }
}

void TaroCBV3::_enqueue(const std::vector<Task*>& tps) {
  {
    std::scoped_lock lock(_qmtx);
    for(auto* tp: tps) {
      _que.push(tp);
    }
  }
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
