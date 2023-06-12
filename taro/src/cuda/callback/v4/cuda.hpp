#pragma once

#include "taro_callback_v4.hpp"
#include "coro.hpp"

namespace taro { // begin of namespace taro ===================================

// TODO cuda event recycle

class cudaScheduler {

  friend class TaroCBV4;
  friend void _cuda_callback(void* void_args);
  friend void _cuda_polling(void* void_args);

  struct cudaCallbackData {
    cudaScheduler* cuda;
    Worker* worker;
    size_t task_id;
    size_t stream_id;
  };
  struct cudaPollingData {
    cudaScheduler* cuda;
    Worker* worker;
    size_t task_id;
    size_t stream_id;
    std::unique_ptr<Task> ptask;
    cudaEvent_t event;
  };

  struct cudaStream{
    size_t id;
    cudaStream_t st;
  };

  friend Coro _cuda_polling_query(cudaPollingData&);

  public:

    cudaScheduler(TaroCBV4& taro, size_t num_streams);
    ~cudaScheduler();

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto suspend_polling(C&&);

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto suspend_callback(C&&);

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto wait(C&&);
    

  private:

    size_t _acquire_stream();
    void _release_stream(size_t stream_id);

    std::vector<cudaStream> _streams;
    std::queue<cudaPollingData> _pevents;
    std::vector<size_t> _in_stream_tasks;
    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;
    TaroCBV4& _taro;
};

cudaScheduler::cudaScheduler(TaroCBV4& taro, size_t num_streams): _taro{taro} {

  //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  _in_stream_tasks.resize(num_streams, 0);
  _streams.resize(num_streams);
  for(size_t i = 0; i < num_streams; ++i) {
    _streams[i].id = i;
    cudaStreamCreateWithFlags(&(_streams[i].st), cudaStreamNonBlocking);
  }
}

cudaScheduler::~cudaScheduler() {
  for(auto& st: _streams) {
    cudaStreamDestroy(st.st);
  }
}

// ==========================================================================
//
// callback
//
// ==========================================================================

// cuda callback
inline
void _cuda_callback(void* void_args) {

  // unpack
  auto* data = (cudaScheduler::cudaCallbackData*) void_args;
  auto* cuda = data->cuda;
  auto& taro = cuda->_taro;
  auto* worker = data->worker;
  size_t task_id = data->task_id;
  size_t stream_id = data->stream_id;
  
  cuda->_release_stream(stream_id);

  {
    // high priortiy queue is owned by the callback function
    // Due to CUDA runtime, we cannot guarntee whether the cuda callback function is called sequentially
    // we need a lock to atomically enqueue the task
    // note that there is no lock in enqueue functions
    
    std::scoped_lock lock(worker->_mtx);
    taro._enqueue(*worker, taro._tasks[task_id].get(), TaskPriority::HIGH);
  }

  //worker->_status.store(Worker::STAT::SIGNALED);
  if(worker->_status.exchange(Worker::STAT::SIGNALED) == Worker::STAT::SLEEP) {
    worker->_status.notify_one();
  }
  taro._notify(*worker);

  taro._cb_cnt.fetch_sub(1);
}

inline
void _cuda_polling(void* void_args) {

  // unpack
  auto* data = (cudaScheduler::cudaPollingData*) void_args;
  auto* cuda = data->cuda;
  auto& taro = cuda->_taro;
  auto* worker = data->worker;
  size_t task_id = data->task_id;
  size_t stream_id = data->stream_id;
  
  cuda->_release_stream(stream_id);

  {
    // high priortiy queue is owned by the callback function
    // Due to CUDA runtime, we cannot guarntee whether the cuda callback function is called sequentially
    // we need a lock to atomically enqueue the task
    // note that there is no lock in enqueue functions
    
    std::scoped_lock lock(worker->_mtx);
    taro._enqueue(*worker, taro._tasks[task_id].get(), TaskPriority::HIGH);
  }

  //worker->_status.store(Worker::STAT::SIGNALED);
  if(worker->_status.exchange(Worker::STAT::SIGNALED) == Worker::STAT::SLEEP) {
    worker->_status.notify_one();
  }
  taro._notify(*worker);
}

Coro _cuda_polling_query(cudaScheduler::cudaPollingData& data) {
  while(cudaEventQuery(data.event) != cudaSuccess) {
    co_await data.cuda->_taro.suspend(data.ptask.get());
  }
  cudaEventDestroy(data.event);
  _cuda_polling((void*)&data);
}


// ==========================================================================
//
// Definition of class cudaScheduler
//
// ==========================================================================

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto cudaScheduler::suspend_polling(C&& c) {

  struct cuda_awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaPollingData data;

    explicit cuda_awaiter(cudaScheduler* cuda, C&& c) noexcept : kernel{std::forward<C>(c)} {
      data.cuda = cuda;
    }
    
    void await_suspend(std::coroutine_handle<>) {
      // choose the best stream id
      size_t stream_id = data.cuda->_acquire_stream();

      cudaEvent_t e;
      cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

      // set polling data
      data.worker = data.cuda->_taro._this_worker();
      data.task_id = data.worker->_work_on_task_id;
      data.stream_id = stream_id;
      data.event = e;

      {
        std::scoped_lock lock(data.cuda->_kernel_mtx);
        kernel(data.cuda->_streams[stream_id].st);
        cudaEventRecord(e, data.cuda->_streams[stream_id].st);
      }

      // TODO: meaning of task id
      // TODO: don't know why I cannot use lambda of coroutine here...
      data.ptask  = std::make_unique<Task>(-1, std::in_place_type_t<Task::CoroTask>{}, std::bind(_cuda_polling_query, std::ref(data)));
      std::get_if<Task::CoroTask>(&data.ptask.get()->_handle)->set_inner();
      
      data.cuda->_taro._enqueue(*data.worker, data.ptask.get(), TaskPriority::LOW);

      return;
    }
  };

  return cuda_awaiter{this, std::forward<C>(c)};
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto cudaScheduler::wait(C&& c) {
  // choose the best stream id
  size_t stream_id = _acquire_stream();
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  c(stream_id);
  cudaEventRecord(e, _streams[stream_id].st);
  cudaEventSynchronize(e);
  _release_stream(stream_id);
  cudaEventDestroy(e);
}

size_t cudaScheduler::_acquire_stream() {
  size_t stream_id;
  {
    std::scoped_lock lock(_stream_mtx);
    stream_id = std::distance(
      _in_stream_tasks.begin(), 
      std::min_element(_in_stream_tasks.begin(), _in_stream_tasks.end())
    );
    ++_in_stream_tasks[stream_id];
  }
  return stream_id;
}

void cudaScheduler::_release_stream(size_t stream_id) {
  {
    std::scoped_lock lock(_stream_mtx);
    --_in_stream_tasks[stream_id];
  }
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto cudaScheduler::suspend_callback(C&& c) {

  struct cuda_awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaCallbackData data;

    explicit cuda_awaiter(cudaScheduler* cuda, C&& c) noexcept : kernel{std::forward<C>(c)} {
      data.cuda = cuda; 
    }
    
    void await_suspend(std::coroutine_handle<>) {
      // choose the best stream id
      size_t stream_id = data.cuda->_acquire_stream();
      // set callback data
      data.worker = data.cuda->_taro._this_worker();
      data.task_id = data.worker->_work_on_task_id;
      data.stream_id = stream_id;

      // enqueue the kernel to the stream
      data.cuda->_taro._cb_cnt.fetch_add(1);
      {
        // TODO: is cudaLaunchHostFunc thread safe?
        std::scoped_lock lock(data.cuda->_kernel_mtx);
        kernel(data.cuda->_streams[stream_id].st);
        cudaLaunchHostFunc(data.cuda->_streams[stream_id].st, _cuda_callback, (void*)&data);
      }

      // TODO: maybe the kernel is super fast
      // there can be a last chance to not suspend
      return;
    }
  };

  return cuda_awaiter{this, std::forward<C>(c)};
}

// ==========================================================================
//
// Definition of cuda_scheduler in TaroCBV4
//
// ==========================================================================

inline
cudaScheduler TaroCBV4::cuda_scheduler(size_t num_streams) {
  return cudaScheduler(*this, num_streams);
}


} // end of namespace taro ==============================================
