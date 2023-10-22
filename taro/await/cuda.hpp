#pragma once

#include "../core/taro.hpp"
#include "../utility/cuda.hpp"

namespace taro { // begin of namespace taro ===================================

// TODO cuda event recycle

class cudaAwait {

  struct cudaCallbackData {
    cudaAwait* cuda;
    Worker* worker;
    size_t task_id;
    size_t stream_id;
  };
  struct cudaPollingData {
    cudaAwait* cuda;
    Task* ptask; // polling task
    Worker* worker;
    size_t task_id;
    size_t stream_id;
    cudaEvent_t event;
  };

  struct cudaStream{
    size_t id;
    cudaStream_t st;
  };

  friend class Taro;
  friend void _cuda_callback(void* void_args);
  friend void _cuda_polling(void* void_args);
  friend Coro _cuda_polling_query(cudaPollingData&);

  public:

    cudaAwait(Taro& taro, size_t num_streams);
    ~cudaAwait();

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto until_polling(C&&);

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto until_callback(C&&);

    template <typename C, std::enable_if_t<is_kernel_v<C>, void>* = nullptr>
    auto wait(C&&);
    

  private:

    size_t _acquire_stream();
    void _release_stream(size_t stream_id);

    std::vector<cudaStream> _streams;
    std::vector<std::unique_ptr<Task>> _ptasks;
    std::vector<size_t> _in_stream_tasks;
    std::mutex _stream_mtx;
    std::mutex _kernel_mtx;
    Taro& _taro;
};

inline
cudaAwait::cudaAwait(Taro& taro, size_t num_streams): _taro{taro} {

  //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  _in_stream_tasks.resize(num_streams, 0);
  _streams.resize(num_streams);
  for(size_t i = 0; i < num_streams; ++i) {
    _streams[i].id = i;
    cudaStreamCreateWithFlags(&(_streams[i].st), cudaStreamNonBlocking);
  }
}

inline
cudaAwait::~cudaAwait() {
  for(auto& st: _streams) {
    cudaStreamDestroy(st.st);
  }
}

// ==========================================================================
//
// callback and polling
//
// ==========================================================================

// cuda callback
inline
void _cuda_callback(void* void_args) {

  // unpack
  auto* data = (cudaAwait::cudaCallbackData*) void_args;
  auto* cuda = data->cuda;
  auto& taro = cuda->_taro;
  auto* worker = data->worker;
  size_t task_id = data->task_id;
  size_t stream_id = data->stream_id;
  
  cuda->_release_stream(stream_id);

  taro._enqueue_back(*worker, task_id);
  taro._callback_polling_cnt.fetch_sub(1);
}

inline
void _cuda_polling(void* void_args) {

  // unpack
  auto* data = (cudaAwait::cudaPollingData*) void_args;
  auto* cuda = data->cuda;
  auto& taro = cuda->_taro;
  auto* worker = data->worker;
  size_t task_id = data->task_id;
  size_t stream_id = data->stream_id;
  
  cuda->_release_stream(stream_id);

  taro._enqueue_back(*worker, task_id);
}

Coro _cuda_polling_query(cudaAwait::cudaPollingData& data) {
  while(cudaEventQuery(data.event) != cudaSuccess) {
    co_await data.cuda->_taro.suspend(data.ptask);
  }
  cudaEventDestroy(data.event);
  _cuda_polling((void*)&data);
}


// ==========================================================================
//
// Definition of class cudaAwait
//
// ==========================================================================

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto cudaAwait::until_callback(C&& c) {

  struct cuda_awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaCallbackData data;

    explicit cuda_awaiter(cudaAwait* cuda, C&& c) noexcept : kernel{std::forward<C>(c)} {
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
      data.cuda->_taro._callback_polling_cnt.fetch_add(1);
      {
        // TODO: is cudaLaunchHostFunc thread safe?
        std::scoped_lock lock(data.cuda->_kernel_mtx);
        kernel(data.cuda->_streams[stream_id].st);
        cudaLaunchHostFunc(data.cuda->_streams[stream_id].st, _cuda_callback, (void*)&data);
      }

      return;
    }
  };

  return cuda_awaiter{this, std::forward<C>(c)};
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto cudaAwait::until_polling(C&& c) {

  struct cuda_awaiter: std::suspend_always {
    std::function<void(cudaStream_t)> kernel;
    cudaPollingData data;

    explicit cuda_awaiter(cudaAwait* cuda, C&& c) noexcept : kernel{std::forward<C>(c)} {
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
        data.ptask = data.cuda->_ptasks.emplace_back(
          std::make_unique<Task>(-1, std::in_place_type_t<Task::CoroTask>{}, std::bind(_cuda_polling_query, std::ref(data)))
        ).get();
      }

      std::get_if<Task::CoroTask>(&data.ptask->_handle)->set_inner();
      
      data.cuda->_taro._enqueue(*data.worker, data.ptask, TaskPriority::LOW);

      return;
    }
  };

  return cuda_awaiter{this, std::forward<C>(c)};
}

template <typename C, std::enable_if_t<is_kernel_v<C>, void>*>
auto cudaAwait::wait(C&& c) {
  // choose the best stream id
  size_t stream_id = _acquire_stream();
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  c(_streams[stream_id].st);
  cudaEventRecord(e, _streams[stream_id].st);
  cudaEventSynchronize(e);
  _release_stream(stream_id);
  cudaEventDestroy(e);
}

inline
size_t cudaAwait::_acquire_stream() {
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

inline
void cudaAwait::_release_stream(size_t stream_id) {
  {
    std::scoped_lock lock(_stream_mtx);
    --_in_stream_tasks[stream_id];
  }
}

// ==========================================================================
//
// Definition of cuda_await in Taro
//
// ==========================================================================

inline
cudaAwait Taro::cuda_await(size_t num_streams) {
  return cudaAwait(*this, num_streams);
}


} // end of namespace taro ==============================================
