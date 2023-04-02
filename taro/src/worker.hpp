#pragma once

namespace taro { // begin of namespace taro ===================================


class cudaWorker {
  friend class Worker;
  friend class TaroPV1;
  friend class TaroPV2;

  cudaStream_t stream;
  Task* cur_task;
  // -1: busy
  // 0: (busy -> available) => need to commit
  // 1:  available
  int status{1};

  public:

    cudaWorker(): cur_task{nullptr} {
      //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~cudaWorker() {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }

    //cudaWorker(const cudaWorker&) = delete;
    //cudaWorker(cudaWorker&&) = delete;
    //cudaWorker& operator = (cudaWorker&&) = delete;
    //cudaWorker& operator = (const cudaWorker&) = delete;
};

// ==========================================================================
//
// Declaration/Definition of class Worker
//
// ==========================================================================

class Worker {

  friend class TaroCBV1;
  friend class TaroCBV2;
  friend class TaroCBV3;
  friend class TaroPV1;
  friend class TaroPV2;

  friend void CUDART_CB _cuda_stream_callback_v3(void* void_args);

  // for TaroCBV3
  enum class Stage: unsigned {
    INIT,
    EXPLORE,
    EXPLOIT,
    SLEEP
  };

  public:

    size_t get_id() const { return _id; }

  private:

    WorkStealingQueue<Task*> _que;
    
    // for TaroPV1
    std::vector<cudaWorker> _gws;

    // for TaroCBV2
    WorkStealingQueue<cudaStream_t> _sque;

    Notifier::Waiter* _waiter;
    std::thread* _thread;
    size_t _id;
    size_t _vtm;
    std::default_random_engine _rdgen{std::random_device{}()};

    // for TaroCBV3
    std::condition_variable _cv;
    std::atomic<Stage> _stage;

    // for TaroCBV3
    void _update_stage(Stage s) { _stage.store(s, std::memory_order_relaxed; }

};

} // end of namespace taro ==============================================
