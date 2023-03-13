#pragma once

#include "../../3rd-party/taskflow/notifier.hpp"
#include "../../3rd-party/taskflow/wsq.hpp"

namespace taro { // begin of namespace taro ===================================


class cudaWorker {
  friend class Worker;
  friend class TaroPV1;

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

  public:

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

};

} // end of namespace taro ==============================================
