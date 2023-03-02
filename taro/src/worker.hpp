#pragma once

#include "../../3rd-party/taskflow/notifier.hpp"
#include "../../3rd-party/taskflow/wsq.hpp"

namespace taro { // begin of namespace taro ===================================


class cudaWorker {
  friend class Worker;
  friend class TaroV7;

  cudaStream_t st;
  Task* cur_task;

  public:

    cudaWorker(): cur_task{nullptr} {
      //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
      cudaStreamCreate(&st);
    }

    ~cudaWorker() {
      cudaStreamSynchronize(st);
      cudaStreamDestroy(st);
    }

    cudaWorker(const cudaWorker&) = default;
    cudaWorker(cudaWorker&&) = default;
    cudaWorker& operator = (cudaWorker&&) = default;
    cudaWorker& operator = (const cudaWorker&) = default;
};

// ==========================================================================
//
// Declaration/Definition of class Worker
//
// ==========================================================================

class Worker {


  friend class TaroV4;
  friend class TaroV5;
  friend class TaroV6;
  friend class TaroV7;

  public:

  private:

    WorkStealingQueue<Task*> _que;
    WorkStealingQueue<std::function<void(cudaWorker&)>*> _gque;

    std::vector<cudaWorker> _gws;

    Notifier::Waiter* _waiter;
    std::thread* _thread;
    size_t _id;
    size_t _vtm;
    std::default_random_engine _rdgen{std::random_device{}()};

};

} // end of namespace taro ==============================================
