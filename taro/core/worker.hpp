#pragma once
#include "wsq.hpp"
#include "task.hpp"

namespace taro { // begin of namespace taro ===================================

// ==========================================================================
//
// Declaration/Definition of class Worker
//
// ==========================================================================

class Worker {

  friend class Taro;
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
  friend void _async_io_consume(void* void_args);

  public:

    size_t get_id() const { return _id; }

  private:

    WorkStealingQueue<Task*> _que;
    
    std::jthread* _thread;
    size_t _id;
    std::mutex _mtx;
    size_t _work_on_task_id;

    enum STAT: unsigned {
      SLEEP,
      BUSY,
      SIGNALED // 
    };
    std::atomic<unsigned> _status{STAT::SLEEP};

};

} // end of namespace taro ==============================================
