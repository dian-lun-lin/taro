#pragma once
#include "wsq.hpp"

namespace taro { // begin of namespace taro ===================================

// ==========================================================================
//
// Declaration/Definition of class Worker
//
// ==========================================================================

class Worker {

  friend class TaroCBV3;

  friend void CUDART_CB _cuda_stream_callback_v3(void* void_args);

  public:

    size_t get_id() const { return _id; }

  private:

    WorkStealingQueue<Task*> _que;
    
    std::jthread* _thread;
    size_t _id;
    std::mutex _mtx;
    std::binary_semaphore _wait_task{0};
};

} // end of namespace taro ==============================================
