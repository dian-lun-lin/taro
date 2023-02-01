#pragma once

namespace cf { // begin of namespace cf ===================================

class ThreadPool {

  friend class Scheduler;

  public:

    ThreadPool(size_t num_workers);
    ~ThreadPool();


};



} // end of namespace coro ==============================================
