#pragma once

#include "../../3rd-party/taskflow/notifier.hpp"
#include "../../3rd-party/taskflow/wsq.hpp"

namespace cf { // begin of namespace cf ===================================

// ==========================================================================
//
// Declaration/Definition of class Worker
//
// ==========================================================================

class Worker {

  friend class CoroflowV4;

  public:

  private:

    WorkStealingQueue<Task*> _que;
    Notifier::Waiter* _waiter;
    std::thread* _thread;
    size_t _id;
    size_t _vtm;
    std::default_random_engine _rdgen{std::random_device{}()};
};

} // end of namespace cf ==============================================
