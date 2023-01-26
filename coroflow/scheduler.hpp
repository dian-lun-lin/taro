#pragma once

#include "thread_pool.hpp"

namespace cf { // begin of namespace cf ===================================

class Scheduler {

  public:

    Scheduler(size_t num_threads = 4): _pool{num_threads} {}

    bool schedule() {
      _schedule();
      return !_pool._jobs.empty();
    }


    auto suspend() {
      struct awaiter: public std::suspend_always {
        Scheduler& sched;
        explicit awaiter(Scheduler& sched): sched{sched} {}
        void await_suspend(std::coroutine_handle<> coro) const noexcept { sched._tasks.push_back(coro); }
      };
      return awaiter{*this};
    }

  private:

    std::list<std::coroutine_handle<>> _tasks{};
    ThreadPool _pool;

    void _schedule() {
      if(!_tasks.empty()) {
        auto task = _tasks.front();
        _tasks.pop_front();
        if(!task.done()) { _pool.enqueue([task](){ task.resume(); }); }
      }
    }

};

} // end of namespace coro ==============================================
