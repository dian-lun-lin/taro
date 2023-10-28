#pragma once

#include <semaphore>
#include "../core/taro.hpp"

namespace taro { // begin of namespace taro ===================================

// ==========================================================================
//
// Declaration/Definition of class Semaphore
//
// ==========================================================================

template <size_t V>
class Semaphore {

  friend class SemaphoreAwait<V>;


  private:

    std::queue<std::pair<Worker*, size_t>> _waiters; // worker and task id
    std::counting_semaphore<V> _sema{V};
    std::mutex _mtx;

    bool try_acquire() {
      return _sema.try_acquire();
    }

    void release() {
      _sema.release();
    }

};

// ==========================================================================
//
// Declaration of class SemaphoreAwait
//
// ==========================================================================

template <size_t V>
class SemaphoreAwait {

  public:

    SemaphoreAwait(Taro& taro, size_t num_semaphores);

    void release(size_t sid);
  
    auto acquire(size_t sid);

  private:

    Taro& _taro;
    std::vector<Semaphore<V>> _semaphores;
};

// ==========================================================================
//
// Definition of class SemaphoreAwait
//
// ==========================================================================

template <size_t V>
SemaphoreAwait<V>::SemaphoreAwait(Taro& taro, size_t num_semaphores): _taro{taro}, _semaphores{num_semaphores} {
}

template <size_t V>
auto SemaphoreAwait<V>::acquire(size_t sid) {
  struct semaphore_awaiter {
    SemaphoreAwait& semaphores;
    size_t sid;
    semaphore_awaiter(SemaphoreAwait& semaphores, size_t sid): semaphores{semaphores}, sid{sid} {}

    bool await_ready() const noexcept {
      return semaphores._semaphores[sid].try_acquire();
    }

    bool await_suspend(std::coroutine_handle<>) {
      std::scoped_lock lock{semaphores._semaphores[sid]._mtx};
      if(!semaphores._semaphores[sid].try_acquire()) {
        semaphores._semaphores[sid]._waiters.emplace(semaphores._taro._this_worker(), semaphores._taro._this_worker()->_work_on_task_id);
        return true;
      }
      return false;
    }

    void await_resume() noexcept {
    }
    
  };

  return semaphore_awaiter{*this, sid};
}

template <size_t V>
void SemaphoreAwait<V>::release(size_t sid) {
  std::scoped_lock lock{_semaphores[sid]._mtx};

  if(!_semaphores[sid]._waiters.empty()) {
    // if not empty, directly enqueue a waiter without release;
    Worker* worker = _semaphores[sid]._waiters.front().first;
    size_t task_id = _semaphores[sid]._waiters.front().second;
    _semaphores[sid]._waiters.pop();
    _taro._enqueue_back(*worker, task_id);
    return;
  }

  _semaphores[sid].release();
  return;
}



// ==========================================================================
//
// Definition of event_await in Taro
//
// ==========================================================================

template <size_t V>
SemaphoreAwait<V> Taro::semaphore_await(size_t num_semaphores) {
  return SemaphoreAwait<V>(*this, num_semaphores);
}

} // end of namespace taro ==============================================
