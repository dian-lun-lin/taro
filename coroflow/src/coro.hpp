#pragma once

namespace cf { // begin of namespace cf ===================================

// ==========================================================================
//
// Decalartion of class Coro
//
// ==========================================================================

struct Coro { // Coroutine needs to be struct

  friend class Coroflow;
  friend class Task;

  public:

    struct promise_type {

      size_t _id;
      Coro get_return_object() { return Coro{this}; }

      std::suspend_always initial_suspend() noexcept { return {}; } // suspend a coroutine now and schedule it after
      std::suspend_always final_suspend() noexcept { return {}; } // suspend to decrement dependencies for a task graph
                                                                  // otherwise we don't know whether a coroutine is finished
      void unhandled_exception() {}
      void return_void() noexcept {}
    };

    explicit Coro(promise_type* p);
    Coro(Coro&& rhs);
    ~Coro();

  private:

    void _resume();
    bool _done();

    std::coroutine_handle<promise_type> _coro_handle;
};

// ==========================================================================
//
// Definition of class Coro
//
// ==========================================================================

Coro::Coro(promise_type* p): _coro_handle{std::coroutine_handle<promise_type>::from_promise(*p)} {
}

Coro::Coro(Coro&& rhs): _coro_handle{std::exchange(rhs._coro_handle, nullptr)} {
}

Coro::~Coro() { 
  if(_coro_handle) { 
    _coro_handle.destroy(); 
  }
}

void Coro::_resume() {
  _coro_handle.resume();
}

bool Coro::_done() {
  return _coro_handle.done();
}

} // end of namespace cf ==============================================
