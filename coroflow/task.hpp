#pragma once

namespace cf { // begin of namespace cf ===================================

struct Task {


  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
  };

};

} // end of namespace cf ==============================================
