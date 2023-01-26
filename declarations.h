#pragma once

#include <iostream>
#include <list>
#include <coroutine>
#include <mutex>
#include <thread>
#include <functional>
#include <future>
#include <vector>
#include <queue>


namespace cf { // begin of namespace cf ===================================

  class Scheduler;
  class ThreadPool;
  struct Task;

} // end of namespace coro ==============================================
