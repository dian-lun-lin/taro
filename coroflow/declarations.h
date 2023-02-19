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
#include <stack>
#include <type_traits>
#include <variant>
#include <taskflow/wsq.hpp>
#include <random>


namespace cf { // begin of namespace cf ===================================

  class Coroflow;
  class Coro;
  class Task;
  class TaskHandle;

  //enum class State {
    //SUSPEND,
    //CONTINUE
  //};


} // end of namespace cf ==============================================
