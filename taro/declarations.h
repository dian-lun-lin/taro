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
#include <cassert>
#include <variant>
#include <taskflow/wsq.hpp>
#include <random>


namespace taro { // begin of namespace taro ===================================

  class Coro;
  class Task;
  class TaskHandle;

  //enum class State {
    //SUSPEND,
    //CONTINUE
  //};


} // end of namespace taro ==============================================
