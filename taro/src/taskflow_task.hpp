#pragma once

#include "coro.hpp"

namespace taro { // begin of namespace taro ===================================

// ==========================================================================
//
// TaskflowTask Traits
//
// ==========================================================================

class cudaWorker;
class Worker;

template <typename C>
constexpr bool is_static_task_v = 
  std::is_invocable_r_v<void, C> &&

template <typename C>
constexpr bool is_coro_task_v = 
  std::is_invocable_r_v<Coro, C>;
  //!std::is_invocable_r_v<void, C>; // TODO: why we cannot add this line?

template <typename T, typename>
struct get_index;

template <size_t I, typename... Ts>
struct get_index_impl {};

template <size_t I, typename T, typename... Ts>
struct get_index_impl<I, T, T, Ts...> : std::integral_constant<size_t, I>{};

template <size_t I, typename T, typename U, typename... Ts>
struct get_index_impl<I, T, U, Ts...> : get_index_impl<I+1, T, Ts...>{};

template <typename T, typename... Ts>
struct get_index<T, std::variant<Ts...>> : get_index_impl<0, T, Ts...>{};

template <typename T, typename... Ts>
constexpr auto get_index_v = get_index<T, Ts...>::value;

// ==========================================================================
//
// Decalartion of class TaskflowTask
//
// TaskflowTask stores a coroutine and handles dependencies of the task graph
// ==========================================================================

class TaskflowTask {

  friend class TaroCBTaskflowRuntime;
  friend class TaskflowTaskHandle;

  struct CoroTask {
    template <typename C>
    CoroTask(C&&);

    Coro coro;
    void resume() {
      coro._resume();
    }

    bool done() {
      return coro._done();
    }
  };

  struct StaticTask {
  };

  using handle_t = std::variant<
    std::monostate,
    CoroTask,
    StaticTask
  >;

  public:

    template <typename... Args>
    TaskflowTask(size_t id, Args&&... args);


    TaskflowTask() = default;
    ~TaskflowTask() = default;

    TaskflowTask(TaskflowTask&& rhs) = delete;
    TaskflowTask& operator=(TaskflowTask&& rhs) = delete;
    TaskflowTask(const TaskflowTask&) = delete;
    TaskflowTask& operator=(const TaskflowTask&) = delete;

    constexpr static auto PLACEHOLDER   = get_index_v<std::monostate, handle_t>;
    constexpr static auto COROTASK   = get_index_v<CoroTask, handle_t>;
    constexpr static auto STATICTASK = get_index_v<StaticTask, handle_t>;

  private:

    void _precede(TaskflowTask* task);
    std::vector<TaskflowTask*> _succs;
    std::vector<TaskflowTask*> _preds;
    std::atomic<int> _join_counter{0};
    size_t _id;

    handle_t _handle;

    // for taro_poll_v2
    size_t _poll_times{0};
};

template <typename C>
TaskflowTask::StaticTask::StaticTask() {
}

template <typename C>
TaskflowTask::CoroTask::CoroTask(C&& c): 
  coro{c()}
{
}

// ==========================================================================
//
// Definition of class TaskflowTask
//
// ==========================================================================

template <typename... Args>
TaskflowTask::TaskflowTask(size_t id, Args&&... args):_id{id}, _handle{std::forward<Args>(args)...} {
}

void TaskflowTask::_precede(TaskflowTask* tp) {
  _succs.push_back(tp);
  tp->_preds.push_back(this);
  tp->_join_counter.fetch_add(1, std::memory_order_relaxed);
}

// ==========================================================================
//
// Decalartion of class TaskflowTaskHandle
//
// ==========================================================================

class TaskflowTaskHandle {

  public:

    TaskflowTaskHandle();
    explicit TaskflowTaskHandle(TaskflowTask* tp);
    TaskflowTaskHandle(TaskflowTaskHandle&&) = default;
    TaskflowTaskHandle(const TaskflowTaskHandle&) = default;
    TaskflowTaskHandle& operator = (const TaskflowTaskHandle&) = default;
    TaskflowTaskHandle& operator = (TaskflowTaskHandle&&) = default;
    ~TaskflowTaskHandle() = default;    

    TaskflowTaskHandle& precede(TaskflowTaskHandle ch);

    TaskflowTaskHandle& succeed(TaskflowTaskHandle ch);

  private:

    TaskflowTask* _tp;
};

// ==========================================================================
//
// Definition of class TaskflowTaskHandle
//
// ==========================================================================
//
TaskflowTaskHandle::TaskflowTaskHandle(): _tp{nullptr} {
}

TaskflowTaskHandle::TaskflowTaskHandle(TaskflowTask* tp): _tp{tp} {
}

TaskflowTaskHandle& TaskflowTaskHandle::precede(TaskflowTaskHandle ch) {
  _tp->_precede(ch._tp);
  return *this;
}

TaskflowTaskHandle& TaskflowTaskHandle::succeed(TaskflowTaskHandle ch) {
  ch._tp->_precede(_tp);
  return *this;
}


} // end of namespace taro ==============================================
