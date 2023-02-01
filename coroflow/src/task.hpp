#pragma once

namespace cf { // begin of namespace cf ===================================

// ==========================================================================
//
// Decalartion of class Task
//
// Task stores a coroutine and handles dependencies of the task graph
// ==========================================================================

class Task {

  friend class Coroflow;
  friend class TaskHandle;

  public:

    Task(Coro&& coro);

  private:

    void _precede(Task* task);
    void _resume();
    bool _done();
    std::vector<Task*> _succs;
    std::vector<Task*> _preds;
    std::atomic<int> _join_counter{0};
    Coro _coro;
};

// ==========================================================================
//
// Definition of class Task
//
// ==========================================================================

Task::Task(Coro&& coro): _coro{std::move(coro)} {
}

void Task::_precede(Task* tp) {
  _preds.push_back(tp);
  tp->_succs.push_back(this);
  _join_counter.fetch_add(1, std::memory_order_relaxed);
}

void Task::_resume() {
  _coro._resume();
}

bool Task::_done() {
  return _coro._done();
}

// ==========================================================================
//
// Decalartion of class TaskHandle
//
// ==========================================================================

class TaskHandle {

  public:

    TaskHandle();
    explicit TaskHandle(Task* tp);
    TaskHandle(TaskHandle&&) = default;
    TaskHandle(const TaskHandle&) = default;
    TaskHandle& operator = (const TaskHandle&) = default;
    TaskHandle& operator = (TaskHandle&&) = default;
    ~TaskHandle() = default;    

    TaskHandle& precede(TaskHandle ch);

    TaskHandle& succeed(TaskHandle ch);

  private:

    Task* _tp;
};

// ==========================================================================
//
// Definition of class TaskHandle
//
// ==========================================================================
//
TaskHandle::TaskHandle(): _tp{nullptr} {
}

TaskHandle::TaskHandle(Task* tp): _tp{tp} {
}

TaskHandle& TaskHandle::precede(TaskHandle ch) {
  _tp->_precede(ch._tp);
  return *this;
}

TaskHandle& TaskHandle::succeed(TaskHandle ch) {
  ch._tp->_precede(_tp);
  return *this;
}


} // end of namespace cf ==============================================
