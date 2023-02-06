#pragma once

namespace cf { // begin of namespace cf ===================================

// ==========================================================================
//
// Declaration of class Coroflow
//
// ==========================================================================
//

class Coroflow {

  public:

    Coroflow(size_t num_threads);

    ~Coroflow();

    template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
    TaskHandle emplace(C&&);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
    TaskHandle emplace(C&&);

    void schedule();

    void wait();

  private:

    void _process(Task* tp);
    void _enqueue(Task* tp);

    void _invoke_coro_task(Task* tp);
    void _invoke_static_task(Task* tp);

    std::vector<std::thread> _workers;
    std::vector<std::unique_ptr<Task>> _tasks;
    std::queue<Task*> _queue;

    std::mutex _mtx;
    std::condition_variable _cv;
    bool _stop{false};
    std::atomic<size_t> _finished{0};
};

// ==========================================================================
//
// Definition of class Coroflow
//
// ==========================================================================

Coroflow::Coroflow(size_t num_threads) {
  _workers.reserve(num_threads);

  for(size_t t = 0; t < num_threads; ++t) {
    _workers.emplace_back([this]() {
        while(true) {
          std::unique_lock<std::mutex> lock(_mtx);
          _cv.wait(lock, [this]{ return this->_stop || (!this->_queue.empty()); });
          if(_stop) {
            return;
          }

          auto tp = _queue.front();
          _queue.pop();
          lock.unlock();
          _process(tp);
        }
      }
    );
  }
}

Coroflow::~Coroflow() {
  for(auto& w: _workers) {
    w.join();
  } 
}

void Coroflow::wait() {
  for(auto& w: _workers) {
    w.join();
  } 
  _workers.clear();
}

template <typename C, std::enable_if_t<is_static_task_v<C>, void>*>
TaskHandle Coroflow::emplace(C&& c) {
  auto t = std::make_unique<Task>(std::in_place_type_t<Task::StaticTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
TaskHandle Coroflow::emplace(C&& c) {
  auto t = std::make_unique<Task>(std::in_place_type_t<Task::CoroTask>{}, std::forward<C>(c));
  _tasks.emplace_back(std::move(t));
  return TaskHandle{_tasks.back().get()};
}

void Coroflow::schedule() {
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      _enqueue(t.get());
    }
  }
}

void Coroflow::_invoke_static_task(Task* tp) {
  std::get_if<Task::StaticTask>(&tp->_handle)->work();
  for(auto succp: tp->_succs) {
    if(succp->_join_counter.fetch_sub(1) == 1) {
      _enqueue(succp);
    }
  }

  if(_finished.fetch_add(1) + 1 == _tasks.size()) {
    _stop = true;
    _cv.notify_all();
  }
}

void Coroflow::_invoke_coro_task(Task* tp) {
  auto* coro = std::get_if<Task::CoroTask>(&tp->_handle);
  if(!coro->done()) {
    coro->resume();
    _enqueue(tp);
  }
  else {
    for(auto succp: tp->_succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        _enqueue(succp);
      }
    }

    if(_finished.fetch_add(1) + 1 == _tasks.size()) {
      _stop = true;
      _cv.notify_all();
    }
  }
}

void Coroflow::_process(Task* tp) {

  switch(tp->_handle.index()) {
    case Task::STATICTASK: {
      _invoke_static_task(tp);
    }
    break;

    case Task::COROTASK: {
      _invoke_coro_task(tp);
    }
    break;
  }
}

void Coroflow::_enqueue(Task* tp) {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _queue.push(tp);
  }
  _cv.notify_one();
}

} // end of namespace cf ==============================================
