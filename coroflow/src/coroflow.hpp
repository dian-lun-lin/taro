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

    TaskHandle emplace(Coro&&);

    void schedule();

    void wait();

  private:

    void _process(Task* tp);
    void _enqueue(Task* tp);

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

TaskHandle Coroflow::emplace(Coro&& coro) {
  _tasks.emplace_back(std::make_unique<Task>(std::move(coro)));

  return TaskHandle{_tasks.back().get()};
}

void Coroflow::schedule() {
  for(auto& t: _tasks) {
    if(t->_join_counter.load() == 0) {
      _enqueue(t.get());
    }
  }
}

void Coroflow::_process(Task* tp) {

  if(!tp->_done()) {
    tp->_resume();
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

void Coroflow::_enqueue(Task* tp) {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _queue.push(tp);
  }
  _cv.notify_one();
}

} // end of namespace cf ==============================================
