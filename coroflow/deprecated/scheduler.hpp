#pragma once

namespace cf { // begin of namespace cf ===================================

// ==========================================================================
//
// Decalartion of class Scheduler
//
// ==========================================================================

class Scheduler {

  public:

    Scheduler(size_t num_workers = 4);

    ~Scheduler();

    auto suspend() {
      struct awaiter: public std::suspend_always {
        Scheduler& sched;
        explicit awaiter(Scheduler& sched): sched{sched} {}
        void await_suspend(std::coroutine_handle<> coro) const noexcept { 
    
        }
      };
      return awaiter{*this};
    }

    void schedule(Coroflow& cf);

  private:

    std::vector<std::thread> _workers;

    std::mutex _coro_mutex;
    std::condition_variable _cv;
    bool _stop{false};

    std::list<std::coroutine_handle<>> _coros{};

    void _enqueue(std::coroutine_handle<> coro) noexcept;


};

// ==========================================================================
//
// Definition of class Scheduler
//
// ==========================================================================

inline
Scheduler::Scheduler(size_t num_workers): _workers{num_workers} {
  _workers.reserve(num_workers);
  for(size_t i = 0; i < num_workers; ++i){
    _workers.emplace_back(
      [this] {
        while(true){
          std::unique_lock<std::mutex> lock(_coro_mutex);
          _cv.wait(lock, [this]{return this->_stop || (!this->_coros.empty());});
          if(_stop && _coros.empty()){
            return;
          }
          auto coro = _coros.front();
          _coros.pop_front();
          lock.unlock();
          coro.resume();
        }
      }
    );
  }
}

inline
Scheduler::~Scheduler() {
  {
    std::unique_lock<std::mutex> lock(_coro_mutex);
    _stop = true;
  }
  _cv.notify_all();
  for(auto &worker:_workers){
    worker.join();
  }

}

inline
void Scheduler::_enqueue(std::coroutine_handle<> coro) noexcept {
  std::unique_lock<std::mutex> lock(_coro_mutex);
  _coros.emplace_back(coro);
  _cv.notify_one();
}

} // end of namespace cf ==============================================
