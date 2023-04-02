#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <boost/assert.hpp>
#include <boost/fiber/all.hpp>
#include <boost/fiber/detail/thread_barrier.hpp>
#include <boost/fiber/cuda/waitfor.hpp>


class FiberTask;
class FiberTaskHandle;
class FiberTaskScheduler;

// ==========================================================================
//
// Decalartion of class FiberTask
//
// ==========================================================================

class FiberTask {

  
  friend class FiberTaskScheduler;
  friend class FiberTaskHandle;

  public:

    template <typename C>
    explicit FiberTask(FiberTaskScheduler& sched, C&& c);

    ~FiberTask() = default;

  private:

    void _precede(FiberTask* tp);
    std::atomic<int> _join_counter{0};

    std::vector<FiberTask*> _preds;
    std::vector<FiberTask*> _succs;
    size_t _id;
    std::function<void()> _work; 


    FiberTaskScheduler& _sched;
};

// ==========================================================================
//
// Declaration of class FiberTaskHandle
//
// ==========================================================================

class FiberTaskHandle {

  friend class FiberTaskScheduler;

  public:

    FiberTaskHandle();
    explicit FiberTaskHandle(FiberTask* tp);
    FiberTaskHandle(FiberTaskHandle&&) = default;
    FiberTaskHandle(const FiberTaskHandle&) = default;
    FiberTaskHandle& operator = (const FiberTaskHandle&) = default;
    FiberTaskHandle& operator = (FiberTaskHandle&&) = default;
    ~FiberTaskHandle() = default;    

    FiberTaskHandle& precede(FiberTaskHandle fth);

    FiberTaskHandle& succeed(FiberTaskHandle fth);

  private:

    FiberTask* _tp;
};

// ==========================================================================
//
// Declaration of class FiberTaskScheduler
//
// ==========================================================================

class FiberTaskScheduler {

  friend class FiberTask;

  public:

    FiberTaskScheduler(size_t num_threads);
    ~FiberTaskScheduler();
  
    template <typename C>
    FiberTaskHandle emplace(C&& c);

    void schedule();
    void wait();

  private:

  std::vector<FiberTask*> _tasks;
  std::atomic<size_t> _finished{0};
  std::vector<std::thread> _threads;
  size_t _num_threads;
  boost::fibers::condition_variable_any _cv;
  std::atomic<bool> _stop{false};
  std::mutex _mtx;
  
};

// ==========================================================================
//
// Definition of class FiberTask
//
// ==========================================================================

void FiberTask::_precede(FiberTask* tp) {
  _succs.push_back(tp);
  tp->_preds.push_back(this);
  tp->_join_counter.fetch_add(1, std::memory_order_relaxed);
}

template <typename C>
FiberTask::FiberTask(FiberTaskScheduler& sched, C&& c): _sched{sched} { 
  _work = [this, c=std::forward<C>(c)]() {
    c();

    for(auto succp: _succs) {
      if(succp->_join_counter.fetch_sub(1) == 1) {
        boost::fibers::fiber([succp](){
          succp->_work();
        }).detach();
      }
    }

    if(_sched._finished.fetch_add(1) + 1 == _sched._tasks.size()) {
      _sched._stop = true;
      _sched._cv.notify_all();
    }
  };
}

// ==========================================================================
//
// Definition of class FiberTaskHandle
//
// ==========================================================================

FiberTaskHandle::FiberTaskHandle(): _tp{nullptr} {
}

FiberTaskHandle::FiberTaskHandle(FiberTask* tp): _tp{tp} {
}

FiberTaskHandle& FiberTaskHandle::precede(FiberTaskHandle fth) {
  _tp->_precede(fth._tp);
  return *this;
}

FiberTaskHandle& FiberTaskHandle::succeed(FiberTaskHandle fth) {
  fth._tp->_precede(_tp);
  return *this;
}



// ==========================================================================
//
// Definition of class FiberTaskScheduler
//
// ==========================================================================

FiberTaskScheduler::FiberTaskScheduler(size_t num_threads): _num_threads{num_threads} {
  _threads.reserve(num_threads);
}

FiberTaskScheduler::~FiberTaskScheduler() {
  for(auto task: _tasks) {
    delete task;
  }
}

template <typename C>
FiberTaskHandle FiberTaskScheduler::emplace(C&& c) {
  _tasks.emplace_back(new FiberTask(*this, std::forward<C>(c)));
  _tasks.back()->_id = _tasks.size() - 1;
  return FiberTaskHandle{_tasks.back()};
}

void FiberTaskScheduler::schedule() {

  for(auto task: _tasks) {
    if(task->_join_counter.load(std::memory_order_relaxed) == 0) {
      boost::fibers::fiber([this, task](){
        task->_work();
      }).detach();
    }
  }


  for(size_t i = 0; i < _num_threads - 1; ++i) {
    _threads.emplace_back([this](){
      boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(_num_threads); 

      std::unique_lock<std::mutex> lock(_mtx);
      _cv.wait(lock, [this](){ return _stop.load(); } ); 
      BOOST_ASSERT(_stop.load());
    });
  }

  boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(_num_threads);
}

void FiberTaskScheduler::wait() {
  {
    std::unique_lock<std::mutex> lock(_mtx);
    _cv.wait(lock, [this](){ return _stop.load(); } ); 
  }
  BOOST_ASSERT(_stop.load());

  for(auto& t: _threads) {
    t.join();
  }
}
