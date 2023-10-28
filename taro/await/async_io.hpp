#pragma once

#include "../core/taro.hpp"
#include <liburing.h>

namespace taro { // begin of namespace taro ===================================

// TODO: single thread scheduling will hang
// as the polling task is always at the top

class AsyncIOAwait {

  struct AsyncIOData {
    AsyncIOAwait* async_io;
    size_t task_id;
    int status;
  };

  friend class Taro;
  friend void _async_io_consume(AsyncIOAwait& async_io);
  friend Coro _async_io_polling_query(AsyncIOAwait& async_io);

  public:

    AsyncIOAwait(Taro& taro, size_t queue_size);
    ~AsyncIOAwait();

    auto read(int fd, char* buf, size_t size);

    auto write(int fd, char* buf, size_t size);

    void finish();

  private:

    // Due to io_uring_for_each_cqe,
    // we only need one polling task to check if there is any asyncio work ready
    Task* _ptask;
    std::atomic<bool> _finish{false};
    io_uring _uring;
    Taro& _taro;

    // liburing is not thread safe
    std::mutex _mtx;
};


// ==========================================================================
//
// polling
//
// ==========================================================================

inline
void _async_io_consume(AsyncIOAwait& async_io) {

  int processed{0};
  io_uring_cqe* cqe;
  unsigned head;
  io_uring_for_each_cqe(&async_io._uring, head, cqe) {
    auto& taro = async_io._taro;
    size_t task_id = io_uring_cqe_get_data64(cqe);

    // We resume the task here directly.
    // If we enqueue the task back to the queue and indicate processed,
    // the cque may be overwritten before we resume the task.
    taro._process(*taro._this_worker(), taro._tasks[task_id].get());
    //taro._enqueue_back(*taro._this_worker(), task_id);
    ++processed;
  }
  io_uring_cq_advance(&async_io._uring, processed);
}

inline
Coro _async_io_polling_query(AsyncIOAwait& async_io) {
  io_uring_cqe* tmp;
  while(!async_io._finish) {
    //io_uring_submit(&async_io._uring);
    if (io_uring_peek_cqe(&async_io._uring, &tmp) == 0) {
      _async_io_consume(async_io);
    }
    else {
      co_await async_io._taro.suspend(async_io._ptask);
    }
  }
}


// ==========================================================================
//
// Definition of class AsyncIOAwait
//
// ==========================================================================

inline
AsyncIOAwait::AsyncIOAwait(Taro& taro, size_t queue_size): _taro{taro} {
  if (auto s = io_uring_queue_init(queue_size, &_uring, 0); s < 0) {
     throw std::runtime_error("error initializing io_uring: " + std::to_string(s));
  }
  //std::get_if<Task::CoroTask>(&_ptask->_handle)->set_inner();
  //_taro._enqueue(_taro._workers[0], _ptask.get(), TaskPriority::LOW);
  _taro.emplace(std::bind(_async_io_polling_query, std::ref(*this)));
  //_ptask = std::make_unique<Task>(-1, std::in_place_type_t<Task::CoroTask>{}, std::bind(_async_io_polling_query, std::ref(*this)));
  _ptask = _taro._tasks.back().get();
}

inline
AsyncIOAwait::~AsyncIOAwait() {
  io_uring_queue_exit(&_uring);
}

inline
auto AsyncIOAwait::read(int fd, char* buf, size_t size) {
  {
    // liburing is not thread safe
    std::scoped_lock lock{_mtx};
    auto* sqe = io_uring_get_sqe(&_uring);
    io_uring_prep_read(sqe, fd, buf, size, 0);
    io_uring_sqe_set_data64(sqe, _taro._this_worker()->_work_on_task_id);
    io_uring_submit(&_uring);
  }
  return std::suspend_always{};
}

inline
auto AsyncIOAwait::write(int fd, char* buf, size_t size) {
  {
    // liburing is not thread safe
    std::scoped_lock lock{_mtx};
    auto* sqe = io_uring_get_sqe(&_uring);
    io_uring_prep_write(sqe, fd, buf, size, 0);
    io_uring_sqe_set_data64(sqe, _taro._this_worker()->_work_on_task_id);
    io_uring_submit(&_uring);
  }
  return std::suspend_always{};
}

inline
void AsyncIOAwait::finish() {
  _finish = true;
}

// ==========================================================================
//
// Definition of async_io_await in Taro
//
// ==========================================================================

inline
AsyncIOAwait Taro::async_io_await(size_t queue_size) {
  return AsyncIOAwait(*this, queue_size);
}


} // end of namespace taro ==============================================
