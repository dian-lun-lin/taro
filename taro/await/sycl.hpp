#pragma once

#include "../core/taro.hpp"
#include <sycl/sycl.hpp>

namespace taro { // begin of namespace taro ===================================

template <typename C>
constexpr bool is_sycl_v = 
  std::is_invocable_r_v<void, C, sycl::handler&>;

class syclAwait {

  struct syclPollingData {
    syclAwait* sycl;
    Task* ptask; // polling task
    Worker* worker;
    size_t task_id;
    sycl::event event;
  };

  friend class Taro;
  friend void _sycl_polling(void* void_args);
  friend Coro _sycl_polling_query(syclPollingData&);

  public:

    syclAwait(Taro& taro, sycl::queue& que);

    template <typename C, std::enable_if_t<is_sycl_v<C>, void>* = nullptr>
    auto until_polling(C&&);

    template <typename C, std::enable_if_t<is_sycl_v<C>, void>* = nullptr>
    auto wait(C&&);
    

  private:

    std::vector<std::unique_ptr<Task>> _ptasks;
    sycl::queue& _que;
    Taro& _taro;
};

inline
syclAwait::syclAwait(Taro& taro, sycl::queue& que): _taro{taro}, _que{que} {
}


// ==========================================================================
//
// polling
//
// ==========================================================================

inline
void _sycl_polling(void* void_args) {

  // unpack
  auto* data = (syclAwait::syclPollingData*) void_args;
  auto* sycl = data->sycl;
  auto& taro = sycl->_taro;
  auto* worker = data->worker;
  size_t task_id = data->task_id;
  
  taro._enqueue_back(*worker, task_id);
}

Coro _sycl_polling_query(syclAwait::syclPollingData& data) {

  while(data.event.get_info<sycl::info::event::command_execution_status>() !=  sycl::info::event_command_status::complete) {
    co_await data.sycl->_taro.suspend(data.ptask);
  }
  _sycl_polling((void*)&data);
}


// ==========================================================================
//
// Definition of class syclAwait
//
// ==========================================================================

template <typename C, std::enable_if_t<is_sycl_v<C>, void>*>
auto syclAwait::until_polling(C&& c) {

  struct sycl_awaiter: std::suspend_always {
    std::function<void(sycl::handler&)> command_group;
    syclPollingData data;

    explicit sycl_awaiter(syclAwait* sycl, C&& c) noexcept : command_group{std::forward<C>(c)} {
      data.sycl = sycl;
    }
    
    void await_suspend(std::coroutine_handle<>) {

      // set polling data
      data.worker = data.sycl->_taro._this_worker();
      data.task_id = data.worker->_work_on_task_id;

      data.event = data.sycl->_que.submit([this](sycl::handler& h){
        command_group(h);
      });

      data.ptask = data.sycl->_ptasks.emplace_back(
        std::make_unique<Task>(-1, std::in_place_type_t<Task::CoroTask>{}, std::bind(_sycl_polling_query, std::ref(data)))
      ).get();

      std::get_if<Task::CoroTask>(&data.ptask->_handle)->set_inner();
      
      data.sycl->_taro._enqueue(*data.worker, data.ptask, TaskPriority::LOW);

      return;
    }
  };

  return sycl_awaiter{this, std::forward<C>(c)};
}

template <typename C, std::enable_if_t<is_sycl_v<C>, void>*>
auto syclAwait::wait(C&& c) {
  // choose the best stream id
  auto event = _que.submit([&](sycl::handler& h) {
    c(h);
  });
  event.wait();
}

// ==========================================================================
//
// Definition of sycl_await in Taro
//
// ==========================================================================

inline
syclAwait Taro::sycl_await(sycl::queue& que) {
  return syclAwait(*this, que);
}


} // end of namespace taro ==============================================
