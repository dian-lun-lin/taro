#pragma once
#include "taro_callback_v4.hpp"

namespace taro { // begin of namespace taro ===================================

struct Token {
  size_t id;
  Token(size_t id): id{id} {}
};


class Pipeline {

  public:

    Pipeline(TaroCBV4& taro, size_t num_pipes, size_t num_tokens);

    template <typename C, std::enable_if_t<is_coro_task_v<C>, void>* = nullptr>
    void set_pipe(size_t pid, C&& c);

    TaskHandle first_pipe() const;
    TaskHandle last_pipe() const;

    size_t num_tokens() const;
    size_t token(size_t pid) const;

    bool done(size_t pid);

    auto step(size_t pid);
    auto final_step(size_t pid);

  private:


    //template <typename C>
    //auto _tmp_suspend(C&& c, size_t pid);

    TaroCBV4& _taro;
    size_t _num_pipes;
    size_t _num_tokens;
    std::vector<TaskHandle> _pipes;
    std::vector<size_t> _ids;
    std::vector<size_t> _tokens;
    std::vector<bool> _setted;
    // binary_semaphore cannot move or copy
    std::vector<std::unique_ptr<std::binary_semaphore>> _awakes;
};

Pipeline::Pipeline(TaroCBV4& taro, size_t num_pipes, size_t num_tokens): 
  _taro{taro}, _num_pipes{num_pipes}, _num_tokens{num_tokens},
  _tokens(num_pipes, 0), _pipes(num_pipes), _setted(num_pipes, false),
  _awakes(num_pipes)
{
  for(auto& a: _awakes) {
    a = std::make_unique<std::binary_semaphore>(1);
  }
}

TaskHandle Pipeline::first_pipe() const {
  return _pipes[0];
}

TaskHandle Pipeline::last_pipe() const {
  return _pipes[_num_pipes - 1];
}

size_t Pipeline::token(size_t pid) const {
  return _tokens[pid];
}

size_t Pipeline::num_tokens() const {
  return _num_tokens;
}

bool Pipeline::done(size_t pid) {
  if(_tokens[pid] == _num_tokens) {
    if(pid == 0) {
      // final_step, finish the coroutine
      for(size_t p = 1; p < _num_pipes; ++p) {
        std::cerr << "hi\n";
        _awakes[p]->acquire();
        _awakes[p - 1]->release();
        std::cerr << "eeeeee\n";
        auto* next = _pipes[p]._tp;
        auto coro_handle = std::get_if<Task::CoroTask>(&next->_handle)->coro._coro_handle;
        Worker* worker   = _taro._this_worker();
        worker->_work_on_task_id = next->_id;
        coro_handle.resume();
      }
    }
    return true;
  }
  return false;
}

auto Pipeline::step(size_t pid) {
  struct awaiter: std::suspend_always {
    Pipeline& _pipeline;
    size_t _pid;
    explicit awaiter(Pipeline& pipeline, size_t pid): 
      _pipeline{pipeline}, _pid{pid} {
      if(_pid == 0) {
        std::cerr << "acquire pid: 0\n";
        _pipeline._awakes[0]->acquire();
      }
    }
    
    std::coroutine_handle<> await_suspend(std::coroutine_handle<>) {
      // resume will destroy the awaiter
      // before enqueue(i.e. resume), we need to get all data first
      auto& pl         = _pipeline;
      auto& taro       = pl._taro;
      Worker* worker   = pl._taro._this_worker();
      auto pid         = _pid;

      // acquire next pipe
      if(pid + 1 != pl._num_pipes) {
        std::cerr << "acquire pid: " << pid + 1  << "\n";
        std::cerr << "token: " << pl._tokens[pid + 1] << "\n";
        pl._awakes[pid + 1]->acquire();
      }
      ++pl._tokens[pid];
      std::cerr << "release pid: " << pid << "\n";
      pl._awakes[pid]->release();

      // if pid is zero, enqueue task back for the new token
      // final token
      if(pid == 0) {
        size_t task_id = worker->_work_on_task_id;
        taro._enqueue(*worker, taro._tasks[task_id].get(), TaskPriority::LOW);
      }


      if(pid + 1 == pl._num_pipes) {
        //std::cerr << "release pid: " << pid << "\n";
        //pl._awakes[pid]->release();
        return std::noop_coroutine();
      }

      auto* next = pl._pipes[pid + 1]._tp;
      auto coro_handle = std::get_if<Task::CoroTask>(&next->_handle)->coro._coro_handle;
      worker->_work_on_task_id = next->_id;
      return coro_handle;
    }
    void await_resume() {
    }
  };

  return awaiter{*this, pid};
}

template <typename C, std::enable_if_t<is_coro_task_v<C>, void>*>
void Pipeline::set_pipe(size_t pid, C&& c) {
  assert(!_setted[pid]);
  _pipes[pid] = _taro.emplace(std::forward<C>(c));
  if(pid != 0) {
    _pipes[pid]._tp->_wait_first = true;
  }

  _setted[pid] = true;
}

Pipeline pipeline(TaroCBV4& taro, size_t num_pipes, size_t num_tokens) {
  return Pipeline{taro, num_pipes, num_tokens};
}

} // end of namespace taro ==============================================
