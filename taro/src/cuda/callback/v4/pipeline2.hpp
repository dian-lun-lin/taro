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

    //template <typename C, std::enable_if_t<is_pipeline_static_task_v<C>, void>* = nullptr>
    //void set_pipe(size_t pid, C&& c);

    template <typename C, std::enable_if_t<is_pipeline_task_v<C>, void>* = nullptr>
    void set_pipe(size_t pid, C&& c);

    TaskHandle get_task() const;
    auto _suspend(size_t pid);

  private:


    TaroCBV4& _taro;
    size_t _num_pipes;
    size_t _num_tokens;
    std::vector<size_t> _ids;
    std::vector<Token> _tokens;
    std::vector<bool> _setted;
    std::vector<std::function<Coro(Token&)>> _pipes;
    TaskHandle _handle;
};

Pipeline::Pipeline(TaroCBV4& taro, size_t num_pipes, size_t num_tokens): 
  _taro{taro}, _num_pipes{num_pipes}, _num_tokens{num_tokens},
  _tokens(num_pipes, 0), _pipes(num_pipes), _setted(num_pipes, false)
{
}

TaskHandle Pipeline::get_task() const {
  return _handle;
}

template <typename C, std::enable_if_t<is_pipeline_task_v<C>, void>*>
void Pipeline::set_pipe(size_t pid, C&& c) {
  assert(!_setted[pid]);

  _pipes[pid] = std::forward<C>(c);


  _setted[pid] = true;
}


void Pipeline::initialize() {
  for(auto s: _setted) {
    assert(s);
  }
  _handle = _taro.emplace([this, pid, c=std::forward<C>(c)]() -> Coro {
 
    // TODO: Don't know why _taro.suspend() needs to take a callable...
    // should be a bug in compiler (nvcc)
    auto coro = c(_tokens[pid]);
  if(pid != 0) {
    while(_tokens[pid - 1].id <= _tokens[pid].id) {
      co_await _taro.suspend([](){});
    }
  }

  while(1) {
    coro._resume();
    if(coro._done()) {
      break;
    }
    co_await _taro._no_pushback_suspend([](){});
  }

  ++_tokens[pid].id;
    std::cerr << "start\n";
    while(_tokens[pid].id < _num_tokens) {
    }
  });

}

Pipeline pipeline(TaroCBV4& taro, size_t num_pipes, size_t num_tokens) {
  return Pipeline{taro, num_pipes, num_tokens};
}

} // end of namespace taro ==============================================
