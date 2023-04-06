#pragma once

#include "../graph.hpp"
#include <taro/src/cuda/callback/v2/taro_callback_v2.hpp>

//#include <thrust/reduce.h>
//#include <thrust/sort.h>
//#include <thrust/functional.h>
//#include <thrust/random.h>
//

class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id, size_t num_threads);

    std::pair<double, double> run(size_t cpu_time, size_t gpu_time);
    std::pair<double, double> run_loop(size_t cpu_time, size_t gpu_time);

  private:
    
    int _dev_id;

    Graph& _g;

    size_t _num_threads;
    size_t _num_streams;
};

GraphExecutor::GraphExecutor(Graph& graph, int dev_id, size_t num_threads): 
  _g{graph}, _dev_id{dev_id}, _num_threads{num_threads} {
}

std::pair<double, double> GraphExecutor::run(size_t cpu_time, size_t gpu_time) {
  return run_loop(cpu_time, gpu_time);
}

std::pair<double, double> GraphExecutor::run_loop(size_t cpu_time, size_t gpu_time) {
  auto constr_tic = std::chrono::steady_clock::now();

  taro::TaroCBV2 taro{_num_threads};

  size_t cnt{0};

  std::vector<std::vector<taro::TaskHandle>> tasks;
  tasks.resize(_g.get_graph().size());
  for(size_t l = 0; l < _g.get_graph().size(); ++l) {
    tasks[l].resize((_g.get_graph())[l].size());
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {

      // GPU computing
      tasks[l][i] = taro.emplace([&taro, cpu_time, gpu_time]() mutable -> taro::Coro {
        cpu_loop(cpu_time);
        co_await taro.cuda_suspend([gpu_time](cudaStream_t st) mutable {
          cuda_loop<<<8, 256, 0, st>>>(gpu_time);
        });
      });
        
      ++cnt;
    }
  }

  //connection
  for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      for(auto&& out_node: _g.at(l, i).out_nodes) {
        tasks[l][i].precede(tasks[l + 1][out_node]);
        //tasks[l][i][3].precede(tasks[l + 1][out_node][1]);
        //tasks[l][i][3].precede(tasks[l + 1][out_node][2]);
      }
    }
  }
  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  taro.schedule();
  taro.wait();

  auto exec_toc = std::chrono::steady_clock::now();

  //assert(_g.traversed());

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();

  return {constr_dur, exec_dur};
}

