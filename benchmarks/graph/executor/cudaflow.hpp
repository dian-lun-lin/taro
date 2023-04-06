#pragma once

#include "../graph.hpp"
#include <taskflow/taskflow/taskflow.hpp>
#include <taskflow/taskflow/cuda/cudaflow.hpp>

class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id, size_t num_threads);

    std::pair<double, double> run(size_t cpu_time, size_t gpu_time);
    //std::pair<double, double> run_cudaflow_reduce(size_t N);
    //std::pair<double, double> run_cudaflow_reduce2(size_t N);
    //std::pair<double, double> run_matmul(size_t N);
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

  tf::Taskflow taskflow;
  tf::Executor executor{_num_threads};

  size_t cnt{0};

  std::vector<std::vector<tf::Task>> tasks;
  tasks.resize(_g.get_graph().size());
  for(size_t l = 0; l < _g.get_graph().size(); ++l) {
    tasks[l].resize((_g.get_graph())[l].size());
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {

      tasks[l][i] = taskflow.emplace_on([cpu_time, gpu_time](tf::cudaFlowCapturer& cf) mutable {
        cpu_loop(cpu_time);
        cf.on([gpu_time](cudaStream_t st) mutable {
          cuda_loop<<<8, 256, 0, st>>>(gpu_time);
        });
      }, _dev_id);

      ++cnt;
    }
  }

  //connection
  for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      for(auto&& out_node: _g.at(l, i).out_nodes) {
        tasks[l][i].precede(tasks[l + 1][out_node]);
      }
    }
  }
  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto exec_toc = std::chrono::steady_clock::now();

  //assert(_g.traversed());

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();

  return {constr_dur, exec_dur};
}
