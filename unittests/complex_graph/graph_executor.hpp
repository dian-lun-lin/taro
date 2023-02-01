#pragma once

#include <simple_graph/base/graph_base.hpp>
#include <coroflow.hpp>
#include <chrono>
#include <cassert>

template <typename CF>
class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id = 0);

    std::pair<double, double> run();

  private:
    
    int _dev_id;

    Graph& _g;

};

template <typename CF>
GraphExecutor<CF>::GraphExecutor(Graph& graph, int dev_id): _g{graph}, _dev_id{dev_id} {
  //TODO: why we cannot put cuda lambda function here?
}

template <typename CF>
std::pair<double, double> GraphExecutor<CF>::run() {

  auto constr_tic = std::chrono::steady_clock::now();

  cf::Coroflow coroflow;

  size_t N{1000000};

  //int* d_res = tf::cuda_malloc_device<int>(1);
  //int* d_input = tf::cuda_malloc_device<int>(N);

  //int h_res;
  //std::vector<int> h_input(N);


  auto trav_t = coroflow.emplace_on([this, N, d_res, d_input, &h_res, &h_input](tf::cudaFlowCapturer& cf) {
    
    std::vector<std::vector<std::vector<tf::cudaTask>>> tasks;
    tasks.resize(_g.get_graph().size());

    for(size_t l = 0; l < _g.get_graph().size(); ++l) {
      tasks[l].resize((_g.get_graph())[l].size());

      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {

        //each node contains 4 tasks
        std::vector<tf::cudaTask> tasks_per_node(4);

        //H2D
        tasks_per_node[0] = cf.copy(d_input, h_input.data(), N).name("H2D");

        //reduce
        tasks_per_node[1] =  cf.reduce(d_input, d_input + N, d_res, [this] __device__ (int a, int b) { 
          return a + b; 
        }).name("reduce");

        //D2H
        tasks_per_node[2] = cf.copy(&h_res, d_res, 1).name("D2H");

        //visited
        bool* v = _g.at(l, i).visited;
        tasks_per_node[3] = cf.single_task([this, v] __device__ (){
          *v = true;
        }).name("visited"); 

        //connection
        tasks_per_node[0].precede(tasks_per_node[1]);
        tasks_per_node[1].precede(tasks_per_node[2]);
        tasks_per_node[2].precede(tasks_per_node[3]);
        

        tasks[l][i] = std::move(tasks_per_node);
      }
    }

    //connection
    for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: _g.at(l, i).out_nodes) {
          tasks[l][i][3].precede(tasks[l + 1][out_node][0]);
        }
      }
    }

  }, _dev_id).name("traverse");

  auto check_t = taskflow.emplace([this](){
    assert(_g.traversed());
  });
  
  trav_t.precede(check_t);

  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto exec_toc = std::chrono::steady_clock::now();

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();


  return {constr_dur, exec_dur};

}

