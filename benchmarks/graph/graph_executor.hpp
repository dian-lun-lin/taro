#pragma once

#include "base/graph_base.hpp"

#include <coroflow/src/cuda/coroflow_v1.hpp>
#include <coroflow/src/cuda/coroflow_v2.hpp>
#include <coroflow/src/cuda/coroflow_v3.hpp>
#include <coroflow/src/cuda/coroflow_v4.hpp>
#include <coroflow/src/cuda/coroflow_v5.hpp>
#include <coroflow/src/cuda/coroflow_v6.hpp>
#include <coroflow/src/cuda/algorithm.hpp>

#include <chrono>
#include <cassert>

template <typename CF>
class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id, size_t num_threads, size_t num_streams);

    std::pair<double, double> run();

  private:
    
    int _dev_id;

    Graph& _g;

    size_t _num_threads;
    size_t _num_streams;

};

template <typename CF>
GraphExecutor<CF>::GraphExecutor(Graph& graph, int dev_id, size_t num_threads, size_t num_streams): 
  _g{graph}, _dev_id{dev_id}, _num_threads{num_threads}, _num_streams{num_streams} {
}

template <typename CF>
std::pair<double, double> GraphExecutor<CF>::run() {

  auto constr_tic = std::chrono::steady_clock::now();

  CF cf{_num_threads, _num_streams};

  size_t N{4096};
  size_t cnt{0};

  size_t BLOCK_SIZE{256};
  dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  std::vector<std::vector<std::vector<cf::TaskHandle>>> tasks;
  tasks.resize(_g.get_graph().size());

  std::vector<int*> d_a(_g.num_nodes()), d_b(_g.num_nodes()), d_res(_g.num_nodes());
  std::vector<std::vector<int>> h_res;
  h_res.reserve(_g.num_nodes());

  for(size_t l = 0; l < _g.get_graph().size(); ++l) {
    tasks[l].resize((_g.get_graph())[l].size());

    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      tasks[l][i].resize(4);

      tasks[l][i][0] = cf.emplace([&d_a, &cf, this, cnt, N]() -> cf::Coro {

        // TODO: seems like there is a bug in NVCC
        // When I declare two vectors and use co_await, compilation will got aborted.
        // For example,
        // std::vector<int> a;
        // std::vector<int> b;
        
        // CPU
        std::vector<int> h_a(N * N);
        //std::generate(h_a.begin(), h_a.end(), []() { return ::rand(); });
        std::iota(h_a.begin(), h_a.end(), 0);

        // GPU 
        co_await cf.cuda_suspend(
          [&h_a, this, &d_a, cnt, N](cudaStream_t st) mutable {
          // memory allocation
          cudaMallocAsync(&d_a[cnt], N * N * sizeof(int), st);
          // H2D
          cudaMemcpyAsync(d_a[cnt], h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, st);
        });
      });

      tasks[l][i][1] =  cf.emplace([&d_b, &cf, this, cnt, N]() -> cf::Coro {

        // CPU
        std::vector<int> h_b(N * N);
        //std::generate(h_b.begin(), h_b.end(), []() { return ::rand(); });
        std::iota(h_b.begin(), h_b.end(), 0);
        
        // GPU 
        co_await cf.cuda_suspend(
          [&h_b, &d_b, this, cnt, N](cudaStream_t st) mutable {
          // memory allocation
          cudaMallocAsync(&d_b[cnt], N * N *  sizeof(int), st);
          // H2D
          cudaMemcpyAsync(d_b[cnt], h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, st);
        });
      });

      tasks[l][i][2] =  cf.emplace([&d_res, &h_res, &cf, this, cnt, N]() -> cf::Coro {
        // CPU
        h_res.emplace_back(std::vector<int>(N * N));

        // GPU 
        co_await cf.cuda_suspend(
          [&d_res, this, cnt, N](cudaStream_t st) mutable {
          // memory allocation
          cudaMallocAsync(&d_res[cnt], N * N *  sizeof(int), st);
        });
      });

      tasks[l][i][3] =  cf.emplace([&d_a, &d_b, &d_res, &h_res, &cf, this, cnt, N, l, i, dim_grid, dim_block]() -> cf::Coro {

        // GPU work c
        co_await cf.cuda_suspend(
          [&d_a, &d_b, &d_res, &h_res, this, cnt, N, dim_grid, dim_block](cudaStream_t st) {
          // matrix multiplication
          cf::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(d_a[cnt], d_b[cnt], d_res[cnt], N, N, N);
          // D2H
          cudaMemcpyAsync(h_res[cnt].data(), d_res[cnt], N * N * sizeof(int), cudaMemcpyDeviceToHost, st);

          cudaFreeAsync(d_a[cnt], st);
          cudaFreeAsync(d_b[cnt], st);
          cudaFreeAsync(d_res[cnt], st);
  
        });

        // CPU work
        h_res[cnt].clear();
        bool* v = _g.at(l, i).visited;
        *v = true;

      });

      tasks[l][i][0].succeed(tasks[l][i][3]);
      tasks[l][i][1].succeed(tasks[l][i][3]);
      tasks[l][i][2].succeed(tasks[l][i][3]);
      ++cnt;
    }
  }

  //connection
  for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      for(auto&& out_node: _g.at(l, i).out_nodes) {
        tasks[l][i][3].succeed(tasks[l + 1][out_node][0]);
        tasks[l][i][3].succeed(tasks[l + 1][out_node][1]);
        tasks[l][i][3].succeed(tasks[l + 1][out_node][2]);
      }
    }
  }

  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  cf.schedule();
  cf.wait();

  auto exec_toc = std::chrono::steady_clock::now();

  assert(_g.traversed());

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();


  return {constr_dur, exec_dur};

}

