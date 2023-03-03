#pragma once

//#include <thrust/reduce.h>
//#include <thrust/sort.h>
//#include <thrust/functional.h>
//#include <thrust/random.h>

#include "base/graph_base.hpp"

#include <taro/src/cuda/taro_v1.hpp>
#include <taro/src/cuda/taro_v2.hpp>
#include <taro/src/cuda/taro_v3.hpp>
#include <taro/src/cuda/taro_v4.hpp>
#include <taro/src/cuda/taro_v5.hpp>
#include <taro/src/cuda/taro_v6.hpp>
#include <taro/src/cuda/taro_v7.hpp>
#include <taro/src/cuda/algorithm.hpp>
#include "../taskflow/taskflow/cuda/algorithm/reduce.hpp"

#include <chrono>
#include <cassert>

template <typename TARO>
class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id, size_t num_threads, size_t num_streams);

    std::pair<double, double> run(size_t N, std::string job);
    std::pair<double, double> run_matmul(size_t N);
    std::pair<double, double> run_cudaflow_reduce(size_t N);

  private:
    
    int _dev_id;

    Graph& _g;

    size_t _num_threads;
    size_t _num_streams;
};

template <typename TARO>
GraphExecutor<TARO>::GraphExecutor(Graph& graph, int dev_id, size_t num_threads, size_t num_streams): 
  _g{graph}, _dev_id{dev_id}, _num_threads{num_threads}, _num_streams{num_streams} {
}

template <typename TARO>
std::pair<double, double> GraphExecutor<TARO>::run(size_t N, std::string job) {
  std::pair<double, double> dur;
  if(job == "matmul") {
    dur =  run_matmul(N);
  }
  else if(job == "cudaflow_reduce") {
    dur =  run_cudaflow_reduce(N);
  }
  else {
    assert(false);
  }
  return dur;
}


template <typename TARO>
std::pair<double, double> GraphExecutor<TARO>::run_matmul(size_t N) {

  auto constr_tic = std::chrono::steady_clock::now();

  TARO taro{_num_threads, _num_streams};

  size_t cnt{0};

  size_t BLOCK_SIZE{128};
  dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  std::vector<std::vector<std::vector<taro::TaskHandle>>> tasks;
  tasks.resize(_g.get_graph().size());

  std::vector<int*> d_a(_g.num_nodes()), d_b(_g.num_nodes()), d_res(_g.num_nodes());
  std::vector<std::vector<int>> h_res;
  h_res.resize(_g.num_nodes(), std::vector<int>(N * N));

  for(size_t l = 0; l < _g.get_graph().size(); ++l) {
    tasks[l].resize((_g.get_graph())[l].size());

    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      tasks[l][i].resize(4);

      tasks[l][i][0] = taro.emplace([&d_a, &taro, this, cnt, N]() -> taro::Coro {

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
        co_await taro.cuda_suspend(
          [&h_a, this, &d_a, cnt, N](cudaStream_t st) mutable {
          // memory allocation
          cudaMallocAsync(&d_a[cnt], N * N * sizeof(int), st);
          // H2D
          cudaMemcpyAsync(d_a[cnt], h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, st);
        });
      });

      tasks[l][i][1] =  taro.emplace([&d_b, &taro, this, cnt, N]() -> taro::Coro {

        // CPU
        std::vector<int> h_b(N * N);
        //std::generate(h_b.begin(), h_b.end(), []() { return ::rand(); });
        std::iota(h_b.begin(), h_b.end(), 0);
        
        // GPU 
        co_await taro.cuda_suspend(
          [&h_b, &d_b, this, cnt, N](cudaStream_t st) mutable {
          // memory allocation
          cudaMallocAsync(&d_b[cnt], N * N *  sizeof(int), st);
          // H2D
          cudaMemcpyAsync(d_b[cnt], h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, st);
        });
      });

      tasks[l][i][2] =  taro.emplace([&d_res, &h_res, &taro, this, cnt, N]() -> taro::Coro {
        // CPU
        //h_res.emplace_back(std::vector<int>(N * N));

        // GPU 
        co_await taro.cuda_suspend(
          [&d_res, this, cnt, N](cudaStream_t st) mutable {
          // memory allocation
          cudaMallocAsync(&d_res[cnt], N * N *  sizeof(int), st);
        });
      });

      tasks[l][i][3] =  taro.emplace([&d_a, &d_b, &d_res, &h_res, &taro, this, cnt, N, l, i, dim_grid, dim_block]() -> taro::Coro {

        // GPU work c
        co_await taro.cuda_suspend(
          [&d_a, &d_b, &d_res, &h_res, this, cnt, N, dim_grid, dim_block](cudaStream_t st) {
          // matrix multiplication
          taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(d_a[cnt], d_b[cnt], d_res[cnt], N, N, N);
          // D2H
          cudaMemcpyAsync(h_res[cnt].data(), d_res[cnt], N * N * sizeof(int), cudaMemcpyDeviceToHost, st);
          // free
          cudaFreeAsync(d_a[cnt], st);
          cudaFreeAsync(d_b[cnt], st);
          cudaFreeAsync(d_res[cnt], st);
        });

        // CPU work
        //h_res[cnt].clear();
        //bool* v = _g.at(l, i).visited;
        //*v = true;

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

  taro.schedule();
  taro.wait();

  auto exec_toc = std::chrono::steady_clock::now();

  assert(_g.traversed());

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();


  return {constr_dur, exec_dur};

}

template <typename TARO>
std::pair<double, double> GraphExecutor<TARO>::run_cudaflow_reduce(size_t N) {

  auto constr_tic = std::chrono::steady_clock::now();

  TARO taro{_num_threads, _num_streams};

  size_t cnt{0};

  std::vector<std::vector<std::vector<taro::TaskHandle>>> tasks;
  tasks.resize(_g.get_graph().size());

  std::vector<std::vector<int>> h_input(_g.num_nodes(), std::vector<int>());
  std::vector<int*> d_input(_g.num_nodes());
  std::vector<int*> d_res(_g.num_nodes());
  std::vector<int> d2h_res(_g.num_nodes());

  std::vector<int> h_res(_g.num_nodes(), 0);

  for(size_t i = 0; i < _g.num_nodes(); ++i) {
    cudaMalloc(&d_input[i], N * sizeof(int));
    cudaMalloc(&d_res[i], sizeof(int));
  }

  for(size_t l = 0; l < _g.get_graph().size(); ++l) {
    tasks[l].resize((_g.get_graph())[l].size());
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      tasks[l][i].resize(4);

      // initialization
      tasks[l][i][0] = taro.emplace([&h_input, cnt, N]() mutable {
        h_input[cnt].resize(N);
        //std::generate(h_input[cnt].begin(), h_input[cnt].end(), []() { return ::rand(); });
        std::iota(h_input[cnt].begin(), h_input[cnt].end(), 0);
      });


      // GPU computing
      tasks[l][i][1] = taro.emplace([&taro, &h_input, &d_input, &d_res, &d2h_res, cnt, N]() mutable -> taro::Coro {
        co_await taro.cuda_suspend([&h_input, &d_input, &d_res, &d2h_res, cnt, N](cudaStream_t st) mutable {
          // H2D
          cudaMemcpyAsync(d_input[cnt], h_input[cnt].data(), N * sizeof(int), cudaMemcpyHostToDevice, st);

          // reduce
          //d_res[cnt] = thrust::reduce(thrust::cuda::par.on(st), d_input[cnt], d_input[cnt] + N, 0, thrust::plus<int>());
          tf::cudaExecutionPolicy<512, 8> policy(st);
          auto bytes  = tf::cuda_reduce_buffer_size<decltype(policy), int>(N);
          //auto buffer = tf::cuda_malloc_device<std::byte>(bytes);
          std::byte* buffer;
          cudaMallocAsync(&buffer, bytes * sizeof(std::byte), st);
          tf::cuda_reduce(policy,
            d_input[cnt], d_input[cnt] + N, d_res[cnt], [] __device__ (int a, int b) { return a + b; }, buffer
          );
          cudaFreeAsync(buffer, st);
  
          // D2H
          cudaMemcpyAsync(d2h_res.data() + cnt, d_res[cnt], sizeof(int), cudaMemcpyDeviceToHost, st);
        });
      });
        
      // CPU computing
      tasks[l][i][2] =  taro.emplace([&h_input, &h_res, cnt]() mutable {
        h_res[cnt] = std::accumulate(h_input[cnt].begin(), h_input[cnt].end(), 0);
      });

      // check result and free
      tasks[l][i][3] =  taro.emplace([&taro, &h_res, &d2h_res, cnt]() mutable  {
        assert(h_res[cnt] == d2h_res[cnt]);
      });

      tasks[l][i][0].succeed(tasks[l][i][1]);
      tasks[l][i][0].succeed(tasks[l][i][2]);
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
        //tasks[l][i][3].succeed(tasks[l + 1][out_node][1]);
        //tasks[l][i][3].succeed(tasks[l + 1][out_node][2]);
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

  for(size_t i = 0; i < _g.num_nodes(); ++i) {
    cudaFree(d_input[i]);
    cudaFree(d_res[i]);
  }

  return {constr_dur, exec_dur};

}

