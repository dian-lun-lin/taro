#pragma once

#include "base/graph_base.hpp"

#include "../taskflow/taskflow/taskflow.hpp"
#include "../taskflow/taskflow/cuda/cudaflow.hpp"
#include "../taskflow/taskflow/cuda/algorithm/reduce.hpp"
#include <taro/src/cuda/algorithm.hpp>

#include <chrono>
#include <cassert>

class cudaFlowGraphExecutor {

  public:
  
    cudaFlowGraphExecutor(Graph& graph, int dev_id, size_t num_threads, size_t num_streams);

    std::pair<double, double> run(size_t N, std::string job);
    std::pair<double, double> run_cudaflow_reduce(size_t N);
    std::pair<double, double> run_cudaflow_reduce2(size_t N);
    std::pair<double, double> run_matmul(size_t N);

  private:
    

    int _dev_id;

    Graph& _g;

    size_t _num_threads;
    size_t _num_streams;

};

cudaFlowGraphExecutor::cudaFlowGraphExecutor(Graph& graph, int dev_id, size_t num_threads, size_t num_streams): 
  _g{graph}, _dev_id{dev_id}, _num_threads{num_threads}, _num_streams{num_streams} {
}

std::pair<double, double> cudaFlowGraphExecutor::run(size_t N, std::string job) {
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

std::pair<double, double> cudaFlowGraphExecutor::run_matmul(size_t N) {

  auto constr_tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor{_num_threads};


  size_t BLOCK_SIZE{128};
  dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);


  std::vector<int*> d_a(_g.num_nodes()), d_b(_g.num_nodes()), d_res(_g.num_nodes());
  std::vector<std::vector<int>> h_res;
  h_res.resize(_g.num_nodes(), std::vector<int>(N * N));
  
  auto cuda_graph = taskflow.emplace_on([&d_a, &d_b, &d_res, &h_res, N, dim_grid, dim_block, this](tf::cudaFlowCapturer& cf) mutable {
    cf.make_optimizer<tf::cudaRoundRobinCapturing>(_num_streams);
    std::vector<std::vector<std::vector<tf::cudaTask>>> tasks;
    tasks.resize(_g.get_graph().size());
    size_t cnt{0};

    for(size_t l = 0; l < _g.get_graph().size(); ++l) {
      tasks[l].resize((_g.get_graph())[l].size());

      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        tasks[l][i].resize(4);
        
        // H2D
        tasks[l][i][0] = cf.on([&d_a, cnt, N](cudaStream_t st) mutable {
          std::vector<int> h_a(N * N);
          std::iota(h_a.begin(), h_a.end(), 0);
          cudaMallocAsync(&d_a[cnt], N * N * sizeof(int), st);
          cudaMemcpyAsync(d_a[cnt], h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, st);
        });

        // H2D
        tasks[l][i][1] = cf.on([&d_b, cnt, N](cudaStream_t st) mutable {
          std::vector<int> h_b(N * N);
          std::iota(h_b.begin(), h_b.end(), 0);
          cudaMallocAsync(&d_b[cnt], N * N * sizeof(int), st);
          cudaMemcpyAsync(d_b[cnt], h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, st);
        });

        // res memory allocation
        tasks[l][i][2] = cf.on([&d_res, &h_res, cnt, N](cudaStream_t st) mutable {
          cudaMallocAsync(&d_res[cnt], N * N * sizeof(int), st);
        });

        // matrix multiplication
        tasks[l][i][3] = cf.on([&d_a, &d_b, &d_res, &h_res, cnt, N, dim_grid, dim_block](cudaStream_t st) mutable {
          taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(d_a[cnt], d_b[cnt], d_res[cnt], N, N, N);
          // D2H
          cudaMemcpyAsync(h_res[cnt].data(), d_res[cnt], N * N * sizeof(int), cudaMemcpyDeviceToHost, st);
          // free
          cudaFreeAsync(d_a[cnt], st);
          cudaFreeAsync(d_b[cnt], st);
          cudaFreeAsync(d_res[cnt], st);
        });

        //tasks[l][i][4] = cf.host([](){
          //// CPU work
          //h_res[cnt].clear();
        //});

          

        tasks[l][i][0].precede(tasks[l][i][3]);
        tasks[l][i][1].precede(tasks[l][i][3]);
        tasks[l][i][2].precede(tasks[l][i][3]);
        ++cnt;
      }
    }

    //connection
    for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: _g.at(l, i).out_nodes) {
          tasks[l][i][3].precede(tasks[l + 1][out_node][0]);
          tasks[l][i][3].precede(tasks[l + 1][out_node][1]);
          tasks[l][i][3].precede(tasks[l + 1][out_node][2]);
        }
      }
    }
  }, _dev_id);

  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto exec_toc = std::chrono::steady_clock::now();

  assert(_g.traversed());

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();


  return {constr_dur, exec_dur};

}

std::pair<double, double> cudaFlowGraphExecutor::run_cudaflow_reduce(size_t N) {

  auto constr_tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor{_num_threads};

  size_t cnt{0};

  std::vector<std::vector<int>> h_input(_g.num_nodes(), std::vector<int>());
  std::vector<int*> d_input(_g.num_nodes());
  std::vector<int*> d_res(_g.num_nodes());
  std::vector<int> d2h_res(_g.num_nodes());
  std::vector<int> h_res(_g.num_nodes(), 0);
  
  for(size_t i = 0; i < _g.num_nodes(); ++i) {
    cudaMalloc(&d_input[i], N * sizeof(int));
    cudaMalloc(&d_res[i], sizeof(int));
  }
  //
  std::vector<std::vector<std::vector<tf::Task>>> tasks;
  tasks.resize(_g.get_graph().size());
  for(size_t l = 0; l < _g.get_graph().size(); ++l) {
    tasks[l].resize((_g.get_graph())[l].size());
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      tasks[l][i].resize(4);

      // initalization
      tasks[l][i][0] = taskflow.emplace([&h_input, cnt, N](){
        h_input[cnt].resize(N);
        // generate is to heavy
        //std::generate(h_input[cnt].begin(), h_input[cnt].end(), []() { return ::rand(); });
        std::iota(h_input[cnt].begin(), h_input[cnt].end(), 0);
      });

      tasks[l][i][1] = taskflow.emplace_on([&h_input, &d_input, &h_res, &d_res, &d2h_res, cnt, N, this](tf::cudaFlowCapturer& cf) mutable {
        //cf.make_optimizer<tf::cudaRoundRobinCapturing>(_num_streams);
        
        //// malloc
        //auto task_a = cf.on([&d_input, &d_res, cnt, N](cudaStream_t st) mutable {
        //});

        //// H2D
        //auto task_b = cf.copy(d_input[cnt], h_input[cnt].data(), N).name("H2D");

        // reduce
        auto task_a = cf.on([&, cnt, N](cudaStream_t st) mutable {
          //checkCudaError(cudaMallocAsync(&d_input[cnt], N * sizeof(int), st));
          //checkCudaError(cudaMallocAsync(&d_res[cnt], sizeof(int), st));

          cudaMemcpyAsync(d_input[cnt], h_input[cnt].data(), N * sizeof(int), cudaMemcpyHostToDevice, st);

          tf::cudaExecutionPolicy<512, 8> policy(st);
          auto bytes  = tf::cuda_reduce_buffer_size<decltype(policy), int>(N);
          std::byte* buffer;
          cudaMallocAsync(&buffer, bytes * sizeof(std::byte), st);

          tf::cuda_reduce(policy,
            d_input[cnt], d_input[cnt] + N, d_res[cnt], [] __device__ (int a, int b) { return a + b; }, buffer
          );

          cudaFreeAsync(buffer, st);
          cudaMemcpyAsync(d2h_res.data() + cnt, d_res[cnt], sizeof(int), cudaMemcpyDeviceToHost, st);
          //checkCudaError(cudaFreeAsync(d_input[cnt], st));
          //checkCudaError(cudaFreeAsync(d_res[cnt], st));
        });

        //// D2H
        //auto task_d = cf.copy(d2h_res.data() + cnt, d_res[cnt], 1).name("D2H");

        //// free
        //auto task_e = cf.on([&d_input, &d_res, cnt, N](cudaStream_t st) mutable {
        //});

        //task_a.precede(task_b);
        //task_b.precede(task_c);
        //task_c.precede(task_d);
        //task_d.precede(task_e);
      }, _dev_id);

      // cpu computing
      tasks[l][i][2] = taskflow.emplace([&h_input, &h_res, cnt]() {
        h_res[cnt] = std::accumulate(h_input[cnt].begin(), h_input[cnt].end(), 0);
      });

      // check result 
      tasks[l][i][3] =  taskflow.emplace([&h_res, &d2h_res, cnt]()  {
        assert(h_res[cnt] == d2h_res[cnt]);
      });

      tasks[l][i][0].precede(tasks[l][i][1]);
      tasks[l][i][0].precede(tasks[l][i][2]);
      tasks[l][i][1].precede(tasks[l][i][3]);
      tasks[l][i][2].precede(tasks[l][i][3]);
    }
  }

  //connection
  for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
    for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
      for(auto&& out_node: _g.at(l, i).out_nodes) {
        tasks[l][i][3].precede(tasks[l + 1][out_node][0]);
        //tasks[l][i][3].precede(tasks[l + 1][out_node][1]);
        //tasks[l][i][3].precede(tasks[l + 1][out_node][2]);
      }
    }
  }
          
        //h_res[cnt] = std::accumulate(h_input[cnt].begin(), h_input[cnt].end(), 0, thrust::plus<int>());
          //assert(h_res[cnt] == d2h_res[cnt]);

  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

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

std::pair<double, double> cudaFlowGraphExecutor::run_cudaflow_reduce2(size_t N) {

  auto constr_tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor{_num_threads};

  size_t cnt{0};

  std::vector<std::vector<int>> h_input(_g.num_nodes(), std::vector<int>());
  std::vector<int*> d_input(_g.num_nodes());
  std::vector<int*> d_res(_g.num_nodes());
  std::vector<int> d2h_res(_g.num_nodes());
  std::vector<int> h_res(_g.num_nodes(), 0);
  
  for(size_t i = 0; i < _g.num_nodes(); ++i) {
    cudaMalloc(&d_input[i], N * sizeof(int));
    cudaMalloc(&d_res[i], sizeof(int));
  }
  //

  taskflow.emplace_on([&, cnt, N, this](tf::cudaFlow& cf) mutable {
    std::vector<std::vector<std::vector<tf::cudaTask>>> tasks;
    tasks.resize(_g.get_graph().size());
    for(size_t l = 0; l < _g.get_graph().size(); ++l) {
      tasks[l].resize((_g.get_graph())[l].size());
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        tasks[l][i].resize(4);

        // initalization
        tasks[l][i][0] = cf.host([&h_input, cnt, N](){
          h_input[cnt].resize(N);
          // generate is to heavy
          //std::generate(h_input[cnt].begin(), h_input[cnt].end(), []() { return ::rand(); });
          std::iota(h_input[cnt].begin(), h_input[cnt].end(), 0);
        });

        // reduce
        tasks[l][i][1] = cf.capture([&, cnt, N](tf::cudaFlowCapturer& cap) {
          cap.on([&, cnt, N](cudaStream_t st) mutable {

          cudaMemcpyAsync(d_input[cnt], h_input[cnt].data(), N * sizeof(int), cudaMemcpyHostToDevice, st);

          tf::cudaExecutionPolicy<512, 8> policy(st);
          auto bytes  = tf::cuda_reduce_buffer_size<decltype(policy), int>(N);
          std::byte* buffer;
          cudaMallocAsync(&buffer, bytes * sizeof(std::byte), st);

          tf::cuda_reduce(policy,
            d_input[cnt], d_input[cnt] + N, d_res[cnt], [] __device__ (int a, int b) { return a + b; }, buffer
          );

          cudaFreeAsync(buffer, st);
          cudaMemcpyAsync(d2h_res.data() + cnt, d_res[cnt], sizeof(int), cudaMemcpyDeviceToHost, st);
          });
        });

        // cpu computing
        tasks[l][i][2] = cf.host([&h_input, &h_res, cnt]() {
          h_res[cnt] = std::accumulate(h_input[cnt].begin(), h_input[cnt].end(), 0);
        });

        // check result 
        tasks[l][i][3] =  cf.host([&h_res, &d2h_res, cnt]()  {
          assert(h_res[cnt] == d2h_res[cnt]);
        });

        tasks[l][i][0].precede(tasks[l][i][1]);
        tasks[l][i][0].precede(tasks[l][i][2]);
        tasks[l][i][1].precede(tasks[l][i][3]);
        tasks[l][i][2].precede(tasks[l][i][3]);
      }
    }

    //connection
    for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: _g.at(l, i).out_nodes) {
          tasks[l][i][3].precede(tasks[l + 1][out_node][0]);
          //tasks[l][i][3].precede(tasks[l + 1][out_node][1]);
          //tasks[l][i][3].precede(tasks[l + 1][out_node][2]);
        }
      }
    }
  }, _dev_id);

  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

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

