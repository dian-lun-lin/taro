#include <coroflow/src/cuda/coroflow_v1.hpp>
#include <coroflow/src/cuda/coroflow_v2.hpp>
#include <coroflow/src/cuda/coroflow_v3.hpp>
#include <coroflow/src/cuda/coroflow_v4.hpp>
#include <coroflow/src/cuda/coroflow_v5.hpp>
#include <coroflow/src/cuda/coroflow_v6.hpp>
#include <coroflow/src/cuda/algorithm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#include <chrono>


// GPU kernel
__global__ void cuda_sleep(
   int ms
) {
  for (int i = 0; i < ms; i++) {
    __nanosleep(1000000U);
  }
}


// CPU task
void cpu_sleep(
  int ms
) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// =============================================================
// 
// benchmark: independent
//
// =============================================================

// without coroutine
// one task, one stream
void func(
  size_t num_threads, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {

  std::chrono::time_point<std::chrono::steady_clock> task_tic;
  std::chrono::time_point<std::chrono::steady_clock> task_toc;

  cf::CoroflowV1 cf{num_threads};
  std::vector<cf::TaskHandle> tasks(num_tasks);
  std::vector<cudaEvent_t>  events(num_tasks);
  std::vector<cudaStream_t> streams(num_tasks);
  for(size_t i = 0; i < num_tasks; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&events[i]);
  }

  // emplace tasks
  for(size_t t = 0; t < num_tasks; ++t) {
    tasks[t] = cf.emplace(
      [&streams, &events, t, chain_size, cpu_ms, gpu_ms](){
      for(size_t i = 0; i < chain_size; i++) {
        // cpu task
        cpu_sleep(cpu_ms);

        // gpu task
        cuda_sleep<<<8, 32, 0, streams[t]>>>(gpu_ms);
        cudaEventRecord(events[t]);
        cudaEventSynchronize(events[t]);
      }
    });
  }

  assert(cf.is_DAG());

  task_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  task_toc = std::chrono::steady_clock::now();
  auto task_dur = std::chrono::duration_cast<std::chrono::milliseconds>(task_toc - task_tic).count();
  std::cout << "function time: " << task_dur << "ms\n";

  for(auto& st: streams) {
    cudaStreamDestroy(st);
  }
}

// without callback
// one task, one stream
void coro_v1(
  size_t num_threads, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::CoroflowV1 cf{num_threads};
  std::vector<cf::TaskHandle> tasks(num_tasks);
  std::vector<cudaStream_t> streams(num_tasks);
  std::vector<cudaEvent_t> events(num_tasks);
  for(size_t i = 0; i < num_tasks; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&events[i]);
  }

  // emplace tasks
  for(size_t c = 0; c < num_tasks; ++c) {
      tasks[c] = cf.emplace([&cf, &streams, &events, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
        for(size_t i = 0; i < chain_size; i++) {
          // cpu task
          cpu_sleep(cpu_ms);

          // gpu task
          cuda_sleep<<<8, 32, 0, streams[c]>>>(gpu_ms);
          cudaEventRecord(events[i]);
          auto isdone = [&events, i]() { return cudaEventQuery(events[i]) == cudaSuccess;  };
          while(!isdone()) {
            co_await cf.suspend();
          }
          
        }
        co_return;
      });
  }

  assert(cf.is_DAG());

  coro_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  coro_toc = std::chrono::steady_clock::now();
  auto coro_dur = std::chrono::duration_cast<std::chrono::milliseconds>(coro_toc - coro_tic).count();
  std::cout << "coroflow v1 time: " << coro_dur << "ms\n";
}

// streams are handled by users
// one task, one stream
void coro_v2(
  size_t num_threads, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::CoroflowV2 cf{num_threads};
  std::vector<cf::TaskHandle> tasks(num_tasks);
  std::vector<cudaStream_t> streams(num_tasks);
  for(size_t i = 0; i < num_tasks; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  // emplace tasks
  for(size_t c = 0; c < num_tasks; ++c) {
      tasks[c] = cf.emplace([&cf, &streams, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
        for(size_t i = 0; i < chain_size; i++) {
          // cpu task
          cpu_sleep(cpu_ms);

          // gpu task
          cuda_sleep<<<8, 32, 0, streams[c]>>>(gpu_ms);
          co_await cf.cuda_suspend(streams[c]);
          
        }
        co_return;
      });
  }

  assert(cf.is_DAG());

  coro_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  coro_toc = std::chrono::steady_clock::now();
  auto coro_dur = std::chrono::duration_cast<std::chrono::milliseconds>(coro_toc - coro_tic).count();
  std::cout << "coroflow v2 time: " << coro_dur << "ms\n";
}


// M CPU threads, N GPU streams
void coro_v3(
  size_t num_threads, 
  size_t num_streams, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::CoroflowV3 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> tasks(num_tasks);

  // emplace tasks
  for(size_t c = 0; c < num_tasks; ++c) {
      tasks[c] = cf.emplace([&cf, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
        for(size_t i = 0; i < chain_size; i++) {
          // cpu task
          cpu_sleep(cpu_ms);

          // gpu task
          co_await cf.cuda_suspend([gpu_ms](cudaStream_t st) {
            cuda_sleep<<<8, 32, 0, st>>>(gpu_ms);
          });
        }
        co_return;
      });
  }

  assert(cf.is_DAG());

  coro_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  coro_toc = std::chrono::steady_clock::now();
  auto coro_dur = std::chrono::duration_cast<std::chrono::milliseconds>(coro_toc - coro_tic).count();
  std::cout << "coroflow v3 time: " << coro_dur << "ms\n";
}

// M CPU threads, N GPU streams
// work-stealing approach
void coro_v4(
  size_t num_threads, 
  size_t num_streams, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::CoroflowV4 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> tasks(num_tasks);

  // emplace tasks
  for(size_t c = 0; c < num_tasks; ++c) {
      tasks[c] = cf.emplace([&cf, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
        for(size_t i = 0; i < chain_size; i++) {
          // cpu task
          cpu_sleep(cpu_ms);

          // gpu task
          co_await cf.cuda_suspend([gpu_ms](cudaStream_t st) {
            cuda_sleep<<<8, 32, 0, st>>>(gpu_ms);
          });
        }
        co_return;
      });
  }

  assert(cf.is_DAG());

  coro_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  coro_toc = std::chrono::steady_clock::now();
  auto coro_dur = std::chrono::duration_cast<std::chrono::milliseconds>(coro_toc - coro_tic).count();
  std::cout << "coroflow v4 time: " << coro_dur << "ms\n";
}

// M CPU threads, N GPU streams
// work-stealing approach
void coro_v5(
  size_t num_threads, 
  size_t num_streams, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::CoroflowV5 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> tasks(num_tasks);

  // emplace tasks
  for(size_t c = 0; c < num_tasks; ++c) {
      tasks[c] = cf.emplace([&cf, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
        for(size_t i = 0; i < chain_size; i++) {
          // cpu task
          cpu_sleep(cpu_ms);

          // gpu task
          co_await cf.cuda_suspend([gpu_ms](cudaStream_t st) {
            cuda_sleep<<<8, 32, 0, st>>>(gpu_ms);
          });
        }
        co_return;
      });
  }

  assert(cf.is_DAG());

  coro_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  coro_toc = std::chrono::steady_clock::now();
  auto coro_dur = std::chrono::duration_cast<std::chrono::milliseconds>(coro_toc - coro_tic).count();
  std::cout << "coroflow v5 time: " << coro_dur << "ms\n";
}

// M CPU threads, N GPU streams
// work-stealing approach
void coro_v6(
  size_t num_threads, 
  size_t num_streams, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::CoroflowV5 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> tasks(num_tasks);

  // emplace tasks
  for(size_t c = 0; c < num_tasks; ++c) {
      tasks[c] = cf.emplace([&cf, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
        for(size_t i = 0; i < chain_size; i++) {
          // cpu task
          cpu_sleep(cpu_ms);

          // gpu task
          co_await cf.cuda_suspend([gpu_ms](cudaStream_t st) {
            cuda_sleep<<<8, 32, 0, st>>>(gpu_ms);
          });
        }
        co_return;
      });
  }

  assert(cf.is_DAG());

  coro_tic = std::chrono::steady_clock::now();
  cf.schedule();
  cf.wait();
  coro_toc = std::chrono::steady_clock::now();
  auto coro_dur = std::chrono::duration_cast<std::chrono::milliseconds>(coro_toc - coro_tic).count();
  std::cout << "coroflow v6 time: " << coro_dur << "ms\n";
}


int main(int argc, char* argv[]) {
  if(argc != 8) {
    std::cerr << "usage: ./bin/independent mode num_threads num_streams num_tasks chain_size cpu_ms gpu_ms\n";
    std::cerr << "mode should be 0, 1, 2, 3, 4, 5, 6, or 7\n";
    std::cerr << "0: function, 1: coroflow v1, 2: corflow v2, 3: corflow v3... 7: all \n";
    std::exit(EXIT_FAILURE);
  }
  size_t mode = std::atoi(argv[1]);
  size_t num_threads = std::atoi(argv[2]);
  size_t num_streams = std::atoi(argv[3]);
  size_t num_tasks = std::atoi(argv[4]);
  size_t chain_size = std::atoi(argv[5]);
  int cpu_ms = std::atoi(argv[6]);
  int gpu_ms = std::atoi(argv[7]);
  std::cout << "(mode, num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms): (" 
            << mode        << ", "
            << num_threads << ", "
            << num_streams << ", "
            << num_tasks   << ", "
            << chain_size  << ", "
            << cpu_ms      << ", "
            << gpu_ms      << "):\n";

  if(mode == 0) {
    std::cout << "function...\n";
    std::cout << "igonre num_streams... each task has its own stream\n";
    func(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 1) {
    std::cout << "coroflow v1...\n";
    std::cout << "igonre num_streams... each task has its own stream\n";
    coro_v1(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 2) {
    std::cout << "coroflow v2...\n";
    std::cout << "igonre num_streams... each task has its own stream\n";
    coro_v2(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 3) {
    std::cout << "coroflow v3...\n";
    coro_v3(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 4) {
    std::cout << "coroflow v4...\n";
    coro_v4(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 5) {
    std::cout << "coroflow v5...\n";
    coro_v5(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 6) {
    std::cout << "coroflow v6...\n";
    coro_v6(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else if(mode == 7) {
    std::cout << "all...\n\n";
    std::cout << "function...\n";
    std::cout << "igonre num_streams... each task has its own stream\n";
    func(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms);
    std::cout << "\n";

    // without callback
    std::cout << "coroflow v1...\n";
    std::cout << "igonre num_streams... each task has its own stream\n";
    coro_v1(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms);
    std::cout << "\n";

    // streams are handled by users
    std::cout << "coroflow v2...\n";
    std::cout << "igonre num_streams... each task has its own stream\n";
    coro_v2(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms);
    std::cout << "\n";

    std::cout << "coroflow v3...\n";
    coro_v3(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);

    std::cout << "coroflow v4...\n";
    coro_v4(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);

    std::cout << "coroflow v5...\n";
    coro_v5(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);

    std::cout << "coroflow v6...\n";
    coro_v6(num_threads, num_streams, num_tasks, chain_size, cpu_ms, gpu_ms);
  }
  else {
    std::cerr << "mode should be 0, 1, 2, 3, 4, 5, 6, or 7\n";
    std::cerr << "0: function, 1: coroflow v1, 2: corflow v2, 3: corflow v3... 7: all \n";
    std::exit(EXIT_FAILURE);
  }


  std::cout << "\n";
}


