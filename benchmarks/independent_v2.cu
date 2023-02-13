#include <coroflow/src/cuda/coroflow.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <coroflow/algorithms/matmul.hpp>
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

void independent_coro(
  size_t num_threads, 
  size_t num_coros, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms, 
  std::vector<cudaStream_t>& streams
) {
  std::chrono::time_point<std::chrono::steady_clock> coro_tic;
  std::chrono::time_point<std::chrono::steady_clock> coro_toc;

  cf::Coroflow cf{num_threads};
  std::vector<cf::TaskHandle> coros(num_coros);

  // emplace tasks
  for(size_t c = 0; c < num_coros; ++c) {
      coros[c] = cf.emplace([&cf, &streams, c, chain_size, cpu_ms, gpu_ms]() -> cf::Coro {
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
  std::cout << "Coro time: " << coro_dur << "ms\n";
}

void independent_task(
  size_t num_threads, 
  size_t num_tasks, 
  size_t chain_size, 
  int cpu_ms, 
  int gpu_ms,
  std::vector<cudaStream_t>& streams,
  std::vector<cudaEvent_t>& events
) {

  std::chrono::time_point<std::chrono::steady_clock> task_tic;
  std::chrono::time_point<std::chrono::steady_clock> task_toc;

  cf::Coroflow cf{num_threads};

  std::vector<cf::TaskHandle> tasks(num_tasks);

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
  std::cout << "Task time: " << task_dur << "ms\n";
}

int main(int argc, char* argv[]) {
  if(argc != 7) {
    std::cerr << "usage: ./bin/independent mode num_threads num_tasks chain_size cpu_ms gpu_ms\n";
    std::cerr << "mode should be 0, 1, or 2\n";
    std::cerr << "0: task, 1: coroutine, 2: both\n";
    std::exit(EXIT_FAILURE);
  }
  size_t mode = std::atoi(argv[1]);
  size_t num_threads = std::atoi(argv[2]);
  size_t num_tasks = std::atoi(argv[3]);
  size_t chain_size = std::atoi(argv[4]);
  int cpu_ms = std::atoi(argv[5]);
  int gpu_ms = std::atoi(argv[6]);
  std::cout << "configuration: (" 
            << num_threads << ", "
            << num_tasks << ", "
            << chain_size << ", "
            << cpu_ms << ", "
            << gpu_ms << "):\n";

  std::vector<cudaStream_t> streams(num_tasks);
  std::vector<cudaEvent_t>  events(num_tasks);
  for(auto& st: streams) {
    cudaStreamCreate(&st);
  }
  for(auto& ev: events) {
    cudaEventCreate(&ev);
  }

  if(mode == 0) {
    std::cout << "enable task...\n";
    independent_task(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms, streams, events);
  }
  else if(mode == 1) {
    std::cout << "enable coroutine...\n";
    independent_coro(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms, streams);
  }
  else if(mode == 2) {
    std::cout << "enable both task and coroutine...\n";
    independent_task(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms, streams, events);
    independent_coro(num_threads, num_tasks, chain_size, cpu_ms, gpu_ms, streams);
  }
  else {
    std::cerr << "mode should be 0, 1, or 2\n";
    std::cerr << "0: task, 1: coroutine, 2: both\n";
    std::exit(EXIT_FAILURE);
  }


  std::cout << "\n";

  for(auto& st: streams) {
    cudaStreamDestroy(st);
  }
}


