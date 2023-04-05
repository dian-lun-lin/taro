#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include "../benchmarks/boost_fiber/fiber.hpp"
#include <taro/src/cuda/algorithm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

// --------------------------------------------------------
// Testcase:: Linear chain
// --------------------------------------------------------

// o - o - o - o

template <typename T>
__global__
void count(T* count) {
  ++(*count);
}

void linear_chain_fiber(size_t num_tasks, size_t num_threads) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  FiberTaskScheduler ft_sched{num_threads};
  std::vector<FiberTaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = ft_sched.emplace([t, counter, &ft_sched]() {
      REQUIRE(*counter == t); 
      cudaStream_t st;
      cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);

      count<<<8, 32, 0, st>>>(counter);

      boost::fibers::cuda::waitfor_all(st);
      cudaStreamDestroy(st);


      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].precede(_tasks[t + 1]);
  }

  ft_sched.schedule();
  ft_sched.wait(); 
}

TEST_CASE("linear_chain_fiber.2thread" * doctest::timeout(300)) {
  linear_chain_fiber(99, 2);
}

TEST_CASE("linear_chain_fiber.3thread" * doctest::timeout(300)) {
  linear_chain_fiber(712, 3);
}

TEST_CASE("linear_chain_fiber.4thread" * doctest::timeout(300)) {
  linear_chain_fiber(443, 4);
}

TEST_CASE("linear_chain_fiber.5thread" * doctest::timeout(300)) {
  linear_chain_fiber(1111, 5);
}

TEST_CASE("linear_chain_fiber.6thread" * doctest::timeout(300)) {
  linear_chain_fiber(2, 6);
}

TEST_CASE("linear_chain_fiber.7thread" * doctest::timeout(300)) {
  linear_chain_fiber(5, 7);
}

TEST_CASE("linear_chain_fiber.8thread" * doctest::timeout(300)) {
  linear_chain_fiber(9211, 8);
}

