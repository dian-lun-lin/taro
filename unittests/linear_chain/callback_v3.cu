#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/src/cuda/callback/v3/taro_callback_v3.hpp>
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

void linear_chain_cbv3(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  taro::TaroCBV3 taro{num_threads, num_streams};
  std::vector<taro::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = taro.emplace([t, counter, &taro]() -> taro::Coro {
      REQUIRE(*counter == t); 

      co_await taro.cuda_suspend([counter](cudaStream_t st) {
        count<<<8, 32, 0, st>>>(counter);
      });

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].precede(_tasks[t + 1]);
  }

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait(); 
}

TEST_CASE("linear_chain_cbv3.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_cbv3(1, 1, 1);
}

TEST_CASE("linear_chain_cbv3.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_cbv3(99, 2, 2);
}

TEST_CASE("linear_chain_cbv3.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_cbv3(712, 3, 4);
}

TEST_CASE("linear_chain_cbv3.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_cbv3(443, 4, 8);
}

TEST_CASE("linear_chain_cbv3.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_cbv3(1111, 5, 2);
}

TEST_CASE("linear_chain_cbv3.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_cbv3(2, 6, 3);
}

TEST_CASE("linear_chain_cbv3.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_cbv3(5, 7, 1);
}

TEST_CASE("linear_chain_cbv3.8threads" * doctest::timeout(300)) {
  linear_chain_cbv3(9211, 8, 9);
}
