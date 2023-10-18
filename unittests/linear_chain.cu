#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/await/cuda.hpp>
#include <taro/algorithm/cuda.hpp>
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

void linear_chain(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  taro::Taro taro{num_threads};
  auto cuda = taro.cuda_await(num_streams);

  std::vector<taro::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = taro.emplace([t, counter, &cuda]() -> taro::Coro {
      REQUIRE(*counter == t); 
      if(t % 3 == 0) {
        co_await cuda.until_callback([counter](cudaStream_t st) {
          count<<<8, 32, 0, st>>>(counter);
        });
      }
      else if(t % 3 == 1) {
        co_await cuda.until_polling([counter](cudaStream_t st) {
          count<<<8, 32, 0, st>>>(counter);
        });
      }
      else {
        cuda.wait([counter](cudaStream_t st) {
          count<<<8, 32, 0, st>>>(counter);
        });
      }

      REQUIRE(*counter == t + 1); 
      co_return;
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].precede(_tasks[t + 1]);
  }

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait(); 
  cudaFree(counter);
}

TEST_CASE("linear_chain.1thread.1stream" * doctest::timeout(300)) {
  linear_chain(1, 1, 1);
}

TEST_CASE("linear_chain.2thread.2stream" * doctest::timeout(300)) {
  linear_chain(99, 2, 2);
}

TEST_CASE("linear_chain.3thread.4stream" * doctest::timeout(300)) {
  linear_chain(712, 3, 4);
}

TEST_CASE("linear_chain.4thread.8stream" * doctest::timeout(300)) {
  linear_chain(443, 4, 8);
}

TEST_CASE("linear_chain.5thread.2stream" * doctest::timeout(300)) {
  linear_chain(1111, 5, 2);
}

TEST_CASE("linear_chain.6thread.3stream" * doctest::timeout(300)) {
  linear_chain(2, 6, 3);
}

TEST_CASE("linear_chain.7thread.1stream" * doctest::timeout(300)) {
  linear_chain(5, 7, 1);
}

TEST_CASE("linear_chain.8threads" * doctest::timeout(300)) {
  linear_chain(9211, 8, 9);
}
