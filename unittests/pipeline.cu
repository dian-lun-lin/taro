#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/pattern/pipeline.hpp>
#include <taro/await/cuda.hpp>
#include <taro/algorithm/cuda.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

// --------------------------------------------------------
// Testcase:: Pipeline
// --------------------------------------------------------

template <typename T>
__global__
void count(T* count) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx == 0) {
    ++(*count);
  }
}

void pipeline(size_t num_threads, size_t num_streams, size_t num_pipes, size_t num_lines, size_t num_tokens) {
  int* counter;
  cudaMallocManaged(&counter, num_tokens * sizeof(int));
  cudaMemset(counter, 0, num_tokens * sizeof(int));

  taro::Taro taro{num_threads};
  auto pipeline = taro.pipeline(num_pipes, num_lines, num_tokens);
  auto cuda = taro.cuda_await(num_streams);

  // first pipe
  pipeline.set_pipe(0, [counter, &cuda, &pipeline]() -> taro::Coro {
    size_t token = pipeline.fetch_token();
    for(;token < pipeline.num_tokens(); token = pipeline.fetch_token()) {

      REQUIRE(*(counter + token) == 0); 

      co_await cuda.until_callback([=](cudaStream_t st) {
        count<<<8, 32, 0, st>>>(counter + token);
      });

      REQUIRE(*(counter + token) == 1); 

      co_await pipeline.step();
    }

    pipeline.stop();

  });

  for(size_t p = 1; p < num_pipes; ++p) {
    pipeline.set_pipe(p, [p, counter, &cuda, &pipeline]() -> taro::Coro {
      while(1) {
        size_t token = pipeline.token();

        REQUIRE(*(counter + token) == p); 

        co_await cuda.until_callback([=](cudaStream_t st) {
          count<<<8, 32, 0, st>>>(counter + token);
        });

        REQUIRE(*(counter + token) == p + 1); 

        co_await pipeline.step();
      }
    });
  }

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait(); 
  cudaFree(counter);
}

TEST_CASE("pipeline.1threads.1streams.1pipes.1lines.1tokens" * doctest::timeout(300)) {
  pipeline(1, 1, 1, 1, 1);
}
TEST_CASE("pipeline.1threads.1streams.1pipes..1lines.2tokens" * doctest::timeout(300)) {
  pipeline(1, 1, 1, 1, 2);
}
TEST_CASE("pipeline.1threads.1streams.2pipes.1lines.1tokens" * doctest::timeout(300)) {
  pipeline(1, 1, 2, 1, 1);
}
TEST_CASE("pipeline.1threads.1streams.2pipes.2lines.2tokens" * doctest::timeout(300)) {
  pipeline(1, 1, 2, 2, 2);
}
TEST_CASE("pipeline.1threads.2streams.1pipes.2lines.2tokens" * doctest::timeout(300)) {
  pipeline(1, 2, 1, 2, 2);
}
TEST_CASE("pipeline.3threads.1streams.2pipes.2lines.2tokens" * doctest::timeout(300)) {
  pipeline(3, 1, 2, 2, 2);
}
TEST_CASE("pipeline.3threads.1streams.2pipes.10lines.2tokens" * doctest::timeout(300)) {
  pipeline(3, 1, 2, 10, 2);
}
TEST_CASE("pipeline.5threads.5streams.3pipes.2lines.2tokens" * doctest::timeout(300)) {
  pipeline(5, 5, 3, 2, 2);
}
TEST_CASE("pipeline.3threads.5streams.13pipes.3lines.8tokens" * doctest::timeout(300)) {
  pipeline(3, 5, 13, 3, 8);
}
TEST_CASE("pipeline.4threads.2streams.19pipes.7lines.199tokens" * doctest::timeout(300)) {
  pipeline(4, 2, 19, 7, 199);
}
TEST_CASE("pipeline.4threads.2streams.19pipes.1lines.199tokens" * doctest::timeout(300)) {
  pipeline(4, 2, 19, 1, 199);
}
