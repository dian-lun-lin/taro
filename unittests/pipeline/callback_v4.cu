#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/src/cuda/callback/v4/pipeline.hpp>
#include <taro/src/cuda/algorithm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

// --------------------------------------------------------
// Testcase:: Pipeline
// --------------------------------------------------------

template <typename T>
__global__
void count(T* count) {
  ++(*count);
}

void pipeline_cbv4(size_t num_threads, size_t num_streams, size_t num_pipes, size_t num_tokens) {
  int* counter;
  cudaMallocManaged(&counter, num_pipes * num_tokens * sizeof(int));
  cudaMemset(counter, 0, num_pipes * num_tokens * sizeof(int));

  taro::TaroCBV4 taro{num_threads, num_streams};
  auto pipeline = taro::pipeline(taro, num_pipes, num_tokens);

  for(size_t p = 0; p < num_pipes; ++p) {
    pipeline.set_pipe(p, [p, counter, &taro, &pipeline]() -> taro::Coro {
      while(!pipeline.done(p)) {
        auto token = pipeline.token(p);
        REQUIRE(*(counter + p * pipeline.num_tokens() + token) == 0); 

        co_await taro.cuda_suspend([=, &pipeline](cudaStream_t st) {
          count<<<8, 32, 0, st>>>(counter + p * pipeline.num_tokens() + token);
        });

        REQUIRE(*(counter + p * pipeline.num_tokens() + token) == 1); 

        co_await pipeline.step(p);
      }
    });
  }

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait(); 
  cudaFree(counter);
}

TEST_CASE("pipeline_cbv4.1threads.1streams.1pipes.1tokens" * doctest::timeout(300)) {
  pipeline_cbv4(1, 1, 1, 1);
}
TEST_CASE("pipeline_cbv4.1threads.1streams.1pipes.2tokens" * doctest::timeout(300)) {
  pipeline_cbv4(1, 1, 1, 2);
}
TEST_CASE("pipeline_cbv4.1threads.1streams.2pipes.1tokens" * doctest::timeout(300)) {
  pipeline_cbv4(1, 1, 2, 1);
}
TEST_CASE("pipeline_cbv4.1threads.1streams.2pipes.2tokens" * doctest::timeout(300)) {
  pipeline_cbv4(1, 1, 2, 2);
}
TEST_CASE("pipeline_cbv4.1threads.2streams.1pipes.2tokens" * doctest::timeout(300)) {
  pipeline_cbv4(1, 2, 1, 2);
}
TEST_CASE("pipeline_cbv4.3threads.1streams.2pipes.2tokens" * doctest::timeout(300)) {
  pipeline_cbv4(3, 1, 2, 2);
}
TEST_CASE("pipeline_cbv4.5threads.5streams.3pipes.2tokens" * doctest::timeout(300)) {
  pipeline_cbv4(5, 5, 3, 2);
}
TEST_CASE("pipeline_cbv4.3threads.5streams.13pipes.8tokens" * doctest::timeout(300)) {
  pipeline_cbv4(3, 5, 13, 8);
}
TEST_CASE("pipeline_cbv4.4threads.2streams.19pipes.199tokens" * doctest::timeout(300)) {
  pipeline_cbv4(4, 2, 19, 199);
}
TEST_CASE("pipeline_cbv4.6threads.4streams.99pipes.99tokens" * doctest::timeout(300)) {
  pipeline_cbv4(6, 4, 99, 99);
}
