#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/await/cuda.hpp>
#include <taro/algorithm/cuda.hpp>
#include <vector>
#include <algorithm>
#include <numeric>


//// --------------------------------------------------------
//// Testcase:: Independent
//// --------------------------------------------------------


void independent(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::Taro taro{num_threads};
  auto cuda = taro.cuda_await(num_streams);

  std::vector<taro::TaskHandle> tasks(num_tasks);

  int* a;
  int* b; 
  int* c;
  size_t M{10};
  size_t K{10};
  size_t N{10};
  size_t BLOCK_SIZE = 32;
  dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  cudaMallocManaged(&a, M * K * sizeof(int));
  cudaMallocManaged(&b, K * N * sizeof(int));
  cudaMallocManaged(&c, M * N * num_tasks * sizeof(int));
  for(size_t i = 0; i < M * K; ++i) {
    a[i] = M + K;
  }
  for(size_t i = 0; i < K * N; ++i) {
    b[i] = K + N;
  }

  for(size_t i = 0; i < num_tasks; ++i) {
    tasks[i] = taro.emplace([&cuda, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      if(i % 3 == 0) {
        co_await cuda.until_callback([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
          taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
        });
      }
      else if (i % 3 == 1){
        co_await cuda.until_polling([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
          taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
        });
      }
      else {
        cuda.wait([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
          taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
        });
      }

      for(size_t k = 0; k < M * N; ++k) {
        REQUIRE(c[k + i * M * N] == (int)(M + K) * (K + N) * K);
      }

      co_return;
    });
  }

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait();

  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

TEST_CASE("independent.cbv4.1thread.1stream.1task" * doctest::timeout(300)) {
  independent(1, 1, 1);
}

TEST_CASE("independent.cbv4.2thread.1stream.3task" * doctest::timeout(300)) {
  independent(2, 1, 3);
}

TEST_CASE("independent.cbv4.2thread.2stream.18task" * doctest::timeout(300)) {
  independent(2, 2, 18);
}

TEST_CASE("independent.cbv4.2thread.3stream.18task" * doctest::timeout(300)) {
  independent(2, 3, 18);
}

TEST_CASE("independent.cbv4.3thread.1stream.2task" * doctest::timeout(300)) {
  independent(3, 1, 2);
}

TEST_CASE("independent.cbv4.3thread.2stream.4task" * doctest::timeout(300)) {
  independent(3, 2, 4);
}

TEST_CASE("independent.cbv4.3thread.3stream.18task" * doctest::timeout(300)) {
  independent(3, 3, 18);
}

TEST_CASE("independent.cbv4.4thread.1stream.1task" * doctest::timeout(300)) {
  independent(4, 1, 1);
}

TEST_CASE("independent.cbv4.4thread.2stream.11task" * doctest::timeout(300)) {
  independent(4, 2, 11);
}

TEST_CASE("independent.cbv4.4thread.8stream.38task" * doctest::timeout(300)) {
  independent(4, 8, 38);
}

TEST_CASE("independent.cbv4.4thread.15stream.123task" * doctest::timeout(300)) {
  independent(4, 15, 123);
}
