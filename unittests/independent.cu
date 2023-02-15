#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <coroflow/src/cuda/coroflow_v1.hpp>
#include <coroflow/src/cuda/coroflow_v2.hpp>
#include <coroflow/src/cuda/coroflow_v3.hpp>
#include <coroflow/src/cuda/algorithm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

// --------------------------------------------------------
// Testcase:: Independent
// --------------------------------------------------------

void independent_v1(size_t num_threads, size_t num_tasks) {
  cf::CoroflowV1 cf{num_threads};

  std::vector<cudaStream_t> streams(num_tasks);
  for(auto& st: streams) {
    cudaStreamCreate(&st);
  }

  std::vector<cf::TaskHandle> tasks(num_tasks);

  int* a;
  int* b; 
  int* c;
  size_t M{10};
  size_t K{10};
  size_t N{10};
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
    tasks[i] = cf.emplace([&streams, &cf, i, a, b, c, M, K, N]() -> cf::Coro {
      cudaEvent_t finish;
      cudaEventCreate(&finish);

      cf::cuda_matmul<<<8, 32, 0, streams[i]>>>(a, b, c + i * M * N, M, K, N);
      cudaEventRecord(finish);
      auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };

      while(!isdone()) {
        co_await cf.suspend();
      }

      for(size_t k = 0; k < M * N; ++k) {
        REQUIRE(c[i * M * N + k] == (int)(M + K) * (K + N) * K);
      }

      co_return;
    });
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait();

  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

TEST_CASE("independent.v1.1thread.1task" * doctest::timeout(300)) {
  independent_v1(1, 1);
}

TEST_CASE("independent.v1.2thread.3task" * doctest::timeout(300)) {
  independent_v1(2, 3);
}

TEST_CASE("independent.v1.2thread.18task" * doctest::timeout(300)) {
  independent_v1(2, 18);
}

TEST_CASE("independent.v1.3thread.2task" * doctest::timeout(300)) {
  independent_v1(3, 2);
}

TEST_CASE("independent.v1.3thread.4task" * doctest::timeout(300)) {
  independent_v1(3, 4);
}

TEST_CASE("independent.v1.3thread.18task" * doctest::timeout(300)) {
  independent_v1(3, 18);
}

TEST_CASE("independent.v1.4thread.1task" * doctest::timeout(300)) {
  independent_v1(4, 1);
}

TEST_CASE("independent.v1.4thread.11task" * doctest::timeout(300)) {
  independent_v1(4, 11);
}

TEST_CASE("independent.v1.4thread.38task" * doctest::timeout(300)) {
  independent_v1(4, 38);
}

TEST_CASE("independent.v1.4thread.123task" * doctest::timeout(300)) {
  independent_v1(4, 123);
}

void independent_v2(size_t num_threads, size_t num_tasks) {
  cf::CoroflowV2 cf{num_threads};

  std::vector<cudaStream_t> streams(num_tasks);
  for(auto& st: streams) {
    cudaStreamCreate(&st);
  }

  std::vector<cf::TaskHandle> tasks(num_tasks);

  int* a;
  int* b; 
  int* c;
  size_t M{10};
  size_t K{10};
  size_t N{10};
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
    tasks[i] = cf.emplace( [&streams, &cf, i, a, b, c, M, K, N]() -> cf::Coro {

      cf::cuda_matmul<<<8, 32, 0, streams[i]>>>(a, b, c + i * M * N, M, K, N);
      co_await cf.cuda_suspend(streams[i]);

      for(size_t k = 0; k < M * N; ++k) {
        REQUIRE(c[i * M * N + k] == (int)(M + K) * (K + N) * K);
      }

      co_return;
    });
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait();

  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

TEST_CASE("independent.v2.1thread.1task" * doctest::timeout(300)) {
  independent_v2(1, 1);
}

TEST_CASE("independent.v2.2thread.3task" * doctest::timeout(300)) {
  independent_v2(2, 3);
}

TEST_CASE("independent.v2.2thread.18task" * doctest::timeout(300)) {
  independent_v2(2, 18);
}

TEST_CASE("independent.v2.3thread.2task" * doctest::timeout(300)) {
  independent_v2(3, 2);
}

TEST_CASE("independent.v2.3thread.4task" * doctest::timeout(300)) {
  independent_v2(3, 4);
}

TEST_CASE("independent.v2.3thread.18task" * doctest::timeout(300)) {
  independent_v2(3, 18);
}

TEST_CASE("independent.v2.4thread.1task" * doctest::timeout(300)) {
  independent_v2(4, 1);
}

TEST_CASE("independent.v2.4thread.11task" * doctest::timeout(300)) {
  independent_v2(4, 11);
}

TEST_CASE("independent.v2.4thread.38task" * doctest::timeout(300)) {
  independent_v2(4, 38);
}

TEST_CASE("independent.v2.4thread.123task" * doctest::timeout(300)) {
  independent_v2(4, 123);
}

void independent_v3(size_t num_threads, size_t num_streams, size_t num_tasks) {
  cf::CoroflowV3 cf{num_threads, num_streams};

  std::vector<cf::TaskHandle> tasks(num_tasks);

  int* a;
  int* b; 
  int* c;
  size_t M{10};
  size_t K{10};
  size_t N{10};
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
    tasks[i] = cf.emplace([&cf, i, a, b, c, M, K, N]() -> cf::Coro {
      co_await cf.cuda_suspend([a, b, c, i, M, K, N](cudaStream_t st) {
        cf::cuda_matmul<<<8, 32, 0, st>>>(a, b, c + i * M * N, M, K, N);
      });

      for(size_t k = 0; k < M * N; ++k) {
        REQUIRE(c[k + i * M * N] == (int)(M + K) * (K + N) * K);
      }

      co_return;
    });
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait();

  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

TEST_CASE("independent.v3.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_v3(1, 1, 1);
}

TEST_CASE("independent.v3.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_v3(2, 1, 3);
}

TEST_CASE("independent.v3.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_v3(2, 2, 18);
}

TEST_CASE("independent.v3.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_v3(2, 3, 18);
}

TEST_CASE("independent.v3.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_v3(3, 1, 2);
}

TEST_CASE("independent.v3.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_v3(3, 2, 4);
}

TEST_CASE("independent.v3.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_v3(3, 3, 18);
}

TEST_CASE("independent.v3.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_v3(4, 1, 1);
}

TEST_CASE("independent.v3.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_v3(4, 2, 11);
}

TEST_CASE("independent.v3.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_v3(4, 8, 38);
}

TEST_CASE("independent.v3.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_v3(4, 15, 123);
}
