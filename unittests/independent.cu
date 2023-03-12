#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro/src/cuda/taro_v1.hpp>
#include <taro/src/cuda/taro_v2.hpp>
#include <taro/src/cuda/taro_v3.hpp>
#include <taro/src/cuda/taro_v4.hpp>
#include <taro/src/cuda/taro_v5.hpp>
#include <taro/src/cuda/taro_v6.hpp>
#include <taro/src/cuda/taro_v7.hpp>
#include <taro/src/cuda/taro_v8.hpp>
#include <taro/src/cuda/algorithm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>


//// --------------------------------------------------------
//// Testcase:: Independent
//// --------------------------------------------------------

//void independent_v1(size_t num_threads, size_t num_tasks) {
  //taro::TaroV1 taro{num_threads};

  //std::vector<cudaStream_t> streams(num_tasks);
  //for(auto& st: streams) {
    //cudaStreamCreate(&st);
  //}

  //std::vector<taro::TaskHandle> tasks(num_tasks);

  //int* a;
  //int* b; 
  //int* c;
  //size_t M{10};
  //size_t K{10};
  //size_t N{10};

  //size_t BLOCK_SIZE = 32;
  //dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  //dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  //cudaMallocManaged(&a, M * K * sizeof(int));
  //cudaMallocManaged(&b, K * N * sizeof(int));
  //cudaMallocManaged(&c, M * N * num_tasks * sizeof(int));
  //for(size_t i = 0; i < M * K; ++i) {
    //a[i] = M + K;
  //}
  //for(size_t i = 0; i < K * N; ++i) {
    //b[i] = K + N;
  //}

  //for(size_t i = 0; i < num_tasks; ++i) {
    //tasks[i] = taro.emplace([&streams, &taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      //cudaEvent_t finish;
      //cudaEventCreate(&finish);

      //taro::cuda_matmul<<<dim_grid, dim_block, 0, streams[i]>>>(a, b, c + i * M * N, M, K, N);
      //cudaEventRecord(finish);
      //auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };

      //while(!isdone()) {
        //co_await taro.suspend();
      //}

      //for(size_t k = 0; k < M * N; ++k) {
        //REQUIRE(c[i * M * N + k] == (int)(M + K) * (K + N) * K);
      //}

      //co_return;
    //});
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait();

  
  //REQUIRE(cudaFree(a) == cudaSuccess);
  //REQUIRE(cudaFree(b) == cudaSuccess);
  //REQUIRE(cudaFree(c) == cudaSuccess);

//}

//TEST_CASE("independent.v1.1thread.1task" * doctest::timeout(300)) {
  //independent_v1(1, 1);
//}

//TEST_CASE("independent.v1.2thread.3task" * doctest::timeout(300)) {
  //independent_v1(2, 3);
//}

//TEST_CASE("independent.v1.2thread.18task" * doctest::timeout(300)) {
  //independent_v1(2, 18);
//}

//TEST_CASE("independent.v1.3thread.2task" * doctest::timeout(300)) {
  //independent_v1(3, 2);
//}

//TEST_CASE("independent.v1.3thread.4task" * doctest::timeout(300)) {
  //independent_v1(3, 4);
//}

//TEST_CASE("independent.v1.3thread.18task" * doctest::timeout(300)) {
  //independent_v1(3, 18);
//}

//TEST_CASE("independent.v1.4thread.1task" * doctest::timeout(300)) {
  //independent_v1(4, 1);
//}

//TEST_CASE("independent.v1.4thread.11task" * doctest::timeout(300)) {
  //independent_v1(4, 11);
//}

//TEST_CASE("independent.v1.4thread.38task" * doctest::timeout(300)) {
  //independent_v1(4, 38);
//}

//TEST_CASE("independent.v1.4thread.123task" * doctest::timeout(300)) {
  //independent_v1(4, 123);
//}

//void independent_v2(size_t num_threads, size_t num_tasks) {
  //taro::TaroV2 taro{num_threads};

  //std::vector<cudaStream_t> streams(num_tasks);
  //for(auto& st: streams) {
    //cudaStreamCreate(&st);
  //}

  //std::vector<taro::TaskHandle> tasks(num_tasks);

  //int* a;
  //int* b; 
  //int* c;
  //size_t M{10};
  //size_t K{10};
  //size_t N{10};
  //size_t BLOCK_SIZE = 32;
  //dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  //dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  //cudaMallocManaged(&a, M * K * sizeof(int));
  //cudaMallocManaged(&b, K * N * sizeof(int));
  //cudaMallocManaged(&c, M * N * num_tasks * sizeof(int));
  //for(size_t i = 0; i < M * K; ++i) {
    //a[i] = M + K;
  //}
  //for(size_t i = 0; i < K * N; ++i) {
    //b[i] = K + N;
  //}

  //for(size_t i = 0; i < num_tasks; ++i) {
    //tasks[i] = taro.emplace( [&streams, &taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {

      //taro::cuda_matmul<<<dim_grid, dim_block, 0, streams[i]>>>(a, b, c + i * M * N, M, K, N);
      //co_await taro.cuda_suspend(streams[i]);

      //for(size_t k = 0; k < M * N; ++k) {
        //REQUIRE(c[i * M * N + k] == (int)(M + K) * (K + N) * K);
      //}

      //co_return;
    //});
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait();

  
  //REQUIRE(cudaFree(a) == cudaSuccess);
  //REQUIRE(cudaFree(b) == cudaSuccess);
  //REQUIRE(cudaFree(c) == cudaSuccess);

//}

//TEST_CASE("independent.v2.1thread.1task" * doctest::timeout(300)) {
  //independent_v2(1, 1);
//}

//TEST_CASE("independent.v2.2thread.3task" * doctest::timeout(300)) {
  //independent_v2(2, 3);
//}

//TEST_CASE("independent.v2.2thread.18task" * doctest::timeout(300)) {
  //independent_v2(2, 18);
//}

//TEST_CASE("independent.v2.3thread.2task" * doctest::timeout(300)) {
  //independent_v2(3, 2);
//}

//TEST_CASE("independent.v2.3thread.4task" * doctest::timeout(300)) {
  //independent_v2(3, 4);
//}

//TEST_CASE("independent.v2.3thread.18task" * doctest::timeout(300)) {
  //independent_v2(3, 18);
//}

//TEST_CASE("independent.v2.4thread.1task" * doctest::timeout(300)) {
  //independent_v2(4, 1);
//}

//TEST_CASE("independent.v2.4thread.11task" * doctest::timeout(300)) {
  //independent_v2(4, 11);
//}

//TEST_CASE("independent.v2.4thread.38task" * doctest::timeout(300)) {
  //independent_v2(4, 38);
//}

//TEST_CASE("independent.v2.4thread.123task" * doctest::timeout(300)) {
  //independent_v2(4, 123);
//}

//void independent_v3(size_t num_threads, size_t num_streams, size_t num_tasks) {
  //taro::TaroV3 taro{num_threads, num_streams};

  //std::vector<taro::TaskHandle> tasks(num_tasks);

  //int* a;
  //int* b; 
  //int* c;
  //size_t M{10};
  //size_t K{10};
  //size_t N{10};
  //size_t BLOCK_SIZE = 32;
  //dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  //dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  //cudaMallocManaged(&a, M * K * sizeof(int));
  //cudaMallocManaged(&b, K * N * sizeof(int));
  //cudaMallocManaged(&c, M * N * num_tasks * sizeof(int));
  //for(size_t i = 0; i < M * K; ++i) {
    //a[i] = M + K;
  //}
  //for(size_t i = 0; i < K * N; ++i) {
    //b[i] = K + N;
  //}

  //for(size_t i = 0; i < num_tasks; ++i) {
    //tasks[i] = taro.emplace([&taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      //co_await taro.cuda_suspend([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
        //taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
      //});

      //for(size_t k = 0; k < M * N; ++k) {
        //REQUIRE(c[k + i * M * N] == (int)(M + K) * (K + N) * K);
      //}

      //co_return;
    //});
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait();

  
  //REQUIRE(cudaFree(a) == cudaSuccess);
  //REQUIRE(cudaFree(b) == cudaSuccess);
  //REQUIRE(cudaFree(c) == cudaSuccess);

//}

//TEST_CASE("independent.v3.1thread.1stream.1task" * doctest::timeout(300)) {
  //independent_v3(1, 1, 1);
//}

//TEST_CASE("independent.v3.2thread.1stream.3task" * doctest::timeout(300)) {
  //independent_v3(2, 1, 3);
//}

//TEST_CASE("independent.v3.2thread.2stream.18task" * doctest::timeout(300)) {
  //independent_v3(2, 2, 18);
//}

//TEST_CASE("independent.v3.2thread.3stream.18task" * doctest::timeout(300)) {
  //independent_v3(2, 3, 18);
//}

//TEST_CASE("independent.v3.3thread.1stream.2task" * doctest::timeout(300)) {
  //independent_v3(3, 1, 2);
//}

//TEST_CASE("independent.v3.3thread.2stream.4task" * doctest::timeout(300)) {
  //independent_v3(3, 2, 4);
//}

//TEST_CASE("independent.v3.3thread.3stream.18task" * doctest::timeout(300)) {
  //independent_v3(3, 3, 18);
//}

//TEST_CASE("independent.v3.4thread.1stream.1task" * doctest::timeout(300)) {
  //independent_v3(4, 1, 1);
//}

//TEST_CASE("independent.v3.4thread.2stream.11task" * doctest::timeout(300)) {
  //independent_v3(4, 2, 11);
//}

//TEST_CASE("independent.v3.4thread.8stream.38task" * doctest::timeout(300)) {
  //independent_v3(4, 8, 38);
//}

//TEST_CASE("independent.v3.4thread.15stream.123task" * doctest::timeout(300)) {
  //independent_v3(4, 15, 123);
//}

//void independent_v4(size_t num_threads, size_t num_streams, size_t num_tasks) {
  //taro::TaroV4 taro{num_threads, num_streams};

  //std::vector<taro::TaskHandle> tasks(num_tasks);

  //int* a;
  //int* b; 
  //int* c;
  //size_t M{10};
  //size_t K{10};
  //size_t N{10};
  //size_t BLOCK_SIZE = 32;
  //dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  //dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  //cudaMallocManaged(&a, M * K * sizeof(int));
  //cudaMallocManaged(&b, K * N * sizeof(int));
  //cudaMallocManaged(&c, M * N * num_tasks * sizeof(int));
  //for(size_t i = 0; i < M * K; ++i) {
    //a[i] = M + K;
  //}
  //for(size_t i = 0; i < K * N; ++i) {
    //b[i] = K + N;
  //}

  //for(size_t i = 0; i < num_tasks; ++i) {
    //tasks[i] = taro.emplace([&taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      //co_await taro.cuda_suspend([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
        //taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
      //});

      //for(size_t k = 0; k < M * N; ++k) {
        //REQUIRE(c[k + i * M * N] == (int)(M + K) * (K + N) * K);
      //}

      //co_return;
    //});
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait();

  
  //REQUIRE(cudaFree(a) == cudaSuccess);
  //REQUIRE(cudaFree(b) == cudaSuccess);
  //REQUIRE(cudaFree(c) == cudaSuccess);

//}

//TEST_CASE("independent.v4.1thread.1stream.1task" * doctest::timeout(300)) {
  //independent_v4(1, 1, 1);
//}

//TEST_CASE("independent.v4.2thread.1stream.3task" * doctest::timeout(300)) {
  //independent_v4(2, 1, 3);
//}

//TEST_CASE("independent.v4.2thread.2stream.18task" * doctest::timeout(300)) {
  //independent_v4(2, 2, 18);
//}

//TEST_CASE("independent.v4.2thread.3stream.18task" * doctest::timeout(300)) {
  //independent_v4(2, 3, 18);
//}

//TEST_CASE("independent.v4.3thread.1stream.2task" * doctest::timeout(300)) {
  //independent_v4(3, 1, 2);
//}

//TEST_CASE("independent.v4.3thread.2stream.4task" * doctest::timeout(300)) {
  //independent_v4(3, 2, 4);
//}

//TEST_CASE("independent.v4.3thread.3stream.18task" * doctest::timeout(300)) {
  //independent_v4(3, 3, 18);
//}

//TEST_CASE("independent.v4.4thread.1stream.1task" * doctest::timeout(300)) {
  //independent_v4(4, 1, 1);
//}

//TEST_CASE("independent.v4.4thread.2stream.11task" * doctest::timeout(300)) {
  //independent_v4(4, 2, 11);
//}

//TEST_CASE("independent.v4.4thread.8stream.38task" * doctest::timeout(300)) {
  //independent_v4(4, 8, 38);
//}

//TEST_CASE("independent.v4.4thread.15stream.123task" * doctest::timeout(300)) {
  //independent_v4(4, 15, 123);
//}


//void independent_v5(size_t num_threads, size_t num_streams, size_t num_tasks) {
  //taro::TaroV5 taro{num_threads, num_streams};

  //std::vector<taro::TaskHandle> tasks(num_tasks);

  //int* a;
  //int* b; 
  //int* c;
  //size_t M{10};
  //size_t K{10};
  //size_t N{10};
  //size_t BLOCK_SIZE = 32;
  //dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  //dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  //cudaMallocManaged(&a, M * K * sizeof(int));
  //cudaMallocManaged(&b, K * N * sizeof(int));
  //cudaMallocManaged(&c, M * N * num_tasks * sizeof(int));
  //for(size_t i = 0; i < M * K; ++i) {
    //a[i] = M + K;
  //}
  //for(size_t i = 0; i < K * N; ++i) {
    //b[i] = K + N;
  //}

  //for(size_t i = 0; i < num_tasks; ++i) {
    //tasks[i] = taro.emplace([&taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      //co_await taro.cuda_suspend([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
        //taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
      //});

      //for(size_t k = 0; k < M * N; ++k) {
        //REQUIRE(c[k + i * M * N] == (int)(M + K) * (K + N) * K);
      //}

      //co_return;
    //});
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait();

  
  //REQUIRE(cudaFree(a) == cudaSuccess);
  //REQUIRE(cudaFree(b) == cudaSuccess);
  //REQUIRE(cudaFree(c) == cudaSuccess);

//}

//TEST_CASE("independent.v5.1thread.1stream.1task" * doctest::timeout(300)) {
  //independent_v5(1, 1, 1);
//}

//TEST_CASE("independent.v5.2thread.1stream.3task" * doctest::timeout(300)) {
  //independent_v5(2, 1, 3);
//}

//TEST_CASE("independent.v5.2thread.2stream.18task" * doctest::timeout(300)) {
  //independent_v5(2, 2, 18);
//}

//TEST_CASE("independent.v5.2thread.3stream.18task" * doctest::timeout(300)) {
  //independent_v5(2, 3, 18);
//}

//TEST_CASE("independent.v5.3thread.1stream.2task" * doctest::timeout(300)) {
  //independent_v5(3, 1, 2);
//}

//TEST_CASE("independent.v5.3thread.2stream.4task" * doctest::timeout(300)) {
  //independent_v5(3, 2, 4);
//}

//TEST_CASE("independent.v5.3thread.3stream.18task" * doctest::timeout(300)) {
  //independent_v5(3, 3, 18);
//}

//TEST_CASE("independent.v5.4thread.1stream.1task" * doctest::timeout(300)) {
  //independent_v5(4, 1, 1);
//}

//TEST_CASE("independent.v5.4thread.2stream.11task" * doctest::timeout(300)) {
  //independent_v5(4, 2, 11);
//}

//TEST_CASE("independent.v5.4thread.8stream.38task" * doctest::timeout(300)) {
  //independent_v5(4, 8, 38);
//}

//TEST_CASE("independent.v5.4thread.15stream.123task" * doctest::timeout(300)) {
  //independent_v5(4, 15, 123);
//}


void independent_v6(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::TaroV6 taro{num_threads, num_streams};

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
    tasks[i] = taro.emplace([&taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      co_await taro.cuda_suspend([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
        taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
      });

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

TEST_CASE("independent.v6.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_v6(1, 1, 1);
}

TEST_CASE("independent.v6.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_v6(2, 1, 3);
}

TEST_CASE("independent.v6.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_v6(2, 2, 18);
}

TEST_CASE("independent.v6.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_v6(2, 3, 18);
}

TEST_CASE("independent.v6.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_v6(3, 1, 2);
}

TEST_CASE("independent.v6.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_v6(3, 2, 4);
}

TEST_CASE("independent.v6.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_v6(3, 3, 18);
}

TEST_CASE("independent.v6.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_v6(4, 1, 1);
}

TEST_CASE("independent.v6.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_v6(4, 2, 11);
}

TEST_CASE("independent.v6.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_v6(4, 8, 38);
}

TEST_CASE("independent.v6.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_v6(4, 15, 123);
}

void independent_v7(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::TaroV7 taro{num_threads, num_streams};

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
    tasks[i] = taro.emplace([&taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      co_await taro.cuda_suspend([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
        taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
      });

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

TEST_CASE("independent.v7.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_v7(1, 1, 1);
}

TEST_CASE("independent.v7.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_v7(2, 1, 3);
}

TEST_CASE("independent.v7.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_v7(2, 2, 18);
}

TEST_CASE("independent.v7.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_v7(2, 3, 18);
}

TEST_CASE("independent.v7.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_v7(3, 1, 2);
}

TEST_CASE("independent.v7.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_v7(3, 2, 4);
}

TEST_CASE("independent.v7.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_v7(3, 3, 18);
}

TEST_CASE("independent.v7.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_v7(4, 1, 1);
}

TEST_CASE("independent.v7.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_v7(4, 2, 11);
}

TEST_CASE("independent.v7.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_v7(4, 8, 38);
}

TEST_CASE("independent.v7.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_v7(4, 15, 123);
}


void independent_v8(size_t num_threads, size_t num_tasks) {
  taro::TaroV8 taro{num_threads};

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
    tasks[i] = taro.emplace([&taro, i, a, b, c, M, K, N, dim_grid, dim_block]() -> taro::Coro {
      co_await taro.cuda_suspend([a, b, c, i, M, K, N, dim_grid, dim_block](cudaStream_t st) {
        taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);
      });

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

TEST_CASE("independent.v8.1thread.1task" * doctest::timeout(300)) {
  independent_v8(1, 1);
}

TEST_CASE("independent.v8.2thread.3task" * doctest::timeout(300)) {
  independent_v8(2, 3);
}

TEST_CASE("independent.v8.2thread.18task" * doctest::timeout(300)) {
  independent_v8(2, 18);
}

TEST_CASE("independent.v8.2thread.18task" * doctest::timeout(300)) {
  independent_v8(2, 18);
}

TEST_CASE("independent.v8.3thread.2task" * doctest::timeout(300)) {
  independent_v8(3, 2);
}

TEST_CASE("independent.v8.3thread.4task" * doctest::timeout(300)) {
  independent_v8(3, 4);
}

TEST_CASE("independent.v8.3thread.18task" * doctest::timeout(300)) {
  independent_v8(3, 18);
}

TEST_CASE("independent.v8.4thread.1task" * doctest::timeout(300)) {
  independent_v8(4, 1);
}

TEST_CASE("independent.v8.4thread.11task" * doctest::timeout(300)) {
  independent_v8(4, 11);
}

TEST_CASE("independent.v8.4thread.38task" * doctest::timeout(300)) {
  independent_v8(4, 38);
}

TEST_CASE("independent.v8.4threadm.123task" * doctest::timeout(300)) {
  independent_v8(4, 123);
}
