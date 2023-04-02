#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/src/cuda/callback/taro_callback_v1.hpp>
#include <taro/src/cuda/callback/taro_callback_v2.hpp>
//#include <taro/src/cuda/callback/taro_callback_v3.hpp>
#include <taro/src/cuda/callback/taro_callback_taskflow.hpp>
//#include <taro/src/cuda/callback/taro_callback_taskflow_runtime.hpp>

#include "../benchmarks/boost_fiber/fiber.hpp"

#include <taro/src/cuda/poll/taro_poll_v1.hpp>
#include <taro/src/cuda/poll/taro_poll_v2.hpp>
#include <taro/src/cuda/algorithm.hpp>
#include <vector>
#include <algorithm>
#include <numeric>


//// --------------------------------------------------------
//// Testcase:: Independent
//// --------------------------------------------------------

void independent_cbv1(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::TaroCBV1 taro{num_threads, num_streams};

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

TEST_CASE("independent.cbv1.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_cbv1(1, 1, 1);
}

TEST_CASE("independent.cbv1.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_cbv1(2, 1, 3);
}

TEST_CASE("independent.cbv1.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_cbv1(2, 2, 18);
}

TEST_CASE("independent.cbv1.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_cbv1(2, 3, 18);
}

TEST_CASE("independent.cbv1.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_cbv1(3, 1, 2);
}

TEST_CASE("independent.cbv1.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_cbv1(3, 2, 4);
}

TEST_CASE("independent.cbv1.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_cbv1(3, 3, 18);
}

TEST_CASE("independent.cbv1.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_cbv1(4, 1, 1);
}

TEST_CASE("independent.cbv1.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_cbv1(4, 2, 11);
}

TEST_CASE("independent.cbv1.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_cbv1(4, 8, 38);
}

TEST_CASE("independent.cbv1.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_cbv1(4, 15, 123);
}

void independent_cbv2(size_t num_threads, size_t num_tasks) {
  taro::TaroCBV2 taro{num_threads};

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

TEST_CASE("independent.cbv2.1thread.1task" * doctest::timeout(300)) {
  independent_cbv2(1, 1);
}

TEST_CASE("independent.cbv2.2thread.3task" * doctest::timeout(300)) {
  independent_cbv2(2, 3);
}

TEST_CASE("independent.cbv2.2thread.18task" * doctest::timeout(300)) {
  independent_cbv2(2, 18);
}

TEST_CASE("independent.cbv2.2thread.18task" * doctest::timeout(300)) {
  independent_cbv2(2, 18);
}

TEST_CASE("independent.cbv2.3thread.2task" * doctest::timeout(300)) {
  independent_cbv2(3, 2);
}

TEST_CASE("independent.cbv2.3thread.4task" * doctest::timeout(300)) {
  independent_cbv2(3, 4);
}

TEST_CASE("independent.cbv2.3thread.18task" * doctest::timeout(300)) {
  independent_cbv2(3, 18);
}

TEST_CASE("independent.cbv2.4thread.1task" * doctest::timeout(300)) {
  independent_cbv2(4, 1);
}

TEST_CASE("independent.cbv2.4thread.11task" * doctest::timeout(300)) {
  independent_cbv2(4, 11);
}

TEST_CASE("independent.cbv2.4thread.38task" * doctest::timeout(300)) {
  independent_cbv2(4, 38);
}

TEST_CASE("independent.cbv2.4threadm.123task" * doctest::timeout(300)) {
  independent_cbv2(4, 123);
}

//void independent_cbv3(size_t num_threads, size_t num_streams, size_t num_tasks) {
  //taro::TaroCBV3 taro{num_threads, num_streams};

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

//TEST_CASE("independent.cbv3.1thread.1stream.1task" * doctest::timeout(300)) {
  //independent_cbv3(1, 1, 1);
//}

//TEST_CASE("independent.cbv3.2thread.1stream.3task" * doctest::timeout(300)) {
  //independent_cbv3(2, 1, 3);
//}

//TEST_CASE("independent.cbv3.2thread.2stream.18task" * doctest::timeout(300)) {
  //independent_cbv3(2, 2, 18);
//}

//TEST_CASE("independent.cbv3.2thread.3stream.18task" * doctest::timeout(300)) {
  //independent_cbv3(2, 3, 18);
//}

//TEST_CASE("independent.cbv3.3thread.1stream.2task" * doctest::timeout(300)) {
  //independent_cbv3(3, 1, 2);
//}

//TEST_CASE("independent.cbv3.3thread.2stream.4task" * doctest::timeout(300)) {
  //independent_cbv3(3, 2, 4);
//}

//TEST_CASE("independent.cbv3.3thread.3stream.18task" * doctest::timeout(300)) {
  //independent_cbv3(3, 3, 18);
//}

//TEST_CASE("independent.cbv3.4thread.1stream.1task" * doctest::timeout(300)) {
  //independent_cbv3(4, 1, 1);
//}

//TEST_CASE("independent.cbv3.4thread.2stream.11task" * doctest::timeout(300)) {
  //independent_cbv3(4, 2, 11);
//}

//TEST_CASE("independent.cbv3.4thread.8stream.38task" * doctest::timeout(300)) {
  //independent_cbv3(4, 8, 38);
//}

//TEST_CASE("independent.cbv3.4thread.15stream.123task" * doctest::timeout(300)) {
  //independent_cbv3(4, 15, 123);
//}

void independent_cbtaskflow(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::TaroCBTaskflow taro{num_threads, num_streams};

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
  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

TEST_CASE("independent.cbtaskflow.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_cbtaskflow(1, 1, 1);
}

TEST_CASE("independent.cbtaskflow.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_cbtaskflow(2, 1, 3);
}

TEST_CASE("independent.cbtaskflow.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_cbtaskflow(2, 2, 18);
}

TEST_CASE("independent.cbtaskflow.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_cbtaskflow(2, 3, 18);
}

TEST_CASE("independent.cbtaskflow.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_cbtaskflow(3, 1, 2);
}

TEST_CASE("independent.cbtaskflow.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_cbtaskflow(3, 2, 4);
}

TEST_CASE("independent.cbtaskflow.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_cbtaskflow(3, 3, 18);
}

TEST_CASE("independent.cbtaskflow.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_cbtaskflow(4, 1, 1);
}

TEST_CASE("independent.cbtaskflow.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_cbtaskflow(4, 2, 11);
}

TEST_CASE("independent.cbtaskflow.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_cbtaskflow(4, 8, 38);
}

TEST_CASE("independent.cbtaskflow.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_cbtaskflow(4, 15, 123);
}

//void independent_cbtaskflowruntime(size_t num_threads, size_t num_streams, size_t num_tasks) {
  //taro::TaroCBTaskflowRuntime taro{num_threads, num_streams};

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
  
  //REQUIRE(cudaFree(a) == cudaSuccess);
  //REQUIRE(cudaFree(b) == cudaSuccess);
  //REQUIRE(cudaFree(c) == cudaSuccess);

//}

//TEST_CASE("independent.cbtaskflowruntime.1thread.1stream.1task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(1, 1, 1);
//}

//TEST_CASE("independent.cbtaskflowruntime.2thread.1stream.3task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(2, 1, 3);
//}

//TEST_CASE("independent.cbtaskflowruntime.2thread.2stream.18task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(2, 2, 18);
//}

//TEST_CASE("independent.cbtaskflowruntime.2thread.3stream.18task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(2, 3, 18);
//}

//TEST_CASE("independent.cbtaskflowruntime.3thread.1stream.2task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(3, 1, 2);
//}

//TEST_CASE("independent.cbtaskflowruntime.3thread.2stream.4task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(3, 2, 4);
//}

//TEST_CASE("independent.cbtaskflowruntime.3thread.3stream.18task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(3, 3, 18);
//}

//TEST_CASE("independent.cbtaskflowruntime.4thread.1stream.1task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(4, 1, 1);
//}

//TEST_CASE("independent.cbtaskflowruntime.4thread.2stream.11task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(4, 2, 11);
//}

//TEST_CASE("independent.cbtaskflowruntime.4thread.8stream.38task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(4, 8, 38);
//}

//TEST_CASE("independent.cbtaskflowruntime.4thread.15stream.123task" * doctest::timeout(300)) {
  //independent_cbtaskflowruntime(4, 15, 123);
//}

void independent_pv1(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::TaroPV1 taro{num_threads, num_streams};

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

TEST_CASE("independent.pv1.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_pv1(1, 1, 1);
}

TEST_CASE("independent.pv1.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_pv1(2, 1, 3);
}

TEST_CASE("independent.pv1.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_pv1(2, 2, 18);
}

TEST_CASE("independent.pv1.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_pv1(2, 3, 18);
}

TEST_CASE("independent.pv1.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_pv1(3, 1, 2);
}

TEST_CASE("independent.pv1.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_pv1(3, 2, 4);
}

TEST_CASE("independent.pv1.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_pv1(3, 3, 18);
}

TEST_CASE("independent.pv1.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_pv1(4, 1, 1);
}

TEST_CASE("independent.pv1.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_pv1(4, 2, 11);
}

TEST_CASE("independent.pv1.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_pv1(4, 8, 38);
}

TEST_CASE("independent.pv1.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_pv1(4, 15, 123);
}

void independent_pv2(size_t num_threads, size_t num_streams, size_t num_tasks) {
  taro::TaroPV2 taro{num_threads, num_streams};

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

TEST_CASE("independent.pv2.1thread.1stream.1task" * doctest::timeout(300)) {
  independent_pv2(1, 1, 1);
}

TEST_CASE("independent.pv2.2thread.1stream.3task" * doctest::timeout(300)) {
  independent_pv2(2, 1, 3);
}

TEST_CASE("independent.pv2.2thread.2stream.18task" * doctest::timeout(300)) {
  independent_pv2(2, 2, 18);
}

TEST_CASE("independent.pv2.2thread.3stream.18task" * doctest::timeout(300)) {
  independent_pv2(2, 3, 18);
}

TEST_CASE("independent.pv2.3thread.1stream.2task" * doctest::timeout(300)) {
  independent_pv2(3, 1, 2);
}

TEST_CASE("independent.pv2.3thread.2stream.4task" * doctest::timeout(300)) {
  independent_pv2(3, 2, 4);
}

TEST_CASE("independent.pv2.3thread.3stream.18task" * doctest::timeout(300)) {
  independent_pv2(3, 3, 18);
}

TEST_CASE("independent.pv2.4thread.1stream.1task" * doctest::timeout(300)) {
  independent_pv2(4, 1, 1);
}

TEST_CASE("independent.pv2.4thread.2stream.11task" * doctest::timeout(300)) {
  independent_pv2(4, 2, 11);
}

TEST_CASE("independent.pv2.4thread.8stream.38task" * doctest::timeout(300)) {
  independent_pv2(4, 8, 38);
}

TEST_CASE("independent.pv2.4thread.15stream.123task" * doctest::timeout(300)) {
  independent_pv2(4, 15, 123);
}


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

//void independent_taskflow(size_t num_threads, size_t num_streams, size_t num_tasks) {
  //taro::TaroTaskflow taro{num_threads, num_streams};

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

//TEST_CASE("independent.taskflow.1thread.1stream.1task" * doctest::timeout(300)) {
  //independent_taskflow(1, 1, 1);
//}

//TEST_CASE("independent.taskflow.2thread.1stream.3task" * doctest::timeout(300)) {
  //independent_taskflow(2, 1, 3);
//}

//TEST_CASE("independent.taskflow.2thread.2stream.18task" * doctest::timeout(300)) {
  //independent_taskflow(2, 2, 18);
//}

//TEST_CASE("independent.taskflow.2thread.3stream.18task" * doctest::timeout(300)) {
  //independent_taskflow(2, 3, 18);
//}

//TEST_CASE("independent.taskflow.3thread.1stream.2task" * doctest::timeout(300)) {
  //independent_taskflow(3, 1, 2);
//}

//TEST_CASE("independent.taskflow.3thread.2stream.4task" * doctest::timeout(300)) {
  //independent_taskflow(3, 2, 4);
//}

//TEST_CASE("independent.taskflow.3thread.3stream.18task" * doctest::timeout(300)) {
  //independent_taskflow(3, 3, 18);
//}

//TEST_CASE("independent.taskflow.4thread.1stream.1task" * doctest::timeout(300)) {
  //independent_taskflow(4, 1, 1);
//}

//TEST_CASE("independent.taskflow.4thread.2stream.11task" * doctest::timeout(300)) {
  //independent_taskflow(4, 2, 11);
//}

//TEST_CASE("independent.taskflow.4thread.8stream.38task" * doctest::timeout(300)) {
  //independent_taskflow(4, 8, 38);
//}

//TEST_CASE("independent.taskflow.4thread.15stream.123task" * doctest::timeout(300)) {
  //independent_taskflow(4, 15, 123);
//}
void independent_fiber(size_t num_threads, size_t num_tasks) {
  FiberTaskScheduler ft_sched{num_threads};

  std::vector<FiberTaskHandle> tasks(num_tasks);

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
    tasks[i] = ft_sched.emplace([&ft_sched, i, a, b, c, M, K, N, dim_grid, dim_block]()  {
      cudaStream_t st;
      cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);

      taro::cuda_matmul<<<dim_grid, dim_block, 0, st>>>(a, b, c + i * M * N, M, K, N);

      boost::fibers::cuda::waitfor_all(st);
      cudaStreamDestroy(st);

      for(size_t k = 0; k < M * N; ++k) {
        REQUIRE(c[k + i * M * N] == (int)(M + K) * (K + N) * K);
      }
    });
  }

  ft_sched.schedule();
  ft_sched.wait();

  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

TEST_CASE("independent.fiber.2thread.3task" * doctest::timeout(300)) {
  independent_fiber(2, 3);
}

TEST_CASE("independent.fiber.2thread.9task" * doctest::timeout(300)) {
  independent_fiber(2, 9);
}

TEST_CASE("independent.fiber.2thread.18task" * doctest::timeout(300)) {
  independent_fiber(2, 18);
}

TEST_CASE("independent.fiber.2thread.19task" * doctest::timeout(300)) {
  independent_fiber(2, 19);
}

TEST_CASE("independent.fiber.3thread.2task" * doctest::timeout(300)) {
  independent_fiber(3, 2);
}

TEST_CASE("independent.fiber.3thread.4task" * doctest::timeout(300)) {
  independent_fiber(3, 4);
}

TEST_CASE("independent.fiber.3thread.18task" * doctest::timeout(300)) {
  independent_fiber(3, 18);
}

TEST_CASE("independent.fiber.4thread.1task" * doctest::timeout(300)) {
  independent_fiber(4, 1);
}

TEST_CASE("independent.fiber.4thread.11task" * doctest::timeout(300)) {
  independent_fiber(4, 11);
}

TEST_CASE("independent.fiber.4thread.38task" * doctest::timeout(300)) {
  independent_fiber(4, 38);
}

TEST_CASE("independent.fiber.4threadm.123task" * doctest::timeout(300)) {
  independent_fiber(4, 123);
}


