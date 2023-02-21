#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <coroflow/src/cuda/coroflow_v1.hpp>
#include <coroflow/src/cuda/coroflow_v2.hpp>
#include <coroflow/src/cuda/coroflow_v3.hpp>
#include <coroflow/src/cuda/coroflow_v4.hpp>
#include <coroflow/src/cuda/coroflow_v5.hpp>
#include <coroflow/src/cuda/coroflow_v6.hpp>
#include <coroflow/src/cuda/algorithm.hpp>
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

void linear_chain_v1(size_t num_tasks, size_t num_threads) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  cudaStream_t st;
  cudaStreamCreate(&st);

  cf::CoroflowV1 cf{num_threads};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, counter, &cf, st]() -> cf::Coro {
      REQUIRE(*counter == t); 

      cudaEvent_t finish;
      cudaEventCreate(&finish);
      count<<<8, 32, 0, st>>>(counter);
      cudaEventRecord(finish);

      auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };
      while(!isdone()) {
        co_await cf.suspend();
      }

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait(); 
  cudaStreamDestroy(st);
}

TEST_CASE("linear_chain_v1.1thread" * doctest::timeout(300)) {
  linear_chain_v1(1, 1);
}

TEST_CASE("linear_chain_v1.2thread" * doctest::timeout(300)) {
  linear_chain_v1(99, 2);
}

TEST_CASE("linear_chain_v1.3thread" * doctest::timeout(300)) {
  linear_chain_v1(712, 3);
}

TEST_CASE("linear_chain_v1.4thread" * doctest::timeout(300)) {
  linear_chain_v1(443, 4);
}

TEST_CASE("linear_chain_v1.5thread" * doctest::timeout(300)) {
  linear_chain_v1(1111, 5);
}

TEST_CASE("linear_chain_v1.6thread" * doctest::timeout(300)) {
  linear_chain_v1(2, 6);
}

TEST_CASE("linear_chain_v1.7thread" * doctest::timeout(300)) {
  linear_chain_v1(5, 7);
}

TEST_CASE("linear_chain_v1.8thread" * doctest::timeout(300)) {
  linear_chain_v1(9211, 8);
}

void linear_chain_v2(size_t num_tasks, size_t num_threads) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  cudaStream_t st;
  cudaStreamCreate(&st);

  cf::CoroflowV2 cf{num_threads};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, counter, &cf, st]() -> cf::Coro {
      REQUIRE(*counter == t); 

      cudaEvent_t finish;
      cudaEventCreate(&finish);
      count<<<8, 32, 0, st>>>(counter);
      cudaEventRecord(finish);

      auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };
      while(!isdone()) {
        co_await cf.suspend();
      }

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait(); 
  cudaStreamDestroy(st);
}

TEST_CASE("linear_chain_v2.1thread" * doctest::timeout(300)) {
  linear_chain_v2(1, 1);
}

TEST_CASE("linear_chain_v2.2thread" * doctest::timeout(300)) {
  linear_chain_v2(99, 2);
}

TEST_CASE("linear_chain_v2.3thread" * doctest::timeout(300)) {
  linear_chain_v2(712, 3);
}

TEST_CASE("linear_chain_v2.4thread" * doctest::timeout(300)) {
  linear_chain_v2(443, 4);
}

TEST_CASE("linear_chain_v2.5thread" * doctest::timeout(300)) {
  linear_chain_v2(1111, 5);
}

TEST_CASE("linear_chain_v2.6thread" * doctest::timeout(300)) {
  linear_chain_v2(2, 6);
}

TEST_CASE("linear_chain_v2.7thread" * doctest::timeout(300)) {
  linear_chain_v2(5, 7);
}

TEST_CASE("linear_chain_v2.8thread" * doctest::timeout(300)) {
  linear_chain_v2(9211, 8);
}

void linear_chain_v3(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  cf::CoroflowV3 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, counter, &cf]() -> cf::Coro {
      REQUIRE(*counter == t); 

      co_await cf.cuda_suspend([counter](cudaStream_t st) {
        count<<<8, 32, 0, st>>>(counter);
      });

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait(); 
}

TEST_CASE("linear_chain_v3.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_v3(1, 1, 1);
}

TEST_CASE("linear_chain_v3.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_v3(99, 2, 2);
}

TEST_CASE("linear_chain_v3.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_v3(712, 3, 4);
}

TEST_CASE("linear_chain_v3.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_v3(443, 4, 8);
}

TEST_CASE("linear_chain_v3.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_v3(1111, 5, 2);
}

TEST_CASE("linear_chain_v3.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_v3(2, 6, 3);
}

TEST_CASE("linear_chain_v3.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_v3(5, 7, 1);
}

TEST_CASE("linear_chain_v3.8threads" * doctest::timeout(300)) {
  linear_chain_v3(9211, 8, 9);
}

void linear_chain_v4(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  cf::CoroflowV4 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, counter, &cf]() -> cf::Coro {
      REQUIRE(*counter == t); 

      co_await cf.cuda_suspend([counter](cudaStream_t st) {
        count<<<8, 32, 0, st>>>(counter);
      });

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait(); 
}

TEST_CASE("linear_chain_v4.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_v4(1, 1, 1);
}

TEST_CASE("linear_chain_v4.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_v4(99, 2, 2);
}

TEST_CASE("linear_chain_v4.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_v4(712, 3, 4);
}

TEST_CASE("linear_chain_v4.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_v4(443, 4, 8);
}

TEST_CASE("linear_chain_v4.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_v4(1111, 5, 2);
}

TEST_CASE("linear_chain_v4.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_v4(2, 6, 3);
}

TEST_CASE("linear_chain_v4.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_v4(5, 7, 1);
}

TEST_CASE("linear_chain_v4.8threads" * doctest::timeout(300)) {
  linear_chain_v4(9211, 8, 9);
}

void linear_chain_v5(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  cf::CoroflowV5 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, counter, &cf]() -> cf::Coro {
      REQUIRE(*counter == t); 

      co_await cf.cuda_suspend([counter](cudaStream_t st) {
        count<<<8, 32, 0, st>>>(counter);
      });

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait(); 
}

TEST_CASE("linear_chain_v5.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_v5(1, 1, 1);
}

TEST_CASE("linear_chain_v5.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_v5(99, 2, 2);
}

TEST_CASE("linear_chain_v5.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_v5(712, 3, 4);
}

TEST_CASE("linear_chain_v5.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_v5(443, 4, 8);
}

TEST_CASE("linear_chain_v5.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_v5(1111, 5, 2);
}

TEST_CASE("linear_chain_v5.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_v5(2, 6, 3);
}

TEST_CASE("linear_chain_v5.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_v5(5, 7, 1);
}

TEST_CASE("linear_chain_v5.8threads" * doctest::timeout(300)) {
  linear_chain_v5(9211, 8, 9);
}

void linear_chain_v6(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  cf::CoroflowV6 cf{num_threads, num_streams};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, counter, &cf]() -> cf::Coro {
      REQUIRE(*counter == t); 

      co_await cf.cuda_suspend([counter](cudaStream_t st) {
        count<<<8, 32, 0, st>>>(counter);
      });

      REQUIRE(*counter == t + 1); 
    });
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  REQUIRE(cf.is_DAG());
  cf.schedule();
  cf.wait(); 
}

TEST_CASE("linear_chain_v6.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_v6(1, 1, 1);
}

TEST_CASE("linear_chain_v6.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_v6(99, 2, 2);
}

TEST_CASE("linear_chain_v6.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_v6(712, 3, 4);
}

TEST_CASE("linear_chain_v6.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_v6(443, 4, 8);
}

TEST_CASE("linear_chain_v6.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_v6(1111, 5, 2);
}

TEST_CASE("linear_chain_v6.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_v6(2, 6, 3);
}

TEST_CASE("linear_chain_v6.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_v6(5, 7, 1);
}

TEST_CASE("linear_chain_v6.8threads" * doctest::timeout(300)) {
  linear_chain_v6(9211, 8, 9);
}
