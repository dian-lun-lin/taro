#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro/src/cuda/callback/taro_callback_v1.hpp>
#include <taro/src/cuda/callback/taro_callback_v2.hpp>
#include <taro/src/cuda/callback/taro_callback_v3.hpp>
#include <taro/src/cuda/poll/taro_poll_v1.hpp>
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

void linear_chain_cbv1(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  taro::TaroCBV1 taro{num_threads, num_streams};
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

TEST_CASE("linear_chain_cbv1.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_cbv1(1, 1, 1);
}

TEST_CASE("linear_chain_cbv1.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_cbv1(99, 2, 2);
}

TEST_CASE("linear_chain_cbv1.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_cbv1(712, 3, 4);
}

TEST_CASE("linear_chain_cbv1.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_cbv1(443, 4, 8);
}

TEST_CASE("linear_chain_cbv1.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_cbv1(1111, 5, 2);
}

TEST_CASE("linear_chain_cbv1.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_cbv1(2, 6, 3);
}

TEST_CASE("linear_chain_cbv1.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_cbv1(5, 7, 1);
}

TEST_CASE("linear_chain_cbv1.8threads" * doctest::timeout(300)) {
  linear_chain_cbv1(9211, 8, 9);
}

void linear_chain_cbv2(size_t num_tasks, size_t num_threads) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  taro::TaroCBV2 taro{num_threads};
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

TEST_CASE("linear_chain_cbv2.1thread" * doctest::timeout(300)) {
  linear_chain_cbv2(1, 1);
}

TEST_CASE("linear_chain_cbv2.2thread" * doctest::timeout(300)) {
  linear_chain_cbv2(99, 2);
}

TEST_CASE("linear_chain_cbv2.3thread" * doctest::timeout(300)) {
  linear_chain_cbv2(712, 3);
}

TEST_CASE("linear_chain_cbv2.4thread" * doctest::timeout(300)) {
  linear_chain_cbv2(443, 4);
}

TEST_CASE("linear_chain_cbv2.5thread" * doctest::timeout(300)) {
  linear_chain_cbv2(1111, 5);
}

TEST_CASE("linear_chain_cbv2.6thread" * doctest::timeout(300)) {
  linear_chain_cbv2(2, 6);
}

TEST_CASE("linear_chain_cbv2.7thread" * doctest::timeout(300)) {
  linear_chain_cbv2(5, 7);
}

TEST_CASE("linear_chain_cbv2.8thread" * doctest::timeout(300)) {
  linear_chain_cbv2(9211, 8);
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

void linear_chain_pv1(size_t num_tasks, size_t num_threads, size_t num_streams) {
  int* counter;
  cudaMallocManaged(&counter, sizeof(int));

  taro::TaroPV1 taro{num_threads, num_streams};
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

TEST_CASE("linear_chain_pv1.1thread.1stream" * doctest::timeout(300)) {
  linear_chain_pv1(1, 1, 1);
}

TEST_CASE("linear_chain_pv1.2thread.2stream" * doctest::timeout(300)) {
  linear_chain_pv1(99, 2, 2);
}

TEST_CASE("linear_chain_pv1.3thread.4stream" * doctest::timeout(300)) {
  linear_chain_pv1(712, 3, 4);
}

TEST_CASE("linear_chain_pv1.4thread.8stream" * doctest::timeout(300)) {
  linear_chain_pv1(443, 4, 8);
}

TEST_CASE("linear_chain_pv1.5thread.2stream" * doctest::timeout(300)) {
  linear_chain_pv1(1111, 5, 2);
}

TEST_CASE("linear_chain_pv1.6thread.3stream" * doctest::timeout(300)) {
  linear_chain_pv1(2, 6, 3);
}

TEST_CASE("linear_chain_pv1.7thread.1stream" * doctest::timeout(300)) {
  linear_chain_pv1(5, 7, 1);
}

TEST_CASE("linear_chain_pv1.8threads" * doctest::timeout(300)) {
  linear_chain_pv1(9211, 8, 9);
}
//void linear_chain_v1(size_t num_tasks, size_t num_threads) {
  //int* counter;
  //cudaMallocManaged(&counter, sizeof(int));

  //cudaStream_t st;
  //cudaStreamCreate(&st);

  //taro::TaroV1 taro{num_threads};
  //std::vector<taro::TaskHandle> _tasks(num_tasks);

  //for(size_t t = 0; t < num_tasks; ++t) {
    //_tasks[t] = taro.emplace([t, counter, &taro, st]() -> taro::Coro {
      //REQUIRE(*counter == t); 

      //cudaEvent_t finish;
      //cudaEventCreate(&finish);
      //count<<<8, 32, 0, st>>>(counter);
      //cudaEventRecord(finish);

      //auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };
      //while(!isdone()) {
        //co_await taro.suspend();
      //}

      //REQUIRE(*counter == t + 1); 
    //});
  //}

  //for(size_t t = 0; t < num_tasks - 1; ++t) {
    //_tasks[t].precede(_tasks[t + 1]);
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait(); 
  //cudaStreamDestroy(st);
//}

//TEST_CASE("linear_chain_v1.1thread" * doctest::timeout(300)) {
  //linear_chain_v1(1, 1);
//}

//TEST_CASE("linear_chain_v1.2thread" * doctest::timeout(300)) {
  //linear_chain_v1(99, 2);
//}

//TEST_CASE("linear_chain_v1.3thread" * doctest::timeout(300)) {
  //linear_chain_v1(712, 3);
//}

//TEST_CASE("linear_chain_v1.4thread" * doctest::timeout(300)) {
  //linear_chain_v1(443, 4);
//}

//TEST_CASE("linear_chain_v1.5thread" * doctest::timeout(300)) {
  //linear_chain_v1(1111, 5);
//}

//TEST_CASE("linear_chain_v1.6thread" * doctest::timeout(300)) {
  //linear_chain_v1(2, 6);
//}

//TEST_CASE("linear_chain_v1.7thread" * doctest::timeout(300)) {
  //linear_chain_v1(5, 7);
//}

//TEST_CASE("linear_chain_v1.8thread" * doctest::timeout(300)) {
  //linear_chain_v1(9211, 8);
//}

//void linear_chain_v2(size_t num_tasks, size_t num_threads) {
  //int* counter;
  //cudaMallocManaged(&counter, sizeof(int));

  //cudaStream_t st;
  //cudaStreamCreate(&st);

  //taro::TaroV2 taro{num_threads};
  //std::vector<taro::TaskHandle> _tasks(num_tasks);

  //for(size_t t = 0; t < num_tasks; ++t) {
    //_tasks[t] = taro.emplace([t, counter, &taro, st]() -> taro::Coro {
      //REQUIRE(*counter == t); 

      //cudaEvent_t finish;
      //cudaEventCreate(&finish);
      //count<<<8, 32, 0, st>>>(counter);
      //cudaEventRecord(finish);

      //auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };
      //while(!isdone()) {
        //co_await taro.suspend();
      //}

      //REQUIRE(*counter == t + 1); 
    //});
  //}

  //for(size_t t = 0; t < num_tasks - 1; ++t) {
    //_tasks[t].precede(_tasks[t + 1]);
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait(); 
  //cudaStreamDestroy(st);
//}

//TEST_CASE("linear_chain_v2.1thread" * doctest::timeout(300)) {
  //linear_chain_v2(1, 1);
//}

//TEST_CASE("linear_chain_v2.2thread" * doctest::timeout(300)) {
  //linear_chain_v2(99, 2);
//}

//TEST_CASE("linear_chain_v2.3thread" * doctest::timeout(300)) {
  //linear_chain_v2(712, 3);
//}

//TEST_CASE("linear_chain_v2.4thread" * doctest::timeout(300)) {
  //linear_chain_v2(443, 4);
//}

//TEST_CASE("linear_chain_v2.5thread" * doctest::timeout(300)) {
  //linear_chain_v2(1111, 5);
//}

//TEST_CASE("linear_chain_v2.6thread" * doctest::timeout(300)) {
  //linear_chain_v2(2, 6);
//}

//TEST_CASE("linear_chain_v2.7thread" * doctest::timeout(300)) {
  //linear_chain_v2(5, 7);
//}

//TEST_CASE("linear_chain_v2.8thread" * doctest::timeout(300)) {
  //linear_chain_v2(9211, 8);
//}

//void linear_chain_v3(size_t num_tasks, size_t num_threads, size_t num_streams) {
  //int* counter;
  //cudaMallocManaged(&counter, sizeof(int));

  //taro::TaroV3 taro{num_threads, num_streams};
  //std::vector<taro::TaskHandle> _tasks(num_tasks);

  //for(size_t t = 0; t < num_tasks; ++t) {
    //_tasks[t] = taro.emplace([t, counter, &taro]() -> taro::Coro {
      //REQUIRE(*counter == t); 

      //co_await taro.cuda_suspend([counter](cudaStream_t st) {
        //count<<<8, 32, 0, st>>>(counter);
      //});

      //REQUIRE(*counter == t + 1); 
    //});
  //}

  //for(size_t t = 0; t < num_tasks - 1; ++t) {
    //_tasks[t].precede(_tasks[t + 1]);
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait(); 
//}

//TEST_CASE("linear_chain_v3.1thread.1stream" * doctest::timeout(300)) {
  //linear_chain_v3(1, 1, 1);
//}

//TEST_CASE("linear_chain_v3.2thread.2stream" * doctest::timeout(300)) {
  //linear_chain_v3(99, 2, 2);
//}

//TEST_CASE("linear_chain_v3.3thread.4stream" * doctest::timeout(300)) {
  //linear_chain_v3(712, 3, 4);
//}

//TEST_CASE("linear_chain_v3.4thread.8stream" * doctest::timeout(300)) {
  //linear_chain_v3(443, 4, 8);
//}

//TEST_CASE("linear_chain_v3.5thread.2stream" * doctest::timeout(300)) {
  //linear_chain_v3(1111, 5, 2);
//}

//TEST_CASE("linear_chain_v3.6thread.3stream" * doctest::timeout(300)) {
  //linear_chain_v3(2, 6, 3);
//}

//TEST_CASE("linear_chain_v3.7thread.1stream" * doctest::timeout(300)) {
  //linear_chain_v3(5, 7, 1);
//}

//TEST_CASE("linear_chain_v3.8threads" * doctest::timeout(300)) {
  //linear_chain_v3(9211, 8, 9);
//}

//void linear_chain_v4(size_t num_tasks, size_t num_threads, size_t num_streams) {
  //int* counter;
  //cudaMallocManaged(&counter, sizeof(int));

  //taro::TaroV4 taro{num_threads, num_streams};
  //std::vector<taro::TaskHandle> _tasks(num_tasks);

  //for(size_t t = 0; t < num_tasks; ++t) {
    //_tasks[t] = taro.emplace([t, counter, &taro]() -> taro::Coro {
      //REQUIRE(*counter == t); 

      //co_await taro.cuda_suspend([counter](cudaStream_t st) {
        //count<<<8, 32, 0, st>>>(counter);
      //});

      //REQUIRE(*counter == t + 1); 
    //});
  //}

  //for(size_t t = 0; t < num_tasks - 1; ++t) {
    //_tasks[t].precede(_tasks[t + 1]);
  //}

  //REQUIRE(taro.is_DAG());
  //taro.schedule();
  //taro.wait(); 
//}

//TEST_CASE("linear_chain_v4.1thread.1stream" * doctest::timeout(300)) {
  //linear_chain_v4(1, 1, 1);
//}

//TEST_CASE("linear_chain_v4.2thread.2stream" * doctest::timeout(300)) {
  //linear_chain_v4(99, 2, 2);
//}

//TEST_CASE("linear_chain_v4.3thread.4stream" * doctest::timeout(300)) {
  //linear_chain_v4(712, 3, 4);
//}

//TEST_CASE("linear_chain_v4.4thread.8stream" * doctest::timeout(300)) {
  //linear_chain_v4(443, 4, 8);
//}

//TEST_CASE("linear_chain_v4.5thread.2stream" * doctest::timeout(300)) {
  //linear_chain_v4(1111, 5, 2);
//}

//TEST_CASE("linear_chain_v4.6thread.3stream" * doctest::timeout(300)) {
  //linear_chain_v4(2, 6, 3);
//}

//TEST_CASE("linear_chain_v4.7thread.1stream" * doctest::timeout(300)) {
  //linear_chain_v4(5, 7, 1);
//}

//TEST_CASE("linear_chain_v4.8threads" * doctest::timeout(300)) {
  //linear_chain_v4(9211, 8, 9);
//}
