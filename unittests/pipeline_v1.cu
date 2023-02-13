#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <coroflow/coroflow.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

void cpu_task(size_t size) {
  std::vector<size_t> data(size, ::rand() % 10);
  std::sort(data.begin(), data.end());
}

cf::Coro hybrid_task(float* a, float* b, float* c, size_t M, size_t K, size_t N) {
}


void pipeline(size_t num_pipes, size_t num_lines) {
  cf::Coroflow cf{4};

  std::vector<cf::TaskHandle> pl(num_pipes * num_lines);
  std::vector<size_t> dataa(M * N * num_lines);
  std::vector<size_t> datab(N * K * num_lines);

  float* a;
  float* b; 
  float* c;
  size_t M{10};
  size_t K{10};
  size_t N{10};
  cudaMallocManaged(&a, M * K * sizeof(float));
  cudaMallocManaged(&b, K * N * sizeof(float));
  cudaMallocManaged(&c, M * N * sizeof(float));
  std::fill_n(a.begin(), M * N, M + N);
  std::fill_n(b.begin(), N * K, N + K);

  for(size_t l = 0; l < num_lines; ++l) {
    for(size_t p = 0; p < num_pipes; ++p) {
      pl[l * num_pipes + p] = cf.emplace( []() -> cf::Coro {
        cudaEvent_t finish;
        cudaEventCreate(&finish);

        cuda_matmul<<<8, 32>>>(a, b, c, M, K, N);
        cudaEventRecord(finish);
        auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };

        while(!isdone()) {
          co_await cf::State::SUSPEND;
        }

        for(const auto& x: c) {
          REQUIRE(x == (int)(M + N) * (N+K) * N);
        }

        cpu_task(2);
        cpu_task(3);
        cpu_task(4);
        co_return;
      }
    }
  }

  // dependencies
  // vertical
  for(size_t l = 0; l < num_lines - 1; ++l) {
    for(size_t p = 0; p < num_pipes; ++p) {
      pl[l * num_pipes + p].succeed(pl[(l + 1) * num_pipes]);
    }
  }

  // horizontal
  for(size_t l = 0; l < num_lines; ++l) {
    for(size_t p = 0; p < num_pipes - 1; ++p) {
      pl[l * num_pipes + p].succeed(pl[l * num_pipes + p + 1]);
    }
  }

  cf.schedule();
  cf.wait();

  
  REQUIRE(cudaFree(a) == cudaSuccess);
  REQUIRE(cudaFree(b) == cudaSuccess);
  REQUIRE(cudaFree(c) == cudaSuccess);

}

int main() {
  pipeline(3, 4);
}
