#include <coroflow/coroflow.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <coroflow/algorithms/matmul.hpp>

void cpu_task(size_t size) {
  std::vector<size_t> data(size, ::rand() % 10);

  std::sort(data.begin(), data.end());
  size_t res = std::reduce(data.begin(), data.end(), 0);
}

cf::Coro hybrid_task() {
  cudaEvent_t finish;
  cudaEventCreate(&finish);

  float* a;
  float* b; 
  float* c;
  size_t M{1000};
  size_t K{1000};
  size_t N{1000};
  cudaMallocManaged(&a, M * K * sizeof(float));
  cudaMallocManaged(&b, K * N * sizeof(float));
  cudaMallocManaged(&c, M * N * sizeof(float));

  cuda_matmul<<<8, 32>>>(a, b, c, M, K, N);
  cudaEventRecord(finish);
  auto isdone = [&finish]() { return cudaEventQuery(finish) == cudaSuccess;  };

  while(!isdone()) {
    co_await cf::State::SUSPEND;
    std::cerr << "waiting...\n";
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cpu_task(1000000);
  cpu_task(1000000);
  cpu_task(1000000);

  std::cerr << "finished!\n";
  co_return;
}

// 0 - 1 - 2 
// |   |   |
// 3 - 4 - 5
// |   |   |
// 6 - 7 - 8 

// TODO: right now I assume we only run pipeline once
void pipeline(size_t num_pipes, size_t num_lines) {
  cf::Coroflow cf{4};

  std::vector<cf::TaskHandle> pl(num_pipes * num_lines);

  for(size_t l = 0; l < num_lines; ++l) {
    for(size_t p = 0; p < num_pipes; ++p) {
      pl[l * num_pipes + p] = cf.emplace(hybrid_task());
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
  
}

int main() {
  pipeline(3, 4);
}


