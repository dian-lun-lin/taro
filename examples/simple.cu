#include <taro.hpp>
#include <taro/scheduler/cuda.hpp>

int main() {
  taro::Taro taro{4}; // number of threads
  auto cuda = taro.cuda_scheduler(4); // number of cuda streams
  int* d_a, *d_b, *d_c;
  std::vector<int> h_c;
  size_t N{100};
  
  // H2D
  auto task_a = taro.emplace([&cuda, d_a, N]() -> taro::Coro {
    std::vector<int> h_a(N * N);
    std::iota(h_a.begin(), h_a.end(), 0);
    co_await cuda.suspend_polling([&d_a, &h_a, N](cudaStream_t stream) {  // polling method
      cudaMallocAsync((void**)&d_a, N * N * sizeof(int), stream);
      cudaMemcpyAsync(d_a, h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, stream);
    });
    std::cout << "task a polling\n";
  });
  
  // H2D
  auto task_b = taro.emplace([&cuda, d_b, N]() -> taro::Coro {
    std::vector<int> h_b(N * N);
    std::iota(h_b.begin(), h_b.end(), 0);
    co_await cuda.suspend_polling([&d_b, &h_b, N](cudaStream_t stream) {  // polling method  
      cudaMallocAsync((void**)&d_b, N * N * sizeof(int), stream);
      cudaMemcpyAsync(d_b, h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, stream);
    });
    std::cout << "task b polling\n";
  });
  
  // D2H
  auto task_c = taro.emplace([&cuda, &h_c, d_c, N]() -> taro::Coro {
    size_t BLOCK_SIZE{128};
    dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    h_c.resize(N * N);
    co_await cuda.suspend_callback([=, &d_c, &h_c](cudaStream_t stream) { // callback method   
      cudaMallocAsync((void**)&d_c, N * N * sizeof(int), stream);
      cudaMemcpyAsync(h_c.data(), d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    });
    std::cout << "task c callback\n";
  });
  
  // free
  auto task_d = taro.emplace([&cuda, d_a, d_b, d_c]() { 
    cuda.wait([=](cudaStream_t stream) {   // synchronous 
      cudaFreeAsync(d_a, stream);
      cudaFreeAsync(d_b, stream);
      cudaFreeAsync(d_c, stream);
    });
    std::cout << "task d wait\n";
  });
  
  // dependency
  // A -> C
  // B -> C
  // C -> D
  task_a.precede(task_c);
  task_b.precede(task_c);
  task_c.precede(task_d);
  
  taro.schedule();
  taro.wait();

  return 0;
}
