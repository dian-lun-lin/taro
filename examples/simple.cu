#include <taro.hpp>
#include <taro/await/cuda.hpp>

int main() {
  taro::Taro taro{4}; // number of threads
  auto cuda = taro.cuda_await(4); // number of cuda streams
  int* d_a;
  size_t N{100};
  size_t BLOCK_SIZE{128};

  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  auto task_a = taro.emplace([&]() {
    std::iota(h_a.begin(), h_a.end(), 0);
    std::cout << "task a\n";
  });
  
  // malloc
  auto task_b = taro.emplace([=, &cuda]() {
    cuda.wait([=](cudaStream_t stream) {  // wait method
      cudaMallocAsync((void**)&d_a, N * N * sizeof(int), stream);
    });
    std::cout << "task b use wait methodg\n";
  });
  
  // H2D
  auto task_c = taro.emplace([=, &h_a, &cuda]() -> taro::Coro {
    co_await cuda.until_polling([=, &h_a](cudaStream_t stream) { // polling method   
      cudaMemcpyAsync(d_a, h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, stream);
    });
    std::cout << "task c use polling method\n";
  });
  
  // D2H and free
  auto task_d = taro.emplace([=, &h_b, &cuda]() -> taro::Coro { 
    co_await cuda.until_callback([=, &h_b](cudaStream_t stream) {   // callback method
      cudaMemcpyAsync(h_b.data(), d_a, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaFreeAsync(d_a, stream);
    });
    std::cout << "task d use callback method\n";
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
