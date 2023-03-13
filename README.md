# Taro
***T***ask-based ***A***synchronous programming system using C++ co***RO***utine.

Taro is a task-based asynchronous programming system that leverages C++ coroutines to achieve high-performance CPU-GPU computing. Taro employs a work-stealing algorithm that dynamically redistributes tasks between threads, avoiding the overhead of thread management and enabling concurrent task execution. By using coroutines, Taro enables the CPU to continue executing other tasks while waiting for the GPU to complete its work, thus avoiding blocking and improving overall performance. 


# Example
```cpp
#include <taro/cuda/taro.hpp>
int main() {
  taro::Taro taro{4, 4}; // (num_threads, num_streams)
  int* d_a, *d_b, *d_c;
  std::vector<int> h_c;
  
  // H2D
  auto task_a = taro.emplace([]() -> taro::Coro {
    std::vector<int> h_a(N * N);
    std::iota(h_a.begin(), h_a.end(), 0);
    co_await taro.cuda_suspend([&d_a, &h_a, N](cudaStream_t stream) {   
      cudaMallocAsync(&d_a, N * N * sizeof(int), stream);
      cudaMemcpyAsync(d_a, h_a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, stream);
    });
  });
  
  // H2D
  auto task_b = taro.emplace([]() -> taro::Coro {
    std::vector<int> h_b(N * N);
    std::iota(h_b.begin(), h_b.end(), 0);
    co_await taro.cuda_suspend([&d_b, &h_b, N](cudaStream_t stream) {    
      cudaMallocAsync(&d_b, N * N * sizeof(int), stream);
      cudaMemcpyAsync(d_b, h_b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice, stream);
    });
  });
  
  // matrix multiplication and D2H
  auto task_c = taro.emplace([&h_c, &d_c, N]() -> taro::Coro {
    size BLOCK_SIZE{128};
    dim3 dim_grid((N - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    h_c.resize(N * N);
    co_await taro.cuda_suspend([&d_c, &h_c, =](cudaStream_t stream) {    
      cudaMallocAsync(&d_c, N * N * sizeof(int), stream)
      taro::cuda_matmul<<<dim_grid, dim_block, 0, stream>>>(d_a, d_b, d_c, N, N, N);
      cudaMemcpyAsync(h_c.data(), d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    });
  });
  
  // free
  auto task_d = taro.emplace([]() -> taro::Coro { 
    co_await taro.cuda_suspend([=](cudaStream_t stream) {    
      cudaFreeAsync(d_a, stream);
      cudaFreeAsync(d_b, stream);
      cudaFreeAsync(d_c, stream);
    });
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
}
```