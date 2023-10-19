#include <taro.hpp>
#include <taro/await/semaphore.hpp>
#include <vector>
#include <algorithm>


void simple(size_t num_threads, size_t num_semaphores) {
  taro::Taro taro{num_threads};
  auto semaphores = taro.semaphore_await<1>(num_semaphores);

  std::vector<int> ans(num_semaphores, 0);

  for(size_t i = 0; i < num_semaphores; ++i) {
    taro.emplace([&, i]() -> taro::Coro {
      co_await semaphores.acquire(i);
      ans[i]++;
      semaphores.release(i);
    });

    taro.emplace([&, i]() -> taro::Coro {
      co_await semaphores.acquire(i);
      ans[i]--;
      semaphores.release(i);
    });
  }
  taro.schedule();
  taro.wait();

}

int main() {
  simple(4, 16);
}
