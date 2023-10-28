#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/await/semaphore.hpp>
#include <vector>
#include <algorithm>


//// --------------------------------------------------------
//// Testcase::Simple 
//// --------------------------------------------------------


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

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait();

  REQUIRE(std::accumulate(ans.begin(), ans.end(), 0) == 0);

}

TEST_CASE("semarphore.simple.1thread.1semaphore" * doctest::timeout(300)) {
  simple(1, 1);
}

TEST_CASE("semarphore.simple.2thread.1semaphore" * doctest::timeout(300)) {
  simple(2, 1);
}

TEST_CASE("semarphore.simple.4thread.5semaphore" * doctest::timeout(300)) {
  simple(4, 5);
}

TEST_CASE("semarphore.simple.3thread.119semaphore" * doctest::timeout(300)) {
  simple(3, 119);
}

TEST_CASE("semarphore.simple.4thread.2779semaphore" * doctest::timeout(300)) {
  simple(4, 2779);
}

