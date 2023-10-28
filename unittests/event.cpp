#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>
#include <taro/await/event.hpp>
#include <vector>
#include <algorithm>


//// --------------------------------------------------------
//// Testcase::Simple 
//// --------------------------------------------------------


void simple(size_t num_threads, size_t num_events) {
  taro::Taro taro{num_threads};
  auto events = taro.event_await(num_events);

  std::vector<int> ans(num_events, 0);

  for(size_t i = 0; i < num_events; ++i) {
    taro.emplace([&, i]() {
      events.set(i);
    });

    taro.emplace([&, i]() -> taro::Coro {
      ans[i] = -2;
      co_await events.until(i);
      ans[i] = 1;
    });
  }

  REQUIRE(taro.is_DAG());
  taro.schedule();
  taro.wait();

  REQUIRE(std::accumulate(ans.begin(), ans.end(), 0) == num_events);

}

TEST_CASE("event.simple.1thread.1event" * doctest::timeout(300)) {
  simple(1, 1);
}

TEST_CASE("event.simple.2thread.1event" * doctest::timeout(300)) {
  simple(2, 1);
}

TEST_CASE("event.simple.4thread.5event" * doctest::timeout(300)) {
  simple(4, 5);
}

TEST_CASE("event.simple.3thread.119event" * doctest::timeout(300)) {
  simple(3, 119);
}

TEST_CASE("event.simple.4thread.2779event" * doctest::timeout(300)) {
  simple(4, 2779);
}

