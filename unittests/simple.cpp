#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <coroflow/coroflow.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

// --------------------------------------------------------
// Testcase:: Linear chain
// --------------------------------------------------------

// o - o - o - o

void linear_chain(size_t num_tasks, size_t num_threads) {
  int counter{0};
  cf::Coroflow cf{num_threads};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([](int t, int& counter) -> cf::Coro {
        REQUIRE(counter++ == t); 
        co_await cf::State::SUSPEND;
    }(t, counter));
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  cf.schedule();
  cf.wait(); 
}

TEST_CASE("linear_chain.1thread" * doctest::timeout(300)) {
  linear_chain(1, 1);
}

TEST_CASE("linear_chain.2threads" * doctest::timeout(300)) {
  linear_chain(99, 2);
}

TEST_CASE("linear_chain.3threads" * doctest::timeout(300)) {
  linear_chain(712, 3);
}

TEST_CASE("linear_chain.4threads" * doctest::timeout(300)) {
  linear_chain(443, 4);
}

TEST_CASE("linear_chain.5threads" * doctest::timeout(300)) {
  linear_chain(1111, 5);
}

TEST_CASE("linear_chain.6threads" * doctest::timeout(300)) {
  linear_chain(2, 6);
}

TEST_CASE("linear_chain.7threads" * doctest::timeout(300)) {
  linear_chain(5, 7);
}

TEST_CASE("linear_chain.8threads" * doctest::timeout(300)) {
  linear_chain(9211, 8);
}

// --------------------------------------------------------
// Testcase:: Map reduce
// --------------------------------------------------------

//   o
// / | \
//o  o  o
// \ | /
//   o
// / | \
//o  o  o
// \ | /
//   o
//  ...

void map_reduce(size_t num_iters, size_t num_parallel_tasks, size_t num_threads) {
  cf::Coroflow cf{num_threads};

  int counter{0};
  std::vector<int> buf(num_parallel_tasks, 0);

  std::vector<int> data(num_parallel_tasks, 0);
  std::iota(data.begin(), data.end(), 0);
  int ans = std::accumulate(data.begin(), data.end(), 0);

  auto src_t = cf.emplace([]() -> cf::Coro {
    co_await cf::State::SUSPEND;
  }());


  for(size_t i = 0; i < num_iters; ++i) {

    auto reduce_t = cf.emplace([](std::vector<int>& buf, int& counter, int& ans, int i) -> cf::Coro {
      counter += std::accumulate(buf.begin(), buf.end(), 0);
      co_await cf::State::SUSPEND;
      int res = ans * (i + 1);
      REQUIRE(counter == res);
      co_await cf::State::SUSPEND;
      std::fill(buf.begin(), buf.end(), 0);
    }(buf, counter, ans, i));

    for(size_t t = 0; t < num_parallel_tasks; ++t) {
      auto map_t = cf.emplace([](std::vector<int>& buf, std::vector<int>& data, int t) -> cf::Coro {
        for(size_t s = 0; s < rand() % 3; ++s) {
          co_await cf::State::SUSPEND;
        }
        buf[t] += data[t];
      }(buf, data, t));

      src_t.succeed(map_t);
      map_t.succeed(reduce_t);
    }

    src_t = reduce_t;
    
  }

  cf.schedule();
  cf.wait();
}

TEST_CASE("map_reduce.1thread" * doctest::timeout(300)) {
  map_reduce(1, 1, 1);
  map_reduce(3, 2, 1);
  map_reduce(4, 13, 1);
  map_reduce(10, 7, 1);
}

TEST_CASE("map_reduce.2threads" * doctest::timeout(300)) {
  map_reduce(1, 1, 2);
  map_reduce(3, 5, 2);
  map_reduce(7, 11, 2);
  map_reduce(20, 9, 2);
}

TEST_CASE("map_reduce.3threads" * doctest::timeout(300)) {
  map_reduce(1, 1, 3);
  map_reduce(7, 20, 3);
  map_reduce(3, 41, 3);
  map_reduce(14, 12, 3);
}

TEST_CASE("map_reduce.4threads" * doctest::timeout(300)) {
  map_reduce(2, 2, 4);
  map_reduce(30, 90, 4);
  map_reduce(11, 5, 4);
  map_reduce(19, 102, 4);
}

TEST_CASE("map_reduce.5threads" * doctest::timeout(300)) {
  map_reduce(3, 210, 4);
  map_reduce(34, 123, 4);
  map_reduce(3, 3, 4);
  map_reduce(1, 999, 4);
}

