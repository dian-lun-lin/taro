#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <coroflow/coroflow.hpp>

// --------------------------------------------------------
// Testcase:: Simple task graph
// --------------------------------------------------------

// o - o - o - o

void linear_chain(size_t num_tasks, size_t num_threads) {
  int counter{0};
  cf::Coroflow cf{num_threads};
  std::vector<cf::TaskHandle> _tasks(num_tasks);

  for(size_t t = 0; t < num_tasks; ++t) {
    _tasks[t] = cf.emplace([t, &counter]() -> cf::Coro {
      REQUIRE(counter++ == t); 
      std::cerr << "counter: " << counter << "t: " << t << "\n";
      co_await cf::State::SUSPEND;
    }());
  }

  for(size_t t = 0; t < num_tasks - 1; ++t) {
    _tasks[t].succeed(_tasks[t + 1]);
  }

  cf.schedule();
  
}

//void pipeline() {
//}

TEST_CASE("linear_chain") {
  linear_chain(3, 4);
}

//TEST_CASE("dependencies") {
  //cf::Coroflow cf{4};

  //auto t1 = cf.emplace([]() -> cf::Coro {
    //std::cout << "(t1) call GPU kernel 1\n";
    //// kernel<<<>>>();
    ////co_await ready;
    //std::cout << "(t1) after finish GPU kernel 1, keep working...\n";

    //std::cout << "(t1) call GPU kernel 2\n";
    ////co_await cf::suspend();
    //co_await cf::State::SUSPEND;
    //std::cout << "(t1) after finish GPU kernel 2, keep working...\n";
  //}());


  //auto t2 = cf.emplace([]() -> cf::Coro {
    //std::cout << "(t2) call GPU kernel 1\n";
    //// kernel<<<>>>();
    ////co_await ready;
    //std::cout << "(t2) after finish GPU kernel 1, keep working...\n";

    //std::cout << "(t2) call GPU kernel 2\n";
    ////co_await cf::suspend();
    //co_await cf::State::SUSPEND;
    //std::cout << "(t2) after finish GPU kernel 2, keep working...\n";
  //}());


  //auto t3 = cf.emplace([]() -> cf::Coro {
    //std::cout << "(t3) call GPU kernel 1\n";
    //// kernel<<<>>>();
    ////co_await ready;
    //std::cout << "(t3) after finish GPU kernel 1, keep working...\n";

    //std::cout << "(t3) call GPU kernel 2\n";
    ////co_await cf::suspend();
    //co_await cf::State::SUSPEND;
    //std::cout << "(t3) after finish GPU kernel 2, keep working...\n";
  //}());

  //auto t4 = cf.emplace([]() -> cf::Coro {
    //std::cout << "(t4) call GPU kernel 1\n";
    //// kernel<<<>>>();
    ////co_await ready;
    //std::cout << "(t4) after finish GPU kernel 1, keep working...\n";

    //std::cout << "(t4) call GPU kernel 2\n";
    ////co_await cf::suspend();
    //co_await cf::State::SUSPEND;
    //std::cout << "(t4) after finish GPU kernel 2, keep working...\n";
  //}());

  //t1.succeed(t2);
  //t1.succeed(t3);
  //t2.succeed(t4);
  //t3.succeed(t4);

  ////scheduler.schedule();
  //cf.schedule();
//}

//int main() {
  //test1();
//}
