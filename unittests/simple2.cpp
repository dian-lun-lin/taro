#include "coroflow.hpp"

void test1() {
  cf::Coroflow cf{4};

  auto t1 = cf.emplace([]() -> cf::Coro {
    std::cout << "(t1) call GPU kernel 1\n";
    // kernel<<<>>>();
    //co_await ready;
    std::cout << "(t1) after finish GPU kernel 1, keep working...\n";

    std::cout << "(t1) call GPU kernel 2\n";
    //co_await cf::suspend();
    co_await cf::State::SUSPEND;
    std::cout << "(t1) after finish GPU kernel 2, keep working...\n";
  }());


  auto t2 = cf.emplace([]() -> cf::Coro {
    std::cout << "(t2) call GPU kernel 1\n";
    // kernel<<<>>>();
    //co_await ready;
    std::cout << "(t2) after finish GPU kernel 1, keep working...\n";

    std::cout << "(t2) call GPU kernel 2\n";
    //co_await cf::suspend();
    co_await cf::State::SUSPEND;
    std::cout << "(t2) after finish GPU kernel 2, keep working...\n";
  }());


  auto t3 = cf.emplace([]() -> cf::Coro {
    std::cout << "(t3) call GPU kernel 1\n";
    // kernel<<<>>>();
    //co_await ready;
    std::cout << "(t3) after finish GPU kernel 1, keep working...\n";

    std::cout << "(t3) call GPU kernel 2\n";
    //co_await cf::suspend();
    co_await cf::State::SUSPEND;
    std::cout << "(t3) after finish GPU kernel 2, keep working...\n";
  }());

  auto t4 = cf.emplace([]() -> cf::Coro {
    std::cout << "(t4) call GPU kernel 1\n";
    // kernel<<<>>>();
    //co_await ready;
    std::cout << "(t4) after finish GPU kernel 1, keep working...\n";

    std::cout << "(t4) call GPU kernel 2\n";
    //co_await cf::suspend();
    co_await cf::State::SUSPEND;
    std::cout << "(t4) after finish GPU kernel 2, keep working...\n";
  }());

  t1.succeed(t2);
  t1.succeed(t3);
  t2.succeed(t4);
  t3.succeed(t4);

  //scheduler.schedule();
  cf.schedule();
}

int main() {
  test1();
}

//void test2() {

  //cf::Coroflow coroflow;
  //cf::Scheduler scheduler{4}; // four threads

  //auto t1 = coroflow.emplace([](cf::Scheduler& sched) -> cf::Task {
    //std::cout << "Hello from task1 \n";
    //co_await sched.suspend();
    //std::cout << "task1 is working..\n";
    //co_await sched.suspend();
    //std::cout << "task1 is finished\n";
  //});

  //auto t2 = coroflow.emplace([](cf::Scheduler& sched) -> cf::Task {
    //std::cout << "Hello from task2 \n";
    //co_await sched.suspend();
    //std::cout << "task2 is working..\n";
    //co_await sched.suspend();
    //std::cout << "task2 is finished...\n";
  //});
//}
