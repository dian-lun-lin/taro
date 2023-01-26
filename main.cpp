#include "coroflow.hpp"

void test1() {
  cf::Scheduler scheduler{4};

  [](cf::Scheduler& sched) -> cf::Task {
    std::cout << "Hello from task1 \n";
    co_await sched.suspend();
    std::cout << "task1 is working..\n";
    co_await sched.suspend();
    std::cout << "task1 is finished\n";
  }(scheduler);

  [](cf::Scheduler& sched) -> cf::Task {
    std::cout << "Hello from task2 \n";
    co_await sched.suspend();
    std::cout << "task2 is working..\n";
    co_await sched.suspend();
    std::cout << "task2 is finished...\n";
  }(scheduler);

  while(scheduler.schedule()) {}

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
