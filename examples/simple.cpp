#include <taro.hpp>
#include <iostream>
int main() {
  taro::Taro taro{4}; // number of threads
  auto task_a = taro.emplace([](){
    std::cout << "task a\n";
  });
  auto task_b = taro.emplace([](){
    std::cout << "task b\n";
  });
  auto task_c = taro.emplace([](){
    std::cout << "task c\n";
  });
  auto task_d = taro.emplace([](){
    std::cout << "task d\n";
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
