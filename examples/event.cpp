
#include <taro.hpp>
#include <taro/await/event.hpp>
#include <vector>
#include <algorithm>

int main() {
  size_t NUM_THREADS{2};
  size_t NUM_EVENTS{4};

  taro::Taro taro{NUM_THREADS};
  auto events = taro.event_await(NUM_EVENTS); // create four events

  for(size_t i = 0; i < NUM_EVENTS; ++i) {
    taro.emplace([&, i]() {
      std::cout << "setting event " << i << "\n";
      events.set(i); // set event i
    });

    taro.emplace([&, i]() -> taro::Coro {
      co_await events.until(i); // wait until event i is set
      std::cout << "event " << i << " is set\n";
    });
  }

  taro.schedule();
  taro.wait();
}
