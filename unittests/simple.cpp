#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro.hpp>


//// --------------------------------------------------------
//// Testcase::Construct
//// --------------------------------------------------------

TEST_CASE("construct" * doctest::timeout(300)) {
  for(int i = 0; i < 331; ++i) {
    taro::Taro taro{i % std::thread::hardware_concurrency()};
    if(i % 3 == 0) {
      taro.schedule();
      taro.wait();
    }
    REQUIRE(taro.is_DAG());
  }
}
