#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <taro/await/sycl.hpp>

void simple(size_t num_threads, size_t num_tasks, size_t data_size) {
  taro::Taro taro{num_threads};
  sycl::queue que;
  auto sycl = taro.sycl_await(que);


  for(size_t n = 0; n < num_tasks; ++n) {
    taro.emplace([&]() -> taro::Coro {
      int* d_data = sycl::malloc_device<int>(data_size, que);
      std::vector<int> h_res(data_size, -2);

      sycl::range<1> num_work_items{data_size};

      co_await sycl.until_polling([&](sycl::handler& cgh) {
        // Executing kernel
        cgh.parallel_for<class FillBuffer>(
          num_work_items, [=](sycl::id<1> wid) {
          d_data[wid] = wid.get(0);
        });
      });
    
      sycl.wait([=, &h_res](sycl::handler& cgh) {
        cgh.memcpy(h_res.data(), d_data, data_size * sizeof(int));
      });


      // Access the buffer data without synchronization
      for(size_t i = 0; i < h_res.size(); ++i) {
        REQUIRE(h_res[i] == i);
      }
    });
  }

  taro.schedule();
  taro.wait();

}

TEST_CASE("simple.1thread.1task.1datasize" * doctest::timeout(300)) {
  simple(1, 1, 1);
}

TEST_CASE("simple.2thread.1task.2datasize" * doctest::timeout(300)) {
  simple(2, 1, 2);
}

TEST_CASE("simple.4thread.3task.8datasize" * doctest::timeout(300)) {
  simple(4, 3, 8);
}

TEST_CASE("simple.4thread.9task.10000datasize" * doctest::timeout(300)) {
  simple(4, 9, 10000);
}

TEST_CASE("simple.4thread.27task.344datasize" * doctest::timeout(300)) {
  simple(4, 27, 344);
}

TEST_CASE("simple.5thread.91task.82datasize" * doctest::timeout(300)) {
  simple(5, 91, 82);
}

TEST_CASE("simple.5thread.1997task.2datasize" * doctest::timeout(300)) {
  simple(5, 1997, 2);
}
