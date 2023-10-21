#include <iostream>
#include <taro.hpp>
#include <taro/await/sycl.hpp>


int main() {
  size_t num_threads{4};
  size_t num_tasks{10};
  size_t data_size{1000};

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
    
      // d2h
      sycl.wait([=, &h_res](sycl::handler& cgh) {
        cgh.memcpy(h_res.data(), d_data, data_size * sizeof(int));
      }); 

      for(size_t i = 0; i < data_size; ++i) {
        if(h_res[i] != i) {
          throw std::runtime_error("results is incorrect!\n");
        }
      }

    });
  }

  taro.schedule();
  taro.wait();

  std::cout << "result is correct!\n";
}

