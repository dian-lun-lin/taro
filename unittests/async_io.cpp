#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <cstdio>
#include <fcntl.h>
#include <stdio.h>
#include <taro/await/async_io.hpp>

// =====================================================================
//
// Testcase: read
// 
// =====================================================================

void read(size_t num_threads, size_t num_files, size_t data_size) {
  taro::Taro taro{num_threads};
  auto async_io = taro.async_io_await(num_files);

  std::vector<char> data(num_files * data_size);
  std::iota(data.begin(), data.end(), 'a');


  std::vector<char> buffer(num_files * data_size);
  std::vector<int> fds(num_files);

  for(size_t n = 0; n < num_files; ++n) {
    std::string file_name{"__taro__tmp__" + std::to_string(n) + ".out"};
    std::remove(file_name.c_str());
    fds[n] = open(file_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    int written = write(fds[n], data.data() + n * data_size, data_size);
    if(written == -1) {
      std::runtime_error("failed to write a file\n");
      return;
    }
  }

  auto finish = taro.emplace([&](){
    async_io.finish();
  });

  for(size_t n = 0; n < num_files; ++n) {
    auto task = taro.emplace([&, n, data_size]() -> taro::Coro {
      co_await async_io.read(fds[n], buffer.data() + n * data_size, data_size);
      close(fds[n]);
    });
    task.precede(finish);
  }

  taro.schedule();
  taro.wait();

  REQUIRE(data == buffer);

  for(size_t n = 0; n < num_files; ++n) {
    std::string file_name{"__taro__tmp__" + std::to_string(n) + ".out"};
    std::remove(file_name.c_str());
  }
}

//TEST_CASE("asyncio.read.1thread.1task.1datasize" * doctest::timeout(300)) {
  //read(1, 1, 1);
//}

TEST_CASE("asyncio.read.2thread.1task.8datasize" * doctest::timeout(300)) {
  read(2, 2, 8);
}

TEST_CASE("asyncio.read.2thread.5task.8datasize" * doctest::timeout(300)) {
  read(2, 5, 8);
}

TEST_CASE("asyncio.read.4thread.3task.8datasize" * doctest::timeout(300)) {
  read(4, 2, 8);
}

TEST_CASE("asyncio.read.4thread.9task.10000datasize" * doctest::timeout(300)) {
  read(4, 9, 1000);
}

TEST_CASE("asyncio.read.4thread.27task.344datasize" * doctest::timeout(300)) {
  read(4, 27, 344);
}

TEST_CASE("asyncio.read.5thread.91task.82datasize" * doctest::timeout(300)) {
  read(5, 91, 82);
}

TEST_CASE("asyncio.read.5thread.1997task.2datasize" * doctest::timeout(300)) {
  read(5, 1997, 200);
}

// =====================================================================
//
// Testcase: write
// 
// =====================================================================

void write(size_t num_threads, size_t num_files, size_t data_size) {
  taro::Taro taro{num_threads};
  auto async_io = taro.async_io_await(num_files);

  std::vector<std::vector<char>> data(num_files);
  int cnt{0};
  for(auto& d: data) {
    d.resize(data_size);
    std::iota(d.begin(), d.end(), 'a' + cnt++);
  }

  std::vector<int> fds(num_files);
  for(size_t n = 0; n < num_files; ++n) {
    std::string file_name{"__taro__tmp__" + std::to_string(n) + ".out"};
    std::remove(file_name.c_str());
    fds[n] = open(file_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  }

  // task graph
  auto finish = taro.emplace([&](){
    async_io.finish();
  });

  for(size_t n = 0; n < num_files; ++n) {
    auto task = taro.emplace([&, n, data_size]() -> taro::Coro {
      co_await async_io.write(fds[n], data[n].data(), data_size);
      close(fds[n]);
    });
    task.precede(finish);
  }

  taro.schedule();
  taro.wait();

  // read results from result files
  for(size_t n = 0; n < num_files; ++n) {
    std::string file_name{"__taro__tmp__" + std::to_string(n) + ".out"};
    std::fstream file;
    file.open(file_name);
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string& s = ss.str();
    std::vector<char> buffer(s.begin(), s.end());
    REQUIRE(data[n] == buffer);
  }

  // remove
  for(size_t n = 0; n < num_files; ++n) {
    std::string file_name{"__taro__tmp__" + std::to_string(n) + ".out"};
    std::remove(file_name.c_str());
  }
}

TEST_CASE("asyncio.write.2thwrite.1task.8datasize" * doctest::timeout(300)) {
  write(2, 2, 8);
}

TEST_CASE("asyncio.write.2thwrite.5task.8datasize" * doctest::timeout(300)) {
  write(2, 5, 8);
}

TEST_CASE("asyncio.write.4thwrite.3task.8datasize" * doctest::timeout(300)) {
  write(4, 2, 8);
}

TEST_CASE("asyncio.write.4thwrite.9task.10000datasize" * doctest::timeout(300)) {
  write(4, 9, 1000);
}

TEST_CASE("asyncio.write.4thwrite.27task.344datasize" * doctest::timeout(300)) {
  write(4, 27, 344);
}

TEST_CASE("asyncio.write.5thwrite.91task.82datasize" * doctest::timeout(300)) {
  write(5, 91, 82);
}

TEST_CASE("asyncio.write.5thwrite.1997task.2datasize" * doctest::timeout(300)) {
  write(5, 500, 200);
}
