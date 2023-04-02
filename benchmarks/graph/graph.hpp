#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>
#include <random>


#include "base/graph_base.hpp"
#include "tree/tree.hpp"
#include "random_DAG/random_DAG.hpp"
#include "extreme_graph/parallel_graph.hpp"
#include "extreme_graph/serial_graph.hpp"
#include "map_reduce/diamond.hpp"
#include "wavefront/wavefront_graph.hpp"

#include "../boost_fiber/fiber.hpp"


#include <taskflow/taskflow/taskflow.hpp>
#include <taskflow/taskflow/cuda/cudaflow.hpp>
#include <taskflow/taskflow/cuda/algorithm/reduce.hpp>

#include <taro/src/cuda/callback/taro_callback_v1.hpp>
#include <taro/src/cuda/callback/taro_callback_v2.hpp>
#include <taro/src/cuda/callback/taro_callback_v3.hpp>
#include <taro/src/cuda/callback/taro_callback_taskflow.hpp>
#include <taro/src/cuda/poll/taro_poll_v1.hpp>
#include <taro/src/cuda/poll/taro_poll_v2.hpp>
#include <taro/src/cuda/algorithm.hpp>

#include <chrono>
#include <cassert>

// GPU sleep kernel
__global__ void cuda_sleep(
   int ms
) {
  for (int i = 0; i < ms; i++) {
    __nanosleep(1000000U);
  }
}

// CPU task
void cpu_sleep(
  int ms
) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
