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

#include <taskflow/taskflow/cuda/algorithm/reduce.hpp>
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
