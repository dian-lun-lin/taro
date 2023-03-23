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



#include <taskflow/taskflow/taskflow.hpp>
#include <taskflow/taskflow/cuda/cudaflow.hpp>
#include <taskflow/taskflow/cuda/algorithm/reduce.hpp>

#include <taro/src/cuda/callback/taro_callback_v1.hpp>
#include <taro/src/cuda/callback/taro_callback_v2.hpp>
#include <taro/src/cuda/callback/taro_callback_taskflow.hpp>
#include <taro/src/cuda/poll/taro_poll_v1.hpp>
#include <taro/src/cuda/poll/taro_poll_v2.hpp>
#include <taro/src/cuda/algorithm.hpp>

#include <chrono>
#include <cassert>
