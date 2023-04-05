#include <taro.hpp>

#include "../executor/taro_callback_v3.hpp"

#include <3rd-party/CLI11/CLI11.hpp>

#include <iostream>

int main(int argc, char* argv[]) {

  CLI::App app{"Graph Benchmark"};

  std::string graph{"ParallelGraph"};
  app.add_option(
    "-g, --graph", 
    graph, 
    "select graph(SerialGraph, ParallelGraph, Tree, RandomDAG, MapReduce, Wavefront), default is ParallelGraph" 
  );

  std::string job{"sleep"};
  app.add_option(
    "-j, --job",
    job,
    "select job(matmul, cudaflow_reduce, sleep), default is sleep"
  );

  size_t N{1024};
  app.add_option(
    "-n, --problem_size", 
    N, 
    "set problem size. ignore this arg if using sleep job"
  );

  std::vector<int> args;
  app.add_option(
    "-a, --args",
    args,
    "args for constructing a graph"
  );

  size_t num_threads;
  app.add_option(
    "-t, --num_threads",
    num_threads,
    "number of threads"
  );

  size_t num_streams;
  app.add_option(
    "-s, --num_streams",
    num_streams,
    "number of streams to run. ignore this arg if using TaroCBV2"
  );

  CLI11_PARSE(app, argc, argv);

  Graph* g_ptr;
  if(graph == "SerialGraph") {
    assert(args.size() == 1);
    g_ptr = new SerialGraph(args[0]);
  }
  else if(graph == "ParallelGraph") {
    assert(args.size() == 1);
    g_ptr = new ParallelGraph(args[0]);
  }
  else if(graph == "Tree") {
    assert(args.size() == 2);
    g_ptr = new Tree(args[0], args[1]);
  }
  else if(graph == "RandomDAG") {
    assert(args.size() == 3);
    g_ptr = new RandomDAG(args[0], args[1], args[2]);
  }
  else if(graph == "MapReduce") {
    assert(args.size() == 2);
    g_ptr = new Diamond(args[0], args[1]);
  }
  else if(graph == "Wavefront") {
    assert(args.size() == 1);
    g_ptr = new WavefrontGraph(args[0]);
  }
  else {
    throw std::runtime_error("No such graph\n");
  }

  std::pair<double, double> time_pair;

  GraphExecutor executor(*g_ptr, 0, num_threads, num_streams); 
  time_pair = executor.run(N, job);
  
  std::cout << "Construction time: " 
            << time_pair.first
            << " ms\n"
            << "Execution time: "
            << time_pair.second
            << " ms\n";
}

