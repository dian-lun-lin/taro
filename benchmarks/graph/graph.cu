#include "graph.hpp"
#include "graph_executor.hpp"
#include "cudaflow_graph_executor.hpp"

#include <3rd-party/CLI11/CLI11.hpp>

#include <iostream>

int main(int argc, char* argv[]) {

  CLI::App app{"Graph Benchmark"};

  std::string graph{"SerialGraph"};
  app.add_option(
    "-g, --graph", 
    graph, 
    "select graph(SerialGraph, ParallelGraph, Tree, RandomDAG, MapReduce), default is SerialGraph" 
  );

  std::string job{"cudaflow_reduce"};
  app.add_option(
    "-j, --job",
    job,
    "select job(matmul, cudaflow_reduce, sleep), default is cudaflow_reduce"
  );

  int mode{8};
  app.add_option(
    "-m, --mode", 
    mode, 
    "select version(1, 2, 3, ..., 7, 8), default is 8. 0: cudaFlow"
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
    "number of streams to run. ignore this arg if using v1, v2, or v8"
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
  else {
    throw std::runtime_error("No such graph\n");
  }

  std::pair<double, double> time_pair;
  switch(mode) {
    case 0:
      {
        cudaFlowGraphExecutor executor(*g_ptr, 0, num_threads, num_streams);
        time_pair = executor.run(N, job);
      }
      break;
    //case 1:
      //{
        //GraphExecutor<taro::TaroV1> executor(*g_ptr, 0, num_threads); 
        //time_pair = executor.run(N, job);
      //}
      //break;
    //case 2:
      //{
        //GraphExecutor<taro::TaroV2> executor(*g_ptr, 0, num_threads); 
        //time_pair = executor.run(N, job);
      //}
      //break;
    case 3:
      {
        GraphExecutor<taro::TaroV3> executor(*g_ptr, 0, num_threads, num_streams); 
        time_pair = executor.run(N, job);
      }
      break;
    case 4:
      {
        GraphExecutor<taro::TaroV4> executor(*g_ptr, 0, num_threads, num_streams); 
        time_pair = executor.run(N, job);
      }
      break;
    case 5:
      {
        GraphExecutor<taro::TaroV5> executor(*g_ptr, 0, num_threads, num_streams); 
        time_pair = executor.run(N, job);
      }
      break;
    case 6:
      {
        GraphExecutor<taro::TaroV6> executor(*g_ptr, 0, num_threads, num_streams); 
        time_pair = executor.run(N, job);
      }
      break;
    case 7:
      {
        GraphExecutor<taro::TaroV7> executor(*g_ptr, 0, num_threads, num_streams); 
        time_pair = executor.run(N, job);
      }
      break;
    case 8:
      {
        GraphExecutor<taro::TaroV8> executor(*g_ptr, 0, num_threads); 
        time_pair = executor.run(N, job);
      }
      break;

    default:
      assert(false);
  }

  std::cout << "Construction time: " 
            << time_pair.first
            << " ms\n"
            << "Execution time: "
            << time_pair.second
            << " ms\n";
  //std::cout << time_pair.second << "\n";

  //g_ptr->print_graph(std::cout);
}

