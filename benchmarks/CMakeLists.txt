
add_subdirectory(scheduler/unittests)

list(APPEND BENCHMARKS
  taro_taskflow
  taro_oblivious
  cudaflow
  taro
)

foreach(benchmark IN LISTS BENCHMARKS)
  cuda_add_executable(graph_${benchmark} ${TARO_BENCHMARK_DIR}/graph/main/${benchmark}.cu)
  target_link_libraries(graph_${benchmark} Threads::Threads cuda)
endforeach()

# fiber
cuda_add_executable(graph_fiber ${TARO_BENCHMARK_DIR}/graph/main/fiber.cu)
target_link_libraries(graph_fiber boost_fiber boost_context cuda)

