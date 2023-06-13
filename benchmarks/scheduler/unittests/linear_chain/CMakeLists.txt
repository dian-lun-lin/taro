list(APPEND LINEAR_CHAIN_UNITTESTS
  taro_taskflow
  taro_oblivious
)

foreach(unittest IN LISTS LINEAR_CHAIN_UNITTESTS)
  cuda_add_executable(linear_chain_${unittest} ${TARO_BENCHMARK_UTEST_DIR}/linear_chain/${unittest}.cu)
  target_include_directories(linear_chain_${unittest} PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
  target_link_libraries(linear_chain_${unittest} Threads::Threads cuda)
  doctest_discover_tests(linear_chain_${unittest})
endforeach()

# fiber
cuda_add_executable(linear_chain_fiber ${TARO_BENCHMARK_UTEST_DIR}/linear_chain/fiber.cu)
target_include_directories(linear_chain_fiber PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
target_link_libraries(linear_chain_fiber boost_fiber boost_context cuda)
doctest_discover_tests(linear_chain_fiber)
