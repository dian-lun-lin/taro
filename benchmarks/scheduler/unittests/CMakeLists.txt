
enable_testing()
include(${TARO_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND BENCHMARK_UNITTESTS
  independent
  linear_chain
)

foreach(bunittest IN LISTS BENCHMARK_UNITTESTS)
  add_subdirectory(${TARO_BENCHMARK_UTEST_DIR}/${bunittest})
endforeach()
