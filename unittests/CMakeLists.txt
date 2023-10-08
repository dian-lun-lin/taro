enable_testing()
include(${TARO_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TARO_UNITTESTS
  events
)

list(APPEND TARO_CUDA_UNITTESTS
  independent
  linear_chain
  pipeline
)

foreach(unittest IN LISTS TARO_UNITTESTS)
  add_executable(${unittest} ${TARO_UTEST_DIR}/${unittest}.cpp)
  target_include_directories(${unittest} PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
  target_link_libraries(${unittest} Threads::Threads)
  doctest_discover_tests(${unittest})
endforeach()

#foreach(unittest IN LISTS TARO_CUDA_UNITTESTS)
  #cuda_add_executable(${unittest} ${TARO_UTEST_DIR}/${unittest}.cu)
  #target_include_directories(${unittest} PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
  #target_link_libraries(${unittest} Threads::Threads cuda)
  #doctest_discover_tests(${unittest})
#endforeach()

