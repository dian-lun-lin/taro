enable_testing()
include(${TARO_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TARO_UNITTESTS
  simple
  event
  #semaphore
)

foreach(unittest IN LISTS TARO_UNITTESTS)
  add_executable(${unittest} ${TARO_UTEST_DIR}/${unittest}.cpp)
  target_include_directories(${unittest} PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
  target_link_libraries(${unittest} Threads::Threads)
  doctest_discover_tests(${unittest})
endforeach()


#### CUDA
if(TARO_BUILD_CUDA)

list(APPEND TARO_CUDA_UNITTESTS
  independent
  linear_chain
  pipeline
)
foreach(unittest IN LISTS TARO_CUDA_UNITTESTS)
  cuda_add_executable(${unittest} ${TARO_UTEST_DIR}/${unittest}.cu)
  target_include_directories(${unittest} PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
  target_link_libraries(${unittest} Threads::Threads cuda)
  doctest_discover_tests(${unittest})
endforeach()

endif()

#### SYCL
if(TARO_BUILD_SYCL)

list(APPEND TARO_SYCL_UNITTESTS
  sycl
)

foreach(unittest IN LISTS TARO_SYCL_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_include_directories(${unittest} PRIVATE ${TARO_3RD_PARTY_DIR}/doctest)
  target_compile_options(${unittest} PRIVATE ${TARO_SYCL_OPTIONS})
  target_link_options(${unittest} PRIVATE ${TARO_SYCL_OPTIONS})
  target_link_libraries(${unittest} Threads::Threads)
  doctest_discover_tests(${unittest})
endforeach()

endif()
