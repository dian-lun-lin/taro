#pragma once
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCudaError(cudaError_t result)
{
  if (result != cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("CUDA Runtime Error : "s + cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}
