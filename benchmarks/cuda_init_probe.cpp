#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <string_view>

namespace {

void stage(const std::string_view name, const std::string_view status) {
  std::cout << "{\"stage\":\"" << name << "\",\"status\":\"" << status
            << "\"}" << std::endl;
}

bool check(const cudaError_t result, const std::string_view operation) {
  if (result == cudaSuccess) {
    stage(operation, "ok");
    return true;
  }
  std::cerr << "{\"stage\":\"" << operation
            << "\",\"status\":\"error\",\"cuda_error\":\""
            << cudaGetErrorName(result) << "\",\"message\":\""
            << cudaGetErrorString(result) << "\"}" << std::endl;
  return false;
}

}  // namespace

int main() {
  int device_count = 0;
  stage("cudaGetDeviceCount", "begin");
  if (!check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return 1;
  }
  if (device_count != 1) {
    std::cerr << "{\"stage\":\"device-count\",\"status\":\"error\","
                 "\"expected\":1,\"actual\":"
              << device_count << "}" << std::endl;
    return 1;
  }

  stage("cudaSetDevice", "begin");
  if (!check(cudaSetDevice(0), "cudaSetDevice")) {
    return 1;
  }
  stage("cudaFree-init", "begin");
  if (!check(cudaFree(nullptr), "cudaFree-init")) {
    return 1;
  }

  void* allocation = nullptr;
  stage("cudaMalloc", "begin");
  if (!check(cudaMalloc(&allocation, 4096U), "cudaMalloc")) {
    return 1;
  }
  stage("cudaMemset", "begin");
  if (!check(cudaMemset(allocation, 0, 4096U), "cudaMemset")) {
    static_cast<void>(cudaFree(allocation));
    return 1;
  }
  stage("cudaDeviceSynchronize", "begin");
  if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    static_cast<void>(cudaFree(allocation));
    return 1;
  }
  stage("cudaFree", "begin");
  if (!check(cudaFree(allocation), "cudaFree")) {
    return 1;
  }
  stage("complete", "ok");
  return 0;
}
