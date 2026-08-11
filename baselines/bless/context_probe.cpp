#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string error_name(const CUresult result) {
  const char* name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "CUDA_ERROR_UNKNOWN";
}

void require_cuda(const CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             error_name(result));
  }
}

std::uint64_t elapsed_ns(const std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

}  // namespace

int main() {
  try {
    require_cuda(cuInit(0), "cuInit");
    CUdevice device{};
    require_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    char name[256]{};
    int multiprocessors = 0;
    int supported = 0;
    require_cuda(cuDeviceGetName(name, sizeof(name), device),
                 "cuDeviceGetName");
    require_cuda(cuDeviceGetAttribute(
                     &multiprocessors,
                     CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
                 "cuDeviceGetAttribute");
    require_cuda(cuDeviceGetExecAffinitySupport(
                     &supported, CU_EXEC_AFFINITY_TYPE_SM_COUNT, device),
                 "cuDeviceGetExecAffinitySupport");

    std::cout << "{\n  \"schema_version\": 1,\n"
              << "  \"kind\": \"bless-thor-context-domain\",\n"
              << "  \"device\": \"" << name << "\",\n"
              << "  \"multiprocessors\": " << multiprocessors << ",\n"
              << "  \"exec_affinity_supported\": "
              << (supported != 0 ? "true" : "false") << ",\n"
              << "  \"requests\": [\n";
    for (int requested = 1; requested <= multiprocessors; ++requested) {
      CUexecAffinityParam parameter{};
      parameter.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      parameter.param.smCount.val = static_cast<unsigned int>(requested);
      CUctxCreateParams create_parameters{};
      create_parameters.execAffinityParams = &parameter;
      create_parameters.numExecAffinityParams = 1;
      CUcontext context{};
      const auto start = std::chrono::steady_clock::now();
      const CUresult create =
          cuCtxCreate(&context, &create_parameters, 0, device);
      const std::uint64_t create_ns = elapsed_ns(start);
      unsigned int actual = 0;
      CUresult query = CUDA_ERROR_NOT_INITIALIZED;
      CUresult destroy = CUDA_ERROR_NOT_INITIALIZED;
      if (create == CUDA_SUCCESS) {
        CUexecAffinityParam observed{};
        observed.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        query = cuCtxGetExecAffinity(&observed,
                                     CU_EXEC_AFFINITY_TYPE_SM_COUNT);
        if (query == CUDA_SUCCESS) {
          actual = observed.param.smCount.val;
        }
        destroy = cuCtxDestroy(context);
      }
      std::cout << "    {\"requested_sms\":" << requested
                << ",\"create_result\":\"" << error_name(create)
                << "\",\"actual_sms\":" << actual
                << ",\"query_result\":\"" << error_name(query)
                << "\",\"destroy_result\":\"" << error_name(destroy)
                << "\",\"create_ns\":" << create_ns << "}"
                << (requested == multiprocessors ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
