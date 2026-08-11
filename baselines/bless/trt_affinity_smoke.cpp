#include <cuda.h>

#include <charconv>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "jetson_dla_green/json.hpp"
#include "jetson_dla_green/trt_benchmark.hpp"

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

int parse_positive(const std::string_view text) {
  int value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value <= 0) {
    throw std::invalid_argument("--affinity-sms expects a positive integer");
  }
  return value;
}

struct Arguments {
  int requested_sms{};
  std::vector<char*> benchmark;
};

Arguments parse_arguments(const int argc, char** argv) {
  Arguments result;
  result.benchmark.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--affinity-sms") {
      if (result.requested_sms != 0 || index + 1 >= argc) {
        throw std::invalid_argument(
            "--affinity-sms must appear exactly once with a value");
      }
      result.requested_sms = parse_positive(argv[++index]);
    } else {
      result.benchmark.push_back(argv[index]);
    }
  }
  if (result.requested_sms == 0) {
    throw std::invalid_argument("--affinity-sms is required");
  }
  return result;
}

}  // namespace

int main(const int argc, char** argv) {
  CUcontext context{};
  try {
    auto arguments = parse_arguments(argc, argv);
    require_cuda(cuInit(0), "cuInit");
    CUdevice device{};
    require_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    int supported = 0;
    require_cuda(cuDeviceGetExecAffinitySupport(
                     &supported, CU_EXEC_AFFINITY_TYPE_SM_COUNT, device),
                 "cuDeviceGetExecAffinitySupport");
    if (supported == 0) {
      throw std::runtime_error("SM execution affinity is unsupported");
    }

    CUexecAffinityParam affinity{};
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val =
        static_cast<unsigned int>(arguments.requested_sms);
    CUctxCreateParams parameters{};
    parameters.execAffinityParams = &affinity;
    parameters.numExecAffinityParams = 1;
    require_cuda(cuCtxCreate(&context, &parameters, 0, device),
                 "cuCtxCreate");
    CUexecAffinityParam observed{};
    observed.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    require_cuda(cuCtxGetExecAffinity(&observed,
                                      CU_EXEC_AFFINITY_TYPE_SM_COUNT),
                 "cuCtxGetExecAffinity");

    std::ostringstream benchmark_output;
    std::ostringstream benchmark_error;
    const int status = jdg::run_trt_benchmark(
        static_cast<int>(arguments.benchmark.size()),
        arguments.benchmark.data(), benchmark_output, benchmark_error);
    if (status != 0) {
      throw std::runtime_error("TensorRT affinity smoke failed: " +
                               benchmark_error.str());
    }
    require_cuda(cuCtxDestroy(context), "cuCtxDestroy");
    context = nullptr;

    std::cout << "{\n  \"schema_version\": 1,\n"
              << "  \"kind\": \"bless-thor-trt-affinity-smoke\",\n"
              << "  \"requested_sms\": " << arguments.requested_sms
              << ",\n  \"actual_sms\": "
              << observed.param.smCount.val << ",\n  \"benchmark\": "
              << benchmark_output.str() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    if (context != nullptr) {
      static_cast<void>(cuCtxDestroy(context));
    }
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
