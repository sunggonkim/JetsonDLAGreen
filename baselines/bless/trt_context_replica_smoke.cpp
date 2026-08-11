#include <cuda.h>

#include <charconv>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "jetson_dla_green/trt_benchmark.hpp"

namespace {

constexpr unsigned int kAffinitySms[] = {2U, 4U, 6U, 8U};

[[nodiscard]] std::string error_name(const CUresult result) {
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

[[nodiscard]] int parse_positive(const std::string_view text,
                                 const char* option) {
  int value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value <= 0) {
    throw std::invalid_argument(std::string(option) +
                                " expects a positive integer");
  }
  return value;
}

struct Arguments {
  int rounds{2};
  std::vector<char*> benchmark;
};

[[nodiscard]] Arguments parse_arguments(const int argc, char** argv) {
  Arguments result;
  result.benchmark.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--replica-rounds") {
      if (index + 1 >= argc) {
        throw std::invalid_argument("--replica-rounds requires a value");
      }
      result.rounds = parse_positive(argv[++index], "--replica-rounds");
    } else {
      result.benchmark.push_back(argv[index]);
    }
  }
  return result;
}

class AffinityContext {
 public:
  AffinityContext(const CUdevice device, const unsigned int requested_sms)
      : requested_sms_(requested_sms) {
    CUexecAffinityParam affinity{};
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val = requested_sms;
    CUctxCreateParams parameters{};
    parameters.execAffinityParams = &affinity;
    parameters.numExecAffinityParams = 1;
    require_cuda(cuCtxCreate(&context_, &parameters, 0, device),
                 "cuCtxCreate(replica)");
    CUexecAffinityParam observed{};
    observed.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    require_cuda(cuCtxGetExecAffinity(&observed, CU_EXEC_AFFINITY_TYPE_SM_COUNT),
                 "cuCtxGetExecAffinity(replica)");
    actual_sms_ = observed.param.smCount.val;
    CUcontext popped = nullptr;
    require_cuda(cuCtxPopCurrent(&popped), "cuCtxPopCurrent(replica create)");
    if (popped != context_) {
      throw std::runtime_error("created BLESS context was not current");
    }
  }

  AffinityContext(const AffinityContext&) = delete;
  AffinityContext& operator=(const AffinityContext&) = delete;
  AffinityContext(AffinityContext&& other) noexcept
      : context_(std::exchange(other.context_, nullptr)),
        requested_sms_(other.requested_sms_),
        actual_sms_(other.actual_sms_) {}
  AffinityContext& operator=(AffinityContext&&) = delete;

  ~AffinityContext() {
    if (context_ != nullptr) {
      static_cast<void>(cuCtxDestroy(context_));
    }
  }

  [[nodiscard]] CUcontext handle() const noexcept { return context_; }
  [[nodiscard]] std::uintptr_t id() const noexcept {
    return reinterpret_cast<std::uintptr_t>(context_);
  }
  [[nodiscard]] unsigned int requested_sms() const noexcept {
    return requested_sms_;
  }
  [[nodiscard]] unsigned int actual_sms() const noexcept { return actual_sms_; }

 private:
  CUcontext context_{};
  unsigned int requested_sms_{};
  unsigned int actual_sms_{};
};

class CurrentContext {
 public:
  explicit CurrentContext(const CUcontext context) {
    require_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent(replica run)");
  }
  CurrentContext(const CurrentContext&) = delete;
  CurrentContext& operator=(const CurrentContext&) = delete;
  ~CurrentContext() {
    CUcontext popped = nullptr;
    static_cast<void>(cuCtxPopCurrent(&popped));
  }
};

}  // namespace

int main(const int argc, char** argv) {
  try {
    Arguments arguments = parse_arguments(argc, argv);
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

    std::vector<AffinityContext> replicas;
    replicas.reserve(std::size(kAffinitySms));
    for (const unsigned int sms : kAffinitySms) {
      replicas.emplace_back(device, sms);
    }

    std::cout << "{\n  \"schema_version\":1,\n"
                 "  \"kind\":\"bless-thor-trt-context-replica-smoke\",\n"
              << "  \"replica_rounds\":" << arguments.rounds
              << ",\n  \"replicas\":[";
    bool first = true;
    for (int round = 0; round < arguments.rounds; ++round) {
      for (const AffinityContext& replica : replicas) {
        CurrentContext current(replica.handle());
        std::ostringstream output;
        std::ostringstream error;
        const int status = jdg::run_trt_benchmark(
            static_cast<int>(arguments.benchmark.size()),
            arguments.benchmark.data(), output, error);
        if (status != 0) {
          throw std::runtime_error("TensorRT replica failed at " +
                                   std::to_string(replica.actual_sms()) +
                                   " SM: " + error.str());
        }
        if (!first) {
          std::cout << ',';
        }
        first = false;
        std::cout << "\n    {\"round\":" << round
                  << ",\"context_id\":" << replica.id()
                  << ",\"requested_sms\":" << replica.requested_sms()
                  << ",\"actual_sms\":" << replica.actual_sms()
                  << ",\"benchmark\":" << output.str() << '}';
      }
    }
    std::cout << "\n  ]\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
