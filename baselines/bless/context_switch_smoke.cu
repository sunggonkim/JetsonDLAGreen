#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::array<unsigned int, 4> kShares{2U, 4U, 6U, 8U};

__global__ void advance_stage(std::uint64_t* state,
                              const std::uint64_t contribution) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *state = (*state * 1315423911ULL) ^ contribution;
  }
}

void require_driver(const CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    static_cast<void>(cuGetErrorName(result, &name));
    throw std::runtime_error(std::string(operation) + ": " +
                             (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name));
  }
}

void require_runtime(const cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorName(result));
  }
}

struct AffinityContext {
  CUcontext context{};
  cudaStream_t stream{};
  unsigned int actual_sms{};
  std::uint64_t* device_state{};
};

void set_current(const CUcontext context) {
  require_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

}  // namespace

int main() {
  std::array<AffinityContext, kShares.size()> contexts{};
  std::uint64_t* host_state = nullptr;
  try {
    require_driver(cuInit(0), "cuInit");
    CUdevice device{};
    require_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    for (std::size_t index = 0; index < contexts.size(); ++index) {
      CUexecAffinityParam affinity{};
      affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      affinity.param.smCount.val = kShares[index];
      CUctxCreateParams parameters{};
      parameters.execAffinityParams = &affinity;
      parameters.numExecAffinityParams = 1;
      require_driver(cuCtxCreate(&contexts[index].context, &parameters,
                                 CU_CTX_MAP_HOST, device),
                     "cuCtxCreate");
      CUexecAffinityParam observed{};
      observed.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      require_driver(cuCtxGetExecAffinity(&observed,
                                          CU_EXEC_AFFINITY_TYPE_SM_COUNT),
                     "cuCtxGetExecAffinity");
      contexts[index].actual_sms = observed.param.smCount.val;
      require_runtime(cudaStreamCreateWithFlags(&contexts[index].stream,
                                                cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
    }

    set_current(contexts.front().context);
    require_runtime(cudaHostAlloc(&host_state, sizeof(*host_state),
                                  cudaHostAllocMapped),
                    "cudaHostAlloc");
    *host_state = 0x123456789abcdef0ULL;
    std::uint64_t expected = *host_state;

    for (std::size_t index = 0; index < contexts.size(); ++index) {
      auto& current = contexts[index];
      set_current(current.context);
      require_runtime(cudaHostGetDevicePointer(&current.device_state,
                                               host_state, 0),
                      "cudaHostGetDevicePointer");
      const std::uint64_t contribution =
          static_cast<std::uint64_t>(index + 1U) * 0x9e3779b97f4a7c15ULL;
      advance_stage<<<1, 1, 0, current.stream>>>(current.device_state,
                                                contribution);
      require_runtime(cudaGetLastError(), "advance_stage launch");
      require_runtime(cudaStreamSynchronize(current.stream),
                      "cudaStreamSynchronize");
      expected = (expected * 1315423911ULL) ^ contribution;
    }

    const bool passed = *host_state == expected;
    std::cout << "{\n  \"schema_version\": 1,\n"
              << "  \"kind\": \"bless-thor-context-switch-smoke\",\n"
              << "  \"requested_sms\": [2,4,6,8],\n"
              << "  \"actual_sms\": [";
    for (std::size_t index = 0; index < contexts.size(); ++index) {
      std::cout << (index == 0 ? "" : ",") << contexts[index].actual_sms;
    }
    std::cout << "],\n  \"stages\": " << contexts.size()
              << ",\n  \"checksum\": " << *host_state
              << ",\n  \"expected_checksum\": " << expected
              << ",\n  \"status\": \"" << (passed ? "passed" : "failed")
              << "\"\n}\n";
    if (!passed) {
      throw std::runtime_error("cross-context activation checksum differs");
    }

    set_current(contexts.front().context);
    require_runtime(cudaFreeHost(host_state), "cudaFreeHost");
    host_state = nullptr;
    for (auto& current : contexts) {
      set_current(current.context);
      require_runtime(cudaStreamDestroy(current.stream), "cudaStreamDestroy");
      current.stream = nullptr;
      require_driver(cuCtxDestroy(current.context), "cuCtxDestroy");
      current.context = nullptr;
    }
    return 0;
  } catch (const std::exception& error) {
    if (host_state != nullptr && contexts.front().context != nullptr) {
      static_cast<void>(cuCtxSetCurrent(contexts.front().context));
      static_cast<void>(cudaFreeHost(host_state));
    }
    for (auto& current : contexts) {
      if (current.stream != nullptr && current.context != nullptr) {
        static_cast<void>(cuCtxSetCurrent(current.context));
        static_cast<void>(cudaStreamDestroy(current.stream));
      }
      if (current.context != nullptr) {
        static_cast<void>(cuCtxDestroy(current.context));
      }
    }
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
