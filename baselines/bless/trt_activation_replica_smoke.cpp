#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef JDG_BLESS_SQUAD
#include "trt_squad_intercept.hpp"
#endif

namespace {

constexpr std::array<unsigned int, 4> kAffinitySms{2U, 4U, 6U, 8U};

template <typename T>
using TrtPtr = std::unique_ptr<T>;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

void require(const bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_cuda(const cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorName(result));
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

[[nodiscard]] std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(input), "cannot open TensorRT engine");
  const auto end = input.tellg();
  require(end > 0, "TensorRT engine is empty");
  std::vector<char> data(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  require(static_cast<bool>(input.read(data.data(), end)),
          "failed to read TensorRT engine");
  return data;
}

[[nodiscard]] bool dynamic(const nvinfer1::Dims& dimensions) {
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] < 0) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::size_t type_size(const nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4U;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2U;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kE8M0:
      return 1U;
    case nvinfer1::DataType::kINT64:
      return 8U;
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
      throw std::runtime_error("packed TensorRT I/O is unsupported");
  }
  throw std::runtime_error("unknown TensorRT data type");
}

[[nodiscard]] std::size_t tensor_bytes(const nvinfer1::Dims& dimensions,
                                       const nvinfer1::DataType type) {
  std::size_t elements = 1U;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    require(dimensions.d[index] > 0, "unresolved TensorRT I/O dimension");
    const auto value = static_cast<std::size_t>(dimensions.d[index]);
    require(elements <= std::numeric_limits<std::size_t>::max() / value,
            "TensorRT I/O size overflow");
    elements *= value;
  }
  const std::size_t bytes = type_size(type);
  require(elements <= std::numeric_limits<std::size_t>::max() / bytes,
          "TensorRT I/O byte size overflow");
  return elements * bytes;
}

[[nodiscard]] std::uint64_t checksum(const void* data,
                                     const std::size_t bytes) {
  const auto* current = static_cast<const std::uint8_t*>(data);
  std::uint64_t result = 1469598103934665603ULL;
  for (std::size_t index = 0; index < bytes; ++index) {
    result ^= current[index];
    result *= 1099511628211ULL;
  }
  return result;
}

class Replica {
 public:
  Replica(const CUdevice device, const unsigned int requested_sms,
          const std::vector<char>& serialized, Logger& logger)
      : requested_sms_(requested_sms) {
    CUexecAffinityParam affinity{};
    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity.param.smCount.val = requested_sms;
    CUctxCreateParams parameters{};
    parameters.execAffinityParams = &affinity;
    parameters.numExecAffinityParams = 1;
    require_driver(cuCtxCreate(&driver_context_, &parameters, 0, device),
                   "cuCtxCreate(activation replica)");
    try {
      CUexecAffinityParam observed{};
      observed.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      require_driver(cuCtxGetExecAffinity(&observed,
                                          CU_EXEC_AFFINITY_TYPE_SM_COUNT),
                     "cuCtxGetExecAffinity(activation replica)");
      actual_sms_ = observed.param.smCount.val;
      require(actual_sms_ == requested_sms_, "activation affinity differs");
      runtime_.reset(nvinfer1::createInferRuntime(logger));
      require(runtime_ != nullptr, "failed to create TensorRT runtime");
      engine_.reset(runtime_->deserializeCudaEngine(serialized.data(),
                                                    serialized.size()));
      require(engine_ != nullptr, "failed to deserialize TensorRT engine");
      context_.reset(engine_->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
      require(context_ != nullptr, "failed to create user-managed context");
      for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
        const char* name = engine_->getIOTensorName(index);
        require(name != nullptr, "TensorRT returned null I/O name");
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          const auto shape = engine_->getTensorShape(name);
          if (dynamic(shape)) {
            const auto selected = engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kOPT);
            require(!dynamic(selected), "unresolved TensorRT profile shape");
            require(context_->setInputShape(name, selected),
                    "failed to set TensorRT input shape");
          }
        }
      }
      require(context_->inferShapes(0, nullptr) == 0,
              "not all TensorRT shapes were specified");
      const auto required_memory = engine_->getDeviceMemorySizeV2();
      require(required_memory > 0, "TensorRT activation memory is empty");
      activation_bytes_ = static_cast<std::size_t>(required_memory);
      require_cuda(cudaMalloc(&activation_, activation_bytes_),
                   "cudaMalloc(TensorRT activation)");
      context_->setDeviceMemoryV2(activation_, required_memory);
      require_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreate(activation replica)");
      for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
        const char* name = engine_->getIOTensorName(index);
        const bool input = engine_->getTensorIOMode(name) ==
                           nvinfer1::TensorIOMode::kINPUT;
        auto shape = context_->getTensorShape(name);
        std::size_t bytes = 0U;
        if (dynamic(shape) && !input) {
          const auto maximum = context_->getMaxOutputSize(name);
          require(maximum > 0, "TensorRT output size is unresolved");
          bytes = static_cast<std::size_t>(maximum);
        } else {
          bytes = tensor_bytes(shape, engine_->getTensorDataType(name));
        }
        Buffer buffer{name, nullptr, bytes, input};
        require_cuda(cudaMalloc(&buffer.device, bytes),
                     "cudaMalloc(TensorRT I/O)");
        require_cuda(cudaMemset(buffer.device, 0, bytes),
                     "cudaMemset(TensorRT I/O)");
        require(context_->setTensorAddress(buffer.name.c_str(), buffer.device),
                "failed to bind TensorRT I/O");
        buffers_.push_back(std::move(buffer));
      }
      CUcontext popped = nullptr;
      require_driver(cuCtxPopCurrent(&popped),
                     "cuCtxPopCurrent(activation replica create)");
      require(popped == driver_context_, "activation context stack differs");
    } catch (...) {
      release_current();
      throw;
    }
  }

  Replica(const Replica&) = delete;
  Replica& operator=(const Replica&) = delete;
  Replica(Replica&&) = delete;
  Replica& operator=(Replica&&) = delete;
  ~Replica() { release(); }

  [[nodiscard]] std::uint64_t infer() {
    push();
    try {
      require(context_->enqueueV3(stream_), "TensorRT enqueueV3 failed");
      require_cuda(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize(activation replica)");
      std::uint64_t result = 1469598103934665603ULL;
      for (const auto& buffer : buffers_) {
        if (!buffer.input) {
          std::vector<std::uint8_t> host(buffer.bytes);
          require_cuda(cudaMemcpy(host.data(), buffer.device, buffer.bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(TensorRT output)");
          result ^= checksum(host.data(), host.size());
          result *= 1099511628211ULL;
        }
      }
      pop();
      return result;
    } catch (...) {
      pop_noexcept();
      throw;
    }
  }

#ifdef JDG_BLESS_SQUAD
  void reset_state() {
    push();
    try {
      require_cuda(cudaMemset(activation_, 0, activation_bytes_),
                   "cudaMemset(TensorRT activation)");
      for (const auto& buffer : buffers_) {
        require_cuda(cudaMemset(buffer.device, 0, buffer.bytes),
                     "cudaMemset(TensorRT squad I/O)");
      }
      require_cuda(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize(TensorRT reset)");
      pop();
    } catch (...) {
      pop_noexcept();
      throw;
    }
  }

  void enqueue_squad() {
    push();
    try {
      require(context_->enqueueV3(stream_), "TensorRT squad enqueue failed");
      require_cuda(cudaStreamSynchronize(stream_),
                   "cudaStreamSynchronize(TensorRT squad)");
      pop();
    } catch (...) {
      pop_noexcept();
      throw;
    }
  }

  [[nodiscard]] std::uint64_t output_checksum() {
    push();
    try {
      std::uint64_t result = 1469598103934665603ULL;
      for (const auto& buffer : buffers_) {
        if (!buffer.input) {
          std::vector<std::uint8_t> host(buffer.bytes);
          require_cuda(cudaMemcpy(host.data(), buffer.device, buffer.bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(TensorRT squad output)");
          result ^= checksum(host.data(), host.size());
          result *= 1099511628211ULL;
        }
      }
      pop();
      return result;
    } catch (...) {
      pop_noexcept();
      throw;
    }
  }
#endif

  [[nodiscard]] std::uint64_t activation_checksum() {
    push();
    try {
      std::vector<std::uint8_t> host(activation_bytes_);
      require_cuda(cudaMemcpy(host.data(), activation_, activation_bytes_,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(TensorRT activation)");
      const auto result = checksum(host.data(), host.size());
      pop();
      return result;
    } catch (...) {
      pop_noexcept();
      throw;
    }
  }

  [[nodiscard]] unsigned int sms() const noexcept { return actual_sms_; }
  [[nodiscard]] CUcontext driver_context() const noexcept {
    return driver_context_;
  }
  [[nodiscard]] CUdeviceptr activation() const noexcept {
    return reinterpret_cast<CUdeviceptr>(activation_);
  }
  [[nodiscard]] std::size_t activation_bytes() const noexcept {
    return activation_bytes_;
  }

 private:
  struct Buffer {
    std::string name;
    void* device{};
    std::size_t bytes{};
    bool input{};
  };

  void push() {
    require_driver(cuCtxPushCurrent(driver_context_),
                   "cuCtxPushCurrent(activation replica)");
  }
  void pop() {
    CUcontext popped = nullptr;
    require_driver(cuCtxPopCurrent(&popped),
                   "cuCtxPopCurrent(activation replica)");
    require(popped == driver_context_, "activation context pop differs");
  }
  void pop_noexcept() noexcept {
    CUcontext popped = nullptr;
    static_cast<void>(cuCtxPopCurrent(&popped));
  }
  void release_current() noexcept {
    for (auto& buffer : buffers_) {
      if (buffer.device != nullptr) {
        static_cast<void>(cudaFree(buffer.device));
      }
    }
    buffers_.clear();
    context_.reset();
    if (activation_ != nullptr) {
      static_cast<void>(cudaFree(activation_));
      activation_ = nullptr;
    }
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
    engine_.reset();
    runtime_.reset();
    CUcontext popped = nullptr;
    static_cast<void>(cuCtxPopCurrent(&popped));
    if (driver_context_ != nullptr) {
      static_cast<void>(cuCtxDestroy(driver_context_));
      driver_context_ = nullptr;
    }
  }
  void release() noexcept {
    if (driver_context_ == nullptr) {
      return;
    }
    if (cuCtxPushCurrent(driver_context_) == CUDA_SUCCESS) {
      release_current();
    } else {
      static_cast<void>(cuCtxDestroy(driver_context_));
      driver_context_ = nullptr;
    }
  }

  CUcontext driver_context_{};
  unsigned int requested_sms_{};
  unsigned int actual_sms_{};
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{};
  void* activation_{};
  std::size_t activation_bytes_{};
  std::vector<Buffer> buffers_;
};

}  // namespace

int main(const int argc, char** argv) {
  try {
#ifdef JDG_BLESS_SQUAD
    if (argc != 3 && argc != 6) {
      throw std::invalid_argument(
          "usage: bless-trt-squad-replica-smoke ENGINE TRACE.jsonl | "
          "ENGINE_2 ENGINE_4 ENGINE_6 ENGINE_8 TRACE.jsonl");
    }
#else
    if (argc != 2 && argc != 5) {
      throw std::invalid_argument(
          "usage: bless-trt-activation-replica-smoke ENGINE | "
          "ENGINE_2 ENGINE_4 ENGINE_6 ENGINE_8");
    }
#endif
    const bool affinity_engines =
#ifdef JDG_BLESS_SQUAD
        argc == 6;
#else
        argc == 5;
#endif
    std::vector<std::vector<char>> serialized;
    serialized.reserve(kAffinitySms.size());
    for (std::size_t index = 0; index < kAffinitySms.size(); ++index) {
      serialized.push_back(read_file(argv[affinity_engines ? index + 1U : 1U]));
    }
    require_driver(cuInit(0), "cuInit");
    CUdevice device{};
    require_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    Logger logger;
    std::vector<std::unique_ptr<Replica>> replicas;
    for (std::size_t index = 0; index < kAffinitySms.size(); ++index) {
      replicas.push_back(
          std::make_unique<Replica>(device, kAffinitySms[index],
                                    serialized[index], logger));
    }
    std::vector<std::uint64_t> output_checksums;
    for (auto& replica : replicas) {
      output_checksums.push_back(replica->infer());
    }
    require(std::all_of(output_checksums.begin(), output_checksums.end(),
                        [&](const std::uint64_t value) {
                          return value == output_checksums.front();
                        }),
            "TensorRT replica output checksums differ");
    require(replicas.front()->activation_bytes() ==
                replicas.back()->activation_bytes(),
            "TensorRT replica activation sizes differ");
    require_driver(cuCtxPushCurrent(replicas.back()->driver_context()),
                   "cuCtxPushCurrent(TensorRT activation copy)");
    const CUresult copy_result = cuMemcpyPeer(
        replicas.back()->activation(), replicas.back()->driver_context(),
        replicas.front()->activation(), replicas.front()->driver_context(),
        replicas.front()->activation_bytes());
    CUcontext copy_context = nullptr;
    require_driver(cuCtxPopCurrent(&copy_context),
                   "cuCtxPopCurrent(TensorRT activation copy)");
    require(copy_context == replicas.back()->driver_context(),
            "TensorRT activation copy context differs");
    require_driver(copy_result, "cuMemcpyPeer(TensorRT activation)");
    const auto source_activation = replicas.front()->activation_checksum();
    const auto destination_activation = replicas.back()->activation_checksum();
    require(source_activation == destination_activation,
            "TensorRT activation peer copy differs");
    const auto post_copy_output = replicas.back()->infer();
    require(post_copy_output == output_checksums.front(),
            "TensorRT output differs after activation peer copy");

#ifdef JDG_BLESS_SQUAD
    for (auto& replica : replicas) {
      replica->reset_state();
      require(bless_trt_squad_register_replica(
                  replica->driver_context(), replica->sms(),
                  replica->activation(), replica->activation_bytes()) == 0,
              "failed to register BLESS TensorRT replica");
    }
    require(bless_trt_squad_start(argv[affinity_engines ? 5 : 2]) == 0,
            "failed to start BLESS TensorRT squad scheduler");
    std::array<std::thread, kAffinitySms.size()> threads;
    std::array<std::exception_ptr, kAffinitySms.size()> failures{};
    for (std::size_t index = 0; index < replicas.size(); ++index) {
      threads[index] = std::thread([&, index] {
        try {
          replicas[index]->enqueue_squad();
        } catch (...) {
          failures[index] = std::current_exception();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    for (const auto& failure : failures) {
      if (failure != nullptr) {
        std::rethrow_exception(failure);
      }
    }
    require(bless_trt_squad_stop() == 0,
            "failed to stop BLESS TensorRT squad scheduler");
    bless::thor::SquadStats squad_stats{};
    require(bless_trt_squad_stats(&squad_stats) == 0,
            "failed to read BLESS TensorRT squad stats");
    require(squad_stats.logical_launches > 0U &&
                squad_stats.logical_launches == squad_stats.physical_launches &&
                squad_stats.shadow_launches ==
                    squad_stats.logical_launches * (replicas.size() - 1U) &&
                squad_stats.signature_mismatches == 0U,
            "BLESS TensorRT squad stats differ");
    std::uint64_t squad_output = 0U;
    bool selected_output_found = false;
    for (const auto& replica : replicas) {
      if (replica->sms() == squad_stats.last_selected_sms) {
        squad_output = replica->output_checksum();
        selected_output_found = true;
        break;
      }
    }
    require(selected_output_found, "BLESS selected replica is missing");
    const bool squad_output_matches = squad_output == output_checksums.front();
    const char* allow_mismatch = std::getenv("BLESS_TRT_ALLOW_MISMATCH");
    require(squad_output_matches ||
                (allow_mismatch != nullptr &&
                 std::string_view(allow_mismatch) == "1"),
            "BLESS selected-only TensorRT output differs");
#endif

    std::cout << "{\n  \"schema_version\":1,\n"
#ifdef JDG_BLESS_SQUAD
                 "  \"kind\":\"bless-thor-trt-squad-replica-smoke\",\n"
#else
                 "  \"kind\":\"bless-thor-trt-activation-replica-smoke\",\n"
#endif
                 "  \"affinity_domain_sms\":[2,4,6,8],\n"
              << "  \"activation_bytes\":"
              << replicas.front()->activation_bytes()
              << ",\n  \"output_checksums\":[";
    for (std::size_t index = 0; index < output_checksums.size(); ++index) {
      std::cout << (index == 0 ? "" : ",") << output_checksums[index];
    }
    std::cout << "],\n  \"activation_source_checksum\":"
              << source_activation
              << ",\n  \"activation_destination_checksum\":"
              << destination_activation
              << ",\n  \"post_copy_output_checksum\":"
              << post_copy_output
              << ",\n  \"restricted_to_unrestricted_copy\":true";
#ifdef JDG_BLESS_SQUAD
    std::cout << ",\n  \"logical_launches\":" << squad_stats.logical_launches
              << ",\n  \"physical_launches\":" << squad_stats.physical_launches
              << ",\n  \"shadow_launches\":" << squad_stats.shadow_launches
              << ",\n  \"restricted_launches\":"
              << squad_stats.restricted_launches
              << ",\n  \"unrestricted_launches\":"
              << squad_stats.unrestricted_launches
              << ",\n  \"activation_copies\":"
              << squad_stats.activation_copies
              << ",\n  \"signature_mismatches\":"
              << squad_stats.signature_mismatches
              << ",\n  \"last_selected_sms\":"
              << squad_stats.last_selected_sms
              << ",\n  \"selected_output_checksum\":" << squad_output
              << ",\n  \"selected_output_matches\":"
              << (squad_output_matches ? "true" : "false");
#endif
    std::cout << ",\n"
                 "  \"status\":\"passed\"\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
