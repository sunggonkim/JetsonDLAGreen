#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kSchemaVersion = 1U;
constexpr std::size_t kInputBytes = 80U * 3000U * sizeof(float);
constexpr std::size_t kActivationBytes = 1500U * 384U * sizeof(float);
constexpr std::size_t kMaxGeneratedTokens = 128U;
constexpr int kDecoderLayers = 4;
constexpr int kDecoderHeads = 6;
constexpr int kHeadDim = 64;
constexpr int kEncoderSequence = 1500;
constexpr int kMaxPastTokens = 224;
constexpr std::uint32_t kEndOfText = 50257U;

struct Options {
  std::filesystem::path encoder_engine;
  std::filesystem::path decoder_initial_engine;
  std::filesystem::path decoder_with_past_engine;
  std::filesystem::path input_trace;
  std::filesystem::path output_trace;
  std::filesystem::path trace_csv;
  std::string producer_uuid;
  std::string consumer_uuid;
  std::string mps_pipe;
  int warmup{2};
  int iterations{10};
  int max_tokens{64};
  double deadline_us{};
};

struct Ready {
  int role{};
  int status{};
};

struct Transfer {
  std::uint32_t iteration{};
  std::uint32_t warmup{};
  std::array<char, 65> input_sha256{};
  std::uint64_t arrival_ns{};
  std::uint64_t producer_start_ns{};
  std::uint64_t producer_done_ns{};
};

struct ConsumerResult {
  Transfer transfer{};
  std::uint64_t consumer_start_ns{};
  std::uint64_t consumer_done_ns{};
  std::uint32_t token_count{};
  std::array<std::uint32_t, kMaxGeneratedTokens> tokens{};
  int status{};
};

class Logger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char* const message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

template <typename T>
using TrtPtr = std::unique_ptr<T>;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    fail(std::string(message));
  }
}

void check_cuda(const cudaError_t status, const std::string_view operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorName(status) + " (" +
         cudaGetErrorString(status) + ")");
  }
}

[[nodiscard]] std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

[[nodiscard]] std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    fail("cannot open file: " + path.string());
  }
  const std::streampos end = input.tellg();
  require(end > 0, "file is empty");
  std::vector<char> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  require(static_cast<bool>(input.read(bytes.data(),
                                       static_cast<std::streamsize>(bytes.size()))),
          "file read failed");
  return bytes;
}

void write_all(const int fd, const void* data, std::size_t bytes) {
  const auto* cursor = static_cast<const std::byte*>(data);
  while (bytes != 0U) {
    const ssize_t written = write(fd, cursor, bytes);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail("pipe write failed");
    }
    require(written > 0, "pipe write made no progress");
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
}

bool read_all(const int fd, void* data, std::size_t bytes) {
  auto* cursor = static_cast<std::byte*>(data);
  while (bytes != 0U) {
    const ssize_t received = read(fd, cursor, bytes);
    if (received == 0) {
      return false;
    }
    if (received < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail("pipe read failed");
    }
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
  return true;
}

void close_fd(const int fd) {
  if (fd >= 0) {
    static_cast<void>(close(fd));
  }
}

void set_cuda_environment(const std::string& uuid, const std::string& pipe) {
  require(setenv("CUDA_VISIBLE_DEVICES", uuid.c_str(), 1) == 0,
          "setenv(CUDA_VISIBLE_DEVICES) failed");
  if (pipe.empty()) {
    static_cast<void>(unsetenv("CUDA_MPS_PIPE_DIRECTORY"));
    static_cast<void>(unsetenv("CUDA_MPS_LOG_DIRECTORY"));
  } else {
    require(setenv("CUDA_MPS_PIPE_DIRECTORY", pipe.c_str(), 1) == 0,
            "setenv(CUDA_MPS_PIPE_DIRECTORY) failed");
  }
  require(setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100", 1) == 0,
          "setenv(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) failed");
}

void initialize_cuda() {
  cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (status != cudaSuccess && status != cudaErrorSetOnActiveProcess) {
    check_cuda(status, "cudaSetDeviceFlags(cudaDeviceMapHost)");
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");
}

class RegisteredMapping {
 public:
  RegisteredMapping(void* const host, const std::size_t bytes)
      : host_(host), bytes_(bytes) {
    check_cuda(cudaHostRegister(host_, bytes_, cudaHostRegisterMapped),
               "cudaHostRegister(shared activation)");
    try {
      check_cuda(cudaHostGetDevicePointer(&device_, host_, 0),
                 "cudaHostGetDevicePointer(shared activation)");
    } catch (...) {
      static_cast<void>(cudaHostUnregister(host_));
      throw;
    }
  }

  RegisteredMapping(const RegisteredMapping&) = delete;
  RegisteredMapping& operator=(const RegisteredMapping&) = delete;

  ~RegisteredMapping() { static_cast<void>(cudaHostUnregister(host_)); }

  [[nodiscard]] void* device() const noexcept { return device_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* host_{};
  void* device_{};
  std::size_t bytes_{};
};

class InputTrace {
 public:
  explicit InputTrace(const std::filesystem::path& path) {
    const std::vector<char> raw = read_file(path);
    static constexpr std::array<char, 8> magic =
        {'J', 'D', 'G', 'I', 'N', 'T', '1', '\0'};
    constexpr std::size_t header_bytes = 8U + 4U + 4U + 8U;
    require(raw.size() >= header_bytes, "JDGINT1 trace is truncated");
    require(std::equal(magic.begin(), magic.end(), raw.begin()),
            "JDGINT1 trace magic differs");
    std::size_t offset = 8U;
    const auto read_u32 = [&]() {
      require(offset + sizeof(std::uint32_t) <= raw.size(),
              "JDGINT1 header is truncated");
      std::uint32_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    const auto read_u64 = [&]() {
      require(offset + sizeof(std::uint64_t) <= raw.size(),
              "JDGINT1 header is truncated");
      std::uint64_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    require(read_u32() == 1U, "JDGINT1 schema differs");
    const std::uint32_t count = read_u32();
    require(count > 0U, "JDGINT1 record count is invalid");
    sample_bytes_ = static_cast<std::size_t>(read_u64());
    require(sample_bytes_ == kInputBytes, "Whisper input trace size differs");
    samples_.resize(static_cast<std::size_t>(count) * sample_bytes_);
    hashes_.reserve(count);
    for (std::uint32_t expected = 0U; expected < count; ++expected) {
      require(offset + 4U + 64U + sample_bytes_ <= raw.size(),
              "JDGINT1 record is truncated");
      std::uint32_t iteration = 0U;
      std::memcpy(&iteration, raw.data() + offset, sizeof(iteration));
      offset += sizeof(iteration);
      require(iteration == expected, "JDGINT1 iterations are not dense");
      std::string digest(raw.data() + offset, 64U);
      offset += 64U;
      require(std::all_of(digest.begin(), digest.end(), [](const char value) {
                return (value >= '0' && value <= '9') ||
                       (value >= 'a' && value <= 'f');
              }),
              "JDGINT1 input hash is invalid");
      hashes_.push_back(std::move(digest));
      std::memcpy(samples_.data() + static_cast<std::size_t>(expected) * sample_bytes_,
                  raw.data() + offset, sample_bytes_);
      offset += sample_bytes_;
    }
    require(offset == raw.size(), "JDGINT1 has trailing bytes");
  }

  [[nodiscard]] std::size_t count() const noexcept { return hashes_.size(); }
  [[nodiscard]] const void* sample(const std::size_t iteration) const {
    require(iteration < count(), "JDGINT1 iteration is out of range");
    return samples_.data() + iteration * sample_bytes_;
  }
  [[nodiscard]] std::string_view hash(const std::size_t iteration) const {
    require(iteration < count(), "JDGINT1 hash is out of range");
    return hashes_[iteration];
  }

 private:
  std::size_t sample_bytes_{};
  std::vector<std::uint8_t> samples_;
  std::vector<std::string> hashes_;
};

[[nodiscard]] std::size_t type_bytes(const nvinfer1::DataType type) {
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
      fail("packed TensorRT data types are unsupported");
  }
  fail("unknown TensorRT data type");
}

[[nodiscard]] std::size_t tensor_bytes(const nvinfer1::Dims& dims,
                                       const nvinfer1::DataType type) {
  std::size_t elements = 1U;
  for (int index = 0; index < dims.nbDims; ++index) {
    require(dims.d[index] > 0, "TensorRT shape remains dynamic");
    const auto dimension = static_cast<std::size_t>(dims.d[index]);
    require(elements <= std::numeric_limits<std::size_t>::max() / dimension,
            "TensorRT tensor size overflows");
    elements *= dimension;
  }
  require(elements <= std::numeric_limits<std::size_t>::max() /
                          type_bytes(type),
          "TensorRT tensor byte size overflows");
  return elements * type_bytes(type);
}

class TensorRuntime {
 public:
  TensorRuntime(const std::filesystem::path& path, Logger& logger)
      : runtime_(nvinfer1::createInferRuntime(logger)) {
    require(runtime_ != nullptr, "failed to create TensorRT runtime");
    const std::vector<char> engine_bytes = read_file(path);
    engine_.reset(runtime_->deserializeCudaEngine(engine_bytes.data(),
                                                  engine_bytes.size()));
    require(engine_ != nullptr, "failed to deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    require(context_ != nullptr, "failed to create TensorRT context");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreate");
  }

  TensorRuntime(const TensorRuntime&) = delete;
  TensorRuntime& operator=(const TensorRuntime&) = delete;

  ~TensorRuntime() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream_));
    }
    for (void* allocation : allocations_) {
      static_cast<void>(cudaFree(allocation));
    }
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  [[nodiscard]] bool has(const std::string_view name) const {
    return engine_->getTensorIOMode(std::string(name).c_str()) !=
           nvinfer1::TensorIOMode::kNONE;
  }

  [[nodiscard]] nvinfer1::Dims shape(const std::string_view name) const {
    const auto value = context_->getTensorShape(std::string(name).c_str());
    require(value.nbDims > 0, "TensorRT tensor shape is invalid");
    return value;
  }

  [[nodiscard]] nvinfer1::DataType type(const std::string_view name) const {
    return engine_->getTensorDataType(std::string(name).c_str());
  }

  [[nodiscard]] std::size_t bytes(const std::string_view name) const {
    return tensor_bytes(shape(name), type(name));
  }

  void set_shape(const std::string_view name,
                 const std::initializer_list<int64_t> dimensions) {
    nvinfer1::Dims dims{};
    require(dimensions.size() <= nvinfer1::Dims::MAX_DIMS,
            "TensorRT dimensions exceed MAX_DIMS");
    dims.nbDims = static_cast<int>(dimensions.size());
    std::size_t index = 0U;
    for (const int64_t dimension : dimensions) {
      require(dimension > 0 && dimension <= std::numeric_limits<int>::max(),
              "TensorRT input dimension is invalid");
      dims.d[index++] = static_cast<int>(dimension);
    }
    require(context_->setInputShape(std::string(name).c_str(), dims),
            "failed to set TensorRT input shape");
  }

  void bind(const std::string_view name, void* const address) {
    require(address != nullptr, "TensorRT binding address is null");
    require(context_->setTensorAddress(std::string(name).c_str(), address),
            "failed to bind TensorRT tensor");
    bound_.push_back(std::string(name));
  }

  [[nodiscard]] void* allocate(const std::string_view name,
                                const std::size_t minimum_bytes = 0U) {
    const std::size_t requested = std::max(bytes(name), minimum_bytes);
    void* allocation = nullptr;
    check_cuda(cudaMalloc(&allocation, requested), "cudaMalloc(TensorRT tensor)");
    allocations_.push_back(allocation);
    bind(name, allocation);
    return allocation;
  }

  void copy_to(const std::string_view name, const void* const source,
               const std::size_t source_bytes) {
    require(source_bytes == bytes(name), "TensorRT input copy size differs");
    check_cuda(cudaMemcpyAsync(address(name), source, source_bytes,
                               cudaMemcpyHostToDevice, stream_),
               "cudaMemcpyAsync(TensorRT input)");
  }

  void copy_from(const std::string_view name, void* const destination,
                 const std::size_t destination_bytes) {
    require(destination_bytes == bytes(name), "TensorRT output copy size differs");
    check_cuda(cudaMemcpy(destination, address(name), destination_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(TensorRT output)");
  }

  void run() {
    require(context_->enqueueV3(stream_), "TensorRT enqueueV3 failed");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

  [[nodiscard]] void* address(const std::string_view name) const {
    const auto iterator = addresses_.find(std::string(name));
    require(iterator != addresses_.end(), "TensorRT address is not bound");
    return iterator->second;
  }

  void bind_recorded(const std::string_view name, void* const address) {
    bind(name, address);
    addresses_[std::string(name)] = address;
  }

  void allocate_recorded(const std::string_view name,
                         const std::size_t minimum_bytes = 0U) {
    void* const address = allocate(name, minimum_bytes);
    addresses_[std::string(name)] = address;
  }

  void validate_bindings() const {
    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char* const name = engine_->getIOTensorName(index);
      require(name != nullptr, "TensorRT I/O tensor name is null");
      require(addresses_.find(name) != addresses_.end(),
              std::string("TensorRT I/O tensor is unbound: ") + name);
    }
  }

 private:
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{};
  std::vector<void*> allocations_;
  std::vector<std::string> bound_;
  std::unordered_map<std::string, void*> addresses_;
};

class EncoderRunner {
 public:
  EncoderRunner(const std::filesystem::path& engine_path, Logger& logger,
                void* const activation_device)
      : runtime_(engine_path, logger), activation_device_(activation_device) {
    require(runtime_.type("input_features") == nvinfer1::DataType::kFLOAT,
            "Whisper input_features must be float32");
    require(runtime_.bytes("input_features") == kInputBytes,
            "Whisper input_features shape differs");
    require(runtime_.type("last_hidden_state") == nvinfer1::DataType::kFLOAT,
            "Whisper encoder output must be float32");
    require(runtime_.bytes("last_hidden_state") == kActivationBytes,
            "Whisper encoder output shape differs");
    runtime_.allocate_recorded("input_features");
    runtime_.bind_recorded("last_hidden_state", activation_device_);
    runtime_.validate_bindings();
  }

  void infer(const void* const input) {
    runtime_.copy_to("input_features", input, kInputBytes);
    runtime_.run();
  }

 private:
  TensorRuntime runtime_;
  void* activation_device_{};
};

struct KeyValueStorage {
  std::array<void*, kDecoderLayers> decoder_key{};
  std::array<void*, kDecoderLayers> decoder_value{};
};

class WhisperDecoderRunner {
 public:
  WhisperDecoderRunner(const std::filesystem::path& initial_path,
                       const std::filesystem::path& with_past_path,
                       Logger& logger, const int max_tokens)
      : initial_(initial_path, logger), with_past_(with_past_path, logger),
        max_tokens_(max_tokens) {
    require(max_tokens_ > 0 &&
                static_cast<std::size_t>(max_tokens_) <= kMaxGeneratedTokens,
            "Whisper max token count is invalid");
    allocate_key_values();
  }

  WhisperDecoderRunner(const WhisperDecoderRunner&) = delete;
  WhisperDecoderRunner& operator=(const WhisperDecoderRunner&) = delete;

  ~WhisperDecoderRunner() {
    const auto release = [](void* const address) {
      if (address != nullptr) {
        static_cast<void>(cudaFree(address));
      }
    };
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      release(current_.decoder_key[layer]);
      release(current_.decoder_value[layer]);
      release(next_.decoder_key[layer]);
      release(next_.decoder_value[layer]);
      release(encoder_key_[layer]);
      release(encoder_value_[layer]);
    }
  }

  std::vector<std::uint32_t> decode(void* const encoder_device) {
    bind_initial(encoder_device);
    static constexpr std::array<std::int64_t, 4> prompt =
        {50258, 50259, 50359, 50363};
    initial_.copy_to("input_ids", prompt.data(), sizeof(prompt));
    initial_.run();
    const std::uint32_t first = read_next_token(initial_, "logits");
    std::vector<std::uint32_t> tokens;
    tokens.reserve(static_cast<std::size_t>(max_tokens_));
    if (first != kEndOfText) {
      tokens.push_back(first);
    }
    std::size_t past_tokens = prompt.size();
    bind_with_past(past_tokens);
    std::int64_t token = static_cast<std::int64_t>(first);
    while (first != kEndOfText && tokens.size() < static_cast<std::size_t>(max_tokens_)) {
      const std::array<std::int64_t, 1> next = {token};
      with_past_.copy_to("input_ids", next.data(), sizeof(next));
      with_past_.run();
      const std::uint32_t generated = read_next_token(with_past_, "logits");
      if (generated == kEndOfText) {
        break;
      }
      tokens.push_back(generated);
      ++past_tokens;
      if (past_tokens >= static_cast<std::size_t>(kMaxPastTokens)) {
        break;
      }
      token = static_cast<std::int64_t>(generated);
      rebind_with_past(past_tokens);
    }
    return tokens;
  }

 private:
  void allocate_key_values() {
    const std::size_t decoder_bytes =
        static_cast<std::size_t>(1) * kDecoderHeads *
        static_cast<std::size_t>(kMaxPastTokens + 1) * kHeadDim * sizeof(float);
    const std::size_t encoder_bytes =
        static_cast<std::size_t>(1) * kDecoderHeads * kEncoderSequence *
        kHeadDim * sizeof(float);
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      check_cuda(cudaMalloc(&current_.decoder_key[layer], decoder_bytes),
                 "cudaMalloc(Whisper decoder key)");
      check_cuda(cudaMalloc(&current_.decoder_value[layer], decoder_bytes),
                 "cudaMalloc(Whisper decoder value)");
      check_cuda(cudaMalloc(&next_.decoder_key[layer], decoder_bytes),
                 "cudaMalloc(Whisper next decoder key)");
      check_cuda(cudaMalloc(&next_.decoder_value[layer], decoder_bytes),
                 "cudaMalloc(Whisper next decoder value)");
      check_cuda(cudaMalloc(&encoder_key_[layer], encoder_bytes),
                 "cudaMalloc(Whisper encoder key)");
      check_cuda(cudaMalloc(&encoder_value_[layer], encoder_bytes),
                 "cudaMalloc(Whisper encoder value)");
    }
  }

  void bind_initial(void* const encoder_device) {
    if (!initial_allocated_) {
      initial_.allocate_recorded("input_ids");
      initial_.allocate_recorded("logits");
      initial_allocated_ = true;
    }
    initial_.bind_recorded("encoder_hidden_states", encoder_device);
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      initial_.bind_recorded("present." + std::to_string(layer) + ".decoder.key",
                             current_.decoder_key[layer]);
      initial_.bind_recorded("present." + std::to_string(layer) + ".decoder.value",
                             current_.decoder_value[layer]);
      initial_.bind_recorded("present." + std::to_string(layer) + ".encoder.key",
                             encoder_key_[layer]);
      initial_.bind_recorded("present." + std::to_string(layer) + ".encoder.value",
                             encoder_value_[layer]);
    }
    if (!initial_validated_) {
      initial_.validate_bindings();
      initial_validated_ = true;
    }
  }

  void bind_with_past(const std::size_t past_tokens) {
    with_past_.set_shape("input_ids", {1, 1});
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      with_past_.set_shape(
          "past_key_values." + std::to_string(layer) + ".decoder.key",
          {1, kDecoderHeads, static_cast<int64_t>(past_tokens), kHeadDim});
      with_past_.set_shape(
          "past_key_values." + std::to_string(layer) + ".decoder.value",
          {1, kDecoderHeads, static_cast<int64_t>(past_tokens), kHeadDim});
    }
    if (!with_past_allocated_) {
      with_past_.allocate_recorded("input_ids");
      with_past_.allocate_recorded("logits");
      with_past_allocated_ = true;
    }
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      with_past_.bind_recorded(
          "past_key_values." + std::to_string(layer) + ".decoder.key",
          current_.decoder_key[layer]);
      with_past_.bind_recorded(
          "past_key_values." + std::to_string(layer) + ".decoder.value",
          current_.decoder_value[layer]);
      with_past_.bind_recorded(
          "past_key_values." + std::to_string(layer) + ".encoder.key",
          encoder_key_[layer]);
      with_past_.bind_recorded(
          "past_key_values." + std::to_string(layer) + ".encoder.value",
          encoder_value_[layer]);
    }
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      with_past_.bind_recorded(
          "present." + std::to_string(layer) + ".decoder.key",
          next_.decoder_key[layer]);
      with_past_.bind_recorded(
          "present." + std::to_string(layer) + ".decoder.value",
          next_.decoder_value[layer]);
    }
    if (!with_past_validated_) {
      with_past_.validate_bindings();
      with_past_validated_ = true;
    }
  }

  void rebind_with_past(const std::size_t past_tokens) {
    // The preceding enqueue wrote the present state into next_.  Rotate the
    // ping-pong buffers before rebinding so the next enqueue consumes that
    // state and writes into the now-free buffer.
    std::swap(current_, next_);
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      with_past_.set_shape(
          "past_key_values." + std::to_string(layer) + ".decoder.key",
          {1, kDecoderHeads, static_cast<int64_t>(past_tokens), kHeadDim});
      with_past_.set_shape(
          "past_key_values." + std::to_string(layer) + ".decoder.value",
          {1, kDecoderHeads, static_cast<int64_t>(past_tokens), kHeadDim});
      with_past_.bind_recorded(
          "past_key_values." + std::to_string(layer) + ".decoder.key",
          current_.decoder_key[layer]);
      with_past_.bind_recorded(
          "past_key_values." + std::to_string(layer) + ".decoder.value",
          current_.decoder_value[layer]);
      with_past_.bind_recorded(
          "present." + std::to_string(layer) + ".decoder.key",
          next_.decoder_key[layer]);
      with_past_.bind_recorded(
          "present." + std::to_string(layer) + ".decoder.value",
          next_.decoder_value[layer]);
    }
  }

  [[nodiscard]] static std::uint32_t read_next_token(TensorRuntime& runtime,
                                                       const std::string_view name) {
    const nvinfer1::Dims shape = runtime.shape(name);
    require(shape.nbDims == 3 && shape.d[0] == 1 && shape.d[1] > 0 &&
                shape.d[2] > 0,
            "Whisper logits shape differs");
    const std::size_t vocabulary = static_cast<std::size_t>(shape.d[2]);
    const std::size_t positions = static_cast<std::size_t>(shape.d[1]);
    std::vector<float> logits(runtime.bytes(name) / sizeof(float));
    require(logits.size() == positions * vocabulary,
            "Whisper logits byte size differs from shape");
    runtime.copy_from(name, logits.data(), logits.size() * sizeof(float));
    const auto last_position = logits.begin() +
                               static_cast<std::ptrdiff_t>((positions - 1U) * vocabulary);
    const auto last_end = last_position + static_cast<std::ptrdiff_t>(vocabulary);
    const std::vector<float> last_logits(last_position, last_end);
    require(std::all_of(last_logits.begin(), last_logits.end(), [](const float value) {
              return std::isfinite(value);
            }),
            "Whisper logits contain a non-finite value");
    return static_cast<std::uint32_t>(
        std::distance(last_logits.begin(),
                      std::max_element(last_logits.begin(), last_logits.end())));
  }

  TensorRuntime initial_;
  TensorRuntime with_past_;
  int max_tokens_{};
  KeyValueStorage current_{};
  KeyValueStorage next_{};
  std::array<void*, kDecoderLayers> encoder_key_{};
  std::array<void*, kDecoderLayers> encoder_value_{};
  bool initial_allocated_{};
  bool initial_validated_{};
  bool with_past_allocated_{};
  bool with_past_validated_{};
};

void write_asr_trace(const std::filesystem::path& path,
                    const std::vector<ConsumerResult>& results) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "failed to open ASR output trace");
  static constexpr char magic[] = "JDGASR1\0";
  output.write(magic, sizeof(magic) - 1U);
  const std::uint32_t schema = 1U;
  const std::uint32_t count = static_cast<std::uint32_t>(results.size());
  output.write(reinterpret_cast<const char*>(&schema), sizeof(schema));
  output.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const ConsumerResult& result : results) {
    output.write(reinterpret_cast<const char*>(&result.transfer.iteration),
                 sizeof(result.transfer.iteration));
    output.write(reinterpret_cast<const char*>(&result.token_count),
                 sizeof(result.token_count));
    output.write(reinterpret_cast<const char*>(result.tokens.data()),
                 static_cast<std::streamsize>(result.token_count * sizeof(std::uint32_t)));
  }
  require(output.good(), "failed to write ASR output trace");
}

void producer_main(const Options& options, void* const mapping,
                   Transfer* const metadata, const int ready_fd,
                   const int go_fd, const int transfer_fd, const int ack_fd) {
  try {
    set_cuda_environment(options.producer_uuid, options.mps_pipe);
    initialize_cuda();
    Ready ready{0, 0};
    write_all(ready_fd, &ready, sizeof(ready));
    char go = 0;
    if (!read_all(go_fd, &go, sizeof(go))) {
      _exit(3);
    }
    InputTrace input(options.input_trace);
    RegisteredMapping activation(mapping, kActivationBytes);
    Logger logger;
    EncoderRunner encoder(options.encoder_engine, logger, activation.device());
    const int total = options.warmup + options.iterations;
    require(input.count() == static_cast<std::size_t>(total),
            "input trace count differs from warmup plus iterations");
    for (int index = 0; index < total; ++index) {
      Transfer transfer{};
      transfer.iteration = static_cast<std::uint32_t>(index);
      transfer.warmup = static_cast<std::uint32_t>(index < options.warmup);
      std::copy(input.hash(index).begin(), input.hash(index).end(),
                transfer.input_sha256.begin());
      transfer.arrival_ns = monotonic_ns();
      transfer.producer_start_ns = monotonic_ns();
      encoder.infer(input.sample(index));
      transfer.producer_done_ns = monotonic_ns();
      metadata[index] = transfer;
      write_all(transfer_fd, &transfer, sizeof(transfer));
      char ack = 0;
      if (!read_all(ack_fd, &ack, sizeof(ack))) {
        _exit(4);
      }
    }
    _exit(0);
  } catch (const std::exception& error) {
    const Ready failure{0, 1};
    try {
      write_all(ready_fd, &failure, sizeof(failure));
    } catch (...) {
    }
    std::cerr << "whisper producer: " << error.what() << '\n';
    _exit(5);
  }
}

void consumer_main(const Options& options, void* const mapping,
                   const int ready_fd, const int go_fd, const int transfer_fd,
                   const int result_fd, const int ack_fd) {
  try {
    // The resident MPS daemon is intentionally scoped to the 1g producer.
    // The 2g consumer must use its direct MIG CUDA context; passing its UUID
    // to that daemon is rejected as an invisible device.
    set_cuda_environment(options.consumer_uuid, "");
    initialize_cuda();
    Ready ready{1, 0};
    write_all(ready_fd, &ready, sizeof(ready));
    char go = 0;
    if (!read_all(go_fd, &go, sizeof(go))) {
      _exit(3);
    }
    RegisteredMapping activation(mapping, kActivationBytes);
    Logger logger;
    WhisperDecoderRunner decoder(options.decoder_initial_engine,
                                 options.decoder_with_past_engine, logger,
                                 options.max_tokens);
    while (true) {
      Transfer transfer{};
      if (!read_all(transfer_fd, &transfer, sizeof(transfer))) {
        break;
      }
      ConsumerResult result{};
      result.transfer = transfer;
      result.consumer_start_ns = monotonic_ns();
      const std::vector<std::uint32_t> tokens = decoder.decode(activation.device());
      result.consumer_done_ns = monotonic_ns();
      require(tokens.size() <= result.tokens.size(), "Whisper token output is too long");
      result.token_count = static_cast<std::uint32_t>(tokens.size());
      std::copy(tokens.begin(), tokens.end(), result.tokens.begin());
      write_all(result_fd, &result, sizeof(result));
      const char ack = 1;
      write_all(ack_fd, &ack, sizeof(ack));
    }
    _exit(0);
  } catch (const std::exception& error) {
    const Ready failure{1, 1};
    try {
      write_all(ready_fd, &failure, sizeof(failure));
    } catch (...) {
    }
    std::cerr << "whisper consumer: " << error.what() << '\n';
    _exit(5);
  }
}

[[nodiscard]] int parse_int(const std::string& value, const std::string_view name,
                            const bool allow_zero = false) {
  std::size_t consumed = 0U;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed < (allow_zero ? 0 : 1) ||
      parsed > std::numeric_limits<int>::max()) {
    fail("invalid " + std::string(name) + ": " + value);
  }
  return static_cast<int>(parsed);
}

Options parse_options(const int argc, char** argv) {
  Options options;
  if (const char* value = std::getenv("JDG_MIG_SMALL_UUID")) {
    options.producer_uuid = value;
  }
  if (const char* value = std::getenv("JDG_MIG_BIG_UUID")) {
    options.consumer_uuid = value;
  }
  if (const char* value = std::getenv("JDG_MPS_PIPE_DIRECTORY")) {
    options.mps_pipe = value;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&]() -> std::string {
      if (++index >= argc) {
        fail("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--encoder-engine") {
      options.encoder_engine = next();
    } else if (argument == "--decoder-initial-engine") {
      options.decoder_initial_engine = next();
    } else if (argument == "--decoder-with-past-engine") {
      options.decoder_with_past_engine = next();
    } else if (argument == "--input-trace") {
      options.input_trace = next();
    } else if (argument == "--output-trace") {
      options.output_trace = next();
    } else if (argument == "--trace-csv") {
      options.trace_csv = next();
    } else if (argument == "--producer") {
      options.producer_uuid = next();
    } else if (argument == "--consumer") {
      options.consumer_uuid = next();
    } else if (argument == "--mps-pipe") {
      options.mps_pipe = next();
    } else if (argument == "--warmup") {
      options.warmup = parse_int(next(), "warmup", true);
    } else if (argument == "--iterations") {
      options.iterations = parse_int(next(), "iterations");
    } else if (argument == "--max-tokens") {
      options.max_tokens = parse_int(next(), "max-tokens");
    } else if (argument == "--deadline-us") {
      const std::string value = next();
      std::size_t consumed = 0U;
      options.deadline_us = std::stod(value, &consumed);
      require(consumed == value.size() && std::isfinite(options.deadline_us) &&
                  options.deadline_us > 0.0,
              "deadline-us must be positive and finite");
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "usage: jdg-mig-whisper-asr --encoder-engine PATH "
             "--decoder-initial-engine PATH --decoder-with-past-engine PATH "
             "--input-trace PATH --output-trace PATH --trace-csv PATH "
             "[--warmup N] [--iterations N] [--max-tokens N] "
             "[--deadline-us US] [--producer UUID] [--consumer UUID] "
             "[--mps-pipe PATH]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + argument);
    }
  }
  require(!options.encoder_engine.empty() &&
              !options.decoder_initial_engine.empty() &&
              !options.decoder_with_past_engine.empty() &&
              !options.input_trace.empty() && !options.output_trace.empty() &&
              !options.trace_csv.empty(),
          "engine, trace, and output paths are required");
  require(!options.producer_uuid.empty() && !options.consumer_uuid.empty(),
          "producer and consumer UUIDs are required");
  require(options.warmup + options.iterations > 0,
          "warmup plus iterations must be positive");
  require(options.max_tokens <= static_cast<int>(kMaxGeneratedTokens),
          "max-tokens exceeds output schema capacity");
  return options;
}

int wait_child(const pid_t child) {
  int status = 0;
  require(waitpid(child, &status, 0) == child, "waitpid failed");
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

void terminate_children(const pid_t producer, const pid_t consumer) {
  if (producer > 0) {
    static_cast<void>(kill(producer, SIGTERM));
  }
  if (consumer > 0) {
    static_cast<void>(kill(consumer, SIGTERM));
  }
  if (producer > 0) {
    static_cast<void>(waitpid(producer, nullptr, 0));
  }
  if (consumer > 0) {
    static_cast<void>(waitpid(consumer, nullptr, 0));
  }
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const InputTrace input(options.input_trace);
    const std::size_t total = static_cast<std::size_t>(options.warmup + options.iterations);
    require(input.count() == total, "input trace count differs from requested run");
    void* const mapping = mmap(nullptr, kActivationBytes, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    require(mapping != MAP_FAILED, "shared activation mmap failed");
    auto* const metadata = static_cast<Transfer*>(mmap(
        nullptr, total * sizeof(Transfer), PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    require(metadata != MAP_FAILED, "metadata mmap failed");
    std::fill_n(metadata, total, Transfer{});

    int ready[2]{}, producer_go[2]{}, consumer_go[2]{}, transfer[2]{},
        result[2]{}, ack[2]{};
    require(pipe(ready) == 0 && pipe(producer_go) == 0 && pipe(consumer_go) == 0 &&
                pipe(transfer) == 0 && pipe(result) == 0 && pipe(ack) == 0,
            "pipe creation failed");

    const pid_t producer = fork();
    require(producer >= 0, "producer fork failed");
    if (producer == 0) {
      close_fd(ready[0]); close_fd(producer_go[1]);
      close_fd(consumer_go[0]); close_fd(consumer_go[1]);
      close_fd(transfer[0]); close_fd(result[0]); close_fd(result[1]);
      close_fd(ack[1]);
      producer_main(options, mapping, metadata, ready[1], producer_go[0],
                    transfer[1], ack[0]);
    }
    const pid_t consumer = fork();
    require(consumer >= 0, "consumer fork failed");
    if (consumer == 0) {
      close_fd(ready[0]); close_fd(producer_go[0]); close_fd(producer_go[1]);
      close_fd(consumer_go[1]); close_fd(transfer[1]); close_fd(result[0]);
      close_fd(ack[0]);
      consumer_main(options, mapping, ready[1], consumer_go[0], transfer[0],
                    result[1], ack[1]);
    }
    if (consumer < 0) {
      terminate_children(producer, -1);
      fail("consumer fork failed");
    }

    close_fd(ready[1]); close_fd(producer_go[0]); close_fd(consumer_go[0]);
    close_fd(transfer[0]); close_fd(transfer[1]); close_fd(result[1]);
    close_fd(ack[0]); close_fd(ack[1]);
    Ready readiness[2]{};
    const bool readiness_received =
        read_all(ready[0], &readiness[0], sizeof(Ready)) &&
        read_all(ready[0], &readiness[1], sizeof(Ready));
    close_fd(ready[0]);
    if (!readiness_received) {
      terminate_children(producer, consumer);
      fail("child readiness failed");
    }
    std::sort(std::begin(readiness), std::end(readiness),
              [](const Ready& left, const Ready& right) {
                return left.role < right.role;
              });
    if (readiness[0].status != 0 || readiness[1].status != 0) {
      terminate_children(producer, consumer);
      fail("child CUDA readiness failed");
    }
    const char go = 1;
    write_all(producer_go[1], &go, sizeof(go));
    write_all(consumer_go[1], &go, sizeof(go));
    close_fd(producer_go[1]); close_fd(consumer_go[1]);

    std::vector<ConsumerResult> results;
    results.reserve(total);
    for (std::size_t index = 0U; index < total; ++index) {
      ConsumerResult item{};
      if (!read_all(result[0], &item, sizeof(item))) {
        break;
      }
      results.push_back(item);
    }
    close_fd(result[0]);
    const int producer_status = wait_child(producer);
    const int consumer_status = wait_child(consumer);
    static_cast<void>(munmap(mapping, kActivationBytes));
    static_cast<void>(munmap(metadata, total * sizeof(Transfer)));
    require(results.size() == total && producer_status == 0 && consumer_status == 0,
            "Whisper ASR child run did not complete all requests");
    std::sort(results.begin(), results.end(), [](const ConsumerResult& left,
                                                 const ConsumerResult& right) {
      return left.transfer.iteration < right.transfer.iteration;
    });

    std::ofstream trace(options.trace_csv, std::ios::out | std::ios::trunc);
    require(trace.is_open(), "failed to open ASR timing trace");
    trace << "request,input_sha256,producer_start_ns,producer_done_ns,"
             "consumer_start_ns,consumer_done_ns,wall_end_to_end_us,deadline_miss\n";
    std::size_t misses = 0U;
    std::vector<double> latency_us;
    for (const ConsumerResult& item : results) {
      require(item.consumer_done_ns >= item.transfer.arrival_ns,
              "ASR completion precedes arrival");
      const double wall = static_cast<double>(item.consumer_done_ns -
                                              item.transfer.arrival_ns) /
                          1000.0;
      const bool miss = options.deadline_us > 0.0 && wall > options.deadline_us;
      misses += miss ? 1U : 0U;
      if (item.transfer.warmup == 0U) {
        latency_us.push_back(wall);
      }
      trace << item.transfer.iteration << ',' << item.transfer.input_sha256.data()
            << ',' << item.transfer.producer_start_ns << ','
            << item.transfer.producer_done_ns << ',' << item.consumer_start_ns
            << ',' << item.consumer_done_ns << ',' << wall << ',' << (miss ? 1 : 0)
            << '\n';
    }
    trace.flush();
    require(trace.good(), "failed to write ASR timing trace");
    write_asr_trace(options.output_trace, results);
    std::sort(latency_us.begin(), latency_us.end());
    const auto percentile = [&latency_us](const double probability) {
      require(!latency_us.empty(), "ASR latency trace is empty");
      const double position = probability * static_cast<double>(latency_us.size() - 1U);
      const auto lower = static_cast<std::size_t>(position);
      const auto upper = std::min(lower + 1U, latency_us.size() - 1U);
      return latency_us[lower] + (latency_us[upper] - latency_us[lower]) *
                                     (position - static_cast<double>(lower));
    };
    std::cout << "{\"schema_version\":1,\"status\":\"ok\","
                 "\"task\":\"asr\","
                 "\"transport\":\"registered-shared-sysmem-direct-binding\","
                 "\"transport_description\":\"full-coherent registered system-memory activation edge\","
                 "\"producer_uuid\":\"" << options.producer_uuid
              << "\",\"consumer_uuid\":\"" << options.consumer_uuid
              << "\",\"payload_bytes\":" << kActivationBytes
              << ",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations
              << ",\"deadline_us\":" << options.deadline_us
              << ",\"deadline_misses\":" << misses
              << ",\"accuracy_validation_placement\":\"post-completion\""
                 ",\"application_output_trace\":\""
              << options.output_trace.string()
              << "\",\"input_trace\":\"" << options.input_trace.string()
              << "\",\"p50_us\":" << percentile(0.50)
              << ",\"p99_us\":" << percentile(0.99) << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema_version\":1,\"status\":\"error\",\"message\":\""
              << error.what() << "\"}\n";
    return 1;
  }
}
