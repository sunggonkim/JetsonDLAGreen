#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <sched.h>
#include <poll.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "jetson_dla_green/json.hpp"
#include "jetson_dla_green/stats.hpp"
#include "jetson_dla_green/trt_benchmark.hpp"

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;
volatile std::sig_atomic_t g_pause_requested = 0;

extern "C" void request_stop(int) { g_stop_requested = 1; }
extern "C" void request_pause(int) { g_pause_requested = 1; }

enum class Priority { kDefault, kHigh, kLow };
enum class Role { kBenchmark, kPressure };
enum class GateMode { kStop, kCooperative };

void honor_pause_request(const Role role) {
  if (role != Role::kPressure || g_pause_requested == 0) {
    return;
  }
  g_pause_requested = 0;
  if (raise(SIGSTOP) != 0) {
    throw std::runtime_error("pressure worker failed to acknowledge pause");
  }
}

void sleep_until_honoring_pause(
    const std::chrono::steady_clock::time_point deadline, const Role role) {
  using Clock = std::chrono::steady_clock;
  while (Clock::now() < deadline) {
    honor_pause_request(role);
    if (role == Role::kPressure && g_stop_requested != 0) {
      return;
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline - Clock::now());
    if (remaining.count() <= 0) {
      return;
    }
    timespec request{
        static_cast<time_t>(remaining.count() / 1'000'000'000LL),
        static_cast<long>(remaining.count() % 1'000'000'000LL),
    };
    timespec unused{};
    if (nanosleep(&request, &unused) != 0 && errno != EINTR) {
      throw std::runtime_error("nanosleep failed: " +
                               std::string(std::strerror(errno)));
    }
  }
  honor_pause_request(role);
}

struct Options {
  std::filesystem::path engine;
  std::filesystem::path trace;
  std::string model_name{"unnamed"};
  std::size_t samples{1000U};
  std::size_t warmup{100U};
  std::size_t burst_size{1U};
  double period_ms{};
  double deadline_ms{};
  double duration_seconds{};
  double guard_ms{};
  std::vector<pid_t> gate_pids;
  std::vector<pid_t> stop_pids;
  Priority priority{Priority::kDefault};
  Role role{Role::kBenchmark};
  GateMode gate_mode{GateMode::kStop};
  bool include_transfers{true};
  bool start_paused{};
  int dependency_wait_fd{-1};
  int dependency_signal_fd{-1};
  bool show_help{};
};

[[noreturn]] void throw_cuda_error(const cudaError_t result,
                                   const std::string_view operation) {
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorName(result) + " (" +
                           cudaGetErrorString(result) + ")");
}

void check_cuda(const cudaError_t result, const std::string_view operation) {
  if (result != cudaSuccess) {
    throw_cuda_error(result, operation);
  }
}

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

[[nodiscard]] std::uint64_t monotonic_now_ns() {
  timespec timestamp{};
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC): " +
                             std::string(std::strerror(errno)));
  }
  require(timestamp.tv_sec >= 0 && timestamp.tv_nsec >= 0 &&
              timestamp.tv_nsec < 1'000'000'000L,
          "CLOCK_MONOTONIC returned an invalid timestamp");
  constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
  const auto seconds = static_cast<std::uint64_t>(timestamp.tv_sec);
  const auto nanoseconds = static_cast<std::uint64_t>(timestamp.tv_nsec);
  require(seconds <=
              (std::numeric_limits<std::uint64_t>::max() - nanoseconds) /
                  kNanosecondsPerSecond,
          "CLOCK_MONOTONIC timestamp overflows uint64_t nanoseconds");
  return seconds * kNanosecondsPerSecond + nanoseconds;
}

template <typename T>
T parse_integer(const std::string_view text, const std::string_view option) {
  T value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(std::string(option) + " expects an integer");
  }
  return value;
}

double parse_double(const std::string_view text,
                    const std::string_view option) {
  std::string copy(text);
  std::size_t consumed = 0U;
  const double value = std::stod(copy, &consumed);
  if (consumed != copy.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(option) +
                                " expects a finite number");
  }
  return value;
}

std::vector<pid_t> parse_pids(const std::string_view text,
                              const std::string_view option) {
  if (text.empty() || text.front() == ',' || text.back() == ',') {
    throw std::invalid_argument(std::string(option) +
                                " contains an empty PID");
  }
  std::vector<pid_t> pids;
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    const std::string_view token = text.substr(begin, end - begin);
    const long value = parse_integer<long>(token, option);
    if (value <= 0L || value > std::numeric_limits<pid_t>::max() ||
        value == static_cast<long>(getpid())) {
      throw std::invalid_argument(std::string(option) +
                                  " contains an invalid PID");
    }
    pids.push_back(static_cast<pid_t>(value));
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1U;
  }
  if (pids.empty()) {
    throw std::invalid_argument(std::string(option) +
                                " requires at least one PID");
  }
  std::sort(pids.begin(), pids.end());
  if (std::adjacent_find(pids.begin(), pids.end()) != pids.end()) {
    throw std::invalid_argument(std::string(option) +
                                " must not contain duplicates");
  }
  return pids;
}

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      options.show_help = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(argument) + " requires a value");
    }
    const std::string_view value(argv[++index]);
    if (argument == "--engine") {
      options.engine = value;
    } else if (argument == "--trace") {
      options.trace = value;
    } else if (argument == "--model-name") {
      options.model_name = value;
    } else if (argument == "--samples") {
      options.samples = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--warmup") {
      options.warmup = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--burst-size") {
      options.burst_size = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--period-ms") {
      options.period_ms = parse_double(value, argument);
    } else if (argument == "--deadline-ms") {
      options.deadline_ms = parse_double(value, argument);
    } else if (argument == "--duration-seconds") {
      options.duration_seconds = parse_double(value, argument);
    } else if (argument == "--guard-ms") {
      options.guard_ms = parse_double(value, argument);
    } else if (argument == "--gate-pids") {
      options.gate_pids = parse_pids(value, argument);
    } else if (argument == "--stop-pids") {
      options.stop_pids = parse_pids(value, argument);
    } else if (argument == "--gate-mode") {
      if (value == "stop") {
        options.gate_mode = GateMode::kStop;
      } else if (value == "cooperative") {
        options.gate_mode = GateMode::kCooperative;
      } else {
        throw std::invalid_argument("--gate-mode expects stop or cooperative");
      }
    } else if (argument == "--role") {
      if (value == "benchmark") {
        options.role = Role::kBenchmark;
      } else if (value == "pressure") {
        options.role = Role::kPressure;
      } else {
        throw std::invalid_argument("--role expects benchmark or pressure");
      }
    } else if (argument == "--priority") {
      if (value == "default") {
        options.priority = Priority::kDefault;
      } else if (value == "high") {
        options.priority = Priority::kHigh;
      } else if (value == "low") {
        options.priority = Priority::kLow;
      } else {
        throw std::invalid_argument("--priority expects default, high, or low");
      }
    } else if (argument == "--include-transfers") {
      if (value == "true") {
        options.include_transfers = true;
      } else if (value == "false") {
        options.include_transfers = false;
      } else {
        throw std::invalid_argument("--include-transfers expects true or false");
      }
    } else if (argument == "--start-paused") {
      if (value == "true") {
        options.start_paused = true;
      } else if (value == "false") {
        options.start_paused = false;
      } else {
        throw std::invalid_argument("--start-paused expects true or false");
      }
    } else if (argument == "--dependency-wait-fd") {
      options.dependency_wait_fd = parse_integer<int>(value, argument);
    } else if (argument == "--dependency-signal-fd") {
      options.dependency_signal_fd = parse_integer<int>(value, argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (!options.show_help) {
    if (options.engine.empty()) {
      throw std::invalid_argument("--engine is required");
    }
    if (options.samples == 0U || options.burst_size == 0U ||
        options.period_ms < 0.0 ||
        options.deadline_ms < 0.0 || options.duration_seconds < 0.0 ||
        options.guard_ms < 0.0) {
      throw std::invalid_argument("sample counts and time values must be valid");
    }
    if (options.role == Role::kPressure && options.duration_seconds <= 0.0) {
      throw std::invalid_argument(
          "pressure role requires a positive --duration-seconds");
    }
    if ((options.dependency_wait_fd >= 0 ||
         options.dependency_signal_fd >= 0) &&
        options.role != Role::kPressure) {
      throw std::invalid_argument(
          "dependency pipes are supported only for pressure workers");
    }
    if (options.dependency_wait_fd == options.dependency_signal_fd &&
        options.dependency_wait_fd >= 0) {
      throw std::invalid_argument("dependency wait and signal fds must differ");
    }
    if (!options.gate_pids.empty() &&
        (options.role != Role::kBenchmark || options.period_ms <= 0.0 ||
         options.guard_ms <= 0.0 || options.guard_ms >= options.period_ms)) {
      throw std::invalid_argument(
          "gating requires benchmark role and 0 < guard-ms < period-ms");
    }
    if (!options.stop_pids.empty() && options.role != Role::kBenchmark) {
      throw std::invalid_argument("--stop-pids requires benchmark role");
    }
    if (options.burst_size > 1U &&
        (options.role != Role::kBenchmark || options.period_ms <= 0.0 ||
         options.samples % options.burst_size != 0U)) {
      throw std::invalid_argument(
          "burst-size requires periodic benchmark role and must divide samples");
    }
  }
  return options;
}

void print_help() {
  std::cout
      << "Usage: jdg-trt-bench --engine PATH [options]\n"
      << "  --model-name NAME\n"
      << "  --role benchmark|pressure\n"
      << "  --samples N                 Measured requests (default: 1000)\n"
      << "  --warmup N                  Warm-up requests (default: 100)\n"
      << "  --burst-size N              Requests released together (default: 1)\n"
      << "  --period-ms MS              Periodic release interval\n"
      << "  --deadline-ms MS            Release-to-completion deadline\n"
      << "  --duration-seconds S        Pressure-mode duration\n"
      << "  --gate-pids PID[,PID...]    Quiesce pressure processes per request\n"
      << "  --stop-pids PID[,PID...]    Stop pressure processes at window end\n"
      << "  --guard-ms MS               Drain time before each release\n"
      << "  --gate-mode MODE            stop or cooperative (default: stop)\n"
      << "  --priority default|high|low CUDA stream priority\n"
      << "  --include-transfers true|false\n"
      << "  --start-paused true|false  Stop after warm-up for an external barrier\n"
      << "  --dependency-wait-fd FD    Wait for one upstream completion token\n"
      << "  --dependency-signal-fd FD Emit one completion token downstream\n"
      << "  --trace PATH                Per-request CSV output\n";
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

class CudaStream {
 public:
  explicit CudaStream(const Priority priority) {
    int least = 0;
    int greatest = 0;
    check_cuda(cudaDeviceGetStreamPriorityRange(&least, &greatest),
               "cudaDeviceGetStreamPriorityRange");
    int selected = 0;
    if (priority == Priority::kHigh) {
      selected = greatest;
    } else if (priority == Priority::kLow) {
      selected = least;
    }
    check_cuda(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking,
                                            selected),
               "cudaStreamCreateWithPriority");
    priority_value_ = selected;
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  ~CudaStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] int priority_value() const noexcept { return priority_value_; }

 private:
  cudaStream_t stream_{};
  int priority_value_{};
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  ~CudaEvent() {
    if (event_ != nullptr) {
      static_cast<void>(cudaEventDestroy(event_));
    }
  }
  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_{};
};

class ProcessGate {
 public:
  ProcessGate(std::vector<pid_t> pids, const GateMode mode)
      : pids_(std::move(pids)), mode_(mode) {
    for (const pid_t pid : pids_) {
      if (kill(pid, 0) != 0) {
        throw std::runtime_error("cannot signal gate PID " +
                                 std::to_string(pid) + ": " +
                                 std::strerror(errno));
      }
    }
  }

  ProcessGate(const ProcessGate&) = delete;
  ProcessGate& operator=(const ProcessGate&) = delete;

  ~ProcessGate() { resume_best_effort(); }

  [[nodiscard]] bool enabled() const noexcept { return !pids_.empty(); }

  void pause(const std::chrono::steady_clock::time_point deadline) {
    const int pause_signal =
        mode_ == GateMode::kCooperative ? SIGUSR1 : SIGSTOP;
    std::size_t paused = 0U;
    for (; paused < pids_.size(); ++paused) {
      if (kill(pids_[paused], pause_signal) != 0) {
        for (std::size_t index = 0U; index < paused; ++index) {
          static_cast<void>(kill(pids_[index], SIGCONT));
        }
        throw std::runtime_error("failed to pause gate PID " +
                                 std::to_string(pids_[paused]) + ": " +
                                 std::strerror(errno));
      }
    }
    paused_ = true;
    if (mode_ == GateMode::kCooperative) {
      wait_for_drain(deadline);
    }
  }

  void resume() {
    for (const pid_t pid : pids_) {
      if (kill(pid, SIGCONT) != 0) {
        throw std::runtime_error("failed to resume gate PID " +
                                 std::to_string(pid) + ": " +
                                 std::strerror(errno));
      }
    }
    paused_ = false;
  }

 private:
  [[nodiscard]] static char process_state(const pid_t pid) {
    std::ifstream status("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(status, line)) {
      throw std::runtime_error("cannot read state for gate PID " +
                               std::to_string(pid));
    }
    const auto command_end = line.rfind(')');
    if (command_end == std::string::npos || command_end + 2U >= line.size()) {
      throw std::runtime_error("malformed state for gate PID " +
                               std::to_string(pid));
    }
    return line[command_end + 2U];
  }

  void wait_for_drain(
      const std::chrono::steady_clock::time_point deadline) {
    using Clock = std::chrono::steady_clock;
    while (true) {
      const bool drained = std::all_of(
          pids_.begin(), pids_.end(), [](const pid_t pid) {
            const char state = process_state(pid);
            return state == 'T' || state == 't';
          });
      if (drained) {
        return;
      }
      if (Clock::now() >= deadline) {
        resume_best_effort();
        paused_ = false;
        throw std::runtime_error(
            "cooperative gate did not drain before guard deadline");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  void resume_best_effort() noexcept {
    if (!paused_) {
      return;
    }
    for (const pid_t pid : pids_) {
      static_cast<void>(kill(pid, SIGCONT));
    }
  }

  std::vector<pid_t> pids_;
  GateMode mode_;
  bool paused_{};
};

class TensorBuffer {
 public:
  TensorBuffer(std::string name, const std::size_t bytes, const bool input)
      : name_(std::move(name)), bytes_(bytes), input_(input) {
    require(bytes_ > 0U, "TensorRT tensor has zero allocation size");
    check_cuda(cudaMalloc(&device_, bytes_), "cudaMalloc(TensorRT tensor)");
    try {
      check_cuda(cudaMallocHost(&host_, bytes_),
                 "cudaMallocHost(TensorRT tensor)");
      std::memset(host_, 0, bytes_);
      check_cuda(cudaMemset(device_, 0, bytes_),
                 "cudaMemset(TensorRT tensor)");
    } catch (...) {
      static_cast<void>(cudaFree(device_));
      device_ = nullptr;
      throw;
    }
  }

  TensorBuffer(const TensorBuffer&) = delete;
  TensorBuffer& operator=(const TensorBuffer&) = delete;
  TensorBuffer(TensorBuffer&&) = delete;
  TensorBuffer& operator=(TensorBuffer&&) = delete;

  ~TensorBuffer() {
    if (host_ != nullptr) {
      static_cast<void>(cudaFreeHost(host_));
    }
    if (device_ != nullptr) {
      static_cast<void>(cudaFree(device_));
    }
  }

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool input() const noexcept { return input_; }
  [[nodiscard]] void* device() const noexcept { return device_; }
  [[nodiscard]] void* host() const noexcept { return host_; }

 private:
  std::string name_;
  std::size_t bytes_{};
  bool input_{};
  void* device_{};
  void* host_{};
};

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open engine: " + path.string());
  }
  const auto end = input.tellg();
  if (end <= 0) {
    throw std::runtime_error("engine is empty: " + path.string());
  }
  const auto size = static_cast<std::size_t>(end);
  std::vector<char> contents(size);
  input.seekg(0, std::ios::beg);
  if (!input.read(contents.data(), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("failed to read engine: " + path.string());
  }
  return contents;
}

std::size_t data_type_size(const nvinfer1::DataType type) {
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
      throw std::runtime_error("packed 4-bit I/O tensors are unsupported");
  }
  throw std::runtime_error("unknown TensorRT data type");
}

std::size_t tensor_bytes(const nvinfer1::Dims& dimensions,
                         const nvinfer1::DataType type) {
  std::size_t elements = 1U;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    const auto dimension = dimensions.d[index];
    if (dimension <= 0) {
      throw std::runtime_error("unresolved TensorRT tensor dimension");
    }
    const auto unsigned_dimension = static_cast<std::size_t>(dimension);
    if (elements > std::numeric_limits<std::size_t>::max() /
                       unsigned_dimension) {
      throw std::overflow_error("TensorRT tensor element count overflow");
    }
    elements *= unsigned_dimension;
  }
  const auto bytes_per_element = data_type_size(type);
  if (elements >
      std::numeric_limits<std::size_t>::max() / bytes_per_element) {
    throw std::overflow_error("TensorRT tensor byte count overflow");
  }
  return elements * bytes_per_element;
}

bool has_dynamic_dimension(const nvinfer1::Dims& dimensions) {
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] < 0) {
      return true;
    }
  }
  return false;
}

class InferenceEngine {
 public:
  InferenceEngine(const Options& options, Logger& logger)
      : serialized_(read_file(options.engine)), stream_(options.priority) {
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    require(runtime_ != nullptr, "failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(serialized_.data(),
                                                  serialized_.size()));
    require(engine_ != nullptr, "failed to deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    require(context_ != nullptr, "failed to create TensorRT context");

    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char* name = engine_->getIOTensorName(index);
      require(name != nullptr, "TensorRT returned a null tensor name");
      const bool input = engine_->getTensorIOMode(name) ==
                         nvinfer1::TensorIOMode::kINPUT;
      if (input) {
        const auto engine_shape = engine_->getTensorShape(name);
        if (has_dynamic_dimension(engine_shape)) {
          const auto shape = engine_->getProfileShape(
              name, 0, nvinfer1::OptProfileSelector::kOPT);
          require(!has_dynamic_dimension(shape),
                  "engine optimization profile has unresolved dimensions");
          require(context_->setInputShape(name, shape),
                  "failed to set TensorRT input shape");
        }
      }
    }
    require(context_->inferShapes(0, nullptr) == 0,
            "not all TensorRT input shapes were specified");

    buffers_.reserve(static_cast<std::size_t>(engine_->getNbIOTensors()));
    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char* name = engine_->getIOTensorName(index);
      const bool input = engine_->getTensorIOMode(name) ==
                         nvinfer1::TensorIOMode::kINPUT;
      auto shape = context_->getTensorShape(name);
      std::size_t bytes = 0U;
      if (has_dynamic_dimension(shape) && !input) {
        const auto maximum = context_->getMaxOutputSize(name);
        require(maximum > 0, "TensorRT output size is unresolved");
        bytes = static_cast<std::size_t>(maximum);
      } else {
        bytes = tensor_bytes(shape, engine_->getTensorDataType(name));
      }
      buffers_.push_back(
          std::make_unique<TensorBuffer>(name, bytes, input));
      require(context_->setTensorAddress(name, buffers_.back()->device()),
              "failed to set TensorRT tensor address");
    }
  }

  [[nodiscard]] double infer(const bool include_transfers) {
    check_cuda(cudaEventRecord(start_.get(), stream_.get()),
               "cudaEventRecord(start)");
    if (include_transfers) {
      for (const auto& buffer : buffers_) {
        if (buffer->input()) {
          check_cuda(cudaMemcpyAsync(buffer->device(), buffer->host(),
                                     buffer->bytes(), cudaMemcpyHostToDevice,
                                     stream_.get()),
                     "cudaMemcpyAsync(input)");
        }
      }
    }
    require(context_->enqueueV3(stream_.get()), "TensorRT enqueueV3 failed");
    if (include_transfers) {
      for (const auto& buffer : buffers_) {
        if (!buffer->input()) {
          check_cuda(cudaMemcpyAsync(buffer->host(), buffer->device(),
                                     buffer->bytes(), cudaMemcpyDeviceToHost,
                                     stream_.get()),
                     "cudaMemcpyAsync(output)");
        }
      }
    }
    check_cuda(cudaEventRecord(stop_.get(), stream_.get()),
               "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_.get()), "cudaEventSynchronize");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start_.get(), stop_.get()),
               "cudaEventElapsedTime");
    return static_cast<double>(elapsed_ms);
  }

  [[nodiscard]] int stream_priority() const noexcept {
    return stream_.priority_value();
  }

 private:
  std::vector<char> serialized_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  CudaStream stream_;
  CudaEvent start_;
  CudaEvent stop_;
  std::vector<std::unique_ptr<TensorBuffer>> buffers_;
};

struct Measurements {
  std::vector<double> release_to_completion_ms;
  std::vector<double> gpu_service_ms;
  std::vector<double> queue_delay_ms;
  std::vector<double> gate_overhead_ms;
  std::vector<double> drain_ms;
  std::vector<double> resume_ms;
  std::size_t deadline_misses{};
  std::uint64_t measurement_start_monotonic_ns{};
  std::uint64_t measurement_end_monotonic_ns{};
  double elapsed_seconds{};
};

bool wait_dependency_token(const int fd) {
  if (fd < 0) {
    return true;
  }
  char token = 0;
  while (true) {
    honor_pause_request(Role::kPressure);
    if (g_stop_requested != 0) {
      return false;
    }
    pollfd descriptor{fd, POLLIN, 0};
    const int ready = poll(&descriptor, 1, 10);
    if (ready == 0) {
      continue;
    }
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("dependency wait poll failed");
    }
    if ((descriptor.revents & POLLIN) == 0) {
      if (g_stop_requested != 0) {
        return false;
      }
      throw std::runtime_error("dependency wait pipe closed unexpectedly");
    }
    const ssize_t count = read(fd, &token, sizeof(token));
    if (count == static_cast<ssize_t>(sizeof(token))) {
      return true;
    }
    if (count < 0 && errno == EINTR) {
      if (g_stop_requested != 0) {
        return false;
      }
      continue;
    }
    if (count == 0 && g_stop_requested != 0) {
      return false;
    }
    throw std::runtime_error("dependency wait pipe closed unexpectedly");
  }
}

void signal_dependency_token(const int fd) {
  if (fd < 0) {
    return;
  }
  const char token = 1;
  while (true) {
    const ssize_t count = write(fd, &token, sizeof(token));
    if (count == static_cast<ssize_t>(sizeof(token))) {
      return;
    }
    if (count < 0 && errno == EINTR) {
      if (g_stop_requested != 0) {
        return;
      }
      continue;
    }
    if (g_stop_requested != 0 && (errno == EPIPE || errno == EBADF)) {
      return;
    }
    throw std::runtime_error("dependency signal pipe failed");
  }
}

Measurements measure(const Options& options, InferenceEngine& engine) {
  for (std::size_t index = 0U; index < options.warmup; ++index) {
    static_cast<void>(engine.infer(options.include_transfers));
    honor_pause_request(options.role);
  }
  if (options.start_paused && raise(SIGSTOP) != 0) {
    throw std::runtime_error("failed to enter the start barrier");
  }

  Measurements measurements;
  measurements.release_to_completion_ms.reserve(options.samples);
  measurements.gpu_service_ms.reserve(options.samples);
  measurements.queue_delay_ms.reserve(options.samples);
  measurements.gate_overhead_ms.reserve(options.samples);
  measurements.drain_ms.reserve(options.samples);
  measurements.resume_ms.reserve(options.samples);
  using Clock = std::chrono::steady_clock;
  measurements.measurement_start_monotonic_ns = monotonic_now_ns();
  const auto measurement_start = Clock::now();
  const auto duration = std::chrono::duration<double>(options.duration_seconds);
  const auto stop = measurement_start +
                    std::chrono::duration_cast<Clock::duration>(duration);
  const auto period = std::chrono::duration<double, std::milli>(
      options.period_ms);
  ProcessGate gate(options.gate_pids, options.gate_mode);
  const auto schedule_start =
      measurement_start + std::chrono::duration_cast<Clock::duration>(period);

  for (std::size_t index = 0U;; ++index) {
    honor_pause_request(options.role);
    if (options.role == Role::kBenchmark && index >= options.samples) {
      break;
    }
    if (options.role == Role::kPressure && Clock::now() >= stop) {
      break;
    }
    if (options.role == Role::kPressure && g_stop_requested != 0) {
      break;
    }
    if (options.role == Role::kPressure &&
        !wait_dependency_token(options.dependency_wait_fd)) {
      break;
    }
    auto release = Clock::now();
    double gate_overhead_ms = 0.0;
    double drain_ms = 0.0;
    double resume_ms = 0.0;
    if (options.period_ms > 0.0) {
      const std::size_t burst_index = index / options.burst_size;
      const bool burst_start = index % options.burst_size == 0U;
      release = schedule_start + std::chrono::duration_cast<Clock::duration>(
                                     period * static_cast<double>(burst_index));
      if (gate.enabled() && burst_start) {
        const auto guard = std::chrono::duration<double, std::milli>(
            options.guard_ms);
        sleep_until_honoring_pause(
            release - std::chrono::duration_cast<Clock::duration>(guard),
            options.role);
        const auto gate_begin = Clock::now();
        gate.pause(release);
        const auto gate_end = Clock::now();
        drain_ms =
            std::chrono::duration<double, std::milli>(gate_end - gate_begin)
                .count();
        gate_overhead_ms += drain_ms;
      }
      if (burst_start) {
        sleep_until_honoring_pause(release, options.role);
      }
    }
    if (options.role == Role::kPressure && g_stop_requested != 0) {
      break;
    }
    const auto dispatch = Clock::now();
    const double gpu_ms = engine.infer(options.include_transfers);
    const auto completion = Clock::now();
    signal_dependency_token(options.dependency_signal_fd);
    const bool burst_end =
        (index + 1U) % options.burst_size == 0U || index + 1U == options.samples;
    if (gate.enabled() && burst_end) {
      const auto gate_begin = Clock::now();
      gate.resume();
      const auto gate_end = Clock::now();
      resume_ms =
          std::chrono::duration<double, std::milli>(gate_end - gate_begin)
              .count();
      gate_overhead_ms += resume_ms;
    }
    const double latency_ms =
        std::chrono::duration<double, std::milli>(completion - release).count();
    const double queue_ms = std::max(
        0.0, std::chrono::duration<double, std::milli>(dispatch - release)
                 .count());
    measurements.release_to_completion_ms.push_back(latency_ms);
    measurements.gpu_service_ms.push_back(gpu_ms);
    measurements.queue_delay_ms.push_back(queue_ms);
    measurements.gate_overhead_ms.push_back(gate_overhead_ms);
    measurements.drain_ms.push_back(drain_ms);
    measurements.resume_ms.push_back(resume_ms);
    if (options.deadline_ms > 0.0 && latency_ms > options.deadline_ms) {
      ++measurements.deadline_misses;
    }
    honor_pause_request(options.role);
  }
  measurements.measurement_end_monotonic_ns = monotonic_now_ns();
  for (const pid_t pid : options.stop_pids) {
    if (kill(pid, SIGINT) != 0 && errno != ESRCH) {
      throw std::runtime_error("failed to stop worker PID " +
                               std::to_string(pid) + ": " +
                               std::strerror(errno));
    }
  }
  require(measurements.measurement_end_monotonic_ns >=
              measurements.measurement_start_monotonic_ns,
          "CLOCK_MONOTONIC regressed during measurement");
  constexpr double kNanosecondsPerSecond = 1'000'000'000.0;
  measurements.elapsed_seconds =
      static_cast<double>(measurements.measurement_end_monotonic_ns -
                          measurements.measurement_start_monotonic_ns) /
      kNanosecondsPerSecond;
  return measurements;
}

void write_trace(const std::filesystem::path& path,
                 const Measurements& measurements) {
  if (path.empty()) {
    return;
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot create trace: " + path.string());
  }
  output << "request,release_to_completion_ms,gpu_service_ms,queue_delay_ms,"
            "gate_overhead_ms,drain_ms,resume_ms\n";
  output << std::setprecision(10);
  for (std::size_t index = 0U;
       index < measurements.release_to_completion_ms.size(); ++index) {
    output << index << ',' << measurements.release_to_completion_ms[index]
           << ',' << measurements.gpu_service_ms[index] << ','
           << measurements.queue_delay_ms[index] << ','
           << measurements.gate_overhead_ms[index] << ','
           << measurements.drain_ms[index] << ','
           << measurements.resume_ms[index] << '\n';
  }
}

void write_summary(std::ostream& output, const jdg::LatencySummary& summary) {
  output << "{\"count\":" << summary.count << ",\"mean_ms\":"
         << summary.mean << ",\"p50_ms\":" << summary.p50
         << ",\"p95_ms\":" << summary.p95 << ",\"p99_ms\":"
         << summary.p99 << ",\"p999_ms\":" << summary.p999
         << ",\"max_ms\":" << summary.maximum << '}';
}

std::string_view role_name(const Role role) {
  return role == Role::kPressure ? "pressure" : "benchmark";
}

std::string_view priority_name(const Priority priority) {
  if (priority == Priority::kHigh) {
    return "high";
  }
  if (priority == Priority::kLow) {
    return "low";
  }
  return "default";
}

std::string_view gate_mode_name(const GateMode mode) {
  return mode == GateMode::kCooperative ? "cooperative" : "stop";
}

[[nodiscard]] std::vector<int> process_cpu_affinity() {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
    throw std::runtime_error("sched_getaffinity: " +
                             std::string(std::strerror(errno)));
  }
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &affinity)) {
      cpus.push_back(cpu);
    }
  }
  require(!cpus.empty(), "process CPU affinity is empty");
  return cpus;
}

[[nodiscard]] int mps_active_thread_percentage() {
  const char* const raw = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
  if (raw == nullptr || raw[0] == '\0') {
    return 100;
  }
  const int percentage =
      parse_integer<int>(raw, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
  if (percentage <= 0 || percentage > 100) {
    throw std::runtime_error(
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE must be in [1, 100]");
  }
  return percentage;
}

void write_integer_array(std::ostream& output, const std::vector<int>& values) {
  output << '[';
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

int run(const Options& options, std::ostream& output) {
  if (options.role == Role::kPressure) {
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    std::signal(SIGUSR1, request_pause);
  }
  Logger logger;
  InferenceEngine engine(options, logger);
  const auto measurements = measure(options, engine);
  require(!measurements.release_to_completion_ms.empty(),
          "measurement produced no samples");
  write_trace(options.trace, measurements);
  const auto latency = jdg::summarize(measurements.release_to_completion_ms);
  const auto gpu = jdg::summarize(measurements.gpu_service_ms);
  const auto queue = jdg::summarize(measurements.queue_delay_ms);
  const auto gate = jdg::summarize(measurements.gate_overhead_ms);
  const auto drain = jdg::summarize(measurements.drain_ms);
  const auto resume = jdg::summarize(measurements.resume_ms);

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  const int mps_percentage = mps_active_thread_percentage();
  const std::vector<int> cpu_affinity = process_cpu_affinity();
  const char* const visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");

  output << std::setprecision(10) << "{\n  \"schema_version\": 1,\n"
            << "  \"model\": ";
  jdg::write_json_string(output, options.model_name);
  output << ",\n  \"role\": ";
  jdg::write_json_string(output, role_name(options.role));
  output << ",\n  \"engine\": ";
  jdg::write_json_string(output, options.engine.string());
  output << ",\n  \"execution_environment\": {\"pid\":" << getpid()
            << ",\"cuda_visible_devices\":";
  if (visible_devices == nullptr) {
    output << "null";
  } else {
    jdg::write_json_string(output, visible_devices);
  }
  output << ",\"mps_active_thread_percentage\":" << mps_percentage
            << ",\"cpu_affinity\":";
  write_integer_array(output, cpu_affinity);
  output << "},\n  \"gpu\": {\"name\": ";
  jdg::write_json_string(output, properties.name);
  output << ",\"multiprocessors\":" << properties.multiProcessorCount
            << "},\n  \"config\": {\"warmup\":" << options.warmup
            << ",\"burst_size\":" << options.burst_size
            << ",\"period_ms\":" << options.period_ms
            << ",\"deadline_ms\":" << options.deadline_ms
            << ",\"duration_seconds\":" << options.duration_seconds
            << ",\"guard_ms\":" << options.guard_ms
            << ",\"gated_processes\":" << options.gate_pids.size()
            << ",\"stopped_processes\":" << options.stop_pids.size()
            << ",\"gate_mode\":";
  jdg::write_json_string(output, gate_mode_name(options.gate_mode));
  output
            << ",\"start_paused\":"
            << (options.start_paused ? "true" : "false")
            << ",\"include_transfers\":"
            << (options.include_transfers ? "true" : "false")
            << ",\"priority\":";
  jdg::write_json_string(output, priority_name(options.priority));
  output << ",\"stream_priority_value\":" << engine.stream_priority()
            << ",\"dependency_wait_enabled\":"
            << (options.dependency_wait_fd >= 0 ? "true" : "false")
            << ",\"dependency_signal_enabled\":"
            << (options.dependency_signal_fd >= 0 ? "true" : "false")
            << "},\n  \"release_to_completion\": ";
  write_summary(output, latency);
  output << ",\n  \"gpu_service\": ";
  write_summary(output, gpu);
  output << ",\n  \"queue_delay\": ";
  write_summary(output, queue);
  output << ",\n  \"gate_overhead\": ";
  write_summary(output, gate);
  output << ",\n  \"drain\": ";
  write_summary(output, drain);
  output << ",\n  \"resume\": ";
  write_summary(output, resume);
  const double throughput =
      static_cast<double>(latency.count) / measurements.elapsed_seconds;
  output << ",\n  \"completed_requests\": " << latency.count
            << ",\n  \"throughput_per_second\": " << throughput
            << ",\n  \"measurement_start_monotonic_ns\": "
            << measurements.measurement_start_monotonic_ns
            << ",\n  \"measurement_end_monotonic_ns\": "
            << measurements.measurement_end_monotonic_ns
            << ",\n  \"elapsed_seconds\": " << measurements.elapsed_seconds
            << ",\n  \"deadline_misses\": "
            << measurements.deadline_misses
            << ",\n  \"deadline_miss_rate\": ";
  if (options.deadline_ms > 0.0) {
    output << static_cast<double>(measurements.deadline_misses) /
                     static_cast<double>(latency.count);
  } else {
    output << "null";
  }
  output << "\n}\n";
  return 0;
}

}  // namespace

int jdg::run_trt_benchmark(const int argc, char** argv, std::ostream& output,
                           std::ostream& error) {
  try {
    const auto options = parse_options(argc, argv);
    if (options.show_help) {
      print_help();
      return 0;
    }
    return run(options, output);
  } catch (const std::exception& exception) {
    error << "error: " << exception.what() << '\n';
    return 1;
  }
}
