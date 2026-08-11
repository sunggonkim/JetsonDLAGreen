#include <cuda_runtime.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "jetson_dla_green/json.hpp"
#include "jetson_dla_green/stats.hpp"

namespace {

constexpr int kThreadsPerBlock = 256;

enum class BackgroundMode { kNone, kCompute, kMemory };
enum class PriorityMode { kDefault, kHigh };
enum class Role { kBenchmark, kPressure };
enum class IsolationMode { kDefault, kGreen };

struct Options {
  int device{};
  std::size_t samples{500U};
  std::size_t warmup{50U};
  std::size_t critical_elements{1U << 20U};
  unsigned int critical_iterations{16U};
  std::size_t background_elements{1U << 24U};
  unsigned int background_iterations{};
  unsigned int green_sm_count{8U};
  double deadline_ms{};
  double duration_seconds{5.0};
  BackgroundMode background{BackgroundMode::kNone};
  PriorityMode priority{PriorityMode::kDefault};
  Role role{Role::kBenchmark};
  IsolationMode isolation{IsolationMode::kDefault};
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

class CudaStream {
 public:
  explicit CudaStream(const int priority,
                      const cudaExecutionContext_t context = nullptr) {
    if (context == nullptr) {
      check_cuda(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking,
                                              priority),
                 "cudaStreamCreateWithPriority");
    } else {
      check_cuda(cudaExecutionCtxStreamCreate(
                     &stream_, context, cudaStreamNonBlocking, priority),
                 "cudaExecutionCtxStreamCreate");
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  ~CudaStream() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_{};
};

class GreenPartitionPair {
 public:
  GreenPartitionPair(const int device, const unsigned int sm_count) {
    cudaDevResource available{};
    check_cuda(cudaDeviceGetDevResource(device, &available,
                                        cudaDevResourceTypeSm),
               "cudaDeviceGetDevResource");

    cudaDevResource partitions[2]{};
    cudaDevResource remaining{};
    cudaDevSmResourceGroupParams parameters[2]{};
    for (auto& parameter : parameters) {
      parameter.smCount = sm_count;
      parameter.coscheduledSmCount = available.sm.smCoscheduledAlignment;
    }
    check_cuda(cudaDevSmResourceSplit(partitions, 2U, &available, &remaining,
                                      0U, parameters),
               "cudaDevSmResourceSplit");

    try {
      for (std::size_t index = 0U; index < 2U; ++index) {
        cudaDevResourceDesc_t descriptor{};
        check_cuda(cudaDevResourceGenerateDesc(&descriptor, &partitions[index],
                                               1U),
                   "cudaDevResourceGenerateDesc");
        check_cuda(cudaGreenCtxCreate(&contexts_[index], descriptor, device,
                                      0U),
                   "cudaGreenCtxCreate");
        sm_counts_[index] = partitions[index].sm.smCount;
      }
    } catch (...) {
      destroy_noexcept();
      throw;
    }
  }

  GreenPartitionPair(const GreenPartitionPair&) = delete;
  GreenPartitionPair& operator=(const GreenPartitionPair&) = delete;

  ~GreenPartitionPair() { destroy_noexcept(); }

  [[nodiscard]] cudaExecutionContext_t context(
      const std::size_t index) const noexcept {
    return contexts_[index];
  }

  [[nodiscard]] unsigned int sm_count(
      const std::size_t index) const noexcept {
    return sm_counts_[index];
  }

 private:
  void destroy_noexcept() noexcept {
    for (auto& context : contexts_) {
      if (context != nullptr) {
        static_cast<void>(cudaExecutionCtxDestroy(context));
        context = nullptr;
      }
    }
  }

  cudaExecutionContext_t contexts_[2]{};
  unsigned int sm_counts_[2]{};
};

class CudaEvent {
 public:
  CudaEvent() {
    check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDefault),
               "cudaEventCreateWithFlags");
  }

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

template <typename Value>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t elements) : elements_(elements) {
    if (elements_ == 0U ||
        elements_ > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
      throw std::invalid_argument("invalid device buffer size");
    }
    check_cuda(cudaMalloc(&data_, elements_ * sizeof(Value)), "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  [[nodiscard]] Value* get() noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return elements_; }

 private:
  Value* data_{};
  std::size_t elements_{};
};

__global__ void critical_kernel(float* const data, const std::size_t elements,
                                const unsigned int iterations) {
  const std::size_t index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= elements) {
    return;
  }

  float value = data[index];
  for (unsigned int iteration = 0U; iteration < iterations; ++iteration) {
    value = fmaf(value, 1.00000011920928955078125F, 0.00000011920928955078125F);
  }
  data[index] = value;
}

__global__ void compute_pressure_kernel(float* const data,
                                        const std::size_t elements,
                                        const unsigned int iterations) {
  const std::size_t index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  float value = data[index % elements] + static_cast<float>(index & 0xffU);
  for (unsigned int iteration = 0U; iteration < iterations; ++iteration) {
    value = fmaf(value, 1.00000011920928955078125F, 0.000000059604644775390625F);
    value = fmaf(value, 0.999999940395355224609375F,
                 -0.0000000298023223876953125F);
  }
  data[index % elements] = value;
}

__global__ void memory_pressure_kernel(float* const data,
                                       const std::size_t elements,
                                       const unsigned int passes) {
  const std::size_t index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (unsigned int pass = 0U; pass < passes; ++pass) {
    for (std::size_t offset = index; offset < elements; offset += stride) {
      data[offset] = fmaf(data[offset], 1.00000011920928955078125F,
                          static_cast<float>(pass) * 0.000001F);
    }
  }
}

int launch_blocks(const std::size_t elements) {
  const std::size_t blocks =
      (elements + static_cast<std::size_t>(kThreadsPerBlock) - 1U) /
      static_cast<std::size_t>(kThreadsPerBlock);
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("element count exceeds CUDA grid limit");
  }
  return static_cast<int>(blocks);
}

std::string_view background_name(const BackgroundMode mode) {
  switch (mode) {
    case BackgroundMode::kNone:
      return "none";
    case BackgroundMode::kCompute:
      return "compute";
    case BackgroundMode::kMemory:
      return "memory";
  }
  return "unknown";
}

std::string_view priority_name(const PriorityMode mode) {
  return mode == PriorityMode::kHigh ? "high" : "default";
}

std::string_view role_name(const Role role) {
  return role == Role::kPressure ? "pressure" : "benchmark";
}

std::string_view isolation_name(const IsolationMode isolation) {
  return isolation == IsolationMode::kGreen ? "green" : "default";
}

template <typename Value>
Value parse_integer(const std::string_view text, const std::string_view option) {
  Value value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(std::string(option) + " expects an integer");
  }
  return value;
}

double parse_double(const std::string_view text, const std::string_view option) {
  double value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(value)) {
    throw std::invalid_argument(std::string(option) +
                                " expects a finite number");
  }
  return value;
}

Options parse_options(const int argc, char** const argv) {
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
    if (argument == "--device") {
      options.device = parse_integer<int>(value, argument);
    } else if (argument == "--samples") {
      options.samples = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--warmup") {
      options.warmup = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--critical-elements") {
      options.critical_elements = parse_integer<std::size_t>(value, argument);
    } else if (argument == "--critical-iterations") {
      options.critical_iterations =
          parse_integer<unsigned int>(value, argument);
    } else if (argument == "--background-elements") {
      options.background_elements =
          parse_integer<std::size_t>(value, argument);
    } else if (argument == "--background-iterations") {
      options.background_iterations =
          parse_integer<unsigned int>(value, argument);
    } else if (argument == "--green-sm-count") {
      options.green_sm_count = parse_integer<unsigned int>(value, argument);
    } else if (argument == "--deadline-ms") {
      options.deadline_ms = parse_double(value, argument);
    } else if (argument == "--duration-seconds") {
      options.duration_seconds = parse_double(value, argument);
    } else if (argument == "--role") {
      if (value == "benchmark") {
        options.role = Role::kBenchmark;
      } else if (value == "pressure") {
        options.role = Role::kPressure;
      } else {
        throw std::invalid_argument(
            "--role expects benchmark or pressure");
      }
    } else if (argument == "--isolation") {
      if (value == "default") {
        options.isolation = IsolationMode::kDefault;
      } else if (value == "green") {
        options.isolation = IsolationMode::kGreen;
      } else {
        throw std::invalid_argument("--isolation expects default or green");
      }
    } else if (argument == "--background") {
      if (value == "none") {
        options.background = BackgroundMode::kNone;
      } else if (value == "compute") {
        options.background = BackgroundMode::kCompute;
      } else if (value == "memory") {
        options.background = BackgroundMode::kMemory;
      } else {
        throw std::invalid_argument(
            "--background expects none, compute, or memory");
      }
    } else if (argument == "--critical-priority") {
      if (value == "default") {
        options.priority = PriorityMode::kDefault;
      } else if (value == "high") {
        options.priority = PriorityMode::kHigh;
      } else {
        throw std::invalid_argument(
            "--critical-priority expects default or high");
      }
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }

  if (options.samples == 0U || options.critical_elements == 0U ||
      options.critical_iterations == 0U ||
      options.background_elements == 0U || options.green_sm_count == 0U ||
      options.device < 0 ||
      options.deadline_ms < 0.0 || options.duration_seconds <= 0.0) {
    throw std::invalid_argument("benchmark sizes and counts must be positive");
  }
  if (options.role == Role::kPressure &&
      options.background == BackgroundMode::kNone) {
    throw std::invalid_argument(
        "--role pressure requires compute or memory background");
  }
  if (options.role == Role::kPressure &&
      options.isolation != IsolationMode::kDefault) {
    throw std::invalid_argument(
        "pressure role uses process-level isolation, not Green Contexts");
  }
  if (options.background_iterations == 0U) {
    options.background_iterations =
        options.background == BackgroundMode::kMemory ? 8U : 16384U;
  }
  return options;
}

void print_usage() {
  std::cout
      << "Usage: jdg-bench [options]\n"
      << "  --role benchmark|pressure\n"
      << "  --isolation default|green\n"
      << "  --background none|compute|memory\n"
      << "  --critical-priority default|high\n"
      << "  --samples N                 Measured requests (default: 500)\n"
      << "  --warmup N                  Warm-up requests (default: 50)\n"
      << "  --deadline-ms MS            Optional release-to-completion SLA\n"
      << "  --duration-seconds S        Pressure-role duration (default: 5)\n"
      << "  --device N                  CUDA device ordinal (default: 0)\n"
      << "  --critical-elements N       Critical working set in floats\n"
      << "  --critical-iterations N     Critical FMAs per element\n"
      << "  --background-elements N     Background working set in floats\n"
      << "  --background-iterations N   Compute iterations or memory passes\n"
      << "  --green-sm-count N          SMs per Green Context (default: 8)\n";
}

class BackgroundLoad {
 public:
  BackgroundLoad(const Options& options, const int multiprocessors,
                 const cudaExecutionContext_t context = nullptr)
      : mode_(options.background),
        device_(options.device),
        iterations_(options.background_iterations),
        compute_blocks_(multiprocessors * 8),
        memory_blocks_(multiprocessors * 8),
        buffer_(options.background_elements),
        stream_(0, context) {
    check_cuda(cudaMemsetAsync(buffer_.get(), 0,
                               buffer_.size() * sizeof(float), stream_.get()),
               "cudaMemsetAsync(background)");
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "cudaStreamSynchronize(background initialization)");
  }

  BackgroundLoad(const BackgroundLoad&) = delete;
  BackgroundLoad& operator=(const BackgroundLoad&) = delete;

  ~BackgroundLoad() { stop_noexcept(); }

  void start() {
    worker_ = std::thread([this] { run(); });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!launched_.load(std::memory_order_acquire) && !has_error()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        stop_noexcept();
        throw std::runtime_error("background CUDA launch timed out");
      }
      std::this_thread::yield();
    }
    rethrow_error();
  }

  void stop() {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
    }
    rethrow_error();
  }

  [[nodiscard]] std::uint64_t completed_launches() const noexcept {
    return completed_launches_.load(std::memory_order_acquire);
  }

 private:
  void launch_once() {
    if (mode_ == BackgroundMode::kCompute) {
      compute_pressure_kernel<<<compute_blocks_, kThreadsPerBlock, 0,
                                stream_.get()>>>(buffer_.get(), buffer_.size(),
                                                iterations_);
    } else {
      memory_pressure_kernel<<<memory_blocks_, kThreadsPerBlock, 0,
                               stream_.get()>>>(buffer_.get(), buffer_.size(),
                                               iterations_);
    }
    check_cuda(cudaGetLastError(), "background kernel launch");
  }

  void run() noexcept {
    try {
      check_cuda(cudaSetDevice(device_), "cudaSetDevice(background)");
      while (!stop_requested_.load(std::memory_order_acquire)) {
        launch_once();
        launched_.store(true, std::memory_order_release);
        check_cuda(cudaStreamSynchronize(stream_.get()),
                   "cudaStreamSynchronize(background)");
        completed_launches_.fetch_add(1U, std::memory_order_release);
      }
    } catch (...) {
      std::lock_guard lock(error_mutex_);
      error_ = std::current_exception();
    }
  }

  [[nodiscard]] bool has_error() {
    std::lock_guard lock(error_mutex_);
    return error_.has_value();
  }

  void rethrow_error() {
    std::lock_guard lock(error_mutex_);
    if (error_.has_value()) {
      std::rethrow_exception(*error_);
    }
  }

  void stop_noexcept() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  BackgroundMode mode_;
  int device_;
  unsigned int iterations_;
  int compute_blocks_;
  int memory_blocks_;
  DeviceBuffer<float> buffer_;
  CudaStream stream_;
  std::thread worker_;
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> launched_{false};
  std::atomic<std::uint64_t> completed_launches_{0U};
  std::mutex error_mutex_;
  std::optional<std::exception_ptr> error_;
};

int run_pressure(const Options& options) {
  check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, options.device),
             "cudaGetDeviceProperties");

  BackgroundLoad pressure(options, properties.multiProcessorCount);
  const auto start = std::chrono::steady_clock::now();
  pressure.start();
  std::this_thread::sleep_for(
      std::chrono::duration<double>(options.duration_seconds));
  pressure.stop();
  const auto stop = std::chrono::steady_clock::now();

  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  std::cout << std::setprecision(10) << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"role\": \"pressure\",\n"
            << "  \"gpu\": {\"device\":" << options.device
            << ",\"name\":";
  jdg::write_json_string(std::cout, properties.name);
  std::cout << ",\"multiprocessors\":" << properties.multiProcessorCount
            << ",\"cuda_runtime_version\":" << runtime_version
            << ",\"cuda_driver_version\":" << driver_version << "},\n"
            << "  \"background\":";
  jdg::write_json_string(std::cout, background_name(options.background));
  std::cout << ",\n  \"requested_duration_seconds\":"
            << options.duration_seconds
            << ",\n  \"actual_duration_seconds\":"
            << std::chrono::duration<double>(stop - start).count()
            << ",\n  \"completed_launches\":"
            << pressure.completed_launches() << "\n}\n";
  return EXIT_SUCCESS;
}

struct Measurement {
  double wall_ms{};
  double service_ms{};
};

Measurement measure_once(DeviceBuffer<float>& buffer, const unsigned int iterations,
                         const CudaStream& stream, const CudaEvent& start,
                         const CudaEvent& stop) {
  const auto release = std::chrono::steady_clock::now();
  check_cuda(cudaEventRecord(start.get(), stream.get()),
             "cudaEventRecord(start)");
  critical_kernel<<<launch_blocks(buffer.size()), kThreadsPerBlock, 0,
                    stream.get()>>>(buffer.get(), buffer.size(), iterations);
  check_cuda(cudaGetLastError(), "critical kernel launch");
  check_cuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");
  const auto completion = std::chrono::steady_clock::now();

  float service_ms = 0.0F;
  check_cuda(cudaEventElapsedTime(&service_ms, start.get(), stop.get()),
             "cudaEventElapsedTime");
  const auto wall_duration =
      std::chrono::duration<double, std::milli>(completion - release);
  return Measurement{.wall_ms = wall_duration.count(),
                     .service_ms = static_cast<double>(service_ms)};
}

void write_summary(const jdg::LatencySummary& summary) {
  std::cout << "{\"count\":" << summary.count << ",\"mean_ms\":"
            << summary.mean << ",\"p50_ms\":" << summary.p50
            << ",\"p95_ms\":" << summary.p95 << ",\"p99_ms\":"
            << summary.p99 << ",\"p999_ms\":" << summary.p999
            << ",\"max_ms\":" << summary.maximum << '}';
}

int run_benchmark(const Options& options) {
  check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, options.device),
             "cudaGetDeviceProperties");
  int least_priority = 0;
  int greatest_priority = 0;
  check_cuda(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
             "cudaDeviceGetStreamPriorityRange");
  const int critical_priority = options.priority == PriorityMode::kHigh
                                    ? greatest_priority
                                    : 0;

  std::unique_ptr<GreenPartitionPair> green_partitions;
  cudaExecutionContext_t critical_context = nullptr;
  cudaExecutionContext_t background_context = nullptr;
  int critical_multiprocessors = properties.multiProcessorCount;
  int background_multiprocessors = properties.multiProcessorCount;
  if (options.isolation == IsolationMode::kGreen) {
    green_partitions = std::make_unique<GreenPartitionPair>(
        options.device, options.green_sm_count);
    critical_context = green_partitions->context(0U);
    background_context = green_partitions->context(1U);
    critical_multiprocessors =
        static_cast<int>(green_partitions->sm_count(0U));
    background_multiprocessors =
        static_cast<int>(green_partitions->sm_count(1U));
  }

  DeviceBuffer<float> critical_buffer(options.critical_elements);
  CudaStream critical_stream(critical_priority, critical_context);
  CudaEvent start_event;
  CudaEvent stop_event;
  check_cuda(cudaMemsetAsync(critical_buffer.get(), 0,
                             critical_buffer.size() * sizeof(float),
                             critical_stream.get()),
             "cudaMemsetAsync(critical)");
  check_cuda(cudaStreamSynchronize(critical_stream.get()),
             "cudaStreamSynchronize(critical initialization)");

  std::unique_ptr<BackgroundLoad> background;
  if (options.background != BackgroundMode::kNone) {
    background = std::make_unique<BackgroundLoad>(
        options, background_multiprocessors, background_context);
    background->start();
  }

  for (std::size_t sample = 0; sample < options.warmup; ++sample) {
    static_cast<void>(measure_once(critical_buffer, options.critical_iterations,
                                   critical_stream, start_event, stop_event));
  }

  std::vector<double> wall_samples;
  std::vector<double> service_samples;
  wall_samples.reserve(options.samples);
  service_samples.reserve(options.samples);
  const auto benchmark_start = std::chrono::steady_clock::now();
  for (std::size_t sample = 0; sample < options.samples; ++sample) {
    const Measurement measurement =
        measure_once(critical_buffer, options.critical_iterations,
                     critical_stream, start_event, stop_event);
    wall_samples.push_back(measurement.wall_ms);
    service_samples.push_back(measurement.service_ms);
  }
  const auto benchmark_stop = std::chrono::steady_clock::now();

  std::uint64_t background_launches = 0U;
  if (background) {
    background->stop();
    background_launches = background->completed_launches();
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  const jdg::LatencySummary wall_summary = jdg::summarize(wall_samples);
  const jdg::LatencySummary service_summary = jdg::summarize(service_samples);
  const double duration_seconds =
      std::chrono::duration<double>(benchmark_stop - benchmark_start).count();
  const double request_rate =
      static_cast<double>(options.samples) / duration_seconds;

  std::size_t deadline_misses = 0U;
  if (options.deadline_ms > 0.0) {
    for (const double latency : wall_samples) {
      if (latency > options.deadline_ms) {
        ++deadline_misses;
      }
    }
  }

  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  std::cout << std::setprecision(10) << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"role\": \"benchmark\",\n"
            << "  \"gpu\": {\"device\":" << options.device
            << ",\"name\":";
  jdg::write_json_string(std::cout, properties.name);
  std::cout << ",\"multiprocessors\":" << properties.multiProcessorCount
            << ",\"critical_multiprocessors\":" << critical_multiprocessors
            << ",\"cuda_runtime_version\":" << runtime_version
            << ",\"cuda_driver_version\":" << driver_version << "},\n"
            << "  \"config\": {\"role\":";
  jdg::write_json_string(std::cout, role_name(options.role));
  std::cout << ",\"background\":";
  jdg::write_json_string(std::cout, background_name(options.background));
  std::cout << ",\"isolation\":";
  jdg::write_json_string(std::cout, isolation_name(options.isolation));
  std::cout << ",\"critical_priority\":";
  jdg::write_json_string(std::cout, priority_name(options.priority));
  std::cout << ",\"samples\":" << options.samples
            << ",\"warmup\":" << options.warmup
            << ",\"critical_elements\":" << options.critical_elements
            << ",\"critical_iterations\":" << options.critical_iterations
            << ",\"background_elements\":" << options.background_elements
            << ",\"background_iterations\":"
            << options.background_iterations << ",\"green_sm_count\":"
            << options.green_sm_count << ",\"deadline_ms\":";
  if (options.deadline_ms > 0.0) {
    std::cout << options.deadline_ms;
  } else {
    std::cout << "null";
  }
  std::cout << "},\n  \"release_to_completion\": ";
  write_summary(wall_summary);
  std::cout << ",\n  \"gpu_service\": ";
  write_summary(service_summary);
  std::cout << ",\n  \"throughput_requests_per_second\": " << request_rate
            << ",\n  \"background_completed_launches\": "
            << background_launches << ",\n  \"deadline_misses\": ";
  if (options.deadline_ms > 0.0) {
    std::cout << deadline_misses << ",\n  \"deadline_miss_rate\": "
              << static_cast<double>(deadline_misses) /
                     static_cast<double>(options.samples);
  } else {
    std::cout << "null,\n  \"deadline_miss_rate\": null";
  }
  std::cout << "\n}\n";
  return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char** const argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage();
      return EXIT_SUCCESS;
    }
    return options.role == Role::kPressure ? run_pressure(options)
                                           : run_benchmark(options);
  } catch (const std::exception& error) {
    std::cerr << "jdg-bench: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
