#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifdef JDG_WITH_ORION
#include "../baselines/orion/driver_capture/scheduler.hpp"
#else
namespace orion::thor {
struct SchedulerStats {
  std::uint64_t arrivals{};
  std::uint64_t decisions{};
  std::uint64_t reordered_decisions{};
  std::uint64_t high_priority_decisions{};
  std::uint64_t profiled_best_effort_admissions{};
  std::uint64_t complementary_admissions{};
  std::uint64_t profile_blocked_polls{};
  std::uint64_t trace_records{};
};
}  // namespace orion::thor
inline int orion_trt_scheduler_start_profiled(const char*, const char*,
                                               const char*, double) {
  return ENOTSUP;
}
inline int orion_trt_register_client(int, int) { return ENOTSUP; }
inline int orion_trt_scheduler_stats(orion::thor::SchedulerStats*) {
  return ENOTSUP;
}
inline int orion_trt_scheduler_stop() { return ENOTSUP; }
#endif

namespace {

constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::size_t kResnetPayloadBytes = 4U * 23U * 40U * sizeof(float);
constexpr std::size_t kResnetDetectionPayloadBytes =
    512U * 23U * 40U * sizeof(float);
constexpr std::size_t kResnet50ClassificationPayloadBytes =
    1024U * 14U * 14U * sizeof(float);
constexpr std::size_t kWhisperPayloadBytes = 1500U * 384U * sizeof(float);
constexpr int kTimeoutMs = 30'000;

enum class Transport { kRegisteredDirect, kPinnedBounce, kPageableBounce };
enum class GateMode { kCooperative, kStop };
enum class GateScope { kProducer, kConsumer, kPipeline };
enum class Workload {
  kResnetControl,
  kResnetDetectionHead,
  kResnet50Classification,
  kWhisperProjection,
};
enum class ChecksumMode { kInline, kSampled, kOff };
enum class DependencyMode { kDependent, kIndependent };

struct Options {
  std::filesystem::path producer_engine;
  std::filesystem::path consumer_engine;
  std::string consumer_input_tensor{"features"};
  std::string producer_uuid;
  std::string consumer_uuid;
  std::string producer_mps_pipe;
  std::string consumer_mps_pipe;
  int warmup{10};
  int iterations{100};
  Transport transport{Transport::kRegisteredDirect};
  std::vector<pid_t> gate_pids;
  double deadline_us{};
  GateMode gate_mode{GateMode::kCooperative};
  GateScope gate_scope{GateScope::kProducer};
  int producer_quota{100};
  int consumer_quota{100};
  Workload workload{Workload::kResnetControl};
  DependencyMode dependency_mode{DependencyMode::kDependent};
  bool validation_excluded_deadline{};
  ChecksumMode checksum_mode{ChecksumMode::kInline};
  int checksum_sample_period{10};
  std::filesystem::path trace_csv;
  std::filesystem::path event_trace_csv;
  std::filesystem::path checksum_trace_csv;
  // Optional post-completion raw TensorRT output capture for application
  // accuracy replay. It is deliberately outside the production wall.
  std::filesystem::path application_output_trace;
  // Optional fixed-size, preprocessed producer input trace.  The binary
  // JDGINT1 contract is generated from externally labelled tensor samples.
  std::filesystem::path producer_input_trace;
  // Optional operational JDGARR1 release schedule.  When present, the
  // producer consumes declared offsets directly instead of deriving arrivals
  // from the completion/ACK loop.
  std::filesystem::path arrival_trace;
  // Optional request-indexed producer activation replay.  JDGACT1 contains
  // the exact bytes published by a producer outside the measured interval.
  std::filesystem::path activation_replay_trace;
  // Capture mode is producer-only and exists to build a JDGACT1 trace before
  // running the dependent/independent causal pair.
  std::filesystem::path activation_capture_trace;
  bool orion_profile_aware{};
  std::filesystem::path orion_background_engine;
  std::filesystem::path orion_best_effort_profile;
  std::filesystem::path orion_high_priority_profile;
  std::filesystem::path orion_decision_trace;
  double orion_max_be_duration_us{1.0};
  std::string orion_trace_mode{"events"};
  double orion_background_period_us{4000.0};
  double validation_delay_us{};
};

struct StageTrace {
  std::uint32_t request{};
  std::string input_sha256;
  double producer_compute_us{};
  double producer_copy_us{};
  double producer_validation_us{};
  double notification_us{};
  double consumer_validation_us{};
  double consumer_copy_us{};
  double edge_transport_us{};
  double consumer_compute_us{};
  double output_verification_us{};
  double validation_excluded_end_to_end_us{};
  double wall_end_to_end_us{};
  bool deadline_miss{};
};

[[noreturn]] void fail(const std::string& message);

[[nodiscard]] std::size_t payload_bytes(const Workload workload) {
  switch (workload) {
    case Workload::kResnetControl:
      return kResnetPayloadBytes;
    case Workload::kResnetDetectionHead:
      return kResnetDetectionPayloadBytes;
    case Workload::kResnet50Classification:
      return kResnet50ClassificationPayloadBytes;
    case Workload::kWhisperProjection:
      return kWhisperPayloadBytes;
  }
  fail("unknown workload");
}

[[nodiscard]] std::string_view producer_output(const Workload workload) {
  switch (workload) {
    case Workload::kResnetControl:
      return "Layer7_cov";
    case Workload::kResnetDetectionHead:
      return "Layer6_relu_Y";
    case Workload::kResnet50Classification:
      return "gpu_0/res4_5_branch2c_bn_2";
    case Workload::kWhisperProjection:
      return "last_hidden_state";
  }
  fail("unknown workload");
}

[[nodiscard]] std::string_view pipeline_name(const Workload workload) {
  switch (workload) {
    case Workload::kResnetControl:
      return "resnet10-layer7-cov-to-control-mlp";
    case Workload::kResnetDetectionHead:
      return "resnet10-backbone-to-learned-detection-head";
    case Workload::kResnet50Classification:
      return "resnet50-backbone-to-classification-head";
    case Workload::kWhisperProjection:
      return "whisper-last-hidden-state-to-projection-mlp";
  }
  fail("unknown workload");
}

[[nodiscard]] bool checksum_enabled(const Options& options,
                                    const std::uint32_t iteration) {
  if (options.checksum_mode == ChecksumMode::kInline) {
    return true;
  }
  if (options.checksum_mode == ChecksumMode::kOff) {
    return false;
  }
  return iteration % static_cast<std::uint32_t>(options.checksum_sample_period) ==
         0U;
}

[[nodiscard]] std::string_view checksum_mode_name(const ChecksumMode mode) {
  switch (mode) {
    case ChecksumMode::kInline:
      return "inline";
    case ChecksumMode::kSampled:
      return "sampled";
    case ChecksumMode::kOff:
      return "off";
  }
  return "unknown";
}

[[nodiscard]] std::string_view dependency_mode_name(const DependencyMode mode) {
  return mode == DependencyMode::kDependent ? "dependent" : "independent";
}

struct Ready {
  int role{};
  int status{};
  int multiprocessors{};
  std::uint64_t payload_bytes{};
};

struct Transfer {
  std::uint32_t iteration{};
  std::uint32_t warmup{};
  std::array<char, 65> request_id{};
  std::uint64_t arrival_ns{};
  std::uint64_t declared_arrival_ns{};
  std::uint64_t actual_release_ns{};
  std::uint64_t queue_delay_ns{};
  std::uint64_t producer_start_ns{};
  std::uint64_t producer_compute_done_ns{};
  std::uint64_t producer_done_ns{};
  std::uint64_t producer_payload_verification_done_ns{};
  std::uint64_t producer_checksum{};
  std::uint64_t pause_begin_ns{};
  std::uint64_t pause_complete_ns{};
  std::uint64_t publication_ns{};
  std::uint64_t resume_issued_ns{};
  std::uint64_t resume_observed_ns{};
  std::uint64_t gate_begin_ns{};
  std::uint64_t gate_done_ns{};
  std::uint64_t resume_done_ns{};
  bool checksum_enabled{};
  std::array<char, 65> input_sha256{};
};

struct Result {
  Transfer transfer;
  std::uint64_t consumer_checksum{};
  std::uint64_t consumer_payload_verification_done_ns{};
  std::uint64_t consumer_output_checksum{};
  std::uint64_t consumer_start_ns{};
  std::uint64_t consumer_compute_start_ns{};
  std::uint64_t consumer_compute_done_ns{};
  std::uint64_t consumer_done_ns{};
  std::uint64_t consumer_pause_begin_ns{};
  std::uint64_t consumer_pause_complete_ns{};
  std::uint64_t consumer_resume_issued_ns{};
  std::uint64_t consumer_resume_observed_ns{};
  int status{};
};

struct OrionEvidence {
  std::uint64_t background_completed{};
  std::uint64_t measured_background_completed{};
  std::uint64_t measurement_start_ns{};
  std::uint64_t measurement_end_ns{};
  orion::thor::SchedulerStats scheduler;
  int status{};
};

class Logger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char* message) noexcept override {
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

[[nodiscard]] std::uint64_t checksum(const void* data, const std::size_t bytes) {
  constexpr std::uint64_t offset = 1469598103934665603ULL;
  constexpr std::uint64_t prime = 1099511628211ULL;
  const auto* input = static_cast<const std::uint8_t*>(data);
  std::uint64_t hash = offset;
  for (std::size_t index = 0; index < bytes; ++index) {
    hash = (hash ^ input[index]) * prime;
  }
  return hash;
}

void write_all(const int fd, const void* data, const std::size_t bytes) {
  const auto* cursor = static_cast<const std::byte*>(data);
  std::size_t remaining = bytes;
  while (remaining != 0U) {
    const ssize_t count = write(fd, cursor, remaining);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      fail("pipe write failed: " + std::string(std::strerror(errno)));
    }
    cursor += count;
    remaining -= static_cast<std::size_t>(count);
  }
}

[[nodiscard]] bool read_all(const int fd, void* data, const std::size_t bytes) {
  auto* cursor = static_cast<std::byte*>(data);
  std::size_t remaining = bytes;
  while (remaining != 0U) {
    pollfd descriptor{fd, POLLIN, 0};
    int polled = 0;
    do {
      polled = poll(&descriptor, 1, kTimeoutMs);
    } while (polled < 0 && errno == EINTR);
    if (polled == 0) {
      fail("pipe read timed out");
    }
    if (polled < 0) {
      fail("pipe poll failed: " + std::string(std::strerror(errno)));
    }
    const ssize_t count = read(fd, cursor, remaining);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count == 0) {
      return false;
    }
    if (count < 0) {
      fail("pipe read failed: " + std::string(std::strerror(errno)));
    }
    cursor += count;
    remaining -= static_cast<std::size_t>(count);
  }
  return true;
}

void close_fd(const int fd) {
  if (fd >= 0) {
    static_cast<void>(close(fd));
  }
}

void wait_until_ns(const std::uint64_t target_ns) {
  while (true) {
    const std::uint64_t now = monotonic_ns();
    if (now >= target_ns) {
      return;
    }
    const std::uint64_t remaining_ns = target_ns - now;
    const auto sleep_ns = std::min<std::uint64_t>(remaining_ns, 1'000'000U);
    std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ns));
  }
}

void correctness_delay(const double delay_us) {
  if (delay_us > 0.0) {
    std::this_thread::sleep_for(
        std::chrono::duration<double, std::micro>(delay_us));
  }
}

[[nodiscard]] char process_state(const pid_t pid) {
  std::ifstream input("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (!std::getline(input, line)) {
    fail("cannot read gate process state for PID " + std::to_string(pid));
  }
  const std::size_t end = line.rfind(')');
  require(end != std::string::npos && end + 2U < line.size(),
          "malformed gate process state");
  return line[end + 2U];
}

void pause_processes(const std::vector<pid_t>& pids, const GateMode mode) {
  for (const pid_t pid : pids) {
    const char state = process_state(pid);
    if (state == 'T' || state == 't') {
      continue;
    }
    const int signal_number =
        mode == GateMode::kCooperative ? SIGUSR1 : SIGSTOP;
    if (kill(pid, signal_number) != 0) {
      fail("failed to pause gate PID " + std::to_string(pid) + ": " +
           std::strerror(errno));
    }
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(100);
  while (true) {
    const bool stopped = std::all_of(pids.begin(), pids.end(), [](const pid_t pid) {
      const char state = process_state(pid);
      return state == 'T' || state == 't';
    });
    if (stopped) {
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      fail("gate processes did not stop within 100 ms");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void resume_processes(const std::vector<pid_t>& pids,
                      std::uint64_t& issued_ns,
                      std::uint64_t& observed_ns) {
  if (pids.empty()) {
    return;
  }
  issued_ns = monotonic_ns();
  for (const pid_t pid : pids) {
    if (kill(pid, SIGCONT) != 0) {
      fail("failed to resume gate PID " + std::to_string(pid) + ": " +
           std::strerror(errno));
    }
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(100);
  while (true) {
    const bool running = std::all_of(pids.begin(), pids.end(), [](const pid_t pid) {
      const char state = process_state(pid);
      return state != 'T' && state != 't';
    });
    if (running) {
      observed_ns = monotonic_ns();
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      fail("gate processes did not resume within 100 ms");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void set_cuda_environment(const std::string& uuid,
                          const std::string& mps_pipe, const int quota) {
  if (setenv("CUDA_VISIBLE_DEVICES", uuid.c_str(), 1) != 0) {
    fail("setenv(CUDA_VISIBLE_DEVICES) failed");
  }
  if (mps_pipe.empty()) {
    unsetenv("CUDA_MPS_PIPE_DIRECTORY");
    unsetenv("CUDA_MPS_LOG_DIRECTORY");
  } else if (setenv("CUDA_MPS_PIPE_DIRECTORY", mps_pipe.c_str(), 1) != 0) {
    fail("setenv(CUDA_MPS_PIPE_DIRECTORY) failed");
  }
  const std::string percentage = std::to_string(quota);
  if (setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", percentage.c_str(), 1) != 0) {
    fail("setenv(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) failed");
  }
}

[[nodiscard]] std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    fail("cannot open TensorRT engine: " + path.string());
  }
  const auto end = input.tellg();
  require(end > 0, "TensorRT engine is empty");
  std::vector<char> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    fail("failed to read TensorRT engine");
  }
  return bytes;
}

class ProducerInputTrace {
 public:
  explicit ProducerInputTrace(const std::filesystem::path& path) {
    const std::vector<char> raw = read_file(path);
    static constexpr std::array<char, 8> kMagic =
        {'J', 'D', 'G', 'I', 'N', 'T', '1', '\0'};
    constexpr std::size_t kHeaderBytes = 8U + 4U + 4U + 8U;
    require(raw.size() >= kHeaderBytes, "producer input trace is truncated");
    require(std::equal(kMagic.begin(), kMagic.end(), raw.begin()),
            "producer input trace magic differs");
    std::size_t offset = 8U;
    const auto read_u32 = [&]() {
      require(offset + sizeof(std::uint32_t) <= raw.size(),
              "producer input trace header is truncated");
      std::uint32_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    const auto read_u64 = [&]() {
      require(offset + sizeof(std::uint64_t) <= raw.size(),
              "producer input trace header is truncated");
      std::uint64_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    require(read_u32() == 1U, "producer input trace schema differs");
    const std::uint32_t count = read_u32();
    require(count > 0U && count <= 1'000'000U,
            "producer input trace record count is invalid");
    sample_bytes_ = static_cast<std::size_t>(read_u64());
    require(sample_bytes_ > 0U && sample_bytes_ <= (1ULL << 34U),
            "producer input trace sample size is invalid");
    samples_.reserve(count);
    hashes_.reserve(count);
    for (std::uint32_t expected = 0U; expected < count; ++expected) {
      require(offset + sizeof(std::uint32_t) + 64U + sample_bytes_ <= raw.size(),
              "producer input trace record is truncated");
      std::uint32_t iteration = 0U;
      std::memcpy(&iteration, raw.data() + offset, sizeof(iteration));
      offset += sizeof(iteration);
      require(iteration == expected,
              "producer input trace iterations must be dense and ordered");
      std::string digest(raw.data() + offset, 64U);
      offset += 64U;
      require(digest.size() == 64U &&
                  std::all_of(digest.begin(), digest.end(), [](const char value) {
                    return (value >= '0' && value <= '9') ||
                           (value >= 'a' && value <= 'f');
                  }),
              "producer input trace input_sha256 is invalid");
      hashes_.push_back(std::move(digest));
      samples_.emplace_back(
          reinterpret_cast<const std::uint8_t*>(raw.data() + offset),
          reinterpret_cast<const std::uint8_t*>(raw.data() + offset + sample_bytes_));
      offset += sample_bytes_;
    }
    require(offset == raw.size(), "producer input trace has trailing bytes");
  }

  [[nodiscard]] std::size_t count() const noexcept { return samples_.size(); }
  [[nodiscard]] std::size_t sample_bytes() const noexcept { return sample_bytes_; }
  [[nodiscard]] const std::uint8_t* sample(const std::size_t iteration) const {
    require(iteration < samples_.size(), "producer input trace iteration is out of range");
    return samples_[iteration].data();
  }
  [[nodiscard]] std::string_view input_sha256(const std::size_t iteration) const {
    require(iteration < hashes_.size(), "producer input trace hash is out of range");
    return hashes_[iteration];
  }

 private:
  std::size_t sample_bytes_{};
  std::vector<std::vector<std::uint8_t>> samples_;
  std::vector<std::string> hashes_;
};

class ActivationReplayTrace {
 public:
  explicit ActivationReplayTrace(const std::filesystem::path& path) {
    const std::vector<char> raw = read_file(path);
    static constexpr std::array<char, 8> kMagic =
        {'J', 'D', 'G', 'A', 'C', 'T', '1', '\0'};
    constexpr std::size_t kHeaderBytes = 8U + 4U + 4U + 8U;
    constexpr std::size_t kRecordPrefixBytes = 4U + 64U + 8U;
    require(raw.size() >= kHeaderBytes, "activation replay trace is truncated");
    require(std::equal(kMagic.begin(), kMagic.end(), raw.begin()),
            "activation replay trace magic differs");
    std::size_t offset = 8U;
    const auto read_u32 = [&]() {
      require(offset + sizeof(std::uint32_t) <= raw.size(),
              "activation replay trace header is truncated");
      std::uint32_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    const auto read_u64 = [&]() {
      require(offset + sizeof(std::uint64_t) <= raw.size(),
              "activation replay trace header is truncated");
      std::uint64_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    require(read_u32() == 1U, "activation replay trace schema differs");
    const std::uint32_t count = read_u32();
    require(count > 0U && count <= 1'000'000U,
            "activation replay trace record count is invalid");
    sample_bytes_ = static_cast<std::size_t>(read_u64());
    require(sample_bytes_ > 0U && sample_bytes_ <= (1ULL << 34U),
            "activation replay trace sample size is invalid");
    require(count <= (std::numeric_limits<std::size_t>::max() - offset) /
                         (kRecordPrefixBytes + sample_bytes_),
            "activation replay trace size overflows");
    samples_.resize(static_cast<std::size_t>(count) * sample_bytes_);
    hashes_.reserve(count);
    activation_checksums_.reserve(count);
    for (std::uint32_t expected = 0U; expected < count; ++expected) {
      require(offset + kRecordPrefixBytes + sample_bytes_ <= raw.size(),
              "activation replay trace record is truncated");
      std::uint32_t iteration = 0U;
      std::memcpy(&iteration, raw.data() + offset, sizeof(iteration));
      offset += sizeof(iteration);
      require(iteration == expected,
              "activation replay trace iterations must be dense and ordered");
      std::string input_sha(raw.data() + offset, 64U);
      offset += 64U;
      require(std::all_of(input_sha.begin(), input_sha.end(), [](const char value) {
                return (value >= '0' && value <= '9') ||
                       (value >= 'a' && value <= 'f');
              }),
              "activation replay trace input_sha256 is invalid");
      hashes_.push_back(std::move(input_sha));
      std::uint64_t activation_checksum = 0U;
      std::memcpy(&activation_checksum, raw.data() + offset,
                  sizeof(activation_checksum));
      offset += sizeof(activation_checksum);
      activation_checksums_.push_back(activation_checksum);
      auto* const destination = samples_.data() +
                                static_cast<std::size_t>(expected) * sample_bytes_;
      std::memcpy(destination, raw.data() + offset, sample_bytes_);
      require(checksum(destination, sample_bytes_) == activation_checksum,
              "activation replay trace checksum differs from payload");
      offset += sample_bytes_;
    }
    require(offset == raw.size(), "activation replay trace has trailing bytes");
  }

  [[nodiscard]] std::size_t count() const noexcept {
    return activation_checksums_.size();
  }
  [[nodiscard]] std::size_t sample_bytes() const noexcept {
    return sample_bytes_;
  }
  [[nodiscard]] std::size_t storage_bytes() const noexcept {
    return samples_.size();
  }
  [[nodiscard]] const std::uint8_t* data() const noexcept {
    return samples_.data();
  }
  [[nodiscard]] const std::uint8_t* sample(const std::size_t iteration) const {
    require(iteration < count(), "activation replay trace iteration is out of range");
    return samples_.data() + iteration * sample_bytes_;
  }
  [[nodiscard]] std::uint64_t activation_checksum(
      const std::size_t iteration) const {
    require(iteration < count(), "activation replay trace checksum is out of range");
    return activation_checksums_[iteration];
  }
  [[nodiscard]] std::string_view input_sha256(const std::size_t iteration) const {
    require(iteration < count(), "activation replay trace hash is out of range");
    return hashes_[iteration];
  }

 private:
  std::size_t sample_bytes_{};
  std::vector<std::uint8_t> samples_;
  std::vector<std::uint64_t> activation_checksums_;
  std::vector<std::string> hashes_;
};

class ActivationTraceWriter {
 public:
  ActivationTraceWriter(const std::filesystem::path& path,
                         const std::size_t count,
                         const std::size_t sample_bytes)
      : output_(path, std::ios::binary | std::ios::out | std::ios::trunc),
        expected_count_(count),
        sample_bytes_(sample_bytes) {
    require(output_.is_open(), "failed to open activation capture trace");
    require(count <= std::numeric_limits<std::uint32_t>::max(),
            "activation capture record count exceeds JDGACT1 schema");
    static constexpr char kMagic[] = "JDGACT1\0";
    output_.write(kMagic, sizeof(kMagic) - 1U);
    const std::uint32_t schema = 1U;
    const auto records = static_cast<std::uint32_t>(count);
    const auto bytes = static_cast<std::uint64_t>(sample_bytes);
    output_.write(reinterpret_cast<const char*>(&schema), sizeof(schema));
    output_.write(reinterpret_cast<const char*>(&records), sizeof(records));
    output_.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    require(output_.good(), "failed to write activation capture header");
  }

  void write(const std::uint32_t iteration, const std::string_view input_sha256,
             const void* const sample) {
    require(iteration == written_, "activation capture iterations are not dense");
    require(input_sha256.size() == 64U,
            "activation capture input_sha256 must be 64 characters");
    const auto activation_checksum = checksum(sample, sample_bytes_);
    output_.write(reinterpret_cast<const char*>(&iteration), sizeof(iteration));
    output_.write(input_sha256.data(), static_cast<std::streamsize>(input_sha256.size()));
    output_.write(reinterpret_cast<const char*>(&activation_checksum),
                  sizeof(activation_checksum));
    output_.write(static_cast<const char*>(sample),
                  static_cast<std::streamsize>(sample_bytes_));
    require(output_.good(), "failed to write activation capture record");
    ++written_;
  }

  void finish() {
    require(written_ == expected_count_, "activation capture record count differs");
    output_.flush();
    require(output_.good(), "failed to flush activation capture trace");
  }

 private:
  std::ofstream output_;
  std::size_t expected_count_{};
  std::size_t sample_bytes_{};
  std::size_t written_{};
};

class OperationalArrivalTrace {
 public:
  struct Record {
    std::uint32_t iteration{};
    std::uint32_t arrival_sequence{};
    std::uint64_t release_offset_ns{};
    std::string input_sha256;
    std::string request_id;
  };

  explicit OperationalArrivalTrace(const std::filesystem::path& path) {
    const std::vector<char> raw = read_file(path);
    static constexpr std::array<char, 8> kMagic =
        {'J', 'D', 'G', 'A', 'R', 'R', '1', '\0'};
    constexpr std::size_t kHeaderBytes = 8U + 4U + 4U + 8U;
    constexpr std::size_t kRecordBytes = 4U + 4U + 8U + 64U + 64U;
    require(raw.size() >= kHeaderBytes, "operational arrival trace is truncated");
    require(std::equal(kMagic.begin(), kMagic.end(), raw.begin()),
            "operational arrival trace magic differs");
    std::size_t offset = 8U;
    const auto read_u32 = [&]() {
      require(offset + sizeof(std::uint32_t) <= raw.size(),
              "operational arrival trace header is truncated");
      std::uint32_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    const auto read_u64 = [&]() {
      require(offset + sizeof(std::uint64_t) <= raw.size(),
              "operational arrival trace header is truncated");
      std::uint64_t value = 0U;
      std::memcpy(&value, raw.data() + offset, sizeof(value));
      offset += sizeof(value);
      return value;
    };
    require(read_u32() == 1U, "operational arrival trace schema differs");
    const std::uint32_t count = read_u32();
    require(count > 0U && count <= 1'000'000U,
            "operational arrival trace record count is invalid");
    require(read_u64() == kRecordBytes,
            "operational arrival trace record size differs");
    require(count <= (std::numeric_limits<std::size_t>::max() - offset) /
                         kRecordBytes,
            "operational arrival trace size overflows");
    records_.reserve(count);
    for (std::uint32_t expected = 0U; expected < count; ++expected) {
      require(offset + kRecordBytes <= raw.size(),
              "operational arrival trace record is truncated");
      Record record{};
      std::memcpy(&record.iteration, raw.data() + offset, sizeof(record.iteration));
      offset += sizeof(record.iteration);
      std::memcpy(&record.arrival_sequence, raw.data() + offset,
                  sizeof(record.arrival_sequence));
      offset += sizeof(record.arrival_sequence);
      std::memcpy(&record.release_offset_ns, raw.data() + offset,
                  sizeof(record.release_offset_ns));
      offset += sizeof(record.release_offset_ns);
      record.input_sha256.assign(raw.data() + offset, 64U);
      offset += 64U;
      record.request_id.assign(raw.data() + offset, 64U);
      offset += 64U;
      while (!record.request_id.empty() && record.request_id.back() == '\0') {
        record.request_id.pop_back();
      }
      require(record.arrival_sequence == expected,
              "operational arrival trace sequences must be dense and ordered");
      require(record.input_sha256.size() == 64U &&
                  std::all_of(record.input_sha256.begin(), record.input_sha256.end(),
                              [](const char value) {
                                return (value >= '0' && value <= '9') ||
                                       (value >= 'a' && value <= 'f');
                              }),
              "operational arrival trace input_sha256 is invalid");
      require(!record.request_id.empty() && record.request_id.size() <= 64U,
              "operational arrival trace request_id is invalid");
      records_.push_back(std::move(record));
    }
    require(offset == raw.size(), "operational arrival trace has trailing bytes");
  }

  [[nodiscard]] std::size_t count() const noexcept { return records_.size(); }

  [[nodiscard]] const Record& record(const std::size_t index) const {
    require(index < records_.size(), "operational arrival trace index is out of range");
    return records_[index];
  }

 private:
  std::vector<Record> records_;
};

[[nodiscard]] std::size_t data_type_bytes(const nvinfer1::DataType type) {
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
      fail("packed TensorRT I/O is unsupported");
  }
  fail("unknown TensorRT data type");
}

[[nodiscard]] std::size_t tensor_bytes(const nvinfer1::Dims& dims,
                                       const nvinfer1::DataType type) {
  std::size_t elements = 1U;
  for (int index = 0; index < dims.nbDims; ++index) {
    require(dims.d[index] > 0, "dynamic TensorRT I/O is unsupported here");
    const auto dimension = static_cast<std::size_t>(dims.d[index]);
    require(elements <= std::numeric_limits<std::size_t>::max() / dimension,
            "TensorRT tensor size overflow");
    elements *= dimension;
  }
  return elements * data_type_bytes(type);
}

class RegisteredMapping {
 public:
  RegisteredMapping(void* host, const std::size_t bytes,
                    const bool register_with_cuda,
                    const std::size_t registration_bytes = 0U)
      : host_(host),
        bytes_(bytes),
        registration_bytes_(registration_bytes == 0U ? bytes
                                                       : registration_bytes),
        registered_(register_with_cuda) {
    require(registration_bytes_ >= bytes_,
            "registered mapping range is smaller than its logical payload");
    if (!registered_) {
      return;
    }
    check_cuda(cudaHostRegister(host_, registration_bytes_, cudaHostRegisterMapped),
               "cudaHostRegister(shared payload)");
    try {
      check_cuda(cudaHostGetDevicePointer(&device_, host_, 0),
                 "cudaHostGetDevicePointer(shared payload)");
    } catch (...) {
      static_cast<void>(cudaHostUnregister(host_));
      throw;
    }
  }

  RegisteredMapping(const RegisteredMapping&) = delete;
  RegisteredMapping& operator=(const RegisteredMapping&) = delete;

  ~RegisteredMapping() {
    if (registered_) {
      static_cast<void>(cudaHostUnregister(host_));
    }
  }

  [[nodiscard]] void* host() const noexcept { return host_; }
  [[nodiscard]] void* device() const noexcept { return device_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  [[nodiscard]] void* device_at(const std::size_t offset) const {
    require(offset <= registration_bytes_ - bytes_,
            "mapped payload offset is outside the registered range");
    return static_cast<void*>(static_cast<std::byte*>(device_) + offset);
  }

 private:
  void* host_{};
  void* device_{};
  std::size_t bytes_{};
  std::size_t registration_bytes_{};
  bool registered_{};
};

class EngineRunner {
 public:
  EngineRunner(const void* serialized, const std::size_t serialized_bytes,
               Logger& logger, RegisteredMapping& payload,
               const std::string_view external_tensor,
               const bool direct_binding, const bool high_priority)
      : runtime_(nvinfer1::createInferRuntime(logger)) {
    require(runtime_ != nullptr, "failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(serialized, serialized_bytes));
    require(engine_ != nullptr, "failed to deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    require(context_ != nullptr, "failed to create TensorRT context");
    if (high_priority) {
      int least_priority = 0;
      int greatest_priority = 0;
      check_cuda(cudaDeviceGetStreamPriorityRange(&least_priority,
                                                   &greatest_priority),
                 "cudaDeviceGetStreamPriorityRange");
      check_cuda(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking,
                                               greatest_priority),
                 "cudaStreamCreateWithPriority");
    } else {
      check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                 "cudaStreamCreate");
    }
    try {
      bool found_external = false;
      for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
        const char* const name = engine_->getIOTensorName(index);
        require(name != nullptr, "TensorRT returned null I/O tensor name");
        const std::size_t bytes = tensor_bytes(
            context_->getTensorShape(name), engine_->getTensorDataType(name));
        if (external_tensor == name && direct_binding) {
          require(bytes == payload.bytes(),
                  "external TensorRT tensor size differs from shared payload");
          require(payload.device() != nullptr,
                  "direct binding requires registered mapped memory");
          handoff_tensor_name_ = name;
          handoff_bytes_ = bytes;
          direct_handoff_ = true;
          bind_direct_handoff(payload.device(), payload.bytes());
          found_external = true;
          continue;
        }
        void* allocation = nullptr;
        check_cuda(cudaMalloc(&allocation, bytes), "cudaMalloc(TensorRT I/O)");
        allocations_.push_back(allocation);
        if (external_tensor == name) {
          require(bytes == payload.bytes(),
                  "handoff TensorRT tensor size differs from shared payload");
          handoff_allocation_ = allocation;
          handoff_bytes_ = bytes;
          handoff_is_input_ =
              engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
          found_external = true;
        }
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          input_allocations_.push_back({allocation, bytes});
        } else {
          output_allocations_.push_back({allocation, bytes});
        }
        check_cuda(cudaMemset(allocation, 0, bytes), "cudaMemset(TensorRT I/O)");
        require(context_->setTensorAddress(name, allocation),
                "failed to bind TensorRT tensor");
      }
      require(external_tensor.empty() || found_external,
              "requested external TensorRT tensor was not found");
    } catch (...) {
      release();
      throw;
    }
  }

  EngineRunner(const EngineRunner&) = delete;
  EngineRunner& operator=(const EngineRunner&) = delete;

  ~EngineRunner() { release(); }

  [[nodiscard]] std::size_t input_bytes() const {
    std::size_t total = 0U;
    for (const auto& [allocation, bytes] : input_allocations_) {
      static_cast<void>(allocation);
      require(total <= std::numeric_limits<std::size_t>::max() - bytes,
              "TensorRT input byte count overflow");
      total += bytes;
    }
    return total;
  }

  void infer(const std::uint32_t iteration, const bool vary_inputs,
             const ProducerInputTrace* input_trace = nullptr) {
    if (input_trace != nullptr) {
      require(input_trace->sample_bytes() == input_bytes(),
              "producer input trace bytes differ from TensorRT input tensors");
      std::size_t offset = 0U;
      const auto* sample = input_trace->sample(iteration);
      for (const auto& [allocation, bytes] : input_allocations_) {
        check_cuda(cudaMemcpyAsync(allocation, sample + offset, bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync(producer input trace)");
        offset += bytes;
      }
    } else if (vary_inputs) {
      constexpr std::uint8_t patterns[] = {0x00U, 0x3eU, 0x3fU, 0x40U};
      const int pattern = patterns[iteration % std::size(patterns)];
      for (const auto& [allocation, bytes] : input_allocations_) {
        check_cuda(cudaMemsetAsync(allocation, pattern, bytes, stream_),
                   "cudaMemsetAsync(request input)");
      }
    }
    require(context_->enqueueV3(stream_), "TensorRT enqueueV3 failed");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

  void bind_direct_handoff(void* const device, const std::size_t bytes) {
    require(direct_handoff_ && !handoff_tensor_name_.empty(),
            "direct handoff binding is unavailable");
    require(bytes == handoff_bytes(), "direct handoff binding size differs");
    require(context_->setTensorAddress(handoff_tensor_name_.c_str(), device),
            "failed to bind replayed activation tensor");
  }

  [[nodiscard]] std::uint64_t output_checksum() const {
    require(!output_allocations_.empty(),
            "TensorRT runner has no locally bound output");
    std::uint64_t combined = 1469598103934665603ULL;
    for (const auto& [allocation, bytes] : output_allocations_) {
      std::vector<std::uint8_t> host(bytes);
      check_cuda(cudaMemcpy(host.data(), allocation, bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(policy output)");
      combined ^= checksum(host.data(), host.size());
      combined *= 1099511628211ULL;
    }
    return combined;
  }

  void write_output_trace_header(std::ofstream& output) const {
    static constexpr char magic[] = "JDGOUT1\0";
    output.write(magic, sizeof(magic) - 1U);
    const std::uint32_t count = static_cast<std::uint32_t>(output_allocations_.size());
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [allocation, bytes] : output_allocations_) {
      static_cast<void>(allocation);
      const std::uint64_t size = static_cast<std::uint64_t>(bytes);
      output.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    require(output.good(), "failed to write application output trace header");
  }

  void write_output_trace_record(std::ofstream& output,
                                 const std::uint32_t iteration) const {
    output.write(reinterpret_cast<const char*>(&iteration), sizeof(iteration));
    for (const auto& [allocation, bytes] : output_allocations_) {
      std::vector<std::uint8_t> host(bytes);
      check_cuda(cudaMemcpy(host.data(), allocation, bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(application output trace)");
      output.write(reinterpret_cast<const char*>(host.data()),
                   static_cast<std::streamsize>(host.size()));
    }
    require(output.good(), "failed to write application output trace record");
  }

  void copy_handoff_to_host(void* host, const std::size_t bytes) const {
    require(handoff_allocation_ != nullptr && !handoff_is_input_,
            "runner has no local handoff output");
    require(bytes == handoff_bytes(), "handoff output size differs");
    check_cuda(cudaMemcpy(host, handoff_allocation_, bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(dependency D2H)");
  }

  void copy_handoff_from_host(const void* host, const std::size_t bytes) const {
    require(handoff_allocation_ != nullptr && handoff_is_input_,
            "runner has no local handoff input");
    require(bytes == handoff_bytes(), "handoff input size differs");
    check_cuda(cudaMemcpy(handoff_allocation_, host, bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(dependency H2D)");
  }

 private:
  void release() noexcept {
    for (void* allocation : allocations_) {
      static_cast<void>(cudaFree(allocation));
    }
    allocations_.clear();
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
  }

  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{};
  std::vector<void*> allocations_;
  std::vector<std::pair<void*, std::size_t>> input_allocations_;
  std::vector<std::pair<void*, std::size_t>> output_allocations_;
  void* handoff_allocation_{};
  bool handoff_is_input_{};
  std::string handoff_tensor_name_;
  std::size_t handoff_bytes_{};
  bool direct_handoff_{};

  [[nodiscard]] std::size_t handoff_bytes() const {
    if (direct_handoff_) {
      require(handoff_bytes_ > 0U, "direct handoff size is unavailable");
      return handoff_bytes_;
    }
    const auto found = std::find_if(
        allocations_.begin(), allocations_.end(),
        [this](const void* allocation) { return allocation == handoff_allocation_; });
    require(found != allocations_.end(), "handoff allocation is not owned");
    for (const auto& [allocation, bytes] : input_allocations_) {
      if (allocation == handoff_allocation_) {
        return bytes;
      }
    }
    for (const auto& [allocation, bytes] : output_allocations_) {
      if (allocation == handoff_allocation_) {
        return bytes;
      }
    }
    fail("handoff allocation size is unavailable");
  }
};

class OrionBackground {
 public:
  OrionBackground(const Options& options, Logger& logger,
                  RegisteredMapping& payload) : enabled_(options.orion_profile_aware) {
    if (!enabled_) {
      return;
    }
    require(setenv("ORION_TRT_TRACE_MODE", options.orion_trace_mode.c_str(), 1) == 0,
            "failed to configure Orion trace mode");
    const int started = orion_trt_scheduler_start_profiled(
        options.orion_decision_trace.c_str(),
        options.orion_best_effort_profile.c_str(),
        options.orion_high_priority_profile.c_str(),
        options.orion_max_be_duration_us);
    require(started == 0, "failed to start profile-aware Orion scheduler");
    scheduler_started_ = true;
    const int registered = orion_trt_register_client(1, 1);
    require(registered == 0, "failed to register Orion HP producer");

    std::vector<char> engine = read_file(options.orion_background_engine);
    worker_ = std::thread(
        [this, engine = std::move(engine), &logger, &payload,
         period_us = options.orion_background_period_us]() mutable {
          try {
            require(orion_trt_register_client(0, 0) == 0,
                    "failed to register Orion BE client");
            EngineRunner runner(engine.data(), engine.size(), logger, payload,
                                "", false, false);
            {
              std::lock_guard<std::mutex> lock(mutex_);
              ready_ = true;
            }
            ready_condition_.notify_one();
            {
              std::unique_lock<std::mutex> lock(mutex_);
              start_condition_.wait(lock, [this] {
                return started_ || stop_.load(std::memory_order_acquire);
              });
            }
            if (stop_.load(std::memory_order_acquire)) {
              return;
            }
            std::uint32_t iteration = 0U;
            auto next_release = std::chrono::steady_clock::now();
            while (!stop_.load(std::memory_order_acquire)) {
              runner.infer(iteration++, true);
              background_completed_.fetch_add(1U, std::memory_order_relaxed);
              next_release += std::chrono::duration_cast<
                  std::chrono::steady_clock::duration>(
                  std::chrono::duration<double, std::micro>(period_us));
              std::this_thread::sleep_until(next_release);
            }
          } catch (const std::exception& error) {
            {
              std::lock_guard<std::mutex> lock(mutex_);
              error_ = error.what();
              ready_ = true;
            }
            ready_condition_.notify_one();
          }
        });
    std::unique_lock<std::mutex> lock(mutex_);
    if (!ready_condition_.wait_for(lock, std::chrono::seconds(10),
                                   [this] { return ready_; })) {
      lock.unlock();
      shutdown(false);
      fail("Orion BE client readiness timed out");
    }
    if (!error_.empty()) {
      const std::string error = error_;
      lock.unlock();
      shutdown(false);
      fail("Orion BE client failed during readiness: " + error);
    }
  }

  OrionBackground(const OrionBackground&) = delete;
  OrionBackground& operator=(const OrionBackground&) = delete;

  ~OrionBackground() { shutdown(false); }

  void start_work() {
    if (!enabled_) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      started_ = true;
    }
    start_condition_.notify_one();
  }

  [[nodiscard]] std::uint64_t completed() const noexcept {
    return background_completed_.load(std::memory_order_relaxed);
  }

  void finish(OrionEvidence& evidence) {
    if (!enabled_) {
      evidence = {};
      return;
    }
    shutdown(true);
    evidence.background_completed =
        background_completed_.load(std::memory_order_relaxed);
    evidence.scheduler = stats_;
    evidence.status = error_.empty() ? 0 : 1;
    require(error_.empty(), "Orion BE client failed: " + error_);
    require(evidence.background_completed > 0U,
            "Orion BE client completed no inference");
  }

 private:
  void shutdown(const bool strict) {
    if (!enabled_ || finished_) {
      return;
    }
    stop_.store(true, std::memory_order_release);
    start_condition_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
    int stats_status = 0;
    int stop_status = 0;
    if (scheduler_started_) {
      stats_status = orion_trt_scheduler_stats(&stats_);
      stop_status = orion_trt_scheduler_stop();
      scheduler_started_ = false;
    }
    finished_ = true;
    if (strict) {
      require(stats_status == 0 && stop_status == 0,
              "failed to finalize profile-aware Orion scheduler");
    }
  }

  bool enabled_{};
  bool scheduler_started_{};
  bool finished_{};
  bool ready_{};
  bool started_{};
  std::atomic<bool> stop_{false};
  std::atomic<std::uint64_t> background_completed_{0U};
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable ready_condition_;
  std::condition_variable start_condition_;
  std::string error_;
  orion::thor::SchedulerStats stats_;
};

[[nodiscard]] bool direct_binding(const Transport transport) {
  return transport == Transport::kRegisteredDirect;
}

[[nodiscard]] bool registered_mapping(const Transport transport) {
  return transport != Transport::kPageableBounce;
}

[[nodiscard]] std::string_view transport_name(const Transport transport) {
  switch (transport) {
    case Transport::kRegisteredDirect:
      return "registered-shared-sysmem-direct-binding";
    case Transport::kPinnedBounce:
      return "pinned-shared-sysmem-d2h-h2d";
    case Transport::kPageableBounce:
      return "pageable-shared-sysmem-d2h-h2d";
  }
  fail("unknown transport");
}

[[nodiscard]] std::string_view transport_description(const Transport transport) {
  switch (transport) {
    case Transport::kRegisteredDirect:
      return "full-coherent registered system-memory activation edge";
    case Transport::kPinnedBounce:
      return "producer device write plus pinned D2H/H2D bounce";
    case Transport::kPageableBounce:
      return "producer device write plus pageable D2H/H2D bounce";
  }
  fail("unknown transport");
}

[[nodiscard]] TrtPtr<nvinfer1::IHostMemory> build_control_policy(
    Logger& logger, const Workload workload) {
  require(workload != Workload::kResnetDetectionHead,
          "resnet-detection-head requires an external learned downstream engine");
  require(workload != Workload::kResnet50Classification,
          "resnet50-classification requires an external learned downstream engine");
  TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  require(builder != nullptr, "failed to create TensorRT builder");
  TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  require(network != nullptr, "failed to create TensorRT network");
  TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  require(config != nullptr, "failed to create TensorRT builder config");
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64U << 20U);

  const bool resnet = workload == Workload::kResnetControl;
  const nvinfer1::Dims input_dims =
      resnet ? nvinfer1::Dims{4, {1, 4, 23, 40}}
             : nvinfer1::Dims{3, {1, 1500, 384}};
  auto* input = network->addInput("features", nvinfer1::DataType::kFLOAT,
                                  input_dims);
  require(input != nullptr, "failed to add control-policy input");
  const std::uint32_t spatial_axes =
      resnet ? ((1U << 2U) | (1U << 3U)) : (1U << 1U);
  auto* reduced = network->addReduce(*input, nvinfer1::ReduceOperation::kAVG,
                                     spatial_axes, false);
  require(reduced != nullptr, "failed to add control-policy reduction");

  const std::size_t feature_width = resnet ? 4U : 384U;
  std::vector<float> weights(feature_width * 16U);
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = static_cast<float>((static_cast<int>(index % 11U) - 5) *
                                        0.03125);
  }
  const nvinfer1::Weights weight_values{nvinfer1::DataType::kFLOAT,
                                        weights.data(),
                                        static_cast<std::int64_t>(weights.size())};
  auto* weight = network->addConstant(
      nvinfer1::Dims2{static_cast<int>(feature_width), 16}, weight_values);
  require(weight != nullptr, "failed to add control-policy weights");
  auto* matrix = network->addMatrixMultiply(
      *reduced->getOutput(0), nvinfer1::MatrixOperation::kNONE,
      *weight->getOutput(0), nvinfer1::MatrixOperation::kNONE);
  require(matrix != nullptr, "failed to add control-policy matrix multiply");

  std::vector<float> biases(16U);
  for (std::size_t index = 0; index < biases.size(); ++index) {
    biases[index] = static_cast<float>(index) * 0.01F;
  }
  const nvinfer1::Weights bias_values{nvinfer1::DataType::kFLOAT,
                                      biases.data(),
                                      static_cast<std::int64_t>(biases.size())};
  auto* bias = network->addConstant(nvinfer1::Dims2{1, 16}, bias_values);
  require(bias != nullptr, "failed to add control-policy bias");
  auto* sum = network->addElementWise(*matrix->getOutput(0), *bias->getOutput(0),
                                      nvinfer1::ElementWiseOperation::kSUM);
  require(sum != nullptr, "failed to add control-policy bias sum");
  auto* activation = network->addActivation(*sum->getOutput(0),
                                            nvinfer1::ActivationType::kSIGMOID);
  require(activation != nullptr, "failed to add control-policy activation");
  activation->getOutput(0)->setName("policy_output");
  network->markOutput(*activation->getOutput(0));

  TrtPtr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*network, *config));
  require(plan != nullptr, "failed to build control-policy TensorRT engine");
  return plan;
}

[[nodiscard]] int initialize_cuda() {
  cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (status != cudaSuccess && status != cudaErrorSetOnActiveProcess) {
    return static_cast<int>(status);
  }
  status = cudaSetDevice(0);
  return static_cast<int>(status);
}

void capture_activation_trace(const Options& options) {
  set_cuda_environment(options.producer_uuid, options.producer_mps_pipe,
                       options.producer_quota);
  const std::size_t bytes = payload_bytes(options.workload);
  void* const mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) {
    fail("activation capture mmap failed: " +
         std::string(std::strerror(errno)));
  }
  try {
    std::memset(mapping, 0, bytes);
    const int status = initialize_cuda();
    require(status == cudaSuccess, "activation capture CUDA initialization failed");
    Logger logger;
    RegisteredMapping payload(mapping, bytes,
                              registered_mapping(options.transport));
    const std::vector<char> engine_bytes = read_file(options.producer_engine);
    EngineRunner runner(engine_bytes.data(), engine_bytes.size(), logger, payload,
                        producer_output(options.workload),
                        direct_binding(options.transport), true);
    ProducerInputTrace input_trace(options.producer_input_trace);
    const std::size_t total =
        static_cast<std::size_t>(options.warmup + options.iterations);
    require(input_trace.count() == total,
            "activation capture input trace count differs from warmup plus iterations");
    require(input_trace.sample_bytes() == runner.input_bytes(),
            "activation capture input trace bytes differ from producer engine inputs");
    ActivationTraceWriter writer(options.activation_capture_trace, total, bytes);
    for (std::size_t index = 0U; index < total; ++index) {
      const auto iteration = static_cast<std::uint32_t>(index);
      runner.infer(iteration, true, &input_trace);
      if (!direct_binding(options.transport)) {
        runner.copy_handoff_to_host(payload.host(), bytes);
      }
      writer.write(iteration, input_trace.input_sha256(iteration), payload.host());
    }
    writer.finish();
    static_cast<void>(munmap(mapping, bytes));
  } catch (...) {
    static_cast<void>(munmap(mapping, bytes));
    throw;
  }
}

[[noreturn]] void producer_main(const Options& options, void* mapping,
                                Transfer* producer_metadata,
                                OrionEvidence* orion_evidence,
                                const int ready_fd, const int go_fd,
                                const int transfer_fd, const int ack_fd) {
  try {
    set_cuda_environment(options.producer_uuid, options.producer_mps_pipe,
                         options.producer_quota);
    const std::size_t bytes = payload_bytes(options.workload);
    Ready ready{0, initialize_cuda(), 0, bytes};
    if (ready.status != cudaSuccess) {
      write_all(ready_fd, &ready, sizeof(ready));
      _exit(2);
    }
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties(producer)");
    ready.multiprocessors = properties.multiProcessorCount;
    Logger logger;
    RegisteredMapping payload(mapping, bytes,
                              registered_mapping(options.transport));
    OrionBackground orion(options, logger, payload);
    const std::vector<char> engine_bytes = read_file(options.producer_engine);
    EngineRunner runner(engine_bytes.data(), engine_bytes.size(), logger, payload,
                        producer_output(options.workload),
                        direct_binding(options.transport), true);
    std::unique_ptr<ProducerInputTrace> input_trace;
    if (!options.producer_input_trace.empty()) {
      input_trace = std::make_unique<ProducerInputTrace>(options.producer_input_trace);
      require(input_trace->count() ==
                  static_cast<std::size_t>(options.warmup + options.iterations),
              "producer input trace count differs from warmup plus iterations");
      require(input_trace->sample_bytes() == runner.input_bytes(),
              "producer input trace bytes differ from producer engine inputs");
    }
    std::unique_ptr<ActivationReplayTrace> activation_replay;
    if (!options.activation_replay_trace.empty()) {
      activation_replay =
          std::make_unique<ActivationReplayTrace>(options.activation_replay_trace);
      require(activation_replay->count() ==
                  static_cast<std::size_t>(options.warmup + options.iterations),
              "activation replay trace count differs from warmup plus iterations");
      require(activation_replay->sample_bytes() == bytes,
              "activation replay trace bytes differ from producer output tensor");
      require(input_trace != nullptr,
              "activation replay requires a producer input trace");
      for (std::size_t index = 0U; index < activation_replay->count(); ++index) {
        require(activation_replay->input_sha256(index) ==
                    input_trace->input_sha256(index),
                "activation replay request binding differs from producer input trace");
      }
    }
    std::unique_ptr<OperationalArrivalTrace> operational_arrival;
    if (!options.arrival_trace.empty()) {
      operational_arrival =
          std::make_unique<OperationalArrivalTrace>(options.arrival_trace);
      require(operational_arrival->count() ==
                  static_cast<std::size_t>(options.iterations),
              "operational arrival trace count differs from measured iterations");
      require(input_trace != nullptr,
              "operational arrival trace requires a producer input trace");
      for (std::size_t index = 0U; index < operational_arrival->count(); ++index) {
        const auto& record = operational_arrival->record(index);
        require(record.iteration ==
                    static_cast<std::uint32_t>(options.warmup + index),
                "operational arrival trace iteration differs from warmup binding");
        require(record.input_sha256 == input_trace->input_sha256(record.iteration),
                "operational arrival trace input binding differs");
      }
    }
    write_all(ready_fd, &ready, sizeof(ready));
    char go = 0;
    if (!read_all(go_fd, &go, sizeof(go))) {
      _exit(3);
    }
    orion.start_work();
    const int total = options.warmup + options.iterations;
    std::uint64_t measured_background_start = 0U;
    std::uint64_t arrival_epoch_ns = 0U;
    for (int index = 0; index < total; ++index) {
      if (index == options.warmup) {
        measured_background_start = orion.completed();
        arrival_epoch_ns = monotonic_ns();
        orion_evidence->measurement_start_ns = arrival_epoch_ns;
      }
      Transfer transfer{};
      transfer.iteration = static_cast<std::uint32_t>(index);
      transfer.warmup = static_cast<std::uint32_t>(index < options.warmup);
      transfer.checksum_enabled = checksum_enabled(options, transfer.iteration);
      if (input_trace != nullptr) {
        const std::string_view digest = input_trace->input_sha256(transfer.iteration);
        std::copy(digest.begin(), digest.end(), transfer.input_sha256.begin());
      }
      if (operational_arrival != nullptr && index >= options.warmup) {
        const auto& record =
            operational_arrival->record(static_cast<std::size_t>(index - options.warmup));
        std::copy(record.request_id.begin(), record.request_id.end(),
                  transfer.request_id.begin());
        transfer.declared_arrival_ns = arrival_epoch_ns + record.release_offset_ns;
        // This scheduler never shifts a late request's declared arrival.  A
        // busy one-buffer pipeline is reported as queue delay instead.
        wait_until_ns(transfer.declared_arrival_ns);
      } else {
        const std::string request_id =
            "legacy-" + std::to_string(static_cast<unsigned int>(index));
        std::copy(request_id.begin(), request_id.end(), transfer.request_id.begin());
        transfer.declared_arrival_ns = monotonic_ns();
      }
      // Arrival is the request's causal entry into the measured service.  It
      // precedes any gate/drain work so the wall deadline includes protection
      // overhead instead of silently starting after the gate.
      transfer.actual_release_ns = monotonic_ns();
      transfer.arrival_ns = transfer.actual_release_ns;
      transfer.queue_delay_ns =
          transfer.actual_release_ns >= transfer.declared_arrival_ns
              ? transfer.actual_release_ns - transfer.declared_arrival_ns
              : 0U;
      transfer.pause_begin_ns = transfer.arrival_ns;
      transfer.gate_begin_ns = transfer.pause_begin_ns;
      if (options.gate_scope != GateScope::kConsumer) {
        pause_processes(options.gate_pids, options.gate_mode);
      }
      transfer.pause_complete_ns = monotonic_ns();
      transfer.gate_done_ns = transfer.pause_complete_ns;
      transfer.producer_start_ns = monotonic_ns();
      if (options.dependency_mode == DependencyMode::kIndependent) {
        // Release the independent consumer before producer inference.  The
        // consumer already has the request-indexed activation replay mapped;
        // this signal is a start barrier rather than a data dependency.
        write_all(transfer_fd, &transfer, sizeof(transfer));
      }
      runner.infer(transfer.iteration, true, input_trace.get());
      transfer.producer_compute_done_ns = monotonic_ns();
      if (!direct_binding(options.transport)) {
        runner.copy_handoff_to_host(payload.host(), payload.bytes());
      }
      transfer.producer_done_ns = monotonic_ns();
      transfer.publication_ns = transfer.producer_done_ns;
      // The dependent consumer can start as soon as the producer's payload
      // is ready.  Correctness checks are deliberately recorded after this
      // production boundary so they cannot inflate wall latency or gate
      // timing.
      if (options.gate_scope == GateScope::kConsumer) {
        transfer.pause_begin_ns = monotonic_ns();
        transfer.gate_begin_ns = transfer.pause_begin_ns;
        pause_processes(options.gate_pids, options.gate_mode);
        transfer.pause_complete_ns = monotonic_ns();
        transfer.gate_done_ns = transfer.pause_complete_ns;
      }
      if (options.dependency_mode == DependencyMode::kDependent) {
        write_all(transfer_fd, &transfer, sizeof(transfer));
      }
      // Publication is the producer protection boundary.  Resume the gated
      // best-effort clients before correctness-only checksum work so the
      // producer scope does not silently include CPU validation time.
      if (options.gate_scope == GateScope::kProducer) {
        resume_processes(options.gate_pids, transfer.resume_issued_ns,
                         transfer.resume_observed_ns);
        transfer.resume_done_ns = transfer.resume_observed_ns;
      }
      if (transfer.checksum_enabled) {
        correctness_delay(options.validation_delay_us);
        transfer.producer_checksum = checksum(payload.host(), payload.bytes());
      }
      if (activation_replay != nullptr) {
        const auto live_checksum = checksum(payload.host(), payload.bytes());
        require(live_checksum ==
                    activation_replay->activation_checksum(transfer.iteration),
                "live producer activation differs from activation replay trace");
      }
      transfer.producer_payload_verification_done_ns = monotonic_ns();
      producer_metadata[transfer.iteration] = transfer;
      char ack = 0;
      if (!read_all(ack_fd, &ack, sizeof(ack))) {
        _exit(4);
      }
    }
    orion_evidence->measurement_end_ns = monotonic_ns();
    orion_evidence->measured_background_completed =
        orion.completed() - measured_background_start;
    orion.finish(*orion_evidence);
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << "producer: " << error.what() << '\n';
    _exit(5);
  }
}

[[noreturn]] void consumer_main(const Options& options, void* mapping,
                                const int ready_fd, const int go_fd,
                                const int transfer_fd, const int result_fd,
                                const int ack_fd) {
  try {
    set_cuda_environment(options.consumer_uuid, options.consumer_mps_pipe,
                         options.consumer_quota);
    const std::size_t bytes = payload_bytes(options.workload);
    Ready ready{1, initialize_cuda(), 0, bytes};
    if (ready.status != cudaSuccess) {
      write_all(ready_fd, &ready, sizeof(ready));
      _exit(2);
    }
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties(consumer)");
    ready.multiprocessors = properties.multiProcessorCount;
    Logger logger;
    std::unique_ptr<ActivationReplayTrace> activation_replay;
    void* replay_storage = nullptr;
    std::unique_ptr<RegisteredMapping> replay_payload;
    if (options.dependency_mode == DependencyMode::kIndependent) {
      activation_replay =
          std::make_unique<ActivationReplayTrace>(options.activation_replay_trace);
      require(activation_replay->count() ==
                  static_cast<std::size_t>(options.warmup + options.iterations),
              "consumer activation replay count differs from warmup plus iterations");
      require(activation_replay->sample_bytes() == bytes,
              "consumer activation replay bytes differ from payload");
      if (direct_binding(options.transport)) {
        replay_storage = mmap(nullptr, activation_replay->storage_bytes(),
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (replay_storage == MAP_FAILED) {
          fail("consumer activation replay mmap failed: " +
               std::string(std::strerror(errno)));
        }
        std::memcpy(replay_storage, activation_replay->data(),
                    activation_replay->storage_bytes());
        replay_payload = std::make_unique<RegisteredMapping>(
            replay_storage, bytes, true, activation_replay->storage_bytes());
      }
    }
    std::unique_ptr<RegisteredMapping> shared_payload;
    if (replay_payload == nullptr) {
      shared_payload = std::make_unique<RegisteredMapping>(
          mapping, bytes, registered_mapping(options.transport));
    }
    RegisteredMapping& payload = replay_payload != nullptr
                                     ? *replay_payload
                                     : *shared_payload;
    std::vector<char> external_engine;
    TrtPtr<nvinfer1::IHostMemory> generated_policy;
    const void* serialized_engine = nullptr;
    std::size_t serialized_bytes = 0U;
    if (!options.consumer_engine.empty()) {
      external_engine = read_file(options.consumer_engine);
      serialized_engine = external_engine.data();
      serialized_bytes = external_engine.size();
    } else {
      generated_policy = build_control_policy(logger, options.workload);
      serialized_engine = generated_policy->data();
      serialized_bytes = generated_policy->size();
    }
    EngineRunner runner(serialized_engine, serialized_bytes, logger, payload,
                        options.consumer_input_tensor,
                        direct_binding(options.transport), true);
    std::ofstream application_output_trace;
    if (!options.application_output_trace.empty()) {
      application_output_trace.open(options.application_output_trace,
                                    std::ios::binary | std::ios::out |
                                        std::ios::trunc);
      require(application_output_trace.is_open(),
              "failed to open application output trace");
      runner.write_output_trace_header(application_output_trace);
    }
    write_all(ready_fd, &ready, sizeof(ready));
    char go = 0;
    if (!read_all(go_fd, &go, sizeof(go))) {
      _exit(3);
    }
    while (true) {
      Transfer transfer{};
      if (!read_all(transfer_fd, &transfer, sizeof(transfer))) {
        break;
      }
      Result result{};
      result.transfer = transfer;
      result.consumer_start_ns = monotonic_ns();
      if (options.dependency_mode == DependencyMode::kIndependent) {
        require(activation_replay != nullptr,
                "independent consumer lacks activation replay");
        require(activation_replay->input_sha256(transfer.iteration) ==
                    std::string_view(transfer.input_sha256.data()),
                "consumer activation replay request binding differs");
        const auto* const replay_sample =
            activation_replay->sample(transfer.iteration);
        if (direct_binding(options.transport)) {
          runner.bind_direct_handoff(
              replay_payload->device_at(static_cast<std::size_t>(transfer.iteration) *
                                         bytes),
              bytes);
        } else {
          runner.copy_handoff_from_host(replay_sample, bytes);
        }
      } else if (!direct_binding(options.transport)) {
        runner.copy_handoff_from_host(payload.host(), payload.bytes());
      }
      result.consumer_compute_start_ns = monotonic_ns();
      runner.infer(transfer.iteration, false);
      result.consumer_compute_done_ns = monotonic_ns();
      // This is the production completion boundary.  Resume a gated
      // submitter before doing any correctness-only checksum work.
      result.consumer_done_ns = monotonic_ns();
      if (options.gate_scope == GateScope::kPipeline ||
          options.gate_scope == GateScope::kConsumer) {
        resume_processes(options.gate_pids, result.consumer_resume_issued_ns,
                         result.consumer_resume_observed_ns);
        result.transfer.resume_done_ns = result.consumer_resume_observed_ns;
      }
      if (transfer.checksum_enabled) {
        correctness_delay(options.validation_delay_us);
        const void* const checksum_source =
            options.dependency_mode == DependencyMode::kIndependent
                ? static_cast<const void*>(
                      activation_replay->sample(transfer.iteration))
                : payload.host();
        result.consumer_checksum = checksum(checksum_source, payload.bytes());
        result.consumer_output_checksum = runner.output_checksum();
      }
      if (application_output_trace.is_open()) {
        // This copy is intentionally after consumer_done_ns and resume. It is
        // an accuracy-audit artifact, never part of the production deadline.
        runner.write_output_trace_record(application_output_trace,
                                         transfer.iteration);
      }
      result.consumer_payload_verification_done_ns = monotonic_ns();
      write_all(result_fd, &result, sizeof(result));
      const char ack = 1;
      write_all(ack_fd, &ack, sizeof(ack));
    }
    // The child uses _exit() to avoid inherited parent destructors. Flush the
    // post-completion accuracy trace explicitly before that exit; otherwise a
    // valid trace path can silently remain a zero-byte file.
    if (application_output_trace.is_open()) {
      application_output_trace.flush();
      require(application_output_trace.good(),
              "failed to flush application output trace");
    }
    if (replay_storage != nullptr) {
      static_cast<void>(munmap(replay_storage,
                               activation_replay->storage_bytes()));
    }
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << "consumer: " << error.what() << '\n';
    _exit(5);
  }
}

[[nodiscard]] int parse_int(const std::string& text,
                            const std::string_view name,
                            const bool allow_zero) {
  std::size_t consumed = 0;
  const long value = std::stol(text, &consumed);
  if (consumed != text.size() || value < (allow_zero ? 0 : 1) ||
      value > std::numeric_limits<int>::max()) {
    fail("invalid " + std::string(name) + ": " + text);
  }
  return static_cast<int>(value);
}

[[nodiscard]] std::vector<pid_t> parse_pids(const std::string& text) {
  std::vector<pid_t> pids;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string token = text.substr(begin, end - begin);
    const int value = parse_int(token, "gate PID", false);
    require(value != getpid(), "gate PID must not be the pipeline parent");
    pids.push_back(static_cast<pid_t>(value));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1U;
  }
  std::sort(pids.begin(), pids.end());
  require(std::adjacent_find(pids.begin(), pids.end()) == pids.end(),
          "gate PIDs must be unique");
  return pids;
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  if (const char* value = std::getenv("JDG_MIG_SMALL_UUID")) {
    options.producer_uuid = value;
  }
  if (const char* value = std::getenv("JDG_MIG_BIG_UUID")) {
    options.consumer_uuid = value;
  }
  if (const char* value = std::getenv("JDG_MPS_PIPE_DIRECTORY")) {
    options.producer_mps_pipe = value;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&]() -> std::string {
      if (++index >= argc) {
        fail("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--producer-engine") {
      options.producer_engine = next();
    } else if (argument == "--consumer-engine") {
      options.consumer_engine = next();
    } else if (argument == "--consumer-input-tensor") {
      options.consumer_input_tensor = next();
      require(!options.consumer_input_tensor.empty(),
              "--consumer-input-tensor must not be empty");
    } else if (argument == "--producer") {
      options.producer_uuid = next();
    } else if (argument == "--consumer") {
      options.consumer_uuid = next();
    } else if (argument == "--producer-mps-pipe") {
      options.producer_mps_pipe = next();
    } else if (argument == "--consumer-mps-pipe") {
      options.consumer_mps_pipe = next();
    } else if (argument == "--warmup") {
      options.warmup = parse_int(next(), "warmup", true);
    } else if (argument == "--iterations") {
      options.iterations = parse_int(next(), "iterations", false);
    } else if (argument == "--transport") {
      const std::string value = next();
      if (value == "registered-direct") {
        options.transport = Transport::kRegisteredDirect;
      } else if (value == "pinned-bounce") {
        options.transport = Transport::kPinnedBounce;
      } else if (value == "pageable-bounce") {
        options.transport = Transport::kPageableBounce;
      } else {
        fail("--transport expects registered-direct, pinned-bounce, or pageable-bounce");
      }
    } else if (argument == "--gate-pids") {
      options.gate_pids = parse_pids(next());
    } else if (argument == "--deadline-us") {
      const std::string value = next();
      std::size_t consumed = 0;
      options.deadline_us = std::stod(value, &consumed);
      require(consumed == value.size() && std::isfinite(options.deadline_us) &&
                  options.deadline_us > 0.0,
              "--deadline-us expects a positive finite number");
    } else if (argument == "--gate-mode") {
      const std::string value = next();
      if (value == "cooperative") {
        options.gate_mode = GateMode::kCooperative;
      } else if (value == "stop") {
        options.gate_mode = GateMode::kStop;
      } else {
        fail("--gate-mode expects cooperative or stop");
      }
    } else if (argument == "--gate-scope") {
      const std::string value = next();
      if (value == "producer") {
        options.gate_scope = GateScope::kProducer;
      } else if (value == "consumer") {
        options.gate_scope = GateScope::kConsumer;
      } else if (value == "pipeline") {
        options.gate_scope = GateScope::kPipeline;
      } else {
        fail("--gate-scope expects producer, consumer, or pipeline");
      }
    } else if (argument == "--producer-quota") {
      options.producer_quota = parse_int(next(), "producer quota", false);
      require(options.producer_quota <= 100,
              "producer quota must be in [1, 100]");
    } else if (argument == "--consumer-quota") {
      options.consumer_quota = parse_int(next(), "consumer quota", false);
      require(options.consumer_quota <= 100,
              "consumer quota must be in [1, 100]");
    } else if (argument == "--workload") {
      const std::string value = next();
      if (value == "resnet-control") {
        options.workload = Workload::kResnetControl;
      } else if (value == "resnet-detection-head") {
        options.workload = Workload::kResnetDetectionHead;
      } else if (value == "resnet50-classification") {
        options.workload = Workload::kResnet50Classification;
      } else if (value == "whisper-projection") {
        options.workload = Workload::kWhisperProjection;
      } else {
        fail("--workload expects resnet-control, resnet-detection-head, resnet50-classification, or whisper-projection");
      }
    } else if (argument == "--dependency-mode") {
      const std::string value = next();
      if (value == "dependent") {
        options.dependency_mode = DependencyMode::kDependent;
      } else if (value == "independent") {
        options.dependency_mode = DependencyMode::kIndependent;
      } else {
        fail("--dependency-mode expects dependent or independent");
      }
    } else if (argument == "--deadline-mode") {
      const std::string value = next();
      if (value == "wall") {
        options.validation_excluded_deadline = false;
      } else if (value == "validation-excluded") {
        options.validation_excluded_deadline = true;
      } else {
        fail("--deadline-mode expects wall or validation-excluded");
      }
    } else if (argument == "--checksum-mode") {
      const std::string value = next();
      if (value == "inline") {
        options.checksum_mode = ChecksumMode::kInline;
      } else if (value == "sampled") {
        options.checksum_mode = ChecksumMode::kSampled;
      } else if (value == "off") {
        options.checksum_mode = ChecksumMode::kOff;
      } else {
        fail("--checksum-mode expects inline, sampled, or off");
      }
    } else if (argument == "--checksum-sample-period") {
      options.checksum_sample_period = parse_int(next(), "checksum sample period", false);
    } else if (argument == "--trace-csv") {
      options.trace_csv = next();
    } else if (argument == "--event-trace-csv") {
      options.event_trace_csv = next();
    } else if (argument == "--checksum-trace-csv") {
      options.checksum_trace_csv = next();
    } else if (argument == "--application-output-trace") {
      options.application_output_trace = next();
    } else if (argument == "--producer-input-trace") {
      options.producer_input_trace = next();
    } else if (argument == "--arrival-trace") {
      options.arrival_trace = next();
    } else if (argument == "--activation-replay-trace") {
      options.activation_replay_trace = next();
    } else if (argument == "--capture-activation-trace") {
      options.activation_capture_trace = next();
    } else if (argument == "--orion-profile-aware") {
      const std::string value = next();
      if (value == "true") {
        options.orion_profile_aware = true;
      } else if (value == "false") {
        options.orion_profile_aware = false;
      } else {
        fail("--orion-profile-aware expects true or false");
      }
    } else if (argument == "--orion-background-engine") {
      options.orion_background_engine = next();
    } else if (argument == "--orion-best-effort-profile") {
      options.orion_best_effort_profile = next();
    } else if (argument == "--orion-high-priority-profile") {
      options.orion_high_priority_profile = next();
    } else if (argument == "--orion-decisions") {
      options.orion_decision_trace = next();
    } else if (argument == "--orion-max-be-duration-us") {
      const std::string value = next();
      std::size_t consumed = 0U;
      options.orion_max_be_duration_us = std::stod(value, &consumed);
      require(consumed == value.size() &&
                  std::isfinite(options.orion_max_be_duration_us) &&
                  options.orion_max_be_duration_us > 0.0,
              "--orion-max-be-duration-us expects a positive finite number");
    } else if (argument == "--orion-trace-mode") {
      options.orion_trace_mode = next();
      require(options.orion_trace_mode == "full" ||
                  options.orion_trace_mode == "events",
              "--orion-trace-mode expects full or events");
    } else if (argument == "--orion-background-period-us") {
      const std::string value = next();
      std::size_t consumed = 0U;
      options.orion_background_period_us = std::stod(value, &consumed);
      require(consumed == value.size() &&
                  std::isfinite(options.orion_background_period_us) &&
                  options.orion_background_period_us > 0.0,
              "--orion-background-period-us expects a positive finite number");
    } else if (argument == "--validation-delay-us") {
      const std::string value = next();
      std::size_t consumed = 0U;
      options.validation_delay_us = std::stod(value, &consumed);
      require(consumed == value.size() &&
                  std::isfinite(options.validation_delay_us) &&
                  options.validation_delay_us >= 0.0,
              "--validation-delay-us expects a nonnegative finite number");
    } else if (argument == "--help") {
      std::cout << "Usage: jdg-mig-trt-pipeline --producer-engine PATH "
                   "[--consumer-engine PATH] "
                   "[--consumer-input-tensor NAME] "
                   "--producer MIG_UUID --consumer MIG_UUID "
                   "[--producer-mps-pipe PATH] [--consumer-mps-pipe PATH] "
                   "[--transport MODE] [--gate-pids PID,...] "
                   "[--gate-mode cooperative|stop] [--deadline-us US] "
                   "[--gate-scope producer|consumer|pipeline] "
                   "[--producer-quota PCT] [--consumer-quota PCT] "
                   "[--workload resnet-control|resnet-detection-head|resnet50-classification|whisper-projection] "
                   "[--dependency-mode dependent|independent] "
                   "[--deadline-mode wall|validation-excluded] "
                   "[--checksum-mode inline|sampled|off] "
                   "[--checksum-sample-period N] "
                   "[--trace-csv PATH] "
                   "[--event-trace-csv PATH] "
                   "[--checksum-trace-csv PATH] "
                   "[--application-output-trace PATH] "
                   "[--producer-input-trace PATH] "
                   "[--arrival-trace PATH] "
                   "[--activation-replay-trace PATH] "
                   "[--capture-activation-trace PATH] "
                   "[--orion-profile-aware true|false] "
                   "[--orion-background-engine PATH] "
                   "[--orion-best-effort-profile PATH] "
                   "[--orion-high-priority-profile PATH] "
                   "[--orion-decisions PATH] "
                   "[--orion-max-be-duration-us US] "
                   "[--orion-trace-mode full|events] "
                   "[--orion-background-period-us US] "
                   "[--validation-delay-us US] "
                   "[--warmup N] [--iterations N]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + argument);
    }
  }
  require(!options.producer_engine.empty(), "--producer-engine is required");
  if (!options.consumer_engine.empty()) {
    require(std::filesystem::is_regular_file(options.consumer_engine),
            "--consumer-engine must point to a regular TensorRT engine");
  }
  require(!options.producer_uuid.empty(), "producer MIG UUID is required");
  if (options.activation_capture_trace.empty()) {
    require(!options.consumer_uuid.empty(), "consumer MIG UUID is required");
  } else {
    require(options.dependency_mode == DependencyMode::kDependent,
            "activation capture does not run an independent arm");
    require(options.activation_replay_trace.empty(),
            "activation capture cannot also consume a replay trace");
    require(!options.producer_input_trace.empty(),
            "activation capture requires --producer-input-trace");
  }
  if (options.dependency_mode == DependencyMode::kIndependent) {
    require(!options.activation_replay_trace.empty(),
            "independent mode requires --activation-replay-trace");
  }
  if (!options.activation_replay_trace.empty()) {
    require(std::filesystem::is_regular_file(options.activation_replay_trace),
            "--activation-replay-trace must point to a regular file");
  }
  if (!options.producer_input_trace.empty()) {
    require(std::filesystem::is_regular_file(options.producer_input_trace),
            "--producer-input-trace must point to a regular file");
  }
  if (!options.arrival_trace.empty()) {
    require(std::filesystem::is_regular_file(options.arrival_trace),
            "--arrival-trace must point to a regular file");
    require(!options.producer_input_trace.empty(),
            "--arrival-trace requires --producer-input-trace");
  }
  require(options.gate_scope != GateScope::kConsumer ||
              options.dependency_mode == DependencyMode::kDependent,
          "consumer gate scope requires dependent mode");
  if (options.producer_uuid == options.consumer_uuid) {
    require(!options.producer_mps_pipe.empty() &&
                options.producer_mps_pipe == options.consumer_mps_pipe,
            "same-instance execution requires one shared MPS pipe");
  }
  if (options.orion_profile_aware) {
#ifndef JDG_WITH_ORION
    fail("this binary lacks the Orion driver interposer; use "
         "jdg-orion-mig-trt-pipeline");
#endif
    require(options.producer_uuid != options.consumer_uuid,
            "Orion dependent smoke requires fixed 1g producer and 2g consumer");
    require(std::filesystem::is_regular_file(options.orion_background_engine) &&
                std::filesystem::is_regular_file(
                    options.orion_best_effort_profile) &&
                std::filesystem::is_regular_file(
                    options.orion_high_priority_profile) &&
                !options.orion_decision_trace.empty() &&
                !std::filesystem::exists(options.orion_decision_trace),
            "Orion requires existing engine/profiles and a new decision trace");
  }
  return options;
}

[[nodiscard]] double percentile(std::vector<double> values, const double q) {
  require(!values.empty(), "percentile input is empty");
  std::sort(values.begin(), values.end());
  const double position = q * static_cast<double>(values.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  return values[lower] + (values[upper] - values[lower]) *
                             (position - static_cast<double>(lower));
}

[[nodiscard]] int wait_child(const pid_t pid) {
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      return 255;
    }
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (!options.activation_capture_trace.empty()) {
      capture_activation_trace(options);
      std::cout << "{\"schema_version\":1,\"status\":\"ok\","
                   "\"kind\":\"producer-activation-capture\","
                   "\"format\":\"JDGACT1\",\"path\":\""
                << options.activation_capture_trace.string()
                << "\",\"warmup\":" << options.warmup
                << ",\"iterations\":" << options.iterations << "}\n";
      return 0;
    }
    const std::size_t bytes = payload_bytes(options.workload);
    void* const mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
      fail("mmap failed: " + std::string(std::strerror(errno)));
    }
    std::memset(mapping, 0, bytes);
    void* const consumer_mapping =
        mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (consumer_mapping == MAP_FAILED) {
      static_cast<void>(munmap(mapping, bytes));
      fail("consumer mmap failed: " + std::string(std::strerror(errno)));
    }
    std::memset(consumer_mapping, 0, bytes);
    const std::size_t expected_transfers =
        static_cast<std::size_t>(options.warmup + options.iterations);
    auto* const producer_metadata = static_cast<Transfer*>(mmap(
        nullptr, expected_transfers * sizeof(Transfer), PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (producer_metadata == MAP_FAILED) {
      static_cast<void>(munmap(mapping, bytes));
      static_cast<void>(munmap(consumer_mapping, bytes));
      fail("producer metadata mmap failed: " +
           std::string(std::strerror(errno)));
    }
    for (std::size_t index = 0; index < expected_transfers; ++index) {
      producer_metadata[index] = {};
    }
    auto* const orion_evidence = static_cast<OrionEvidence*>(
        mmap(nullptr, sizeof(OrionEvidence), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (orion_evidence == MAP_FAILED) {
      static_cast<void>(munmap(mapping, bytes));
      static_cast<void>(munmap(consumer_mapping, bytes));
      static_cast<void>(munmap(producer_metadata,
                               expected_transfers * sizeof(Transfer)));
      fail("Orion evidence mmap failed: " + std::string(std::strerror(errno)));
    }
    *orion_evidence = {};

    int ready[2]{}, producer_go[2]{}, consumer_go[2]{}, transfers[2]{},
        results[2]{}, acknowledgements[2]{};
    if (pipe(ready) != 0 || pipe(producer_go) != 0 || pipe(consumer_go) != 0 ||
        pipe(transfers) != 0 || pipe(results) != 0 ||
        pipe(acknowledgements) != 0) {
      fail("pipe creation failed: " + std::string(std::strerror(errno)));
    }

    const pid_t producer = fork();
    if (producer == 0) {
      close_fd(ready[0]);
      close_fd(producer_go[1]);
      close_fd(consumer_go[0]);
      close_fd(consumer_go[1]);
      close_fd(transfers[0]);
      close_fd(results[0]);
      close_fd(results[1]);
      close_fd(acknowledgements[1]);
      producer_main(options, mapping, producer_metadata, orion_evidence, ready[1],
                    producer_go[0], transfers[1], acknowledgements[0]);
    }
    require(producer > 0, "producer fork failed");

    const pid_t consumer = fork();
    if (consumer == 0) {
      close_fd(ready[0]);
      close_fd(consumer_go[1]);
      close_fd(producer_go[0]);
      close_fd(producer_go[1]);
      close_fd(transfers[1]);
      close_fd(results[0]);
      close_fd(acknowledgements[0]);
      consumer_main(options,
                    options.dependency_mode == DependencyMode::kIndependent
                        ? consumer_mapping
                        : mapping,
                    ready[1], consumer_go[0], transfers[0],
                    results[1], acknowledgements[1]);
    }
    if (consumer < 0) {
      kill(producer, SIGTERM);
      fail("consumer fork failed");
    }

    close_fd(ready[1]);
    close_fd(producer_go[0]);
    close_fd(consumer_go[0]);
    close_fd(transfers[0]);
    close_fd(transfers[1]);
    close_fd(results[1]);
    close_fd(acknowledgements[0]);
    close_fd(acknowledgements[1]);

    Ready readiness[2]{};
    if (!read_all(ready[0], &readiness[0], sizeof(Ready)) ||
        !read_all(ready[0], &readiness[1], sizeof(Ready))) {
      kill(producer, SIGTERM);
      kill(consumer, SIGTERM);
      fail("child exited before TensorRT readiness");
    }
    close_fd(ready[0]);
    std::sort(std::begin(readiness), std::end(readiness),
              [](const Ready& left, const Ready& right) {
                return left.role < right.role;
              });
    require(readiness[0].status == cudaSuccess &&
                readiness[1].status == cudaSuccess,
            "child CUDA initialization failed");

    const char go = 1;
    write_all(producer_go[1], &go, sizeof(go));
    write_all(consumer_go[1], &go, sizeof(go));
    close_fd(producer_go[1]);
    close_fd(consumer_go[1]);

    const std::size_t expected = expected_transfers;
    std::vector<Result> collected;
    collected.reserve(expected);
    for (std::size_t index = 0; index < expected; ++index) {
      Result result{};
      if (!read_all(results[0], &result, sizeof(result))) {
        break;
      }
      collected.push_back(result);
    }
    close_fd(results[0]);
    const int producer_status = wait_child(producer);
    const int consumer_status = wait_child(consumer);
    static_cast<void>(munmap(mapping, bytes));
    static_cast<void>(munmap(consumer_mapping, bytes));
    for (Result& result : collected) {
      if (result.transfer.iteration < expected_transfers) {
        // The child sends the transfer before post-completion validation so
        // the consumer is not delayed by checksum work. Merge the producer
        // metadata captured in shared memory before replaying correctness.
        const std::uint64_t consumer_resume_issued_ns =
            result.consumer_resume_issued_ns;
        const std::uint64_t consumer_resume_observed_ns =
            result.consumer_resume_observed_ns;
        result.transfer = producer_metadata[result.transfer.iteration];
        if (options.gate_scope == GateScope::kPipeline ||
            options.gate_scope == GateScope::kConsumer) {
          result.transfer.resume_issued_ns = consumer_resume_issued_ns;
          result.transfer.resume_observed_ns = consumer_resume_observed_ns;
          result.transfer.resume_done_ns = consumer_resume_observed_ns;
        }
      }
    }
    static_cast<void>(munmap(producer_metadata,
                             expected_transfers * sizeof(Transfer)));

    std::vector<double> handoff_us;
    std::vector<double> end_to_end_us;
    std::vector<double> producer_compute_us;
    std::vector<double> transport_ready_us;
    std::vector<double> consumer_compute_us;
    std::vector<double> output_verification_us;
    std::vector<double> producer_payload_verification_us;
    std::vector<double> producer_handoff_copy_us;
    std::vector<double> transport_notification_us;
    std::vector<double> consumer_payload_verification_us;
    std::vector<double> consumer_handoff_copy_us;
    std::vector<double> edge_transport_us;
    std::vector<double> validation_excluded_end_to_end_us;
    std::vector<double> gate_us;
    std::vector<double> gate_acquire_us;
    std::vector<double> queue_delay_us;
    std::vector<std::uint64_t> payload_checksums;
    std::vector<std::uint64_t> output_checksums;
    std::vector<StageTrace> stage_traces;
    stage_traces.reserve(static_cast<std::size_t>(options.iterations));
    std::size_t checksum_failures = 0;
    std::size_t validated_requests = 0;
    std::size_t deadline_misses = 0;
    std::uint64_t first_start_ns = 0;
    std::uint64_t last_done_ns = 0;
    bool success = collected.size() == expected && producer_status == 0 &&
                   consumer_status == 0;
    for (const Result& result : collected) {
      const bool validated = result.transfer.checksum_enabled;
      const bool independent =
          options.dependency_mode == DependencyMode::kIndependent;
      if (validated) {
        ++validated_requests;
        const bool checksum_ok =
            result.consumer_checksum == result.transfer.producer_checksum;
        success = success && result.status == 0 && checksum_ok;
        if (result.status != 0 || !checksum_ok) {
          ++checksum_failures;
        }
      } else {
        success = success && result.status == 0;
      }
      if (result.transfer.warmup == 0) {
        const bool timing_valid =
          independent
                ? (result.transfer.arrival_ns > 0U &&
                   result.transfer.producer_start_ns >= result.transfer.arrival_ns &&
                   result.transfer.producer_compute_done_ns >=
                       result.transfer.producer_start_ns &&
                   result.transfer.producer_done_ns >=
                       result.transfer.producer_compute_done_ns &&
                   result.transfer.producer_payload_verification_done_ns >=
                       result.transfer.producer_done_ns &&
                   result.consumer_start_ns > 0U &&
                   result.consumer_compute_start_ns >= result.consumer_start_ns &&
                   result.consumer_compute_done_ns >=
                       result.consumer_compute_start_ns &&
                   result.consumer_done_ns >= result.consumer_compute_done_ns)
                : (result.transfer.arrival_ns > 0U &&
                   result.transfer.producer_start_ns >= result.transfer.arrival_ns &&
                   result.transfer.producer_payload_verification_done_ns >=
                       result.transfer.producer_done_ns &&
                   result.transfer.producer_done_ns >=
                       result.transfer.producer_compute_done_ns &&
                   result.consumer_start_ns >= result.transfer.producer_done_ns &&
                   result.consumer_compute_start_ns >= result.consumer_start_ns &&
                   result.consumer_compute_done_ns >=
                       result.consumer_compute_start_ns &&
                   result.consumer_done_ns >= result.consumer_compute_done_ns);
        if (result.status != 0 || !timing_valid) {
          success = false;
          continue;
        }
        if (first_start_ns == 0) {
          first_start_ns = result.transfer.arrival_ns;
        } else if (independent) {
          first_start_ns = std::min(first_start_ns, result.transfer.arrival_ns);
        }
        last_done_ns = std::max(
            last_done_ns,
            independent
                ? std::max(result.transfer.producer_done_ns,
                           result.consumer_done_ns)
                : result.consumer_done_ns);
        if (validated) {
          payload_checksums.push_back(result.transfer.producer_checksum);
          output_checksums.push_back(result.consumer_output_checksum);
        }
        handoff_us.push_back(independent
                                 ? 0.0
                                 : static_cast<double>(
                                       result.consumer_done_ns -
                                       result.transfer.producer_done_ns) /
                                       1000.0);
        producer_compute_us.push_back(
            static_cast<double>(result.transfer.producer_compute_done_ns -
                                result.transfer.producer_start_ns) /
            1000.0);
        transport_ready_us.push_back(
            independent
                ? 0.0
                : static_cast<double>(result.consumer_compute_start_ns -
                                      result.transfer.producer_done_ns) /
                      1000.0);
        consumer_compute_us.push_back(
            static_cast<double>(result.consumer_compute_done_ns -
                                result.consumer_compute_start_ns) /
            1000.0);
        output_verification_us.push_back(
            static_cast<double>(result.consumer_payload_verification_done_ns -
                                result.consumer_compute_done_ns) /
            1000.0);
        producer_payload_verification_us.push_back(
            static_cast<double>(
                result.transfer.producer_payload_verification_done_ns -
                result.transfer.producer_done_ns) /
            1000.0);
        producer_handoff_copy_us.push_back(
            static_cast<double>(result.transfer.producer_done_ns -
                                result.transfer.producer_compute_done_ns) /
            1000.0);
        transport_notification_us.push_back(
            independent
                ? 0.0
                : static_cast<double>(
                      result.consumer_start_ns -
                      result.transfer.producer_done_ns) /
                      1000.0);
        consumer_payload_verification_us.push_back(
            static_cast<double>(
                result.consumer_payload_verification_done_ns -
                result.consumer_start_ns) /
            1000.0);
        consumer_handoff_copy_us.push_back(
            independent
                ? 0.0
                : (direct_binding(options.transport)
                       ? 0.0
                       : static_cast<double>(result.consumer_compute_start_ns -
                                             result.consumer_start_ns) /
                             1000.0));
        edge_transport_us.push_back(
            producer_handoff_copy_us.back() + transport_notification_us.back() +
            consumer_handoff_copy_us.back());
        end_to_end_us.push_back(
            static_cast<double>(
                (independent
                     ? std::max(result.transfer.producer_done_ns,
                                result.consumer_done_ns)
                     : result.consumer_done_ns) -
                result.transfer.arrival_ns) /
            1000.0);
        // Wall latency starts at arrival and ends at production completion.
        // Checksums and output verification run afterward and are reported
        // separately; retain the legacy field as an alias for this interval.
        validation_excluded_end_to_end_us.push_back(end_to_end_us.back());
        if (!options.gate_pids.empty()) {
          require(result.transfer.pause_begin_ns > 0U &&
                      result.transfer.resume_observed_ns >=
                          result.transfer.pause_begin_ns,
                  "gate event timestamps are incomplete");
          gate_us.push_back(
              static_cast<double>(result.transfer.resume_observed_ns -
                                  result.transfer.pause_begin_ns) /
              1000.0);
          require(result.transfer.pause_complete_ns >=
                      result.transfer.pause_begin_ns,
                  "gate pause timestamps are out of order");
          gate_acquire_us.push_back(
              static_cast<double>(result.transfer.pause_complete_ns -
                                  result.transfer.pause_begin_ns) /
              1000.0);
        }
        queue_delay_us.push_back(
            static_cast<double>(result.transfer.queue_delay_ns) / 1000.0);
        const double deadline_latency_us =
            options.validation_excluded_deadline
                ? validation_excluded_end_to_end_us.back()
                : end_to_end_us.back();
        const bool deadline_miss = options.deadline_us > 0.0 &&
                                   deadline_latency_us > options.deadline_us;
        if (deadline_miss) {
          ++deadline_misses;
        }
        stage_traces.push_back(StageTrace{
            result.transfer.iteration,
            std::string(result.transfer.input_sha256.data()),
            producer_compute_us.back(),
            producer_handoff_copy_us.back(),
            producer_payload_verification_us.back(),
            transport_notification_us.back(),
            consumer_payload_verification_us.back(),
            consumer_handoff_copy_us.back(),
            edge_transport_us.back(),
            consumer_compute_us.back(),
            output_verification_us.back(),
            validation_excluded_end_to_end_us.back(),
            end_to_end_us.back(),
            deadline_miss,
        });
      }
    }
    success = success && handoff_us.size() ==
                             static_cast<std::size_t>(options.iterations);
    require(last_done_ns > first_start_ns, "invalid pipeline measurement window");
    const double elapsed_seconds =
        static_cast<double>(last_done_ns - first_start_ns) / 1.0e9;
    const double pipeline_rps =
        static_cast<double>(options.iterations) / elapsed_seconds;
    const auto request_payload_checksums = payload_checksums;
    const auto request_output_checksums = output_checksums;
    std::sort(payload_checksums.begin(), payload_checksums.end());
    const std::size_t unique_payload_checksums = static_cast<std::size_t>(
        std::distance(payload_checksums.begin(),
                      std::unique(payload_checksums.begin(),
                                  payload_checksums.end())));
    std::sort(output_checksums.begin(), output_checksums.end());
    const std::size_t unique_output_checksums = static_cast<std::size_t>(
        std::distance(output_checksums.begin(),
                      std::unique(output_checksums.begin(),
                                  output_checksums.end())));
    if (options.checksum_mode != ChecksumMode::kOff) {
      success = success &&
                (validated_requests >= 1U &&
                 (validated_requests == 1U ||
                  // Input diversity is a useful guard against accidentally
                  // replaying one request.  Output diversity is not a
                  // correctness invariant: a valid classifier may emit the
                  // same class/logit checksum for several distinct inputs.
                  unique_payload_checksums >= 2U));
    }

    if (!options.trace_csv.empty()) {
      std::ofstream trace(options.trace_csv, std::ios::out | std::ios::trunc);
      require(trace.is_open(), "failed to open trace CSV");
      trace << "request,producer_compute_us,producer_copy_us,"
               "input_sha256,"
               "producer_validation_us,notification_us,"
               "consumer_validation_us,consumer_copy_us,edge_transport_us,"
               "consumer_compute_us,output_verification_us,"
               "validation_excluded_end_to_end_us,wall_end_to_end_us,"
               "deadline_miss\n";
      trace.precision(17);
      for (const StageTrace& row : stage_traces) {
        trace << row.request << ',' << row.producer_compute_us << ','
              << row.producer_copy_us << ',' << row.input_sha256 << ','
              << row.producer_validation_us
              << ',' << row.notification_us << ','
              << row.consumer_validation_us << ',' << row.consumer_copy_us
              << ',' << row.edge_transport_us << ','
              << row.consumer_compute_us << ',' << row.output_verification_us
              << ',' << row.validation_excluded_end_to_end_us << ','
              << row.wall_end_to_end_us << ',' << (row.deadline_miss ? 1 : 0)
              << '\n';
      }
      trace.flush();
      require(trace.good(), "failed to write trace CSV");
    }

    if (!options.checksum_trace_csv.empty()) {
      std::ofstream trace(options.checksum_trace_csv,
                          std::ios::out | std::ios::trunc);
      require(trace.is_open(), "failed to open checksum trace CSV");
      trace << "request,payload_checksum,output_checksum\n";
      for (std::size_t index = 0; index < request_payload_checksums.size();
           ++index) {
        trace << (options.warmup + static_cast<int>(index)) << ','
              << request_payload_checksums[index] << ','
              << request_output_checksums[index] << '\n';
      }
      trace.flush();
      require(trace.good(), "failed to write checksum trace CSV");
    }

    if (!options.event_trace_csv.empty()) {
      std::ofstream trace(options.event_trace_csv,
                          std::ios::out | std::ios::trunc);
      require(trace.is_open(), "failed to open event trace CSV");
      trace << "request,request_id,input_sha256,declared_arrival_ns,"
               "actual_release_ns,queue_delay_us,producer_start_ns,"
               "producer_compute_done_ns,publication_ns,consumer_start_ns,"
               "consumer_compute_done_ns,completion_ns,pause_begin_ns,"
               "pause_complete_ns,resume_issued_ns,resume_observed_ns,"
               "producer_validation_done_ns,consumer_validation_done_ns,"
               "gate_hold_us,gate_acquire_us\n";
      trace.precision(17);
      for (const Result& result : collected) {
        if (result.transfer.warmup != 0U) {
          continue;
        }
        const std::uint64_t completion_ns =
            options.dependency_mode == DependencyMode::kIndependent
                ? std::max(result.transfer.producer_done_ns,
                           result.consumer_done_ns)
                : result.consumer_done_ns;
        const bool has_gate = !options.gate_pids.empty();
        const double gate_hold =
            has_gate && result.transfer.resume_observed_ns >=
                            result.transfer.pause_begin_ns
                ? static_cast<double>(result.transfer.resume_observed_ns -
                                      result.transfer.pause_begin_ns) /
                      1000.0
                : 0.0;
        const double gate_acquire =
            has_gate && result.transfer.pause_complete_ns >=
                            result.transfer.pause_begin_ns
                ? static_cast<double>(result.transfer.pause_complete_ns -
                                      result.transfer.pause_begin_ns) /
                      1000.0
                : 0.0;
        trace << result.transfer.iteration << ','
              << result.transfer.request_id.data() << ','
              << result.transfer.input_sha256.data() << ','
              << result.transfer.declared_arrival_ns << ','
              << result.transfer.actual_release_ns << ','
              << static_cast<double>(result.transfer.queue_delay_ns) / 1000.0
              << ',' << result.transfer.producer_start_ns << ','
              << result.transfer.producer_compute_done_ns << ','
              << result.transfer.publication_ns << ',' << result.consumer_start_ns
              << ',' << result.consumer_compute_done_ns << ',' << completion_ns
              << ',' << result.transfer.pause_begin_ns << ','
              << result.transfer.pause_complete_ns << ','
              << result.transfer.resume_issued_ns << ','
              << result.transfer.resume_observed_ns << ','
              << result.transfer.producer_payload_verification_done_ns << ','
              << result.consumer_payload_verification_done_ns << ','
              << gate_hold << ',' << gate_acquire << '\n';
      }
      trace.flush();
      require(trace.good(), "failed to write event trace CSV");
    }

    std::cout.precision(17);
    std::cout << "{\"schema_version\":" << kSchemaVersion
              << ",\"status\":\"" << (success ? "ok" : "error")
              << "\",\"pipeline\":\"" << pipeline_name(options.workload)
              << "\""
              << ",\"transport\":\"" << transport_name(options.transport)
              << "\""
              << ",\"transport_description\":\""
              << transport_description(options.transport) << "\""
              << ",\"producer_uuid\":\"" << options.producer_uuid
              << "\",\"consumer_uuid\":\"" << options.consumer_uuid
              << "\",\"producer_sms\":" << readiness[0].multiprocessors
              << ",\"consumer_sms\":" << readiness[1].multiprocessors
              << ",\"dependency_mode\":\""
              << dependency_mode_name(options.dependency_mode) << "\""
              << ",\"producer_quota\":" << options.producer_quota
              << ",\"consumer_quota\":" << options.consumer_quota
              << ",\"payload_bytes\":" << bytes
              << ",\"producer_output_tensor\":\""
              << producer_output(options.workload) << "\""
              << ",\"consumer_input_tensor\":\""
              << options.consumer_input_tensor << "\""
              << ",\"consumer_output_tensor\":\""
              << (options.consumer_engine.empty() ? "policy_output" : "external-output")
              << "\""
              << ",\"consumer_engine_mode\":\""
              << (options.consumer_engine.empty() ? "generated-control-policy"
                                                   : "external-trained-engine")
              << "\""
              << ",\"application_output_trace\":";
    if (options.application_output_trace.empty()) {
      std::cout << "null";
    } else {
      std::cout << "\"" << options.application_output_trace.string() << "\"";
    }
    std::cout << ",\"producer_input_trace\":";
    if (options.producer_input_trace.empty()) {
      std::cout << "null";
    } else {
      std::cout << "\"" << options.producer_input_trace.string() << "\"";
    }
    std::cout << ",\"activation_replay_trace\":";
    if (options.activation_replay_trace.empty()) {
      std::cout << "null";
    } else {
      std::cout << "\"" << options.activation_replay_trace.string() << "\"";
    }
    std::cout << ",\"arrival_trace\":";
    if (options.arrival_trace.empty()) {
      std::cout << "null";
    } else {
      std::cout << "\"" << options.arrival_trace.string() << "\"";
    }
    std::cout << ",\"event_trace_csv\":";
    if (options.event_trace_csv.empty()) {
      std::cout << "null";
    } else {
      std::cout << "\"" << options.event_trace_csv.string() << "\"";
    }
    std::cout << ",\"arrival_schedule_mode\":\""
              << (options.arrival_trace.empty() ? "legacy-unpaced"
                                                 : "operational-trace")
              << "\"";
    std::cout
              << ",\"payload_shape\":"
              << (options.workload == Workload::kResnetControl
                      ? "[1,4,23,40]"
                      : options.workload == Workload::kResnetDetectionHead
                            ? "[1,512,23,40]"
                            : options.workload == Workload::kResnet50Classification
                                  ? "[1,1024,14,14]"
                                  : "[1,1500,384]")
              << ",\"dependency_edge\":{\"present\":"
              << (options.dependency_mode == DependencyMode::kDependent
                      ? "true"
                      : "false")
              << ",\"payload_bytes\":" << bytes
              << ",\"transport\":\""
              << transport_name(options.transport) << "\"}"
              << ",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations
              << ",\"checksum_failures\":" << checksum_failures
              << ",\"validated_requests\":" << validated_requests
              << ",\"checksum_mode\":\""
              << checksum_mode_name(options.checksum_mode) << "\""
              << ",\"checksum_sample_period\":"
              << options.checksum_sample_period
              << ",\"validation_delay_us\":"
              << options.validation_delay_us
              << ",\"correctness_validated\":"
              << (options.checksum_mode == ChecksumMode::kInline ? "true" : "false")
              << ",\"correctness_scope\":\""
              << (options.dependency_mode == DependencyMode::kDependent
                      ? "producer-output-consumer-input-equality"
                      : "producer-activation-replay-output-oracle")
              << "\""
              << ",\"activation_replay_verified_requests\":"
              << (options.activation_replay_trace.empty()
                      ? 0
                      : options.warmup + options.iterations)
              << ",\"deadline_us\":";
    if (options.deadline_us > 0.0) {
      std::cout << options.deadline_us;
    } else {
      std::cout << "null";
    }
    std::cout << ",\"deadline_misses\":" << deadline_misses
              << ",\"deadline_mode\":\""
              << (options.validation_excluded_deadline ? "validation-excluded"
                                                       : "wall")
              << "\""
              << ",\"gated_processes\":" << options.gate_pids.size()
              << ",\"gate_mode\":\""
              << (options.gate_mode == GateMode::kCooperative ? "cooperative"
                                                               : "stop")
              << "\""
              << ",\"gate_scope\":\""
              << (options.gate_scope == GateScope::kProducer
                      ? "producer"
                      : options.gate_scope == GateScope::kConsumer ? "consumer"
                                                                    : "pipeline")
              << "\""
              << ",\"latency_contract\":\"production-wall-arrival-to-completion\""
              << ",\"production_wall_definition\":\"arrival-to-consumer-completion-excludes-correctness-validation\""
              << ",\"correctness_validation_placement\":\"post-completion\""
              << ",\"measurement_start_monotonic_ns\":" << first_start_ns
              << ",\"measurement_end_monotonic_ns\":" << last_done_ns
              << ",\"elapsed_seconds\":" << elapsed_seconds
              << ",\"pipeline_rps\":" << pipeline_rps
              << ",\"unique_payload_checksums\":"
              << unique_payload_checksums
              << ",\"unique_policy_output_checksums\":"
              << unique_output_checksums
              << ",\"orion\":{"
              << "\"enabled\":"
              << (options.orion_profile_aware ? "true" : "false")
              << ",\"background_completed\":"
              << orion_evidence->background_completed
              << ",\"measured_background_completed\":"
              << orion_evidence->measured_background_completed
              << ",\"measurement_start_monotonic_ns\":"
              << orion_evidence->measurement_start_ns
              << ",\"measurement_end_monotonic_ns\":"
              << orion_evidence->measurement_end_ns
              << ",\"measured_background_goodput_rps\":";
    if (options.orion_profile_aware &&
        orion_evidence->measurement_end_ns >
            orion_evidence->measurement_start_ns) {
      std::cout << static_cast<double>(
                       orion_evidence->measured_background_completed) /
                       (static_cast<double>(orion_evidence->measurement_end_ns -
                                            orion_evidence->measurement_start_ns) /
                        1.0e9);
    } else {
      std::cout << "null";
    }
    std::cout
              << ",\"status\":" << orion_evidence->status
              << ",\"background_period_us\":"
              << options.orion_background_period_us
              << ",\"scheduler\":{"
              << "\"arrivals\":" << orion_evidence->scheduler.arrivals
              << ",\"decisions\":" << orion_evidence->scheduler.decisions
              << ",\"reordered_decisions\":"
              << orion_evidence->scheduler.reordered_decisions
              << ",\"high_priority_decisions\":"
              << orion_evidence->scheduler.high_priority_decisions
              << ",\"profiled_best_effort_admissions\":"
              << orion_evidence->scheduler.profiled_best_effort_admissions
              << ",\"complementary_admissions\":"
              << orion_evidence->scheduler.complementary_admissions
              << ",\"profile_blocked_polls\":"
              << orion_evidence->scheduler.profile_blocked_polls
              << ",\"trace_records\":"
              << orion_evidence->scheduler.trace_records << "}}";
    if (!handoff_us.empty()) {
      std::cout << ",\"handoff_us\":{\"p50\":"
                << percentile(handoff_us, 0.50) << ",\"p95\":"
                << percentile(handoff_us, 0.95) << ",\"p99\":"
                << percentile(handoff_us, 0.99) << ",\"max\":"
                << *std::max_element(handoff_us.begin(), handoff_us.end())
                << "},\"end_to_end_us\":{\"p50\":"
                << percentile(end_to_end_us, 0.50) << ",\"p95\":"
                << percentile(end_to_end_us, 0.95) << ",\"p99\":"
                << percentile(end_to_end_us, 0.99) << ",\"max\":"
                << *std::max_element(end_to_end_us.begin(),
                                    end_to_end_us.end())
                << "},\"stage_latency_us\":{"
                << "\"producer_compute_p50\":"
                << percentile(producer_compute_us, 0.50)
                << ",\"producer_compute_p99\":"
                << percentile(producer_compute_us, 0.99)
                << ",\"transport_ready_p50\":"
                << percentile(transport_ready_us, 0.50)
                << ",\"transport_ready_p99\":"
                << percentile(transport_ready_us, 0.99)
                << ",\"consumer_compute_p50\":"
                << percentile(consumer_compute_us, 0.50)
                << ",\"consumer_compute_p99\":"
                << percentile(consumer_compute_us, 0.99)
                << ",\"output_verification_p50\":"
                << percentile(output_verification_us, 0.50)
                << ",\"output_verification_p99\":"
                << percentile(output_verification_us, 0.99)
                << ",\"producer_payload_verification_p50\":"
                << percentile(producer_payload_verification_us, 0.50)
                << ",\"producer_payload_verification_p99\":"
                << percentile(producer_payload_verification_us, 0.99)
                << ",\"producer_handoff_copy_p50\":"
                << percentile(producer_handoff_copy_us, 0.50)
                << ",\"producer_handoff_copy_p99\":"
                << percentile(producer_handoff_copy_us, 0.99)
                << ",\"transport_notification_p50\":"
                << percentile(transport_notification_us, 0.50)
                << ",\"transport_notification_p99\":"
                << percentile(transport_notification_us, 0.99)
                << ",\"consumer_payload_verification_p50\":"
                << percentile(consumer_payload_verification_us, 0.50)
                << ",\"consumer_payload_verification_p99\":"
                << percentile(consumer_payload_verification_us, 0.99)
                << ",\"consumer_handoff_copy_p50\":"
                << percentile(consumer_handoff_copy_us, 0.50)
                << ",\"consumer_handoff_copy_p99\":"
                << percentile(consumer_handoff_copy_us, 0.99)
                << ",\"edge_transport_p50\":"
                << percentile(edge_transport_us, 0.50)
                << ",\"edge_transport_p99\":"
                << percentile(edge_transport_us, 0.99)
                << ",\"validation_excluded_end_to_end_p50\":"
                << percentile(validation_excluded_end_to_end_us, 0.50)
                << ",\"validation_excluded_end_to_end_p99\":"
                << percentile(validation_excluded_end_to_end_us, 0.99) << '}';
      if (!gate_us.empty()) {
        std::cout << ",\"gate_us\":{\"p50\":" << percentile(gate_us, 0.50)
                  << ",\"p99\":" << percentile(gate_us, 0.99)
                  << ",\"max\":"
                  << *std::max_element(gate_us.begin(), gate_us.end()) << '}';
      }
      if (!gate_acquire_us.empty()) {
        std::cout << ",\"gate_acquire_us\":{\"p50\":"
                  << percentile(gate_acquire_us, 0.50) << ",\"p99\":"
                  << percentile(gate_acquire_us, 0.99) << ",\"max\":"
                  << *std::max_element(gate_acquire_us.begin(),
                                      gate_acquire_us.end()) << '}';
      }
      if (!queue_delay_us.empty()) {
        std::cout << ",\"queue_delay_us\":{\"p50\":"
                  << percentile(queue_delay_us, 0.50) << ",\"p99\":"
                  << percentile(queue_delay_us, 0.99) << ",\"max\":"
                  << *std::max_element(queue_delay_us.begin(),
                                      queue_delay_us.end()) << '}';
      }
    }
    std::cout << "}\n";
    static_cast<void>(munmap(orion_evidence, sizeof(OrionEvidence)));
    return success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema_version\":" << kSchemaVersion
              << ",\"status\":\"error\",\"message\":\"" << error.what()
              << "\"}\n";
    return 1;
  }
}
