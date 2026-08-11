#include "scheduler.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <cmath>
#include <fstream>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <mutex>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace orion::thor {
namespace {

constexpr std::size_t kClients = 2U;
constexpr std::size_t kTraceBufferBytes = 1U << 20U;

struct OperationProfile {
  std::string api;
  std::array<unsigned int, 3U> grid{};
  std::array<unsigned int, 3U> block{};
  unsigned int shared_mem_bytes{};
  int resource_class{-1};
  int sm_used{};
  double duration_us{};
};

struct LaunchRequest {
  const char* api{};
  LaunchEx real{};
  CUlaunchConfig config{};
  std::vector<CUlaunchAttribute> attributes;
  CUfunction function{};
  CUcontext context{};
  void** kernel_params{};
  void** extra{};
  std::uint64_t arrival_sequence{};
  int client_id{-1};
  bool high_priority{};
  bool complete{};
  CUresult result{CUDA_ERROR_UNKNOWN};
  std::size_t profile_position{};
  OperationProfile operation_profile;
  std::string admission_reason{"unprofiled"};
  int active_sm_at_admission{};
  double active_be_duration_us_at_admission{};
  bool high_priority_active_at_admission{};
  cudaEvent_t completion_event{};
  std::condition_variable completed;
};

struct ActiveOperation {
  cudaEvent_t completion_event{};
  CUcontext context{};
  int client_id{-1};
  OperationProfile profile;
};

struct State {
  std::mutex mutex;
  std::condition_variable ready;
  std::array<std::deque<LaunchRequest*>, kClients> queues;
  std::thread scheduler;
  std::atomic<std::uint64_t> next_arrival{0};
  SchedulerStats stats;
  bool running{};
  bool stopping{};
  int initial_gate_clients{};
  int trace_fd{-1};
  int profile_fd{-1};
  int device_sms{};
  std::array<std::uint64_t, kClients> client_operation_indices{};
  std::array<std::uint64_t, kClients> client_submission_indices{};
  std::array<std::vector<OperationProfile>, kClients> profiles;
  std::vector<ActiveOperation> active;
  double max_be_duration_us{};
  bool profiled{};
  bool event_trace{};
  std::vector<char> trace_buffer;
};

State g_state;
thread_local int g_client_id = -1;
thread_local bool g_high_priority = false;

class CurrentContext {
 public:
  explicit CurrentContext(const CUcontext context) {
    result_ = cuCtxPushCurrent(context);
    active_ = result_ == CUDA_SUCCESS;
  }
  CurrentContext(const CurrentContext&) = delete;
  CurrentContext& operator=(const CurrentContext&) = delete;
  ~CurrentContext() {
    if (active_) {
      CUcontext popped = nullptr;
      static_cast<void>(cuCtxPopCurrent(&popped));
    }
  }
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] CUresult result() const noexcept { return result_; }

 private:
  CUresult result_{CUDA_ERROR_UNKNOWN};
  bool active_{};
};

[[nodiscard]] std::uint64_t monotonic_ns() noexcept {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

void write_all(const int fd, const char* data, const std::size_t bytes) {
  std::size_t offset = 0U;
  while (offset < bytes) {
    const ssize_t result = write(fd, data + offset, bytes - offset);
    if (result > 0) {
      offset += static_cast<std::size_t>(result);
    } else if (result < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
}

void flush_trace_buffer() {
  if (g_state.trace_fd >= 0 && !g_state.trace_buffer.empty()) {
    write_all(g_state.trace_fd, g_state.trace_buffer.data(),
              g_state.trace_buffer.size());
    g_state.trace_buffer.clear();
  }
}

void buffer_trace(const char* data, const std::size_t bytes) {
  if (g_state.trace_buffer.size() + bytes > kTraceBufferBytes) {
    flush_trace_buffer();
  }
  g_state.trace_buffer.insert(g_state.trace_buffer.end(), data, data + bytes);
}

[[nodiscard]] std::size_t nonempty_clients_locked() {
  std::size_t count = 0U;
  for (const auto& queue : g_state.queues) {
    count += queue.empty() ? 0U : 1U;
  }
  return count;
}

[[nodiscard]] bool has_work_locked() {
  return nonempty_clients_locked() != 0U;
}

[[nodiscard]] std::vector<OperationProfile> load_profile(
    const char* path, const int device_sms) {
  if (path == nullptr || path[0] == '\0' || device_sms <= 0) {
    throw std::invalid_argument("missing Orion profile path or GPU width");
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open Orion scheduler profile");
  }
  std::string line;
  if (!std::getline(input, line) || line != "orion-thor-profile-v1" ||
      !std::getline(input, line) ||
      line != "position\tapi\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\t"
              "block_z\tshared_mem_bytes\tprofile\tsm_used\tduration_us") {
    throw std::runtime_error("invalid Orion scheduler profile header");
  }
  std::vector<OperationProfile> result;
  while (std::getline(input, line)) {
    if (line.empty()) {
      throw std::runtime_error("empty Orion scheduler profile row");
    }
    std::array<std::string_view, 12U> fields;
    std::string_view remaining(line);
    for (std::size_t index = 0U; index < fields.size(); ++index) {
      const std::size_t separator = remaining.find('\t');
      if (index + 1U == fields.size()) {
        if (separator != std::string_view::npos) {
          throw std::runtime_error("invalid Orion scheduler profile row");
        }
        fields[index] = remaining;
      } else {
        if (separator == std::string_view::npos) {
          throw std::runtime_error("invalid Orion scheduler profile row");
        }
        fields[index] = remaining.substr(0U, separator);
        remaining.remove_prefix(separator + 1U);
      }
    }
    std::size_t position = 0U;
    OperationProfile profile;
    const auto parse = [](const std::string_view field, auto& output) {
      const auto parsed = std::from_chars(field.data(),
                                          field.data() + field.size(), output);
      return parsed.ec == std::errc{} &&
             parsed.ptr == field.data() + field.size();
    };
    profile.api = fields[1];
    if (!parse(fields[0], position) || profile.api != "cuLaunchKernelEx" ||
        !parse(fields[2], profile.grid[0]) ||
        !parse(fields[3], profile.grid[1]) ||
        !parse(fields[4], profile.grid[2]) ||
        !parse(fields[5], profile.block[0]) ||
        !parse(fields[6], profile.block[1]) ||
        !parse(fields[7], profile.block[2]) ||
        !parse(fields[8], profile.shared_mem_bytes) ||
        !parse(fields[9], profile.resource_class) ||
        !parse(fields[10], profile.sm_used) ||
        !parse(fields[11], profile.duration_us) || position != result.size() ||
        std::any_of(profile.grid.begin(), profile.grid.end(),
                    [](const unsigned int value) { return value == 0U; }) ||
        std::any_of(profile.block.begin(), profile.block.end(),
                    [](const unsigned int value) { return value == 0U; }) ||
        profile.resource_class < -1 || profile.resource_class > 1 ||
        profile.sm_used <= 0 || profile.sm_used > device_sms ||
        !std::isfinite(profile.duration_us) || profile.duration_us <= 0.0) {
      throw std::runtime_error("invalid Orion scheduler profile row");
    }
    result.push_back(profile);
  }
  if (!input.eof() || result.empty()) {
    throw std::runtime_error("empty or unreadable Orion scheduler profile");
  }
  return result;
}

void reap_active_locked() {
  auto current = g_state.active.begin();
  while (current != g_state.active.end()) {
    CurrentContext context(current->context);
    if (!context.active()) {
      ++current;
      continue;
    }
    const cudaError_t status = cudaEventQuery(current->completion_event);
    if (status == cudaSuccess) {
      static_cast<void>(cudaEventDestroy(current->completion_event));
      current = g_state.active.erase(current);
    } else {
      static_cast<void>(cudaGetLastError());
      ++current;
    }
  }
}

[[nodiscard]] int active_sm_locked() {
  int total = 0;
  for (const auto& operation : g_state.active) {
    total += operation.profile.sm_used;
  }
  return total;
}

[[nodiscard]] double active_be_duration_locked() {
  double total = 0.0;
  for (const auto& operation : g_state.active) {
    if (operation.client_id == 0) {
      total += operation.profile.duration_us;
    }
  }
  return total;
}

[[nodiscard]] int active_be_sm_locked() {
  int total = 0;
  for (const auto& operation : g_state.active) {
    if (operation.client_id == 0) {
      total += operation.profile.sm_used;
    }
  }
  return total;
}

[[nodiscard]] const OperationProfile* active_hp_profile_locked() {
  for (auto current = g_state.active.rbegin(); current != g_state.active.rend();
       ++current) {
    if (current->client_id == 1) {
      return &current->profile;
    }
  }
  return nullptr;
}

void annotate_admission_locked(LaunchRequest& request,
                               const char* reason) {
  request.admission_reason = reason;
  request.active_sm_at_admission = active_sm_locked();
  request.active_be_duration_us_at_admission = active_be_duration_locked();
  request.high_priority_active_at_admission =
      active_hp_profile_locked() != nullptr;
}

[[nodiscard]] LaunchRequest* select_locked() {
  // Orion assigns the final client as HP and serves it first whenever it has
  // an operation ready. With two clients, index 1 is therefore the HP queue.
  if (!g_state.queues[1].empty()) {
    LaunchRequest* request = g_state.queues[1].front();
    g_state.queues[1].pop_front();
    annotate_admission_locked(*request,
                              g_state.profiled ? "high-priority" : "unprofiled");
    return request;
  }
  if (!g_state.queues[0].empty()) {
    LaunchRequest* request = g_state.queues[0].front();
    if (g_state.profiled) {
      const OperationProfile* hp = active_hp_profile_locked();
      if (hp != nullptr) {
        const int occupied_be_sms = active_be_sm_locked();
        const double be_duration = active_be_duration_locked();
        const bool complementary =
            hp->resource_class == -1 ||
            request->operation_profile.resource_class == -1 ||
            hp->resource_class != request->operation_profile.resource_class;
        if (be_duration > g_state.max_be_duration_us || !complementary ||
            occupied_be_sms + request->operation_profile.sm_used >
                g_state.device_sms) {
          ++g_state.stats.profile_blocked_polls;
          return nullptr;
        }
        annotate_admission_locked(*request, "complementary-with-high-priority");
        ++g_state.stats.complementary_admissions;
      } else {
        annotate_admission_locked(*request, "no-active-high-priority");
      }
      ++g_state.stats.profiled_best_effort_admissions;
    } else {
      annotate_admission_locked(*request, "unprofiled");
    }
    g_state.queues[0].pop_front();
    return request;
  }
  return nullptr;
}

[[nodiscard]] std::uint64_t oldest_arrival_locked() {
  std::uint64_t oldest = UINT64_MAX;
  for (const auto& queue : g_state.queues) {
    if (!queue.empty() && queue.front()->arrival_sequence < oldest) {
      oldest = queue.front()->arrival_sequence;
    }
  }
  return oldest;
}

void trace_decision(const LaunchRequest& request,
                    const std::uint64_t decision_sequence,
                    const bool reordered, const std::uint64_t start_ns,
                    const std::uint64_t end_ns) {
  if (g_state.trace_fd < 0) {
    return;
  }
  if (g_state.event_trace && !reordered &&
      request.admission_reason != "complementary-with-high-priority") {
    return;
  }
  char line[768];
  const int length = std::snprintf(
      line, sizeof(line),
      "{\"schema_version\":1,\"decision_sequence\":%llu,"
      "\"arrival_sequence\":%llu,\"client_id\":%d,"
      "\"priority\":\"%s\",\"api\":\"%s\",\"reordered\":%s,"
      "\"profile_position\":%zu,\"resource_class\":%d,"
      "\"sm_used\":%d,\"profile_duration_us\":%.9g,"
      "\"admission_reason\":\"%s\",\"active_sm_at_admission\":%d,"
      "\"active_be_duration_us_at_admission\":%.9g,"
      "\"high_priority_active_at_admission\":%s,"
      "\"initial_gate_clients\":%d,\"start_monotonic_ns\":%llu,"
      "\"end_monotonic_ns\":%llu,\"result\":%d}\n",
      static_cast<unsigned long long>(decision_sequence),
      static_cast<unsigned long long>(request.arrival_sequence),
      request.client_id, request.high_priority ? "high" : "best-effort",
      request.api, reordered ? "true" : "false",
      request.profile_position, request.operation_profile.resource_class,
      request.operation_profile.sm_used, request.operation_profile.duration_us,
      request.admission_reason.c_str(), request.active_sm_at_admission,
      request.active_be_duration_us_at_admission,
      request.high_priority_active_at_admission ? "true" : "false",
      g_state.initial_gate_clients,
      static_cast<unsigned long long>(start_ns),
      static_cast<unsigned long long>(end_ns),
      static_cast<int>(request.result));
  if (length > 0 && static_cast<std::size_t>(length) < sizeof(line)) {
    buffer_trace(line, static_cast<std::size_t>(length));
    ++g_state.stats.trace_records;
  }
}

void trace_profile(const LaunchRequest& request, const std::uint64_t operation,
                   const float duration_us, const int active_blocks_per_sm,
                   const int estimated_sms) {
  if (g_state.profile_fd < 0) {
    return;
  }
  const std::uint64_t grid_blocks =
      static_cast<std::uint64_t>(request.config.gridDimX) *
      request.config.gridDimY * request.config.gridDimZ;
  const std::uint64_t block_threads =
      static_cast<std::uint64_t>(request.config.blockDimX) *
      request.config.blockDimY * request.config.blockDimZ;
  char line[768];
  const int length = std::snprintf(
      line, sizeof(line),
      "{\"schema_version\":1,\"client_id\":%d,"
      "\"operation_index\":%llu,\"api\":\"%s\","
      "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
      "\"grid_blocks\":%llu,\"block_threads\":%llu,"
      "\"shared_mem_bytes\":%u,\"active_blocks_per_sm\":%d,"
      "\"device_sms\":%d,\"estimated_sms\":%d,"
      "\"kernel_duration_us\":%.9g}\n",
      request.client_id, static_cast<unsigned long long>(operation),
      request.api, request.config.gridDimX, request.config.gridDimY,
      request.config.gridDimZ, request.config.blockDimX,
      request.config.blockDimY, request.config.blockDimZ,
      static_cast<unsigned long long>(grid_blocks),
      static_cast<unsigned long long>(block_threads),
      request.config.sharedMemBytes, active_blocks_per_sm, g_state.device_sms,
      estimated_sms, static_cast<double>(duration_us));
  if (length > 0 && static_cast<std::size_t>(length) < sizeof(line)) {
    write_all(g_state.profile_fd, line, static_cast<std::size_t>(length));
  }
}

[[nodiscard]] CUresult execute_request(LaunchRequest& request) {
  CurrentContext context(request.context);
  if (!context.active()) {
    return context.result();
  }
  if (g_state.profile_fd < 0) {
    if (g_state.profiled &&
        cudaEventCreateWithFlags(&request.completion_event,
                                 cudaEventDisableTiming) != cudaSuccess) {
      return CUDA_ERROR_UNKNOWN;
    }
    const CUresult result = request.real(&request.config, request.function,
                                         request.kernel_params, request.extra);
    if (g_state.profiled &&
        (result != CUDA_SUCCESS ||
         cudaEventRecord(request.completion_event,
                         reinterpret_cast<cudaStream_t>(
                             request.config.hStream)) != cudaSuccess)) {
      static_cast<void>(cudaEventDestroy(request.completion_event));
      request.completion_event = nullptr;
      return result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result;
    }
    return result;
  }

  cudaEvent_t begin{};
  cudaEvent_t end{};
  if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
    if (begin != nullptr) {
      static_cast<void>(cudaEventDestroy(begin));
    }
    if (end != nullptr) {
      static_cast<void>(cudaEventDestroy(end));
    }
    return CUDA_ERROR_UNKNOWN;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(request.config.hStream);
  CUresult result = CUDA_ERROR_UNKNOWN;
  float duration_us = 0.0F;
  if (cudaEventRecord(begin, stream) == cudaSuccess) {
    result = request.real(&request.config, request.function,
                          request.kernel_params, request.extra);
    if (result == CUDA_SUCCESS && cudaEventRecord(end, stream) == cudaSuccess &&
        cudaEventSynchronize(end) == cudaSuccess) {
      float duration_ms = 0.0F;
      if (cudaEventElapsedTime(&duration_ms, begin, end) == cudaSuccess) {
        duration_us = duration_ms * 1000.0F;
      }
    }
  }

  int active_blocks = 0;
  const int threads = static_cast<int>(
      request.config.blockDimX * request.config.blockDimY *
      request.config.blockDimZ);
  if (threads > 0) {
    static_cast<void>(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, request.function, threads,
        request.config.sharedMemBytes));
  }
  const std::uint64_t grid_blocks =
      static_cast<std::uint64_t>(request.config.gridDimX) *
      request.config.gridDimY * request.config.gridDimZ;
  int estimated_sms = g_state.device_sms;
  if (active_blocks > 0 && g_state.device_sms > 0) {
    const auto required = static_cast<int>(
        (grid_blocks + static_cast<std::uint64_t>(active_blocks) - 1U) /
        static_cast<std::uint64_t>(active_blocks));
    estimated_sms = std::min(required, g_state.device_sms);
  }
  const auto operation =
      g_state.client_operation_indices[static_cast<std::size_t>(request.client_id)]++;
  trace_profile(request, operation, duration_us, active_blocks, estimated_sms);
  static_cast<void>(cudaEventDestroy(end));
  static_cast<void>(cudaEventDestroy(begin));
  return result;
}

void schedule_loop() {
  std::unique_lock<std::mutex> lock(g_state.mutex);
  bool initial_gate_open = g_state.initial_gate_clients == 0;
  while (true) {
    g_state.ready.wait_for(lock, std::chrono::microseconds(50), [&] {
      if (g_state.stopping) {
        return true;
      }
      if (!initial_gate_open) {
        return nonempty_clients_locked() >=
               static_cast<std::size_t>(g_state.initial_gate_clients);
      }
      return has_work_locked();
    });
    if (g_state.stopping && !has_work_locked()) {
      break;
    }
    if (!initial_gate_open) {
      if (nonempty_clients_locked() <
          static_cast<std::size_t>(g_state.initial_gate_clients)) {
        continue;
      }
      initial_gate_open = true;
    }
    if (g_state.profiled) {
      reap_active_locked();
    }
    const std::uint64_t oldest = oldest_arrival_locked();
    LaunchRequest* request = select_locked();
    if (request == nullptr) {
      // A queued BE operation can be temporarily inadmissible while an HP or
      // long BE event is in flight. Avoid turning the queue-presence predicate
      // into a CPU spin; a new HP arrival still wakes this timed wait early.
      g_state.ready.wait_for(lock, std::chrono::microseconds(50));
      continue;
    }
    const std::uint64_t decision = g_state.stats.decisions++;
    const bool reordered = request->arrival_sequence != oldest;
    g_state.stats.reordered_decisions += reordered ? 1U : 0U;
    g_state.stats.high_priority_decisions += request->high_priority ? 1U : 0U;
    lock.unlock();

    const std::uint64_t start_ns = monotonic_ns();
    request->result = execute_request(*request);
    const std::uint64_t end_ns = monotonic_ns();
    trace_decision(*request, decision, reordered, start_ns, end_ns);

    lock.lock();
    if (request->completion_event != nullptr) {
      g_state.active.push_back(ActiveOperation{
          request->completion_event, request->context, request->client_id,
          request->operation_profile});
      request->completion_event = nullptr;
    }
    request->complete = true;
    request->completed.notify_one();
  }
}

}  // namespace

CUresult submit_launch_ex(const char* api, const LaunchEx real,
                          const CUlaunchConfig* config,
                          const CUfunction function, void** kernel_params,
                          void** extra) {
  if (real == nullptr || config == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::unique_lock<std::mutex> lock(g_state.mutex);
  if (!g_state.running || g_client_id < 0 ||
      g_client_id >= static_cast<int>(kClients)) {
    lock.unlock();
    return real(config, function, kernel_params, extra);
  }

  LaunchRequest request;
  request.api = api;
  request.real = real;
  request.config = *config;
  request.function = function;
  if (cuCtxGetCurrent(&request.context) != CUDA_SUCCESS ||
      request.context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  request.kernel_params = kernel_params;
  request.extra = extra;
  request.client_id = g_client_id;
  request.high_priority = g_high_priority;
  request.arrival_sequence =
      g_state.next_arrival.fetch_add(1, std::memory_order_relaxed);
  if (g_state.profiled) {
    const auto client = static_cast<std::size_t>(g_client_id);
    request.profile_position = static_cast<std::size_t>(
        g_state.client_submission_indices[client]++ %
        g_state.profiles[client].size());
    request.operation_profile = g_state.profiles[client][request.profile_position];
    const auto& profile = request.operation_profile;
    if (profile.api != api ||
        profile.grid != std::array<unsigned int, 3U>{
                            config->gridDimX, config->gridDimY,
                            config->gridDimZ} ||
        profile.block != std::array<unsigned int, 3U>{
                             config->blockDimX, config->blockDimY,
                             config->blockDimZ} ||
        profile.shared_mem_bytes != config->sharedMemBytes) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  if (config->numAttrs > 0U && config->attrs != nullptr) {
    request.attributes.assign(config->attrs, config->attrs + config->numAttrs);
    request.config.attrs = request.attributes.data();
  }
  g_state.queues[static_cast<std::size_t>(g_client_id)].push_back(&request);
  ++g_state.stats.arrivals;
  g_state.ready.notify_one();
  request.completed.wait(lock, [&] { return request.complete; });
  return request.result;
}

}  // namespace orion::thor

extern "C" int orion_trt_scheduler_start(const char* decision_trace,
                                           const int initial_gate_clients) {
  using namespace orion::thor;
  if (initial_gate_clients < 0 || initial_gate_clients > 2) {
    return EINVAL;
  }
  std::lock_guard<std::mutex> lock(g_state.mutex);
  if (g_state.running) {
    return EALREADY;
  }
  if (decision_trace == nullptr || decision_trace[0] == '\0') {
    return EINVAL;
  }
  g_state.trace_fd =
      open(decision_trace, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (g_state.trace_fd < 0) {
    return errno;
  }
  g_state.initial_gate_clients = initial_gate_clients;
  g_state.profiled = false;
  g_state.event_trace = false;
  g_state.trace_buffer.clear();
  g_state.trace_buffer.reserve(kTraceBufferBytes);
  g_state.profiles = {};
  g_state.max_be_duration_us = 0.0;
  if (const char* profile_trace = std::getenv("ORION_TRT_PROFILE_TRACE");
      profile_trace != nullptr && profile_trace[0] != '\0') {
    g_state.profile_fd =
        open(profile_trace, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (g_state.profile_fd < 0) {
      const int error = errno;
      static_cast<void>(close(g_state.trace_fd));
      g_state.trace_fd = -1;
      return error;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&g_state.device_sms,
                               cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess) {
      static_cast<void>(close(g_state.profile_fd));
      static_cast<void>(close(g_state.trace_fd));
      g_state.profile_fd = -1;
      g_state.trace_fd = -1;
      return EIO;
    }
  }
  g_state.stopping = false;
  g_state.stats = {};
  g_state.client_operation_indices = {};
  g_state.client_submission_indices = {};
  g_state.next_arrival.store(0, std::memory_order_relaxed);
  g_state.running = true;
  g_state.scheduler = std::thread(schedule_loop);
  return 0;
}

extern "C" int orion_trt_scheduler_start_profiled(
    const char* decision_trace, const char* best_effort_profile,
    const char* high_priority_profile, const double max_be_duration_us) {
  using namespace orion::thor;
  if (decision_trace == nullptr || decision_trace[0] == '\0' ||
      !std::isfinite(max_be_duration_us) || max_be_duration_us <= 0.0) {
    return EINVAL;
  }
  int device = 0;
  int device_sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      device_sms <= 0) {
    return EIO;
  }
  std::array<std::vector<OperationProfile>, kClients> profiles;
  try {
    profiles[0] = load_profile(best_effort_profile, device_sms);
    profiles[1] = load_profile(high_priority_profile, device_sms);
  } catch (const std::exception&) {
    return EINVAL;
  }
  std::lock_guard<std::mutex> lock(g_state.mutex);
  if (g_state.running) {
    return EALREADY;
  }
  g_state.trace_fd =
      open(decision_trace, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (g_state.trace_fd < 0) {
    return errno;
  }
  g_state.initial_gate_clients = 0;
  g_state.profiled = true;
  const char* trace_mode = std::getenv("ORION_TRT_TRACE_MODE");
  if (trace_mode == nullptr || std::strcmp(trace_mode, "full") == 0) {
    g_state.event_trace = false;
  } else if (std::strcmp(trace_mode, "events") == 0) {
    g_state.event_trace = true;
  } else {
    static_cast<void>(close(g_state.trace_fd));
    g_state.trace_fd = -1;
    return EINVAL;
  }
  g_state.trace_buffer.clear();
  g_state.trace_buffer.reserve(kTraceBufferBytes);
  g_state.profiles = std::move(profiles);
  g_state.max_be_duration_us = max_be_duration_us;
  g_state.device_sms = device_sms;
  g_state.stopping = false;
  g_state.stats = {};
  g_state.client_operation_indices = {};
  g_state.client_submission_indices = {};
  g_state.next_arrival.store(0, std::memory_order_relaxed);
  g_state.active.clear();
  g_state.running = true;
  g_state.scheduler = std::thread(schedule_loop);
  return 0;
}

extern "C" int orion_trt_register_client(const int client_id,
                                           const int high_priority) {
  using namespace orion::thor;
  if (client_id < 0 || client_id >= 2 ||
      (high_priority != 0 && high_priority != 1) ||
      client_id != high_priority) {
    return EINVAL;
  }
  g_client_id = client_id;
  g_high_priority = high_priority != 0;
  return 0;
}

extern "C" int orion_trt_scheduler_stop() {
  using namespace orion::thor;
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    if (!g_state.running) {
      return EINVAL;
    }
    g_state.stopping = true;
    g_state.ready.notify_all();
  }
  g_state.scheduler.join();
  std::lock_guard<std::mutex> lock(g_state.mutex);
  for (const auto& operation : g_state.active) {
    CurrentContext context(operation.context);
    if (context.active()) {
      static_cast<void>(cudaEventDestroy(operation.completion_event));
    }
  }
  g_state.active.clear();
  g_state.running = false;
  flush_trace_buffer();
  const int close_result = close(g_state.trace_fd);
  g_state.trace_fd = -1;
  int profile_close_result = 0;
  if (g_state.profile_fd >= 0) {
    profile_close_result = close(g_state.profile_fd);
    g_state.profile_fd = -1;
  }
  return close_result == 0 && profile_close_result == 0 ? 0 : errno;
}

extern "C" int orion_trt_scheduler_stats(
    orion::thor::SchedulerStats* output) {
  using namespace orion::thor;
  if (output == nullptr) {
    return EINVAL;
  }
  std::lock_guard<std::mutex> lock(g_state.mutex);
  *output = g_state.stats;
  return 0;
}
