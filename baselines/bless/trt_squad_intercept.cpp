#include "trt_squad_intercept.hpp"

#include <cuda.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <string_view>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#undef cuGetProcAddress

namespace {

constexpr std::size_t kReplicas = 4U;
constexpr unsigned int kRestrictedSms = 2U;
constexpr unsigned int kUnrestrictedSms = 8U;
constexpr std::uint64_t kSquadKernels = 6U;
constexpr std::uint64_t kRestrictedKernels = 3U;

using LaunchEx = CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**);
using Launch = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, CUstream, void**, void**);
using GetProcV2 = CUresult (*)(const char*, void**, int, cuuint64_t,
                              CUdriverProcAddressQueryResult*);
using GetProcLegacy = CUresult (*)(const char*, void**, int, cuuint64_t);
using Dlsym = void* (*)(void*, const char*);

enum class Api { kLaunchEx, kLaunch };

struct Replica {
  CUcontext context{};
  unsigned int sms{};
  CUdeviceptr activation{};
  std::size_t activation_bytes{};
};

struct Request {
  bool present{};
  Api api{};
  const char* api_name{};
  CUcontext context{};
  CUfunction function{};
  CUlaunchConfig launch_ex{};
  unsigned int grid_x{};
  unsigned int grid_y{};
  unsigned int grid_z{};
  unsigned int block_x{};
  unsigned int block_y{};
  unsigned int block_z{};
  unsigned int shared_bytes{};
  CUstream stream{};
  void** kernel_params{};
  void** extra{};
  LaunchEx real_ex{};
  Launch real{};
};

std::array<Replica, kReplicas> g_replicas{};
std::array<Request, kReplicas> g_requests{};
std::size_t g_registered = 0U;
std::size_t g_arrived = 0U;
std::uint64_t g_generation = 0U;
std::uint64_t g_operation = 0U;
CUresult g_generation_result = CUDA_SUCCESS;
CUcontext g_previous_context = nullptr;
CUstream g_previous_stream = nullptr;
std::atomic<bool> g_active{false};
std::mutex g_mutex;
std::condition_variable g_condition;
bless::thor::SquadStats g_stats{};
int g_trace_fd = -1;
thread_local bool g_internal = false;

std::atomic<LaunchEx> g_launch_ex{nullptr};
std::atomic<LaunchEx> g_launch_ex_ptsz{nullptr};
std::atomic<Launch> g_launch{nullptr};
std::atomic<Launch> g_launch_ptsz{nullptr};

[[nodiscard]] std::uint64_t monotonic_ns() noexcept {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0U;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

[[nodiscard]] long thread_id() noexcept {
  return static_cast<long>(syscall(SYS_gettid));
}

template <typename Function>
[[nodiscard]] Function resolve_next(const char* symbol) noexcept {
  static const Dlsym system_dlsym = []() noexcept {
    void* raw = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    Dlsym function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
  }();
  if (system_dlsym == nullptr) {
    return nullptr;
  }
  void* raw = system_dlsym(RTLD_NEXT, symbol);
  if (raw == nullptr) {
    return nullptr;
  }
  Function function{};
  static_assert(sizeof(function) == sizeof(raw));
  std::memcpy(&function, &raw, sizeof(function));
  return function;
}

[[nodiscard]] bool symbol_is(const char* symbol, const std::string_view name) {
  return symbol != nullptr && std::string_view(symbol) == name;
}

[[nodiscard]] int replica_index(const CUcontext context) noexcept {
  for (std::size_t index = 0; index < g_registered; ++index) {
    if (g_replicas[index].context == context) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

[[nodiscard]] bool same_signature(const Request& left,
                                  const Request& right) noexcept {
  return left.api == right.api && left.grid_x == right.grid_x &&
         left.grid_y == right.grid_y && left.grid_z == right.grid_z &&
         left.block_x == right.block_x && left.block_y == right.block_y &&
         left.block_z == right.block_z &&
         left.shared_bytes == right.shared_bytes &&
         (left.api != Api::kLaunchEx ||
          left.launch_ex.numAttrs == right.launch_ex.numAttrs);
}

void write_trace(const Request& selected, const unsigned int selected_sms,
                 const bool copied, const CUresult result,
                 const std::uint64_t start_ns,
                 const std::uint64_t end_ns) noexcept {
  if (g_trace_fd < 0) {
    return;
  }
  char line[768];
  const int length = std::snprintf(
      line, sizeof(line),
      "{\"schema_version\":1,\"operation\":%llu,\"tid\":%ld,"
      "\"api\":\"%s\",\"selected_sms\":%u,\"activation_copy\":%s,"
      "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
      "\"shared_mem_bytes\":%u,\"start_monotonic_ns\":%llu,"
      "\"end_monotonic_ns\":%llu,\"result\":%d}\n",
      static_cast<unsigned long long>(g_operation), thread_id(),
      selected.api_name, selected_sms, copied ? "true" : "false",
      selected.grid_x, selected.grid_y, selected.grid_z, selected.block_x,
      selected.block_y, selected.block_z, selected.shared_bytes,
      static_cast<unsigned long long>(start_ns),
      static_cast<unsigned long long>(end_ns), static_cast<int>(result));
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    return;
  }
  std::size_t offset = 0U;
  while (offset < static_cast<std::size_t>(length)) {
    const ssize_t written =
        write(g_trace_fd, line + offset, static_cast<std::size_t>(length) - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
    } else if (written < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
}

[[nodiscard]] CUresult execute_generation() noexcept {
  const Request& first = g_requests.front();
  for (std::size_t index = 1U; index < kReplicas; ++index) {
    if (!same_signature(first, g_requests[index])) {
      ++g_stats.signature_mismatches;
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  const unsigned int selected_sms = []() noexcept {
    const char* fixed = std::getenv("BLESS_TRT_FIXED_SMS");
    if (fixed != nullptr) {
      for (const auto& replica : g_replicas) {
        if ((replica.sms == 2U && std::string_view(fixed) == "2") ||
            (replica.sms == 4U && std::string_view(fixed) == "4") ||
            (replica.sms == 6U && std::string_view(fixed) == "6") ||
            (replica.sms == 8U && std::string_view(fixed) == "8")) {
          return replica.sms;
        }
      }
    }
    const char* switch_text = std::getenv("BLESS_TRT_SWITCH_OPERATION");
    if (switch_text != nullptr && switch_text[0] != '\0') {
      char* end = nullptr;
      errno = 0;
      const unsigned long long operation =
          std::strtoull(switch_text, &end, 10);
      if (errno == 0 && end != switch_text && *end == '\0') {
        return g_operation < operation ? kRestrictedSms : kUnrestrictedSms;
      }
    }
    return (g_operation % kSquadKernels) < kRestrictedKernels
               ? kRestrictedSms
               : kUnrestrictedSms;
  }();
  std::size_t selected_index = kReplicas;
  for (std::size_t index = 0U; index < kReplicas; ++index) {
    if (g_replicas[index].sms == selected_sms) {
      selected_index = index;
      break;
    }
  }
  if (selected_index == kReplicas) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  const Replica& selected_replica = g_replicas[selected_index];
  const Request& selected = g_requests[selected_index];
  bool copied = false;
  g_internal = true;
  CUresult result = CUDA_SUCCESS;
  if (g_previous_context != nullptr &&
      g_previous_context != selected_replica.context) {
    result = cuCtxPushCurrent(g_previous_context);
    if (result == CUDA_SUCCESS) {
      result = cuStreamSynchronize(g_previous_stream);
      CUcontext popped = nullptr;
      const CUresult pop_result = cuCtxPopCurrent(&popped);
      if (result == CUDA_SUCCESS) {
        result = pop_result;
      }
    }
    if (result == CUDA_SUCCESS) {
      result = cuCtxPushCurrent(selected_replica.context);
    }
    if (result == CUDA_SUCCESS) {
      const Replica* previous = nullptr;
      for (const auto& replica : g_replicas) {
        if (replica.context == g_previous_context) {
          previous = &replica;
          break;
        }
      }
      if (previous == nullptr ||
          previous->activation_bytes != selected_replica.activation_bytes) {
        result = CUDA_ERROR_INVALID_VALUE;
      } else {
        result = cuMemcpyPeer(selected_replica.activation,
                              selected_replica.context, previous->activation,
                              previous->context, previous->activation_bytes);
        copied = result == CUDA_SUCCESS;
      }
      CUcontext popped = nullptr;
      const CUresult pop_result = cuCtxPopCurrent(&popped);
      if (result == CUDA_SUCCESS) {
        result = pop_result;
      }
    }
  }
  const std::uint64_t start_ns = monotonic_ns();
  if (result == CUDA_SUCCESS) {
    result = cuCtxPushCurrent(selected_replica.context);
  }
  if (result == CUDA_SUCCESS) {
    if (selected.api == Api::kLaunchEx) {
      result = selected.real_ex(&selected.launch_ex, selected.function,
                                selected.kernel_params, selected.extra);
    } else {
      result = selected.real(
          selected.function, selected.grid_x, selected.grid_y, selected.grid_z,
          selected.block_x, selected.block_y, selected.block_z,
          selected.shared_bytes, selected.stream, selected.kernel_params,
          selected.extra);
    }
    const char* profile_sync = std::getenv("BLESS_TRT_PROFILE_SYNC");
    if (result == CUDA_SUCCESS && profile_sync != nullptr &&
        std::string_view(profile_sync) == "1") {
      result = cuStreamSynchronize(selected.stream);
    }
    CUcontext popped = nullptr;
    const CUresult pop_result = cuCtxPopCurrent(&popped);
    if (result == CUDA_SUCCESS) {
      result = pop_result;
    }
  }
  const std::uint64_t end_ns = monotonic_ns();
  g_internal = false;

  ++g_stats.logical_launches;
  ++g_stats.physical_launches;
  g_stats.shadow_launches += kReplicas - 1U;
  if (selected_sms == kRestrictedSms) {
    ++g_stats.restricted_launches;
  } else {
    ++g_stats.unrestricted_launches;
  }
  if (copied) {
    ++g_stats.activation_copies;
  }
  g_stats.last_selected_sms = selected_sms;
  g_previous_context = selected_replica.context;
  g_previous_stream = selected.stream;
  write_trace(selected, selected_sms, copied, result, start_ns, end_ns);
  return result;
}

[[nodiscard]] CUresult submit(Request request) {
  if (!g_active.load(std::memory_order_acquire) || g_internal) {
    if (request.api == Api::kLaunchEx) {
      return request.real_ex(&request.launch_ex, request.function,
                             request.kernel_params, request.extra);
    }
    return request.real(request.function, request.grid_x, request.grid_y,
                        request.grid_z, request.block_x, request.block_y,
                        request.block_z, request.shared_bytes, request.stream,
                        request.kernel_params, request.extra);
  }
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  std::unique_lock<std::mutex> lock(g_mutex);
  const int index = replica_index(context);
  if (index < 0 || g_requests[static_cast<std::size_t>(index)].present) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  const std::uint64_t generation = g_generation;
  request.context = context;
  request.present = true;
  g_requests[static_cast<std::size_t>(index)] = request;
  ++g_arrived;
  if (g_arrived == kReplicas) {
    g_generation_result = execute_generation();
    for (auto& item : g_requests) {
      item = Request{};
    }
    g_arrived = 0U;
    ++g_generation;
    ++g_operation;
    lock.unlock();
    g_condition.notify_all();
    return g_generation_result;
  }
  g_condition.wait(lock, [&] {
    return g_generation != generation ||
           !g_active.load(std::memory_order_acquire);
  });
  return g_active.load(std::memory_order_acquire) ? g_generation_result
                                                  : CUDA_ERROR_DEINITIALIZED;
}

void replace_launch_pointer(const char* symbol, void** pointer) noexcept;

}  // namespace

extern "C" int bless_trt_squad_register_replica(
    const CUcontext context, const unsigned int sms,
    const CUdeviceptr activation, const std::size_t activation_bytes) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_active.load(std::memory_order_acquire) || context == nullptr ||
      activation == 0U || activation_bytes == 0U || g_registered >= kReplicas ||
      replica_index(context) >= 0) {
    return -1;
  }
  g_replicas[g_registered++] = {context, sms, activation, activation_bytes};
  return 0;
}

extern "C" int bless_trt_squad_start(const char* trace_path) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_registered != kReplicas || g_active.load(std::memory_order_acquire)) {
    return -1;
  }
  if (trace_path != nullptr && trace_path[0] != '\0') {
    g_trace_fd = open(trace_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (g_trace_fd < 0) {
      return -1;
    }
  }
  g_stats = {};
  g_operation = 0U;
  g_generation = 0U;
  g_previous_context = nullptr;
  g_previous_stream = nullptr;
  g_active.store(true, std::memory_order_release);
  return 0;
}

extern "C" int bless_trt_squad_stop() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_active.exchange(false, std::memory_order_acq_rel) || g_arrived != 0U) {
    return -1;
  }
  if (g_previous_context != nullptr) {
    g_internal = true;
    CUresult result = cuCtxPushCurrent(g_previous_context);
    if (result == CUDA_SUCCESS) {
      result = cuStreamSynchronize(g_previous_stream);
      CUcontext popped = nullptr;
      const CUresult pop_result = cuCtxPopCurrent(&popped);
      if (result == CUDA_SUCCESS) {
        result = pop_result;
      }
    }
    g_internal = false;
    if (result != CUDA_SUCCESS) {
      return -1;
    }
  }
  if (g_trace_fd >= 0) {
    if (close(g_trace_fd) != 0) {
      g_trace_fd = -1;
      return -1;
    }
    g_trace_fd = -1;
  }
  g_condition.notify_all();
  return 0;
}

extern "C" int bless_trt_squad_stats(bless::thor::SquadStats* output) {
  if (output == nullptr) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  *output = g_stats;
  return 0;
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig* config,
                                      CUfunction function, void** kernel_params,
                                      void** extra) {
  LaunchEx real = g_launch_ex.load(std::memory_order_acquire);
  if (real == nullptr) {
    real = resolve_next<LaunchEx>("cuLaunchKernelEx");
    g_launch_ex.store(real, std::memory_order_release);
  }
  if (real == nullptr || config == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Request request{};
  request.api = Api::kLaunchEx;
  request.api_name = "cuLaunchKernelEx";
  request.function = function;
  request.launch_ex = *config;
  request.grid_x = config->gridDimX;
  request.grid_y = config->gridDimY;
  request.grid_z = config->gridDimZ;
  request.block_x = config->blockDimX;
  request.block_y = config->blockDimY;
  request.block_z = config->blockDimZ;
  request.shared_bytes = config->sharedMemBytes;
  request.stream = config->hStream;
  request.kernel_params = kernel_params;
  request.extra = extra;
  request.real_ex = real;
  return submit(request);
}

extern "C" CUresult cuLaunchKernel(
    CUfunction function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_bytes, CUstream stream,
    void** kernel_params, void** extra) {
  Launch real = g_launch.load(std::memory_order_acquire);
  if (real == nullptr) {
    real = resolve_next<Launch>("cuLaunchKernel");
    g_launch.store(real, std::memory_order_release);
  }
  if (real == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Request request{};
  request.api = Api::kLaunch;
  request.api_name = "cuLaunchKernel";
  request.function = function;
  request.grid_x = grid_x;
  request.grid_y = grid_y;
  request.grid_z = grid_z;
  request.block_x = block_x;
  request.block_y = block_y;
  request.block_z = block_z;
  request.shared_bytes = shared_bytes;
  request.stream = stream;
  request.kernel_params = kernel_params;
  request.extra = extra;
  request.real = real;
  return submit(request);
}

namespace {

void replace_launch_pointer(const char* symbol, void** pointer) noexcept {
  if (pointer == nullptr || *pointer == nullptr) {
    return;
  }
  if (symbol_is(symbol, "cuLaunchKernelEx") ||
      symbol_is(symbol, "cuLaunchKernelEx_ptsz")) {
    LaunchEx real{};
    std::memcpy(&real, pointer, sizeof(real));
    if (symbol_is(symbol, "cuLaunchKernelEx_ptsz")) {
      g_launch_ex_ptsz.store(real, std::memory_order_release);
    } else {
      g_launch_ex.store(real, std::memory_order_release);
    }
    *pointer = reinterpret_cast<void*>(&cuLaunchKernelEx);
  } else if (symbol_is(symbol, "cuLaunchKernel") ||
             symbol_is(symbol, "cuLaunchKernel_ptsz")) {
    Launch real{};
    std::memcpy(&real, pointer, sizeof(real));
    if (symbol_is(symbol, "cuLaunchKernel_ptsz")) {
      g_launch_ptsz.store(real, std::memory_order_release);
    } else {
      g_launch.store(real, std::memory_order_release);
    }
    *pointer = reinterpret_cast<void*>(&cuLaunchKernel);
  }
}

}  // namespace

extern "C" CUresult cuGetProcAddress_v2(
    const char* symbol, void** pointer, int cuda_version, cuuint64_t flags,
    CUdriverProcAddressQueryResult* status) {
  const GetProcV2 real = resolve_next<GetProcV2>("cuGetProcAddress_v2");
  if (real == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  const CUresult result = real(symbol, pointer, cuda_version, flags, status);
  if (result == CUDA_SUCCESS) {
    replace_launch_pointer(symbol, pointer);
  }
  return result;
}

extern "C" CUresult cuGetProcAddress(const char* symbol, void** pointer,
                                      int cuda_version, cuuint64_t flags) {
  GetProcLegacy real = resolve_next<GetProcLegacy>("cuGetProcAddress");
  if (real == nullptr) {
    real = resolve_next<GetProcLegacy>("cuGetProcAddress_v1");
  }
  if (real == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  const CUresult result = real(symbol, pointer, cuda_version, flags);
  if (result == CUDA_SUCCESS) {
    replace_launch_pointer(symbol, pointer);
  }
  return result;
}

extern "C" void* dlsym(void* handle, const char* symbol) noexcept {
  static const Dlsym real = []() noexcept {
    void* raw = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    Dlsym function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
  }();
  if (real == nullptr) {
    return nullptr;
  }
  void* result = real(handle, symbol);
  if (result == nullptr) {
    return nullptr;
  }
  void* intercepted = result;
  replace_launch_pointer(symbol, &intercepted);
  return intercepted;
}

__attribute__((destructor)) static void close_trace() {
  if (g_trace_fd >= 0) {
    close(g_trace_fd);
    g_trace_fd = -1;
  }
}
