#include <cuda.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
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

#include "scheduler.hpp"

// CUDA 13 maps the source-level name to the versioned ABI symbol.
#undef cuGetProcAddress

namespace {

using LaunchEx = orion::thor::LaunchEx;
using Launch = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, CUstream, void**, void**);
using GetProcV2 = CUresult (*)(const char*, void**, int, cuuint64_t,
                              CUdriverProcAddressQueryResult*);
using GetProcLegacy = CUresult (*)(const char*, void**, int, cuuint64_t);
using Dlsym = void* (*)(void*, const char*);

std::atomic<LaunchEx> g_launch_ex{nullptr};
std::atomic<LaunchEx> g_launch_ex_ptsz{nullptr};
std::atomic<Launch> g_launch{nullptr};
std::atomic<Launch> g_launch_ptsz{nullptr};
std::atomic<std::uint64_t> g_sequence{0};
std::once_flag g_trace_once;
std::mutex g_trace_mutex;
int g_trace_fd = -1;

[[nodiscard]] std::uint64_t monotonic_ns() noexcept {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

[[nodiscard]] long thread_id() noexcept {
  return static_cast<long>(syscall(SYS_gettid));
}

void open_trace() noexcept {
  const char* path = std::getenv("ORION_TRT_DRIVER_TRACE");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  g_trace_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
}

void write_all(const char* data, std::size_t bytes) noexcept {
  if (g_trace_fd < 0) {
    return;
  }
  std::size_t written = 0;
  while (written < bytes) {
    const ssize_t result = write(g_trace_fd, data + written, bytes - written);
    if (result > 0) {
      written += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    break;
  }
}

void record_launch(const char* api, const CUfunction function,
                   const unsigned int grid_x, const unsigned int grid_y,
                   const unsigned int grid_z, const unsigned int block_x,
                   const unsigned int block_y, const unsigned int block_z,
                   const unsigned int shared_bytes, const CUstream stream,
                   const unsigned int attributes, const std::uint64_t start_ns,
                   const std::uint64_t end_ns, const CUresult result) noexcept {
  std::call_once(g_trace_once, open_trace);
  if (g_trace_fd < 0) {
    return;
  }
  char line[768];
  const std::uint64_t sequence = g_sequence.fetch_add(1, std::memory_order_relaxed);
  const int length = std::snprintf(
      line, sizeof(line),
      "{\"schema_version\":1,\"sequence\":%" PRIu64
      ",\"api\":\"%s\",\"tid\":%ld,\"start_monotonic_ns\":%" PRIu64
      ",\"end_monotonic_ns\":%" PRIu64
      ",\"function\":\"%p\",\"stream\":\"%p\",\"grid\":[%u,%u,%u],"
      "\"block\":[%u,%u,%u],\"shared_mem_bytes\":%u,\"attributes\":%u,"
      "\"result\":%d}\n",
      sequence, api, thread_id(), start_ns, end_ns,
      reinterpret_cast<void*>(function), reinterpret_cast<void*>(stream),
      grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
      attributes, static_cast<int>(result));
  if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_trace_mutex);
  write_all(line, static_cast<std::size_t>(length));
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

void replace_launch_pointer(const char* symbol, void** pointer) noexcept;

}  // namespace

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
  const std::uint64_t start = monotonic_ns();
  const CUresult result = orion::thor::submit_launch_ex(
      "cuLaunchKernelEx", real, config, function, kernel_params, extra);
  const std::uint64_t end = monotonic_ns();
  record_launch("cuLaunchKernelEx", function, config->gridDimX, config->gridDimY,
                config->gridDimZ, config->blockDimX, config->blockDimY,
                config->blockDimZ, config->sharedMemBytes, config->hStream,
                config->numAttrs, start, end, result);
  return result;
}

extern "C" CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config,
                                           CUfunction function,
                                           void** kernel_params, void** extra) {
  LaunchEx real = g_launch_ex_ptsz.load(std::memory_order_acquire);
  if (real == nullptr) {
    real = resolve_next<LaunchEx>("cuLaunchKernelEx_ptsz");
    g_launch_ex_ptsz.store(real, std::memory_order_release);
  }
  if (real == nullptr || config == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::uint64_t start = monotonic_ns();
  const CUresult result = orion::thor::submit_launch_ex(
      "cuLaunchKernelEx_ptsz", real, config, function, kernel_params, extra);
  const std::uint64_t end = monotonic_ns();
  record_launch("cuLaunchKernelEx_ptsz", function, config->gridDimX,
                config->gridDimY, config->gridDimZ, config->blockDimX,
                config->blockDimY, config->blockDimZ, config->sharedMemBytes,
                config->hStream, config->numAttrs, start, end, result);
  return result;
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
  const std::uint64_t start = monotonic_ns();
  const CUresult result = real(function, grid_x, grid_y, grid_z, block_x,
                               block_y, block_z, shared_bytes, stream,
                               kernel_params, extra);
  const std::uint64_t end = monotonic_ns();
  record_launch("cuLaunchKernel", function, grid_x, grid_y, grid_z, block_x,
                block_y, block_z, shared_bytes, stream, 0, start, end, result);
  return result;
}

extern "C" CUresult cuLaunchKernel_ptsz(
    CUfunction function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_bytes, CUstream stream,
    void** kernel_params, void** extra) {
  Launch real = g_launch_ptsz.load(std::memory_order_acquire);
  if (real == nullptr) {
    real = resolve_next<Launch>("cuLaunchKernel_ptsz");
    g_launch_ptsz.store(real, std::memory_order_release);
  }
  if (real == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::uint64_t start = monotonic_ns();
  const CUresult result = real(function, grid_x, grid_y, grid_z, block_x,
                               block_y, block_z, shared_bytes, stream,
                               kernel_params, extra);
  const std::uint64_t end = monotonic_ns();
  record_launch("cuLaunchKernel_ptsz", function, grid_x, grid_y, grid_z,
                block_x, block_y, block_z, shared_bytes, stream, 0, start, end,
                result);
  return result;
}

namespace {

void replace_launch_pointer(const char* symbol, void** pointer) noexcept {
  if (pointer == nullptr || *pointer == nullptr) {
    return;
  }
  if (symbol_is(symbol, "cuLaunchKernelEx")) {
    LaunchEx real{};
    std::memcpy(&real, pointer, sizeof(real));
    g_launch_ex.store(real, std::memory_order_release);
    *pointer = reinterpret_cast<void*>(&cuLaunchKernelEx);
  } else if (symbol_is(symbol, "cuLaunchKernelEx_ptsz")) {
    LaunchEx real{};
    std::memcpy(&real, pointer, sizeof(real));
    g_launch_ex_ptsz.store(real, std::memory_order_release);
    *pointer = reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz);
  } else if (symbol_is(symbol, "cuLaunchKernel")) {
    Launch real{};
    std::memcpy(&real, pointer, sizeof(real));
    g_launch.store(real, std::memory_order_release);
    *pointer = reinterpret_cast<void*>(&cuLaunchKernel);
  } else if (symbol_is(symbol, "cuLaunchKernel_ptsz")) {
    Launch real{};
    std::memcpy(&real, pointer, sizeof(real));
    g_launch_ptsz.store(real, std::memory_order_release);
    *pointer = reinterpret_cast<void*>(&cuLaunchKernel_ptsz);
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
  const GetProcLegacy real = resolve_next<GetProcLegacy>("cuGetProcAddress");
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
    return result;
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
