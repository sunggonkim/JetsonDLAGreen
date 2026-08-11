#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kProtocolVersion = 1;
constexpr int kTimeoutMs = 30'000;

enum class TransportMode {
  kRegisteredDirect,
  kPageableDirectControl,
  kPinnedBounce,
  kPageableBounce,
  kManagedUvmControl,
  kHostMaterializeControl,
  kP2pIpcNegativeControl,
};

enum class CacheState { kWarm, kCold };

struct Options {
  std::string producer_uuid;
  std::string consumer_uuid;
  std::string producer_mps_pipe;
  std::vector<std::size_t> sizes{64, 4096, 65536, 1U << 20U, 8U << 20U};
  int warmup = 10;
  int iterations = 100;
  TransportMode transport = TransportMode::kRegisteredDirect;
  CacheState cache_state = CacheState::kWarm;
  std::size_t cache_flush_bytes = 64U << 20U;
};

struct Command {
  std::uint64_t size;
  std::uint32_t iteration;
  std::uint32_t warmup;
  std::uint64_t producer_start_ns;
  std::uint64_t producer_write_done_ns;
  std::uint64_t producer_visibility_done_ns;
  std::uint64_t producer_done_ns;
};

struct Result {
  Command command;
  std::uint64_t consumer_read_start_ns;
  std::uint64_t consumer_read_done_ns;
  std::uint64_t consumer_done_ns;
  std::uint64_t mismatches;
  int cuda_error;
};

struct Ready {
  int role;
  int cuda_error;
  int can_map_host_memory;
  int pageable_memory_access;
  int pageable_memory_access_uses_host_page_tables;
  int concurrent_managed_access;
  int multiprocessor_count;
};

[[nodiscard]] bool direct_transport(const TransportMode mode) {
  return mode == TransportMode::kRegisteredDirect;
}

[[nodiscard]] bool pageable_direct_transport(const TransportMode mode) {
  return mode == TransportMode::kPageableDirectControl;
}

[[nodiscard]] bool mapped_direct_transport(const TransportMode mode) {
  return direct_transport(mode) || pageable_direct_transport(mode);
}

[[nodiscard]] bool registered_transport(const TransportMode mode) {
  return mode == TransportMode::kRegisteredDirect ||
         mode == TransportMode::kPinnedBounce;
}

[[nodiscard]] bool managed_transport(const TransportMode mode) {
  return mode == TransportMode::kManagedUvmControl;
}

[[nodiscard]] std::string_view transport_mode_name(const TransportMode mode) {
  switch (mode) {
    case TransportMode::kRegisteredDirect:
      return "registered-direct";
    case TransportMode::kPageableDirectControl:
      return "pageable-direct-control";
    case TransportMode::kPinnedBounce:
      return "pinned-bounce";
    case TransportMode::kPageableBounce:
      return "pageable-bounce";
    case TransportMode::kManagedUvmControl:
      return "managed-uvm-control";
    case TransportMode::kHostMaterializeControl:
      return "host-materialize-control";
    case TransportMode::kP2pIpcNegativeControl:
      return "p2p-ipc-negative-control";
  }
  return "unknown";
}

[[nodiscard]] std::string_view transport_description(const TransportMode mode) {
  switch (mode) {
    case TransportMode::kRegisteredDirect:
      return "full-coherent registered system-memory activation edge";
    case TransportMode::kPageableDirectControl:
      return "direct pageable system-memory control; runtime capability probe";
    case TransportMode::kPinnedBounce:
      return "producer device write plus pinned D2H/H2D bounce";
    case TransportMode::kPageableBounce:
      return "producer device write plus pageable D2H/H2D bounce";
    case TransportMode::kManagedUvmControl:
      return "cudaMallocManaged/UVM allocation plus host materialization control";
    case TransportMode::kHostMaterializeControl:
      return "host materialize and private consumer copy control";
    case TransportMode::kP2pIpcNegativeControl:
      return "negative control: cross-MIG CUDA P2P/IPC intentionally not attempted";
  }
  return "unknown";
}

[[nodiscard]] std::string_view cache_state_name(const CacheState state) {
  return state == CacheState::kCold ? "cold" : "warm";
}

[[nodiscard]] std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void write_all(const int fd, const void* data, const std::size_t bytes) {
  const auto* cursor = static_cast<const std::byte*>(data);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t written = write(fd, cursor, remaining);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      fail("pipe write failed: " + std::string(std::strerror(errno)));
    }
    cursor += written;
    remaining -= static_cast<std::size_t>(written);
  }
}

[[nodiscard]] bool read_all(const int fd, void* data, const std::size_t bytes,
                            const int timeout_ms = kTimeoutMs) {
  auto* cursor = static_cast<std::byte*>(data);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    pollfd descriptor{fd, POLLIN, 0};
    int polled = 0;
    do {
      polled = poll(&descriptor, 1, timeout_ms);
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

void set_device_environment(const std::string& uuid,
                            const std::string& mps_pipe) {
  if (setenv("CUDA_VISIBLE_DEVICES", uuid.c_str(), 1) != 0) {
    fail("setenv(CUDA_VISIBLE_DEVICES) failed");
  }
  if (mps_pipe.empty()) {
    unsetenv("CUDA_MPS_PIPE_DIRECTORY");
    unsetenv("CUDA_MPS_LOG_DIRECTORY");
  } else if (setenv("CUDA_MPS_PIPE_DIRECTORY", mps_pipe.c_str(), 1) != 0) {
    fail("setenv(CUDA_MPS_PIPE_DIRECTORY) failed");
  }
}

[[nodiscard]] cudaError_t initialize_device(Ready* ready,
                                            const bool require_host_mapping) {
  cudaError_t status = cudaSetDeviceFlags(
      require_host_mapping ? cudaDeviceMapHost : cudaDeviceScheduleAuto);
  if (status != cudaSuccess && status != cudaErrorSetOnActiveProcess) {
    return status;
  }
  status = cudaSetDevice(0);
  if (status != cudaSuccess) {
    return status;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess) {
    return status;
  }
  ready->can_map_host_memory = properties.canMapHostMemory;
  ready->pageable_memory_access = properties.pageableMemoryAccess;
  ready->pageable_memory_access_uses_host_page_tables =
      properties.pageableMemoryAccessUsesHostPageTables;
  ready->concurrent_managed_access = properties.concurrentManagedAccess;
  ready->multiprocessor_count = properties.multiProcessorCount;
  if (require_host_mapping && properties.canMapHostMemory == 0) {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t prefetch_managed(void* pointer,
                                           const std::size_t bytes,
                                           const bool to_host) {
  cudaMemLocation location{};
  location.type = to_host ? cudaMemLocationTypeHost
                          : cudaMemLocationTypeDevice;
  location.id = 0;
  return cudaMemPrefetchAsync(pointer, bytes, location, 0);
}

__global__ void write_pattern(std::uint8_t* output, const std::size_t size,
                              const std::uint32_t iteration) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < size; index += stride) {
    output[index] = static_cast<std::uint8_t>(
        (index * 131U + static_cast<std::size_t>(iteration) * 17U + size) &
        0xffU);
  }
}

__global__ void verify_pattern(const std::uint8_t* input, const std::size_t size,
                               const std::uint32_t iteration,
                               unsigned long long* mismatches) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  unsigned long long local = 0;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < size; index += stride) {
    const auto expected = static_cast<std::uint8_t>(
        (index * 131U + static_cast<std::size_t>(iteration) * 17U + size) &
        0xffU);
    local += input[index] == expected ? 0ULL : 1ULL;
  }
  if (local != 0) {
    atomicAdd(mismatches, local);
  }
}

// The cold-cache control deliberately touches a working set larger than the
// measured activation.  The touch is completed before the timed producer or
// consumer interval, so it changes the cache precondition without becoming a
// hidden transport term.
__global__ void evict_cache(std::uint8_t* output, const std::size_t size,
                            const std::uint32_t iteration) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < size; index += stride) {
    output[index] = static_cast<std::uint8_t>(
        (index * 17U + static_cast<std::size_t>(iteration) * 29U) & 0xffU);
  }
}

void prepare_cache(const Options& options, std::uint8_t* const scratch,
                   const std::uint32_t iteration) {
  if (options.cache_state != CacheState::kCold) {
    return;
  }
  evict_cache<<<256, 256>>>(scratch, options.cache_flush_bytes, iteration);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    fail("cache-state preparation failed: " +
         std::string(cudaGetErrorName(status)));
  }
}

[[noreturn]] void producer_main(const Options& options, void* mapping,
                                const std::size_t mapping_bytes,
                                const int ready_fd, const int go_fd,
                                const int command_fd, const int ack_fd) {
  try {
    set_device_environment(options.producer_uuid, options.producer_mps_pipe);
    Ready ready{};
    ready.role = 0;
    cudaError_t status = initialize_device(&ready,
                                           direct_transport(options.transport));
    const bool registered = registered_transport(options.transport);
    if (status == cudaSuccess && registered) {
      const unsigned int flags = direct_transport(options.transport)
                                     ? cudaHostRegisterMapped
                                     : cudaHostRegisterDefault;
      status = cudaHostRegister(mapping, mapping_bytes, flags);
    }
    ready.cuda_error = static_cast<int>(status);
    write_all(ready_fd, &ready, sizeof(ready));
    char go = 0;
    if (status != cudaSuccess || !read_all(go_fd, &go, sizeof(go))) {
      _exit(status == cudaSuccess ? 2 : 3);
    }
    void* device_mapping = nullptr;
    void* device_payload = nullptr;
    if (direct_transport(options.transport)) {
      status = cudaHostGetDevicePointer(&device_mapping, mapping, 0);
      if (status != cudaSuccess) {
        _exit(4);
      }
      device_payload = device_mapping;
    } else if (pageable_direct_transport(options.transport)) {
      device_payload = mapping;
    } else if (managed_transport(options.transport)) {
      status = cudaMallocManaged(&device_payload, mapping_bytes);
      if (status != cudaSuccess) {
        _exit(4);
      }
      status = prefetch_managed(device_payload, mapping_bytes, false);
      if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
      }
      if (status != cudaSuccess) {
        static_cast<void>(cudaFree(device_payload));
        _exit(4);
      }
    } else {
      status = cudaMalloc(&device_payload, mapping_bytes);
      if (status != cudaSuccess) {
        _exit(4);
      }
    }
    std::uint8_t* cache_scratch = nullptr;
    if (options.cache_state == CacheState::kCold) {
      status = cudaMalloc(reinterpret_cast<void**>(&cache_scratch),
                          options.cache_flush_bytes);
      if (status != cudaSuccess) {
        if (device_payload != nullptr && !mapped_direct_transport(options.transport)) {
          static_cast<void>(cudaFree(device_payload));
        }
        _exit(4);
      }
    }
    auto* payload = static_cast<std::uint8_t*>(device_payload);
    const int blocks = 256;
    for (const std::size_t size : options.sizes) {
      const int total = options.warmup + options.iterations;
      for (int index = 0; index < total; ++index) {
        Command command{size, static_cast<std::uint32_t>(index),
                        static_cast<std::uint32_t>(index < options.warmup), 0, 0,
                        0, 0};
        prepare_cache(options, cache_scratch,
                      static_cast<std::uint32_t>(index));
        command.producer_start_ns = monotonic_ns();
        write_pattern<<<blocks, 256>>>(payload, size, command.iteration);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
          status = cudaDeviceSynchronize();
        }
        if (status != cudaSuccess) {
          _exit(5);
        }
        command.producer_write_done_ns = monotonic_ns();
        if (!mapped_direct_transport(options.transport)) {
          status = cudaMemcpy(mapping, device_payload, size,
                              cudaMemcpyDeviceToHost);
          if (status != cudaSuccess) {
            static_cast<void>(cudaFree(device_payload));
            _exit(5);
          }
          if (managed_transport(options.transport)) {
            status = prefetch_managed(device_payload, mapping_bytes, true);
            if (status == cudaSuccess) {
              status = cudaDeviceSynchronize();
            }
            if (status != cudaSuccess) {
              static_cast<void>(cudaFree(device_payload));
              _exit(5);
            }
          }
        }
        command.producer_visibility_done_ns = monotonic_ns();
        command.producer_done_ns = command.producer_visibility_done_ns;
        write_all(command_fd, &command, sizeof(command));
        char ack = 0;
        if (!read_all(ack_fd, &ack, sizeof(ack))) {
          _exit(8);
        }
      }
    }
    if (registered) {
      status = cudaHostUnregister(mapping);
    }
    if (device_payload != nullptr && !mapped_direct_transport(options.transport)) {
      const cudaError_t free_status = cudaFree(device_payload);
      if (status == cudaSuccess) {
        status = free_status;
      }
    }
    if (cache_scratch != nullptr) {
      const cudaError_t free_status = cudaFree(cache_scratch);
      if (status == cudaSuccess) {
        status = free_status;
      }
    }
    _exit(status == cudaSuccess ? 0 : 6);
  } catch (...) {
    _exit(7);
  }
}

[[noreturn]] void consumer_main(const Options& options, void* mapping,
                                const std::size_t mapping_bytes,
                                const int ready_fd, const int go_fd,
                                const int command_fd, const int result_fd,
                                const int ack_fd) {
  try {
    set_device_environment(
        options.consumer_uuid,
        options.producer_uuid == options.consumer_uuid
            ? options.producer_mps_pipe
            : "");
    Ready ready{};
    ready.role = 1;
    cudaError_t status = initialize_device(&ready,
                                           direct_transport(options.transport));
    const bool registered = registered_transport(options.transport);
    if (status == cudaSuccess && registered) {
      const unsigned int flags = direct_transport(options.transport)
                                     ? cudaHostRegisterMapped
                                     : cudaHostRegisterDefault;
      status = cudaHostRegister(mapping, mapping_bytes, flags);
    }
    ready.cuda_error = static_cast<int>(status);
    write_all(ready_fd, &ready, sizeof(ready));
    char go = 0;
    if (status != cudaSuccess || !read_all(go_fd, &go, sizeof(go))) {
      _exit(status == cudaSuccess ? 2 : 3);
    }
    void* device_mapping = nullptr;
    void* device_payload = nullptr;
    if (direct_transport(options.transport)) {
      status = cudaHostGetDevicePointer(&device_mapping, mapping, 0);
      if (status != cudaSuccess) {
        _exit(4);
      }
      device_payload = device_mapping;
    } else if (pageable_direct_transport(options.transport)) {
      device_payload = mapping;
    } else if (managed_transport(options.transport)) {
      status = cudaMallocManaged(&device_payload, mapping_bytes);
      if (status != cudaSuccess) {
        _exit(4);
      }
    } else {
      status = cudaMalloc(&device_payload, mapping_bytes);
      if (status != cudaSuccess) {
        _exit(4);
      }
    }
    std::uint8_t* cache_scratch = nullptr;
    if (options.cache_state == CacheState::kCold) {
      status = cudaMalloc(reinterpret_cast<void**>(&cache_scratch),
                          options.cache_flush_bytes);
      if (status != cudaSuccess) {
        if (device_payload != nullptr && !mapped_direct_transport(options.transport)) {
          static_cast<void>(cudaFree(device_payload));
        }
        _exit(4);
      }
    }
    unsigned long long* device_mismatches = nullptr;
    status = cudaMalloc(&device_mismatches, sizeof(*device_mismatches));
    if (status != cudaSuccess) {
      _exit(5);
    }
    std::vector<std::uint8_t> materialized;
    if (options.transport == TransportMode::kHostMaterializeControl) {
      materialized.resize(mapping_bytes);
    }
    const auto* payload = static_cast<const std::uint8_t*>(device_payload);
    while (true) {
      Command command{};
      if (!read_all(command_fd, &command, sizeof(command))) {
        break;
      }
      Result result{command, 0, 0, 0, 0, 0};
      prepare_cache(options, cache_scratch, command.iteration);
      result.consumer_read_start_ns = monotonic_ns();
      if (!mapped_direct_transport(options.transport)) {
        const void* source = mapping;
        if (options.transport == TransportMode::kHostMaterializeControl) {
          std::memcpy(materialized.data(), mapping, command.size);
          source = materialized.data();
        }
        status = cudaMemcpy(device_payload, source, command.size,
                            cudaMemcpyHostToDevice);
        if (status == cudaSuccess && managed_transport(options.transport)) {
          status = prefetch_managed(device_payload, mapping_bytes, false);
        }
        if (status == cudaSuccess) {
          status = cudaDeviceSynchronize();
        }
      }
      result.consumer_read_done_ns = monotonic_ns();
      status = status == cudaSuccess
                   ? cudaMemset(device_mismatches, 0, sizeof(*device_mismatches))
                   : status;
      if (status == cudaSuccess) {
        verify_pattern<<<256, 256>>>(payload, command.size, command.iteration,
                                    device_mismatches);
        status = cudaGetLastError();
      }
      if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
      }
      if (status == cudaSuccess) {
        status = cudaMemcpy(&result.mismatches, device_mismatches,
                            sizeof(result.mismatches), cudaMemcpyDeviceToHost);
      }
      result.consumer_done_ns = monotonic_ns();
      result.cuda_error = static_cast<int>(status);
      write_all(result_fd, &result, sizeof(result));
      const char ack = 1;
      write_all(ack_fd, &ack, sizeof(ack));
      if (status != cudaSuccess) {
        break;
      }
    }
    static_cast<void>(cudaFree(device_mismatches));
    if (registered) {
      status = cudaHostUnregister(mapping);
    }
    if (device_payload != nullptr && !mapped_direct_transport(options.transport)) {
      const cudaError_t free_status = cudaFree(device_payload);
      if (status == cudaSuccess) {
        status = free_status;
      }
    }
    if (cache_scratch != nullptr) {
      const cudaError_t free_status = cudaFree(cache_scratch);
      if (status == cudaSuccess) {
        status = free_status;
      }
    }
    _exit(status == cudaSuccess ? 0 : 6);
  } catch (...) {
    _exit(7);
  }
}

[[nodiscard]] std::vector<std::size_t> parse_sizes(const std::string& text) {
  std::vector<std::size_t> sizes;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string token = text.substr(begin, end - begin);
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(token, &consumed);
    if (consumed != token.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
      fail("invalid payload size: " + token);
    }
    sizes.push_back(static_cast<std::size_t>(value));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  if (sizes.empty()) {
    fail("at least one payload size is required");
  }
  return sizes;
}

[[nodiscard]] int parse_positive_int(const std::string& value,
                                     const std::string_view name,
                                     const bool allow_zero) {
  std::size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed < (allow_zero ? 0 : 1) ||
      parsed > std::numeric_limits<int>::max()) {
    fail("invalid " + std::string(name) + ": " + value);
  }
  return static_cast<int>(parsed);
}

[[nodiscard]] TransportMode parse_transport(const std::string& value) {
  if (value == "registered-direct") {
    return TransportMode::kRegisteredDirect;
  }
  if (value == "pageable-direct-control") {
    return TransportMode::kPageableDirectControl;
  }
  if (value == "pinned-bounce") {
    return TransportMode::kPinnedBounce;
  }
  if (value == "pageable-bounce") {
    return TransportMode::kPageableBounce;
  }
  if (value == "managed-uvm-control") {
    return TransportMode::kManagedUvmControl;
  }
  if (value == "host-materialize-control") {
    return TransportMode::kHostMaterializeControl;
  }
  if (value == "p2p-ipc-negative-control") {
    return TransportMode::kP2pIpcNegativeControl;
  }
  fail("unsupported transport mode: " + value);
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
    auto require_value = [&]() -> std::string {
      if (++index >= argc) {
        fail("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--producer") {
      options.producer_uuid = require_value();
    } else if (argument == "--consumer") {
      options.consumer_uuid = require_value();
    } else if (argument == "--producer-mps-pipe") {
      options.producer_mps_pipe = require_value();
    } else if (argument == "--sizes") {
      options.sizes = parse_sizes(require_value());
    } else if (argument == "--warmup") {
      options.warmup = parse_positive_int(require_value(), "warmup", true);
    } else if (argument == "--iterations") {
      options.iterations = parse_positive_int(require_value(), "iterations", false);
    } else if (argument == "--transport") {
      options.transport = parse_transport(require_value());
    } else if (argument == "--cache-state") {
      const std::string value = require_value();
      if (value == "warm") {
        options.cache_state = CacheState::kWarm;
      } else if (value == "cold") {
        options.cache_state = CacheState::kCold;
      } else {
        fail("cache-state expects warm or cold");
      }
    } else if (argument == "--cache-flush-bytes") {
      const std::vector<std::size_t> values = parse_sizes(require_value());
      if (values.size() != 1U) {
        fail("cache-flush-bytes expects exactly one size");
      }
      options.cache_flush_bytes = values.front();
    } else if (argument == "--help") {
      std::cout << "Usage: jdg-mig-sysmem-handoff --producer MIG_UUID "
                   "--consumer MIG_UUID [--producer-mps-pipe PATH] "
                   "[--transport registered-direct|pinned-bounce|"
                   "pageable-direct-control|pageable-bounce|"
                   "managed-uvm-control|"
                   "host-materialize-control|p2p-ipc-negative-control] "
                   "[--sizes BYTES,...] [--warmup N] [--iterations N] "
                   "[--cache-state warm|cold] [--cache-flush-bytes BYTES]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + argument);
    }
  }
  if (options.producer_uuid.empty() || options.consumer_uuid.empty()) {
    fail("producer and consumer MIG UUIDs are required");
  }
  if (options.producer_uuid == options.consumer_uuid &&
      options.producer_mps_pipe.empty()) {
    fail("same-instance transport control requires --producer-mps-pipe");
  }
  return options;
}

[[nodiscard]] double percentile(std::vector<double> values, const double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = q * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] + (values[upper] - values[lower]) * fraction;
}

[[nodiscard]] int child_status(const pid_t pid) {
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
    if (options.transport == TransportMode::kP2pIpcNegativeControl) {
      std::cout << "{\"schema_version\":" << kProtocolVersion
                << ",\"status\":\"negative-control\",\"transport_mode\":\""
                << transport_mode_name(options.transport)
                << "\",\"transport_description\":\""
                << transport_description(options.transport)
                << "\",\"p2p_ipc_attempted\":false,\"producer_uuid\":\""
                << options.producer_uuid << "\",\"consumer_uuid\":\""
                << options.consumer_uuid
                << "\",\"reason\":\"cross-MIG CUDA P2P/IPC is intentionally not"
                   " attempted by this benchmark\"}\n";
      return 0;
    }
    const std::size_t mapping_bytes =
        *std::max_element(options.sizes.begin(), options.sizes.end());
    void* const mapping = mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
      fail("mmap failed: " + std::string(std::strerror(errno)));
    }

    int ready_pipe[2]{};
    int producer_go[2]{};
    int consumer_go[2]{};
    int commands[2]{};
    int acknowledgements[2]{};
    int results[2]{};
    if (pipe(ready_pipe) != 0 || pipe(producer_go) != 0 ||
        pipe(consumer_go) != 0 || pipe(commands) != 0 ||
        pipe(acknowledgements) != 0 || pipe(results) != 0) {
      fail("pipe creation failed: " + std::string(std::strerror(errno)));
    }

    const pid_t producer = fork();
    if (producer == 0) {
      close_fd(ready_pipe[0]);
      close_fd(producer_go[1]);
      close_fd(consumer_go[0]);
      close_fd(consumer_go[1]);
      close_fd(commands[0]);
      close_fd(acknowledgements[1]);
      close_fd(results[0]);
      close_fd(results[1]);
      producer_main(options, mapping, mapping_bytes, ready_pipe[1],
                    producer_go[0], commands[1], acknowledgements[0]);
    }
    if (producer < 0) {
      fail("producer fork failed");
    }

    const pid_t consumer = fork();
    if (consumer == 0) {
      close_fd(ready_pipe[0]);
      close_fd(consumer_go[1]);
      close_fd(producer_go[0]);
      close_fd(producer_go[1]);
      close_fd(commands[1]);
      close_fd(acknowledgements[0]);
      close_fd(results[0]);
      consumer_main(options, mapping, mapping_bytes, ready_pipe[1],
                    consumer_go[0], commands[0], results[1],
                    acknowledgements[1]);
    }
    if (consumer < 0) {
      kill(producer, SIGTERM);
      fail("consumer fork failed");
    }

    close_fd(ready_pipe[1]);
    close_fd(producer_go[0]);
    close_fd(consumer_go[0]);
    close_fd(commands[0]);
    close_fd(commands[1]);
    close_fd(acknowledgements[0]);
    close_fd(acknowledgements[1]);
    close_fd(results[1]);

    Ready readiness[2]{};
    if (!read_all(ready_pipe[0], &readiness[0], sizeof(Ready)) ||
        !read_all(ready_pipe[0], &readiness[1], sizeof(Ready))) {
      kill(producer, SIGTERM);
      kill(consumer, SIGTERM);
      fail("a child exited before CUDA readiness");
    }
    close_fd(ready_pipe[0]);
    std::sort(std::begin(readiness), std::end(readiness),
              [](const Ready& left, const Ready& right) {
                return left.role < right.role;
              });
    if (readiness[0].cuda_error != cudaSuccess ||
        readiness[1].cuda_error != cudaSuccess) {
      close_fd(producer_go[1]);
      close_fd(consumer_go[1]);
      kill(producer, SIGTERM);
      kill(consumer, SIGTERM);
      std::cerr << "{\"schema_version\":" << kProtocolVersion
                << ",\"status\":\"unsupported\",\"producer_cuda_error\":\""
                << cudaGetErrorName(static_cast<cudaError_t>(readiness[0].cuda_error))
                << "\",\"consumer_cuda_error\":\""
                << cudaGetErrorName(static_cast<cudaError_t>(readiness[1].cuda_error))
                << "\"}\n";
      static_cast<void>(child_status(producer));
      static_cast<void>(child_status(consumer));
      munmap(mapping, mapping_bytes);
      return 2;
    }

    const char go = 1;
    write_all(producer_go[1], &go, sizeof(go));
    write_all(consumer_go[1], &go, sizeof(go));
    close_fd(producer_go[1]);
    close_fd(consumer_go[1]);

    std::vector<Result> all_results;
    const std::size_t expected = options.sizes.size() *
                                 static_cast<std::size_t>(options.warmup +
                                                          options.iterations);
    all_results.reserve(expected);
    for (std::size_t index = 0; index < expected; ++index) {
      Result result{};
      if (!read_all(results[0], &result, sizeof(result))) {
        break;
      }
      all_results.push_back(result);
    }
    close_fd(results[0]);
    const int producer_status = child_status(producer);
    const int consumer_status = child_status(consumer);
    munmap(mapping, mapping_bytes);

    bool success = all_results.size() == expected && producer_status == 0 &&
                   consumer_status == 0;
    for (const Result& result : all_results) {
      success = success && result.cuda_error == cudaSuccess &&
                result.mismatches == 0 &&
                result.command.producer_visibility_done_ns >=
                    result.command.producer_write_done_ns &&
                result.consumer_done_ns >= result.command.producer_done_ns &&
                result.consumer_read_start_ns >=
                    result.command.producer_done_ns &&
                result.consumer_read_done_ns >= result.consumer_read_start_ns &&
                result.consumer_done_ns >= result.consumer_read_done_ns &&
                result.command.producer_done_ns >=
                    result.command.producer_start_ns;
    }

    std::cout << "{\"schema_version\":" << kProtocolVersion
              << ",\"status\":\"" << (success ? "ok" : "error")
              << "\",\"transport_mode\":\""
              << transport_mode_name(options.transport)
              << "\",\"transport_description\":\""
              << transport_description(options.transport)
              << "\",\"transport\":\""
              << (direct_transport(options.transport)
                      ? "full-coherent-registered-system-memory"
                      : pageable_direct_transport(options.transport)
                            ? "direct-pageable-system-memory-control"
                            : "explicit-copy-control")
              << "\""
              << ",\"producer_uuid\":\"" << options.producer_uuid
              << "\",\"consumer_uuid\":\"" << options.consumer_uuid
              << "\",\"topology\":\""
              << (options.producer_uuid == options.consumer_uuid
                      ? "same-instance-mps"
                      : "cross-mig")
              << "\""
              << ",\"producer_capabilities\":{\"canMapHostMemory\":"
              << readiness[0].can_map_host_memory
              << ",\"pageableMemoryAccess\":"
              << readiness[0].pageable_memory_access
              << ",\"pageableMemoryAccessUsesHostPageTables\":"
              << readiness[0].pageable_memory_access_uses_host_page_tables
              << ",\"concurrentManagedAccess\":"
              << readiness[0].concurrent_managed_access
              << ",\"multiprocessor_count\":"
              << readiness[0].multiprocessor_count << "}"
              << ",\"consumer_capabilities\":{\"canMapHostMemory\":"
              << readiness[1].can_map_host_memory
              << ",\"pageableMemoryAccess\":"
              << readiness[1].pageable_memory_access
              << ",\"pageableMemoryAccessUsesHostPageTables\":"
              << readiness[1].pageable_memory_access_uses_host_page_tables
              << ",\"concurrentManagedAccess\":"
              << readiness[1].concurrent_managed_access
              << ",\"multiprocessor_count\":"
              << readiness[1].multiprocessor_count << "}"
              << ",\"producer_sms\":" << readiness[0].multiprocessor_count
              << ",\"consumer_sms\":" << readiness[1].multiprocessor_count
              << ",\"capabilities\":{\"canMapHostMemory\":"
              << readiness[0].can_map_host_memory
              << ",\"pageableMemoryAccess\":"
              << readiness[0].pageable_memory_access
              << ",\"pageableMemoryAccessUsesHostPageTables\":"
              << readiness[0].pageable_memory_access_uses_host_page_tables
              << ",\"concurrentManagedAccess\":"
              << readiness[0].concurrent_managed_access << "}"
              << ",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations
              << ",\"cache_state\":\""
              << cache_state_name(options.cache_state)
              << "\",\"cache_flush_bytes\":"
              << options.cache_flush_bytes
              << ",\"results\":[";
    for (std::size_t size_index = 0; size_index < options.sizes.size();
         ++size_index) {
      if (size_index != 0) {
        std::cout << ',';
      }
      const std::size_t size = options.sizes[size_index];
      std::vector<double> handoff_us;
      std::vector<double> end_to_end_us;
      std::vector<double> producer_visibility_us;
      std::vector<double> consumer_read_us;
      std::uint64_t mismatches = 0;
      for (const Result& result : all_results) {
        if (result.command.size != size || result.command.warmup != 0) {
          continue;
        }
        handoff_us.push_back(static_cast<double>(
                                 result.consumer_done_ns -
                                 result.command.producer_done_ns) /
                             1000.0);
        producer_visibility_us.push_back(static_cast<double>(
                                             result.command
                                                 .producer_visibility_done_ns -
                                             result.command
                                                 .producer_write_done_ns) /
                                         1000.0);
        consumer_read_us.push_back(static_cast<double>(
                                       result.consumer_read_done_ns -
                                       result.consumer_read_start_ns) /
                                   1000.0);
        end_to_end_us.push_back(static_cast<double>(
                                    result.consumer_done_ns -
                                    result.command.producer_start_ns) /
                                1000.0);
        mismatches += result.mismatches;
      }
      const double handoff_p50 = percentile(handoff_us, 0.50);
      const double handoff_p95 = percentile(handoff_us, 0.95);
      const double handoff_p99 = percentile(handoff_us, 0.99);
      const double handoff_max = handoff_us.empty()
                                     ? 0.0
                                     : *std::max_element(handoff_us.begin(),
                                                         handoff_us.end());
      const double e2e_p50 = percentile(end_to_end_us, 0.50);
      const double e2e_p99 = percentile(end_to_end_us, 0.99);
      const double producer_visibility_p99 =
          percentile(producer_visibility_us, 0.99);
      const double consumer_read_p99 = percentile(consumer_read_us, 0.99);
      std::cout << "{\"bytes\":" << size << ",\"samples\":"
                << handoff_us.size() << ",\"mismatches\":" << mismatches
                << ",\"handoff_us\":{\"p50\":" << handoff_p50
                << ",\"p95\":" << handoff_p95 << ",\"p99\":"
                << handoff_p99 << ",\"max\":" << handoff_max
                << "},\"producer_visibility_us\":{\"p99\":"
                << producer_visibility_p99
                << "},\"consumer_read_us\":{\"p99\":"
                << consumer_read_p99
                << "},\"end_to_end_us\":{\"p50\":" << e2e_p50
                << ",\"p99\":" << e2e_p99 << "}}";
    }
    std::cout << "]}\n";
    return success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema_version\":" << kProtocolVersion
              << ",\"status\":\"error\",\"message\":\"" << error.what()
              << "\"}\n";
    return 1;
  }
}
