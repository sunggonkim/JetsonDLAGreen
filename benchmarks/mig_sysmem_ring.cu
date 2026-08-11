#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::uint32_t kRingSlots = 3;
constexpr std::uint32_t kFree = 0;
constexpr std::uint32_t kReady = 1;
constexpr std::uint32_t kConsuming = 2;
constexpr std::uint32_t kDone = 3;
constexpr std::uint32_t kAborted = 4;
constexpr int kFaultExit = 42;
constexpr std::uint64_t kNanosecondsPerMicrosecond = 1000;
constexpr std::uint64_t kDefaultTimeoutMs = 30'000;
constexpr std::uint64_t kDefaultStaleTimeoutUs = 100'000;
constexpr std::string_view kTransportDescription =
    "full-coherent registered system-memory activation edge";

enum class CacheState { kWarm, kCold };

struct Options {
  std::string producer_uuid;
  std::string consumer_uuid;
  std::string mps_pipe;
  std::size_t payload_bytes = 65'536;
  std::uint64_t requests = 100;
  std::uint64_t timeout_ms = kDefaultTimeoutMs;
  std::uint64_t stale_timeout_us = kDefaultStaleTimeoutUs;
  std::uint64_t consumer_delay_us = 0;
  std::int64_t fail_consumer_after = -1;
  CacheState cache_state = CacheState::kWarm;
  std::size_t cache_flush_bytes = 64U << 20U;
};

struct alignas(64) RingHeader {
  std::atomic<std::uint32_t> start;
  std::atomic<std::uint32_t> producer_ready;
  std::atomic<std::uint32_t> consumer_ready;
  std::atomic<std::uint32_t> producer_alive;
  std::atomic<std::uint32_t> consumer_alive;
  std::atomic<std::uint32_t> stop;
  std::atomic<std::uint64_t> published;
  std::atomic<std::uint64_t> completed;
  std::atomic<std::uint64_t> ready_transitions;
  std::atomic<std::uint64_t> consuming_transitions;
  std::atomic<std::uint64_t> backpressure_events;
  std::atomic<std::uint64_t> timeout_events;
  std::atomic<std::uint64_t> stale_reclaims;
  std::atomic<std::uint64_t> peer_death_events;
  std::atomic<std::int32_t> producer_error;
  std::atomic<std::int32_t> consumer_error;
  pid_t producer_pid;
  pid_t consumer_pid;
  std::uint64_t requests;
  std::uint64_t payload_bytes;
  std::size_t events_offset;
  std::size_t payload_offset;
};

struct alignas(64) RingSlot {
  std::atomic<std::uint64_t> sequence;
  std::atomic<std::uint32_t> state;
  std::uint32_t reserved;
  std::uint64_t ticket;
  std::uint64_t payload_bytes;
  std::uint64_t producer_start_ns;
  std::uint64_t producer_write_done_ns;
  std::uint64_t producer_publish_ns;
  std::uint64_t consumer_start_ns;
  std::uint64_t consumer_done_ns;
  std::uint64_t mismatches;
};

struct RingEvent {
  std::uint64_t ticket;
  std::uint32_t slot_index;
  std::uint32_t state;
  std::uint64_t producer_start_ns;
  std::uint64_t producer_write_done_ns;
  std::uint64_t producer_publish_ns;
  std::uint64_t consumer_start_ns;
  std::uint64_t consumer_done_ns;
  std::uint64_t mismatches;
};

struct Mapping {
  void* base = nullptr;
  std::size_t bytes = 0;
  RingHeader* header = nullptr;
  RingSlot* slots = nullptr;
  RingEvent* events = nullptr;
  std::uint8_t* payload = nullptr;
};

static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

[[nodiscard]] std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

[[nodiscard]] std::size_t align_up(const std::size_t value,
                                   const std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    fail("ring mapping size overflow");
  }
  return value + increment;
}

[[nodiscard]] std::size_t checked_add(const std::size_t left,
                                      const std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    fail("ring mapping size overflow");
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_mul(const std::size_t left,
                                      const std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    fail("ring mapping size overflow");
  }
  return left * right;
}

[[nodiscard]] std::uint64_t parse_u64(const std::string& value,
                                      const std::string_view name,
                                      const bool allow_zero) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || (!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<std::uint64_t>::max()) {
    fail("invalid " + std::string(name) + ": " + value);
  }
  return static_cast<std::uint64_t>(parsed);
}

[[nodiscard]] std::int64_t parse_i64(const std::string& value,
                                     const std::string_view name) {
  std::size_t consumed = 0;
  const long long parsed = std::stoll(value, &consumed);
  if (consumed != value.size() || parsed < -1 ||
      parsed > std::numeric_limits<std::int64_t>::max()) {
    fail("invalid " + std::string(name) + ": " + value);
  }
  return static_cast<std::int64_t>(parsed);
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
    options.mps_pipe = value;
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
    } else if (argument == "--mps-pipe") {
      options.mps_pipe = require_value();
    } else if (argument == "--payload-bytes") {
      options.payload_bytes = static_cast<std::size_t>(
          parse_u64(require_value(), "payload-bytes", false));
    } else if (argument == "--requests") {
      options.requests = parse_u64(require_value(), "requests", false);
    } else if (argument == "--timeout-ms") {
      options.timeout_ms = parse_u64(require_value(), "timeout-ms", false);
    } else if (argument == "--stale-timeout-us") {
      options.stale_timeout_us =
          parse_u64(require_value(), "stale-timeout-us", false);
    } else if (argument == "--consumer-delay-us") {
      options.consumer_delay_us =
          parse_u64(require_value(), "consumer-delay-us", true);
    } else if (argument == "--fail-consumer-after") {
      options.fail_consumer_after =
          parse_i64(require_value(), "fail-consumer-after");
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
      options.cache_flush_bytes = static_cast<std::size_t>(
          parse_u64(require_value(), "cache-flush-bytes", false));
    } else if (argument == "--help") {
      std::cout
          << "Usage: jdg-mig-sysmem-ring --producer MIG_UUID --consumer MIG_UUID "
             "[--mps-pipe PATH] [--payload-bytes N] [--requests N] "
             "[--timeout-ms N] [--stale-timeout-us N] "
             "[--consumer-delay-us N] [--fail-consumer-after N] "
             "[--cache-state warm|cold] [--cache-flush-bytes N]\n";
      std::exit(0);
    } else {
      fail("unknown argument: " + argument);
    }
  }
  if (options.producer_uuid.empty() || options.consumer_uuid.empty()) {
    fail("producer and consumer MIG UUIDs are required");
  }
  if (options.producer_uuid == options.consumer_uuid && options.mps_pipe.empty()) {
    fail("same-instance ring requires --mps-pipe");
  }
  if (options.payload_bytes > std::numeric_limits<std::size_t>::max() / 3U) {
    fail("payload-bytes is too large for a three-slot ring");
  }
  if (options.fail_consumer_after >= 0 &&
      static_cast<std::uint64_t>(options.fail_consumer_after) >=
          options.requests) {
    fail("fail-consumer-after must be less than requests");
  }
  return options;
}

[[nodiscard]] Mapping make_mapping(const Options& options) {
  const std::size_t slots_offset =
      align_up(sizeof(RingHeader), alignof(RingSlot));
  const std::size_t events_offset = align_up(
      checked_add(slots_offset, checked_mul(kRingSlots, sizeof(RingSlot))),
      alignof(RingEvent));
  const std::size_t payload_offset = align_up(
      checked_add(events_offset, checked_mul(options.requests, sizeof(RingEvent))),
      4096);
  const std::size_t total_bytes = checked_add(
      payload_offset, checked_mul(kRingSlots, options.payload_bytes));
  void* const base = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) {
    fail("ring mmap failed: " + std::string(std::strerror(errno)));
  }
  auto* const bytes = static_cast<std::byte*>(base);
  Mapping mapping;
  mapping.base = base;
  mapping.bytes = total_bytes;
  mapping.header = new (bytes) RingHeader{};
  mapping.slots = reinterpret_cast<RingSlot*>(bytes + slots_offset);
  mapping.events = reinterpret_cast<RingEvent*>(bytes + events_offset);
  mapping.payload = reinterpret_cast<std::uint8_t*>(bytes + payload_offset);
  mapping.header->requests = options.requests;
  mapping.header->payload_bytes = options.payload_bytes;
  mapping.header->events_offset = events_offset;
  mapping.header->payload_offset = payload_offset;
  for (std::uint32_t index = 0; index < kRingSlots; ++index) {
    new (&mapping.slots[index]) RingSlot{};
    mapping.slots[index].sequence.store(index, std::memory_order_relaxed);
    mapping.slots[index].state.store(kFree, std::memory_order_relaxed);
  }
  for (std::uint64_t ticket = 0; ticket < options.requests; ++ticket) {
    new (&mapping.events[ticket]) RingEvent{};
    mapping.events[ticket].ticket = ticket;
    mapping.events[ticket].slot_index =
        static_cast<std::uint32_t>(ticket % kRingSlots);
  }
  return mapping;
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

[[nodiscard]] cudaError_t initialize_device() {
  cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (status != cudaSuccess && status != cudaErrorSetOnActiveProcess) {
    return status;
  }
  return cudaSetDevice(0);
}

__global__ void write_pattern(std::uint8_t* output, const std::size_t size,
                              const std::uint64_t ticket) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < size; index += stride) {
    output[index] = static_cast<std::uint8_t>(
        (index * 131U + static_cast<std::size_t>(ticket) * 17U + size) &
        0xffU);
  }
}

__global__ void verify_pattern(const std::uint8_t* input, const std::size_t size,
                               const std::uint64_t ticket,
                               unsigned long long* mismatches) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  unsigned long long local = 0;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < size; index += stride) {
    const auto expected = static_cast<std::uint8_t>(
        (index * 131U + static_cast<std::size_t>(ticket) * 17U + size) &
        0xffU);
    local += input[index] == expected ? 0ULL : 1ULL;
  }
  if (local != 0) {
    atomicAdd(mismatches, local);
  }
}

__global__ void evict_cache(std::uint8_t* output, const std::size_t size,
                            const std::uint64_t iteration) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < size; index += stride) {
    output[index] = static_cast<std::uint8_t>(
        (index * 17U + static_cast<std::size_t>(iteration) * 29U) & 0xffU);
  }
}

void prepare_cache(const Options& options, std::uint8_t* const scratch,
                   const std::uint64_t ticket) {
  if (options.cache_state != CacheState::kCold) {
    return;
  }
  evict_cache<<<256, 256>>>(scratch, options.cache_flush_bytes, ticket);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    fail("cache-state preparation failed: " +
         std::string(cudaGetErrorName(status)));
  }
}

[[nodiscard]] std::string_view cache_state_name(const CacheState state) {
  return state == CacheState::kCold ? "cold" : "warm";
}

[[nodiscard]] bool peer_alive(const std::atomic<std::uint32_t>& ready,
                              const std::atomic<std::uint32_t>& alive,
                              const pid_t pid) {
  if (ready.load(std::memory_order_acquire) == 0) {
    return true;
  }
  if (alive.load(std::memory_order_acquire) != 0) {
    if (pid <= 0) {
      return true;
    }
    std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
    std::string stat_line;
    if (!std::getline(stat_file, stat_line)) {
      return false;
    }
    const std::size_t comm_end = stat_line.rfind(')');
    return comm_end != std::string::npos &&
           comm_end + 2 < stat_line.size() &&
           stat_line[comm_end + 2] != 'Z';
  }
  return false;
}

void reclaim_stale_slots(Mapping* mapping, const Options& options) {
  const std::uint64_t now = monotonic_ns();
  const std::uint64_t stale_ns =
      options.stale_timeout_us * kNanosecondsPerMicrosecond;
  for (std::uint32_t index = 0; index < kRingSlots; ++index) {
    RingSlot& slot = mapping->slots[index];
    const std::uint32_t state = slot.state.load(std::memory_order_acquire);
    if (state != kReady && state != kConsuming) {
      continue;
    }
    if (now < slot.producer_publish_ns ||
        now - slot.producer_publish_ns < stale_ns) {
      continue;
    }
    std::uint32_t expected = state;
    if (!slot.state.compare_exchange_strong(expected, kAborted,
                                            std::memory_order_acq_rel)) {
      continue;
    }
    const std::uint64_t ticket = slot.ticket;
    mapping->events[ticket].state = kAborted;
    slot.state.store(kFree, std::memory_order_relaxed);
    slot.sequence.store(ticket + kRingSlots, std::memory_order_release);
    mapping->header->stale_reclaims.fetch_add(1, std::memory_order_relaxed);
  }
}

[[nodiscard]] bool wait_for_producer_slot(Mapping* mapping, RingSlot* slot,
                                           const std::uint64_t expected_sequence,
                                           const Options& options) {
  const std::uint64_t deadline =
      monotonic_ns() + options.timeout_ms * 1'000'000U;
  bool waited = false;
  while (true) {
    if (slot->sequence.load(std::memory_order_acquire) == expected_sequence &&
        slot->state.load(std::memory_order_acquire) == kFree) {
      return true;
    }
    if (!peer_alive(mapping->header->consumer_ready,
                    mapping->header->consumer_alive,
                    mapping->header->consumer_pid)) {
      mapping->header->peer_death_events.fetch_add(1,
                                                   std::memory_order_relaxed);
      const std::uint64_t initial_reclaims =
          mapping->header->stale_reclaims.load(std::memory_order_acquire);
      const std::uint64_t stale_deadline =
          monotonic_ns() + options.stale_timeout_us *
                              kNanosecondsPerMicrosecond;
      while (mapping->header->stale_reclaims.load(std::memory_order_acquire) ==
                 initial_reclaims &&
             monotonic_ns() < stale_deadline) {
        reclaim_stale_slots(mapping, options);
        std::this_thread::yield();
      }
      reclaim_stale_slots(mapping, options);
      return false;
    }
    if (monotonic_ns() >= deadline) {
      mapping->header->timeout_events.fetch_add(1,
                                                std::memory_order_relaxed);
      return false;
    }
    if (!waited) {
      mapping->header->backpressure_events.fetch_add(1,
                                                     std::memory_order_relaxed);
      waited = true;
    }
    std::this_thread::yield();
  }
}

[[nodiscard]] bool wait_for_consumer_slot(Mapping* mapping, RingSlot* slot,
                                           const std::uint64_t expected_sequence,
                                           const Options& options) {
  const std::uint64_t deadline =
      monotonic_ns() + options.timeout_ms * 1'000'000U;
  while (true) {
    if (slot->sequence.load(std::memory_order_acquire) == expected_sequence &&
        slot->state.load(std::memory_order_acquire) == kReady) {
      return true;
    }
    if (!peer_alive(mapping->header->producer_ready,
                    mapping->header->producer_alive,
                    mapping->header->producer_pid)) {
      mapping->header->peer_death_events.fetch_add(1,
                                                   std::memory_order_relaxed);
      return false;
    }
    if (monotonic_ns() >= deadline) {
      mapping->header->timeout_events.fetch_add(1,
                                                std::memory_order_relaxed);
      return false;
    }
    std::this_thread::yield();
  }
}

int producer_main(const Options& options, Mapping* mapping) {
  bool registered = false;
  void* device_base = nullptr;
  std::uint8_t* cache_scratch = nullptr;
  cudaError_t status = cudaSuccess;
  try {
    set_device_environment(options.producer_uuid, options.mps_pipe);
    status = initialize_device();
    if (status == cudaSuccess) {
      status = cudaHostRegister(mapping->base, mapping->bytes,
                                cudaHostRegisterMapped);
      registered = status == cudaSuccess;
    }
    if (status == cudaSuccess) {
      status = cudaHostGetDevicePointer(&device_base, mapping->base, 0);
    }
    if (status == cudaSuccess && options.cache_state == CacheState::kCold) {
      status = cudaMalloc(reinterpret_cast<void**>(&cache_scratch),
                          options.cache_flush_bytes);
    }
    mapping->header->producer_error.store(static_cast<std::int32_t>(status),
                                          std::memory_order_release);
    mapping->header->producer_ready.store(1, std::memory_order_release);
    if (status != cudaSuccess) {
      mapping->header->producer_alive.store(0, std::memory_order_release);
      if (registered) {
        static_cast<void>(cudaHostUnregister(mapping->base));
      }
      if (cache_scratch != nullptr) {
        static_cast<void>(cudaFree(cache_scratch));
      }
      return 5;
    }
    mapping->header->producer_alive.store(1, std::memory_order_release);
    while (mapping->header->start.load(std::memory_order_acquire) == 0) {
      std::this_thread::yield();
    }
    auto* const device_bytes = static_cast<std::uint8_t*>(device_base);
    for (std::uint64_t ticket = 0; ticket < options.requests; ++ticket) {
      RingSlot& slot = mapping->slots[ticket % kRingSlots];
      if (!wait_for_producer_slot(mapping, &slot, ticket, options)) {
        break;
      }
      RingEvent& event = mapping->events[ticket];
      const std::size_t offset = mapping->header->payload_offset +
                                 static_cast<std::size_t>(ticket % kRingSlots) *
                                     options.payload_bytes;
      auto* const device_payload = device_bytes + offset;
      slot.ticket = ticket;
      slot.payload_bytes = options.payload_bytes;
      event.ticket = ticket;
      event.slot_index = static_cast<std::uint32_t>(ticket % kRingSlots);
      event.state = kReady;
      prepare_cache(options, cache_scratch, ticket);
      slot.producer_start_ns = monotonic_ns();
      event.producer_start_ns = slot.producer_start_ns;
      write_pattern<<<256, 256>>>(device_payload, options.payload_bytes,
                                  ticket);
      status = cudaGetLastError();
      if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
      }
      if (status != cudaSuccess) {
        mapping->header->producer_error.store(static_cast<std::int32_t>(status),
                                              std::memory_order_release);
        break;
      }
      slot.producer_write_done_ns = monotonic_ns();
      event.producer_write_done_ns = slot.producer_write_done_ns;
      slot.producer_publish_ns = monotonic_ns();
      event.producer_publish_ns = slot.producer_publish_ns;
      slot.state.store(kReady, std::memory_order_relaxed);
      mapping->header->ready_transitions.fetch_add(1,
                                                   std::memory_order_relaxed);
      slot.sequence.store(ticket + 1, std::memory_order_release);
      mapping->header->published.fetch_add(1, std::memory_order_relaxed);
    }
  } catch (...) {
    status = cudaErrorUnknown;
    mapping->header->producer_error.store(static_cast<std::int32_t>(status),
                                          std::memory_order_release);
  }
  mapping->header->producer_alive.store(0, std::memory_order_release);
  if (registered) {
    const cudaError_t unregister_status = cudaHostUnregister(mapping->base);
    if (status == cudaSuccess) {
      status = unregister_status;
    }
  }
  if (cache_scratch != nullptr) {
    const cudaError_t free_status = cudaFree(cache_scratch);
    if (status == cudaSuccess) {
      status = free_status;
    }
  }
  return status == cudaSuccess ? 0 : 6;
}

int consumer_main(const Options& options, Mapping* mapping) {
  bool registered = false;
  void* device_base = nullptr;
  unsigned long long* device_mismatches = nullptr;
  std::uint8_t* cache_scratch = nullptr;
  cudaError_t status = cudaSuccess;
  try {
    set_device_environment(
        options.consumer_uuid,
        options.producer_uuid == options.consumer_uuid ? options.mps_pipe : "");
    status = initialize_device();
    if (status == cudaSuccess) {
      status = cudaHostRegister(mapping->base, mapping->bytes,
                                cudaHostRegisterMapped);
      registered = status == cudaSuccess;
    }
    if (status == cudaSuccess) {
      status = cudaHostGetDevicePointer(&device_base, mapping->base, 0);
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(&device_mismatches, sizeof(*device_mismatches));
    }
    if (status == cudaSuccess && options.cache_state == CacheState::kCold) {
      status = cudaMalloc(reinterpret_cast<void**>(&cache_scratch),
                          options.cache_flush_bytes);
    }
    mapping->header->consumer_error.store(static_cast<std::int32_t>(status),
                                          std::memory_order_release);
    mapping->header->consumer_ready.store(1, std::memory_order_release);
    if (status != cudaSuccess) {
      mapping->header->consumer_alive.store(0, std::memory_order_release);
      if (device_mismatches != nullptr) {
        static_cast<void>(cudaFree(device_mismatches));
      }
      if (registered) {
        static_cast<void>(cudaHostUnregister(mapping->base));
      }
      if (cache_scratch != nullptr) {
        static_cast<void>(cudaFree(cache_scratch));
      }
      return 7;
    }
    mapping->header->consumer_alive.store(1, std::memory_order_release);
    while (mapping->header->start.load(std::memory_order_acquire) == 0) {
      std::this_thread::yield();
    }
    auto* const device_bytes = static_cast<std::uint8_t*>(device_base);
    for (std::uint64_t ticket = 0; ticket < options.requests; ++ticket) {
      if (options.fail_consumer_after >= 0 &&
          ticket >= static_cast<std::uint64_t>(options.fail_consumer_after)) {
        mapping->header->consumer_error.store(kFaultExit,
                                              std::memory_order_release);
        mapping->header->consumer_alive.store(0, std::memory_order_release);
        static_cast<void>(cudaFree(device_mismatches));
        static_cast<void>(cudaHostUnregister(mapping->base));
        if (cache_scratch != nullptr) {
          static_cast<void>(cudaFree(cache_scratch));
        }
        return kFaultExit;
      }
      RingSlot& slot = mapping->slots[ticket % kRingSlots];
      if (!wait_for_consumer_slot(mapping, &slot, ticket + 1, options)) {
        break;
      }
      std::uint32_t expected_state = kReady;
      if (!slot.state.compare_exchange_strong(expected_state, kConsuming,
                                              std::memory_order_acq_rel)) {
        --ticket;
        continue;
      }
      mapping->header->consuming_transitions.fetch_add(1,
                                                       std::memory_order_relaxed);
      RingEvent& event = mapping->events[ticket];
      slot.consumer_start_ns = monotonic_ns();
      event.consumer_start_ns = slot.consumer_start_ns;
      const std::size_t offset = mapping->header->payload_offset +
                                 static_cast<std::size_t>(ticket % kRingSlots) *
                                     options.payload_bytes;
      const auto* const device_payload = device_bytes + offset;
      prepare_cache(options, cache_scratch, ticket);
      status = cudaMemset(device_mismatches, 0, sizeof(*device_mismatches));
      if (status == cudaSuccess) {
        verify_pattern<<<256, 256>>>(device_payload, options.payload_bytes,
                                     ticket, device_mismatches);
        status = cudaGetLastError();
      }
      if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
      }
      if (status == cudaSuccess) {
        status = cudaMemcpy(&slot.mismatches, device_mismatches,
                            sizeof(slot.mismatches), cudaMemcpyDeviceToHost);
      }
      slot.consumer_done_ns = monotonic_ns();
      event.consumer_done_ns = slot.consumer_done_ns;
      event.mismatches = slot.mismatches;
      event.state = status == cudaSuccess ? kDone : kAborted;
      if (status != cudaSuccess) {
        mapping->header->consumer_error.store(static_cast<std::int32_t>(status),
                                              std::memory_order_release);
        break;
      }
      if (options.consumer_delay_us != 0) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(options.consumer_delay_us));
      }
      slot.state.store(kDone, std::memory_order_relaxed);
      slot.state.store(kFree, std::memory_order_relaxed);
      slot.sequence.store(ticket + kRingSlots,
                          std::memory_order_release);
      mapping->header->completed.fetch_add(1, std::memory_order_relaxed);
    }
  } catch (...) {
    status = cudaErrorUnknown;
    mapping->header->consumer_error.store(static_cast<std::int32_t>(status),
                                          std::memory_order_release);
  }
  mapping->header->consumer_alive.store(0, std::memory_order_release);
  if (device_mismatches != nullptr) {
    const cudaError_t free_status = cudaFree(device_mismatches);
    if (status == cudaSuccess) {
      status = free_status;
    }
  }
  if (registered) {
    const cudaError_t unregister_status = cudaHostUnregister(mapping->base);
    if (status == cudaSuccess) {
      status = unregister_status;
    }
  }
  if (cache_scratch != nullptr) {
    const cudaError_t free_status = cudaFree(cache_scratch);
    if (status == cudaSuccess) {
      status = free_status;
    }
  }
  return status == cudaSuccess ? 0 : 8;
}

[[nodiscard]] int wait_child(const pid_t pid) {
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      return 255;
    }
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return 255;
}

[[nodiscard]] bool wait_ready(const Mapping& mapping, const pid_t producer,
                              const pid_t consumer) {
  const std::uint64_t deadline =
      monotonic_ns() + kDefaultTimeoutMs * 1'000'000U;
  while (mapping.header->producer_ready.load(std::memory_order_acquire) == 0 ||
         mapping.header->consumer_ready.load(std::memory_order_acquire) == 0) {
    if (monotonic_ns() >= deadline) {
      kill(producer, SIGTERM);
      kill(consumer, SIGTERM);
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

void print_event(const RingEvent& event) {
  std::cout << "{\"ticket\":" << event.ticket
            << ",\"slot\":" << event.slot_index << ",\"state\":"
            << event.state << ",\"producer_start_ns\":"
            << event.producer_start_ns << ",\"producer_write_done_ns\":"
            << event.producer_write_done_ns << ",\"producer_publish_ns\":"
            << event.producer_publish_ns << ",\"consumer_start_ns\":"
            << event.consumer_start_ns << ",\"consumer_done_ns\":"
            << event.consumer_done_ns << ",\"mismatches\":"
            << event.mismatches << "}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Mapping mapping = make_mapping(options);
    const pid_t producer = fork();
    if (producer == 0) {
      _exit(producer_main(options, &mapping));
    }
    if (producer < 0) {
      munmap(mapping.base, mapping.bytes);
      fail("producer fork failed");
    }
    mapping.header->producer_pid = producer;
    const pid_t consumer = fork();
    if (consumer == 0) {
      _exit(consumer_main(options, &mapping));
    }
    if (consumer < 0) {
      kill(producer, SIGTERM);
      static_cast<void>(wait_child(producer));
      munmap(mapping.base, mapping.bytes);
      fail("consumer fork failed");
    }
    mapping.header->consumer_pid = consumer;
    if (!wait_ready(mapping, producer, consumer)) {
      static_cast<void>(wait_child(producer));
      static_cast<void>(wait_child(consumer));
      munmap(mapping.base, mapping.bytes);
      fail("ring child readiness timed out");
    }
    mapping.header->start.store(1, std::memory_order_release);
    const int producer_status = wait_child(producer);
    const int consumer_status = wait_child(consumer);

    const std::uint64_t published =
        mapping.header->published.load(std::memory_order_acquire);
    const std::uint64_t completed =
        mapping.header->completed.load(std::memory_order_acquire);
    const std::uint64_t stale_reclaims =
        mapping.header->stale_reclaims.load(std::memory_order_acquire);
    const bool fault_mode = options.fail_consumer_after >= 0;
    const bool normal_success =
        !fault_mode && producer_status == 0 && consumer_status == 0 &&
        published == options.requests && completed == options.requests &&
        mapping.header->timeout_events.load(std::memory_order_acquire) == 0;
    const bool fault_success =
        fault_mode && producer_status == 0 && consumer_status == kFaultExit &&
        stale_reclaims > 0;
    const bool success = normal_success || fault_success;

    std::cout << "{\"schema_version\":" << kProtocolVersion
              << ",\"ring_schema\":\"JDG_RING1\",\"status\":\""
              << (normal_success ? "ok"
                                  : fault_success ? "fault-ok" : "error")
              << "\",\"transport\":\"full-coherent-registered-system-memory"
                 "\",\"transport_description\":\""
              << kTransportDescription << "\",\"producer_uuid\":\""
              << options.producer_uuid << "\",\"consumer_uuid\":\""
              << options.consumer_uuid << "\",\"topology\":\""
              << (options.producer_uuid == options.consumer_uuid
                      ? "same-instance-mps"
                      : "cross-mig")
              << "\",\"ring_slots\":" << kRingSlots
              << ",\"queue_depth\":" << kRingSlots
              << ",\"cache_state\":\""
              << cache_state_name(options.cache_state)
              << "\",\"cache_flush_bytes\":"
              << options.cache_flush_bytes
              << ",\"payload_bytes\":" << options.payload_bytes
              << ",\"requests\":" << options.requests
              << ",\"producer_status\":" << producer_status
              << ",\"consumer_status\":" << consumer_status
              << ",\"counters\":{\"published\":" << published
              << ",\"completed\":"
              << completed << ",\"ready_transitions\":"
              << mapping.header->ready_transitions.load()
              << ",\"consuming_transitions\":"
              << mapping.header->consuming_transitions.load()
              << ",\"backpressure_events\":"
              << mapping.header->backpressure_events.load()
              << ",\"timeout_events\":"
              << mapping.header->timeout_events.load()
              << ",\"stale_reclaims\":" << stale_reclaims
              << ",\"peer_death_events\":"
              << mapping.header->peer_death_events.load() << "}"
              << ",\"consumer_delay_us\":" << options.consumer_delay_us
              << ",\"fail_consumer_after\":" << options.fail_consumer_after
              << ",\"events\":[";
    for (std::uint64_t ticket = 0; ticket < options.requests; ++ticket) {
      if (ticket != 0) {
        std::cout << ',';
      }
      print_event(mapping.events[ticket]);
    }
    std::cout << "]}\n";
    munmap(mapping.base, mapping.bytes);
    return success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema_version\":" << kProtocolVersion
              << ",\"ring_schema\":\"JDG_RING1\",\"status\":\"error\""
                 ",\"message\":\""
              << error.what() << "\"}\n";
    return 1;
  }
}
