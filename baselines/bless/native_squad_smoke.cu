#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<unsigned int, 4> kShares{2U, 4U, 6U, 8U};
constexpr std::array<std::array<unsigned int, 2>, 3> kStrictSplits{
    std::array<unsigned int, 2>{2U, 6U},
    std::array<unsigned int, 2>{4U, 4U},
    std::array<unsigned int, 2>{6U, 2U}};
constexpr std::size_t kApplications = 2U;
constexpr std::size_t kKernelsPerRequest = 12U;
constexpr std::size_t kMaximumSquadKernels = 6U;
constexpr unsigned long long kCycles = 120'000ULL;

__global__ void profiled_kernel(std::uint64_t* state,
                                const std::uint64_t contribution,
                                const unsigned long long cycles) {
  const unsigned long long begin = clock64();
  while (clock64() - begin < cycles) {
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *state = (*state * 1315423911ULL) ^ contribution;
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

void require_runtime(const cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorName(result));
  }
}

struct Context {
  CUcontext handle{};
  cudaStream_t stream{};
  unsigned int sms{};
  std::uint64_t* state{};
};

struct Application {
  std::array<Context, kShares.size()> contexts;
  std::array<std::array<double, kShares.size()>, kKernelsPerRequest>
      duration_us{};
  std::array<double, kKernelsPerRequest> cumulative_us{};
  std::uint64_t* host_state{};
  std::uint64_t expected_state{};
  std::size_t cursor{};
  double elapsed_us{};
};

struct Scheduled {
  std::size_t application{};
  std::size_t kernel{};
};

struct Configuration {
  bool unrestricted{};
  std::array<unsigned int, kApplications> shares{8U, 8U};
  double predicted_us{std::numeric_limits<double>::infinity()};
};

std::size_t share_index(const unsigned int share) {
  const auto found = std::find(kShares.begin(), kShares.end(), share);
  if (found == kShares.end()) {
    throw std::runtime_error("unprofiled BLESS share");
  }
  return static_cast<std::size_t>(found - kShares.begin());
}

void set_current(const CUcontext context) {
  require_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

std::uint64_t contribution(const std::size_t application,
                           const std::size_t kernel) {
  return (static_cast<std::uint64_t>(application + 1U) << 56U) ^
         ((static_cast<std::uint64_t>(kernel) + 1U) *
          0x9e3779b97f4a7c15ULL);
}

void launch(Context& context, const std::size_t application,
            const std::size_t kernel) {
  set_current(context.handle);
  profiled_kernel<<<32, 128, 0, context.stream>>>(
      context.state, contribution(application, kernel), kCycles);
  require_runtime(cudaGetLastError(), "profiled_kernel launch");
}

void synchronize(Context& context) {
  set_current(context.handle);
  require_runtime(cudaStreamSynchronize(context.stream),
                  "cudaStreamSynchronize");
}

std::vector<Scheduled> form_squad(
    const std::array<Application, kApplications>& applications) {
  std::array<std::size_t, kApplications> cursors{};
  for (std::size_t app = 0; app < applications.size(); ++app) {
    cursors[app] = applications[app].cursor;
  }
  std::vector<Scheduled> squad;
  while (squad.size() < kMaximumSquadKernels) {
    std::size_t selected = kApplications;
    double smallest = std::numeric_limits<double>::infinity();
    for (std::size_t app = 0; app < applications.size(); ++app) {
      if (cursors[app] >= kKernelsPerRequest) {
        continue;
      }
      const double expected = applications[app].cumulative_us[cursors[app]];
      const double relative = applications[app].elapsed_us / expected;
      if (relative < smallest) {
        smallest = relative;
        selected = app;
      }
    }
    if (selected == kApplications) {
      break;
    }
    squad.push_back({selected, cursors[selected]});
    ++cursors[selected];
    if (cursors[selected] == kKernelsPerRequest) {
      break;
    }
  }
  return squad;
}

Configuration choose_configuration(
    const std::array<Application, kApplications>& applications,
    const std::vector<Scheduled>& squad) {
  Configuration best;
  for (const auto& split : kStrictSplits) {
    std::array<double, kApplications> stacks{};
    for (const auto& item : squad) {
      stacks[item.application] +=
          applications[item.application]
              .duration_us[item.kernel][share_index(split[item.application])];
    }
    const double predicted = *std::max_element(stacks.begin(), stacks.end());
    if (predicted < best.predicted_us) {
      best = {false, split, predicted};
    }
  }
  double unrestricted = 0.0;
  for (const auto& item : squad) {
    unrestricted += applications[item.application]
                        .duration_us[item.kernel][share_index(8U)];
  }
  if (unrestricted < best.predicted_us) {
    best = {true, {8U, 8U}, unrestricted};
  }
  return best;
}

void execute_squad(std::array<Application, kApplications>& applications,
                   const std::vector<Scheduled>& squad,
                   const Configuration& configuration) {
  std::array<std::vector<Scheduled>, kApplications> by_application;
  for (const auto& item : squad) {
    by_application[item.application].push_back(item);
  }
  std::array<bool, kApplications> restricted_launched{};
  for (std::size_t app = 0; app < applications.size(); ++app) {
    const auto count = by_application[app].size() / 2U;
    auto& context = applications[app].contexts[share_index(
        configuration.unrestricted ? 8U : configuration.shares[app])];
    for (std::size_t index = 0; index < count; ++index) {
      launch(context, app, by_application[app][index].kernel);
      restricted_launched[app] = true;
    }
  }
  for (std::size_t app = 0; app < applications.size(); ++app) {
    if (restricted_launched[app]) {
      auto& context = applications[app].contexts[share_index(
          configuration.unrestricted ? 8U : configuration.shares[app])];
      synchronize(context);
    }
  }
  for (std::size_t app = 0; app < applications.size(); ++app) {
    const auto count = by_application[app].size() / 2U;
    auto& context = applications[app].contexts[share_index(8U)];
    for (std::size_t index = count; index < by_application[app].size(); ++index) {
      launch(context, app, by_application[app][index].kernel);
    }
  }
  for (std::size_t app = 0; app < applications.size(); ++app) {
    if (by_application[app].size() > by_application[app].size() / 2U) {
      synchronize(applications[app].contexts[share_index(8U)]);
    }
  }
  for (const auto& item : squad) {
    auto& application = applications[item.application];
    application.expected_state =
        (application.expected_state * 1315423911ULL) ^
        contribution(item.application, item.kernel);
    application.cursor = std::max(application.cursor, item.kernel + 1U);
  }
}

void create_contexts(std::array<Application, kApplications>& applications,
                     const CUdevice device) {
  for (auto& application : applications) {
    for (std::size_t index = 0; index < kShares.size(); ++index) {
      auto& context = application.contexts[index];
      CUexecAffinityParam affinity{};
      affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      affinity.param.smCount.val = kShares[index];
      CUctxCreateParams parameters{};
      parameters.execAffinityParams = &affinity;
      parameters.numExecAffinityParams = 1;
      require_driver(cuCtxCreate(&context.handle, &parameters, CU_CTX_MAP_HOST,
                                 device),
                     "cuCtxCreate");
      CUexecAffinityParam observed{};
      observed.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      require_driver(cuCtxGetExecAffinity(&observed,
                                          CU_EXEC_AFFINITY_TYPE_SM_COUNT),
                     "cuCtxGetExecAffinity");
      context.sms = observed.param.smCount.val;
      if (context.sms != kShares[index]) {
        throw std::runtime_error("unexpected BLESS context affinity");
      }
      require_runtime(cudaStreamCreateWithFlags(&context.stream,
                                                cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
    }
  }
}

void allocate_states(std::array<Application, kApplications>& applications) {
  for (std::size_t app = 0; app < applications.size(); ++app) {
    auto& application = applications[app];
    set_current(application.contexts.front().handle);
    require_runtime(cudaHostAlloc(&application.host_state,
                                  sizeof(*application.host_state),
                                  cudaHostAllocMapped),
                    "cudaHostAlloc");
    *application.host_state = 0x123456789abcdef0ULL + app;
    application.expected_state = *application.host_state;
    for (auto& context : application.contexts) {
      set_current(context.handle);
      require_runtime(cudaHostGetDevicePointer(&context.state,
                                               application.host_state, 0),
                      "cudaHostGetDevicePointer");
    }
  }
}

double profile_one(Context& context) {
  set_current(context.handle);
  cudaEvent_t begin{};
  cudaEvent_t end{};
  require_runtime(cudaEventCreate(&begin), "cudaEventCreate(begin)");
  require_runtime(cudaEventCreate(&end), "cudaEventCreate(end)");
  require_runtime(cudaEventRecord(begin, context.stream),
                  "cudaEventRecord(begin)");
  profiled_kernel<<<32, 128, 0, context.stream>>>(context.state, 0U, kCycles);
  require_runtime(cudaGetLastError(), "profile kernel launch");
  require_runtime(cudaEventRecord(end, context.stream), "cudaEventRecord(end)");
  require_runtime(cudaEventSynchronize(end), "cudaEventSynchronize");
  float milliseconds = 0.0F;
  require_runtime(cudaEventElapsedTime(&milliseconds, begin, end),
                  "cudaEventElapsedTime");
  require_runtime(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
  require_runtime(cudaEventDestroy(end), "cudaEventDestroy(end)");
  return static_cast<double>(milliseconds) * 1000.0;
}

void profile(std::array<Application, kApplications>& applications) {
  for (auto& application : applications) {
    std::array<double, kShares.size()> durations{};
    for (std::size_t share = 0; share < kShares.size(); ++share) {
      durations[share] = profile_one(application.contexts[share]);
    }
    double cumulative = 0.0;
    for (std::size_t kernel = 0; kernel < kKernelsPerRequest; ++kernel) {
      application.duration_us[kernel] = durations;
      cumulative += durations.back();
      application.cumulative_us[kernel] = cumulative;
    }
  }
}

void cleanup(std::array<Application, kApplications>& applications) noexcept {
  for (auto& application : applications) {
    if (application.host_state != nullptr &&
        application.contexts.front().handle != nullptr) {
      static_cast<void>(cuCtxSetCurrent(application.contexts.front().handle));
      static_cast<void>(cudaFreeHost(application.host_state));
      application.host_state = nullptr;
    }
    for (auto& context : application.contexts) {
      if (context.handle != nullptr) {
        static_cast<void>(cuCtxSetCurrent(context.handle));
      }
      if (context.stream != nullptr) {
        static_cast<void>(cudaStreamDestroy(context.stream));
        context.stream = nullptr;
      }
      if (context.handle != nullptr) {
        static_cast<void>(cuCtxDestroy(context.handle));
        context.handle = nullptr;
      }
    }
  }
}

}  // namespace

int main(const int argc, char** argv) {
  std::array<Application, kApplications> applications{};
  try {
    if (argc != 2) {
      throw std::invalid_argument("usage: bless-native-squad-smoke TRACE.jsonl");
    }
    const std::filesystem::path trace_path(argv[1]);
    std::ofstream trace(trace_path, std::ios::out | std::ios::trunc);
    if (!trace) {
      throw std::runtime_error("failed to create BLESS squad trace");
    }
    require_driver(cuInit(0), "cuInit");
    CUdevice device{};
    require_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    create_contexts(applications, device);
    allocate_states(applications);
    profile(applications);
    for (std::size_t app = 0; app < applications.size(); ++app) {
      auto& application = applications[app];
      *application.host_state =
          0x123456789abcdef0ULL + static_cast<std::uint64_t>(app);
      application.expected_state = *application.host_state;
    }

    const auto start = std::chrono::steady_clock::now();
    std::size_t sequence = 0U;
    while (std::any_of(applications.begin(), applications.end(),
                       [](const Application& app) {
                         return app.cursor < kKernelsPerRequest;
                       })) {
      const auto squad = form_squad(applications);
      const auto configuration = choose_configuration(applications, squad);
      execute_squad(applications, squad, configuration);
      const double elapsed = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count()) /
          1000.0;
      for (auto& application : applications) {
        application.elapsed_us = elapsed;
      }
      trace << "{\"schema_version\":1,\"sequence\":" << sequence++
            << ",\"kernel_count\":" << squad.size()
            << ",\"estimator\":\""
            << (configuration.unrestricted ? "workload-equivalence"
                                           : "interference-free")
            << "\",\"shares\":[" << configuration.shares[0] << ','
            << configuration.shares[1] << "],\"predicted_us\":"
            << configuration.predicted_us << ",\"cursor\":["
            << applications[0].cursor << ',' << applications[1].cursor
            << "]}\n";
    }
    if (!trace) {
      throw std::runtime_error("failed to write BLESS squad trace");
    }
    const bool passed = std::all_of(
        applications.begin(), applications.end(), [](const Application& app) {
          return *app.host_state == app.expected_state &&
                 app.cursor == kKernelsPerRequest;
        });
    std::cout << "{\n  \"schema_version\": 1,\n"
              << "  \"kind\": \"bless-thor-native-squad-smoke\",\n"
              << "  \"algorithm\": \"relative-progress-kernel-squads\",\n"
              << "  \"maximum_squad_kernels\": 6,\n"
              << "  \"restricted_fraction\": 0.5,\n"
              << "  \"affinity_domain_sms\": [2,4,6,8],\n"
              << "  \"requests\": 2,\n  \"kernels_per_request\": 12,\n"
              << "  \"squads\": " << sequence
              << ",\n  \"checksums\": [" << *applications[0].host_state
              << ',' << *applications[1].host_state
              << "],\n  \"expected_checksums\": ["
              << applications[0].expected_state << ','
              << applications[1].expected_state << "],\n"
              << "  \"status\": \"" << (passed ? "passed" : "failed")
              << "\"\n}\n";
    if (!passed) {
      throw std::runtime_error("BLESS squad checksum differs");
    }
    cleanup(applications);
    return 0;
  } catch (const std::exception& error) {
    cleanup(applications);
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
