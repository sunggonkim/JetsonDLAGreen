#pragma once

#include <cuda.h>

#include <cstdint>

namespace orion::thor {

using LaunchEx = CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**);

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

CUresult submit_launch_ex(const char* api, LaunchEx real,
                          const CUlaunchConfig* config, CUfunction function,
                          void** kernel_params, void** extra);

}  // namespace orion::thor

extern "C" {

// Starts the in-process scheduler used by the TensorRT managed-client port.
// initial_gate_clients is a positive-control barrier; production evaluations
// must pass zero and report that fact in their provenance.
int orion_trt_scheduler_start(const char* decision_trace,
                              int initial_gate_clients);
// Starts Orion's profile-aware HP/BE scheduler. Profile files use the strict
// orion-thor-profile-v1 TSV emitted by profile_thor.py. max_be_duration_us is
// the upstream Orion bound on aggregate in-flight BE kernel duration.
int orion_trt_scheduler_start_profiled(const char* decision_trace,
                                       const char* best_effort_profile,
                                       const char* high_priority_profile,
                                       double max_be_duration_us);
int orion_trt_register_client(int client_id, int high_priority);
int orion_trt_scheduler_stop();
int orion_trt_scheduler_stats(orion::thor::SchedulerStats* output);

}
