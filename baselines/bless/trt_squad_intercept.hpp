#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace bless::thor {

struct SquadStats {
  std::uint64_t logical_launches{};
  std::uint64_t physical_launches{};
  std::uint64_t shadow_launches{};
  std::uint64_t restricted_launches{};
  std::uint64_t unrestricted_launches{};
  std::uint64_t activation_copies{};
  std::uint64_t signature_mismatches{};
  unsigned int last_selected_sms{};
};

}  // namespace bless::thor

extern "C" {

int bless_trt_squad_register_replica(CUcontext context, unsigned int sms,
                                     CUdeviceptr activation,
                                     std::size_t activation_bytes);
int bless_trt_squad_start(const char* trace_path);
int bless_trt_squad_stop();
int bless_trt_squad_stats(bless::thor::SquadStats* output);

}
