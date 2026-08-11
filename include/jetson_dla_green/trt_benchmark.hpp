#pragma once

#include <iosfwd>

namespace jdg {

// Runs the TensorRT benchmark without owning process-global stdout/stderr.
// This entry point lets faithful in-process schedulers host multiple clients
// while retaining the exact standalone benchmark implementation.
int run_trt_benchmark(int argc, char** argv, std::ostream& output,
                      std::ostream& error);

}  // namespace jdg
