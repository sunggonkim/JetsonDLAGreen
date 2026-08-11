#include <iostream>

#include "jetson_dla_green/trt_benchmark.hpp"

int main(const int argc, char** argv) {
  return jdg::run_trt_benchmark(argc, argv, std::cout, std::cerr);
}
