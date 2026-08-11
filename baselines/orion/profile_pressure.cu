#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

enum class Mode { kCompute, kMemory };

struct Options {
  Mode mode{Mode::kCompute};
  double duration_seconds{20.0};
  std::filesystem::path ready_file;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check(const cudaError_t status, const std::string_view operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

Options parse(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) {
      fail(std::string(argv[index]) + " requires a value");
    }
    const std::string_view key(argv[index]);
    const std::string value(argv[++index]);
    if (key == "--mode") {
      if (value == "compute") {
        options.mode = Mode::kCompute;
      } else if (value == "memory") {
        options.mode = Mode::kMemory;
      } else {
        fail("--mode expects compute or memory");
      }
    } else if (key == "--duration-seconds") {
      options.duration_seconds = std::stod(value);
    } else if (key == "--ready-file") {
      options.ready_file = value;
    } else {
      fail("unknown option: " + std::string(key));
    }
  }
  if (!(options.duration_seconds > 0.0) || options.ready_file.empty()) {
    fail("positive --duration-seconds and --ready-file are required");
  }
  return options;
}

__global__ void compute_pressure(float* data, const std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float value = data[index % elements] + static_cast<float>(index & 255U);
#pragma unroll 4
  for (int iteration = 0; iteration < 4096; ++iteration) {
    value = fmaf(value, 1.00000011920928955078125F,
                 0.000000059604644775390625F);
    value = fmaf(value, 0.999999940395355224609375F,
                 -0.0000000298023223876953125F);
  }
  data[index % elements] = value;
}

__global__ void memory_pressure(float* data, const std::size_t elements,
                                const std::size_t stride) {
  std::size_t index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) %
      elements;
  float value = data[index];
  for (int iteration = 0; iteration < 64; ++iteration) {
    index = (index + stride) % elements;
    value += data[index];
    data[index] = value * 0.999999940395355224609375F;
  }
}

int run(const Options& options) {
  int device = 0;
  check(cudaSetDevice(device), "cudaSetDevice");
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
  constexpr std::size_t elements = 32U * 1024U * 1024U;
  float* data = nullptr;
  check(cudaMalloc(&data, elements * sizeof(float)), "cudaMalloc");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  check(cudaMemsetAsync(data, 1, elements * sizeof(float), stream),
        "cudaMemsetAsync");
  const int blocks = properties.multiProcessorCount * 8;
  std::uint64_t launches = 0;
  const auto begin = std::chrono::steady_clock::now();
  const auto deadline = begin + std::chrono::duration<double>(options.duration_seconds);
  bool ready = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (options.mode == Mode::kCompute) {
      compute_pressure<<<blocks, 256, 0, stream>>>(data, elements);
    } else {
      memory_pressure<<<blocks, 256, 0, stream>>>(data, elements, 4099U);
    }
    check(cudaGetLastError(), "profile pressure launch");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    ++launches;
    if (!ready) {
      std::ofstream marker(options.ready_file, std::ios::out | std::ios::trunc);
      if (!marker) {
        fail("failed to create ready file");
      }
      marker << "ready\n";
      marker.flush();
      if (!marker) {
        fail("failed to flush ready file");
      }
      ready = true;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFree(data), "cudaFree");
  const double elapsed = std::chrono::duration<double>(end - begin).count();
  std::cout << "{\"schema_version\":1,\"mode\":\""
            << (options.mode == Mode::kCompute ? "compute" : "memory")
            << "\",\"gpu\":\"" << properties.name
            << "\",\"multiprocessors\":" << properties.multiProcessorCount
            << ",\"elapsed_seconds\":" << elapsed
            << ",\"completed_launches\":" << launches << "}\n";
  return 0;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
