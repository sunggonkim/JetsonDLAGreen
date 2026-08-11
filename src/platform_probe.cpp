#include <cuda.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "jetson_dla_green/json.hpp"

namespace {

struct GreenContextInfo {
  bool resource_query_supported{};
  bool context_creation_supported{};
  unsigned int sm_count{};
  unsigned int minimum_partition_size{};
  unsigned int coscheduled_alignment{};
  std::optional<std::string> resource_query_error;
  std::optional<std::string> context_creation_error;
};

struct DeviceInfo {
  int ordinal{};
  std::string name;
  int compute_major{};
  int compute_minor{};
  int multiprocessors{};
  std::size_t memory_bytes{};
  bool unified_addressing{};
  bool concurrent_kernels{};
  GreenContextInfo green_context;
};

struct CudaInfo {
  bool available{};
  int driver_version{};
  std::vector<DeviceInfo> devices;
  std::optional<std::string> error;
};

struct TensorRtInfo {
  bool available{};
  int dla_cores{};
  std::optional<std::string> error;
};

std::string trim(std::string value) {
  const auto is_padding = [](const unsigned char character) {
    return character == 0U || std::isspace(character) != 0;
  };
  while (!value.empty() &&
         is_padding(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  std::size_t first = 0U;
  while (first < value.size() &&
         is_padding(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  return value.substr(first);
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::ostringstream content;
  content << input.rdbuf();
  return trim(content.str());
}

std::string cuda_error(const CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &description));
  std::string message = name == nullptr ? "CUDA_ERROR_UNKNOWN" : name;
  if (description != nullptr) {
    message += ": ";
    message += description;
  }
  return message;
}

int device_attribute(const CUdevice device, const CUdevice_attribute attribute) {
  int value = 0;
  const CUresult result = cuDeviceGetAttribute(&value, attribute, device);
  if (result != CUDA_SUCCESS) {
    return 0;
  }
  return value;
}

CudaInfo probe_cuda() {
  CudaInfo info;
  const CUresult init_result = cuInit(0U);
  if (init_result != CUDA_SUCCESS) {
    info.error = cuda_error(init_result);
    return info;
  }

  info.available = true;
  const CUresult version_result = cuDriverGetVersion(&info.driver_version);
  if (version_result != CUDA_SUCCESS) {
    info.error = cuda_error(version_result);
  }

  int device_count = 0;
  const CUresult count_result = cuDeviceGetCount(&device_count);
  if (count_result != CUDA_SUCCESS) {
    info.available = false;
    info.error = cuda_error(count_result);
    return info;
  }

  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    CUdevice device{};
    if (const CUresult result = cuDeviceGet(&device, ordinal);
        result != CUDA_SUCCESS) {
      continue;
    }

    std::array<char, 256> name{};
    if (cuDeviceGetName(name.data(), static_cast<int>(name.size()), device) !=
        CUDA_SUCCESS) {
      name[0] = '\0';
    }

    std::size_t memory_bytes = 0U;
    static_cast<void>(cuDeviceTotalMem(&memory_bytes, device));

    GreenContextInfo green_context;
    CUdevResource resource{};
    const bool is_mig_device = std::string_view(name.data()).find(" MIG ") !=
                               std::string_view::npos;
    const CUresult resource_result = is_mig_device
                                         ? CUDA_ERROR_NOT_SUPPORTED
                                         : cuDeviceGetDevResource(
                                               device, &resource,
                                               CU_DEV_RESOURCE_TYPE_SM);
    if (is_mig_device) {
      green_context.resource_query_error =
          "skipped: Green Context probing is not valid inside a MIG device";
    } else if (resource_result == CUDA_SUCCESS &&
               resource.type == CU_DEV_RESOURCE_TYPE_SM) {
      green_context.resource_query_supported = true;
      green_context.sm_count = resource.sm.smCount;
      green_context.minimum_partition_size = resource.sm.minSmPartitionSize;
      green_context.coscheduled_alignment =
          resource.sm.smCoscheduledAlignment;

      CUdevResource partition{};
      CUdevResource remaining{};
      unsigned int partition_count = 1U;
      CUresult lifecycle_result = cuDevSmResourceSplitByCount(
          &partition, &partition_count, &resource, &remaining, 0U,
          resource.sm.minSmPartitionSize);
      CUdevResourceDesc descriptor{};
      if (lifecycle_result == CUDA_SUCCESS && partition_count == 1U) {
        lifecycle_result =
            cuDevResourceGenerateDesc(&descriptor, &partition, 1U);
      }
      CUgreenCtx context{};
      if (lifecycle_result == CUDA_SUCCESS) {
        lifecycle_result = cuGreenCtxCreate(
            &context, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM);
      }
      if (lifecycle_result == CUDA_SUCCESS) {
        lifecycle_result = cuGreenCtxDestroy(context);
      }
      if (lifecycle_result == CUDA_SUCCESS) {
        green_context.context_creation_supported = true;
      } else {
        green_context.context_creation_error = cuda_error(lifecycle_result);
      }
    } else {
      green_context.resource_query_error = cuda_error(resource_result);
    }

    info.devices.push_back(DeviceInfo{
        .ordinal = ordinal,
        .name = name.data(),
        .compute_major = device_attribute(
            device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        .compute_minor = device_attribute(
            device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        .multiprocessors = device_attribute(
            device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        .memory_bytes = memory_bytes,
        .unified_addressing =
            device_attribute(device, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) !=
            0,
        .concurrent_kernels =
            device_attribute(device, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS) !=
            0,
        .green_context = std::move(green_context),
    });
  }
  return info;
}

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(const Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR && message != nullptr && error_.empty()) {
      error_ = message;
    }
  }

  [[nodiscard]] const std::string& error() const noexcept { return error_; }

 private:
  std::string error_;
};

TensorRtInfo probe_tensorrt() {
  TensorRtInfo info;
  TensorRtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    info.error = logger.error().empty() ? "createInferRuntime returned null"
                                        : logger.error();
    return info;
  }
  info.available = true;
  info.dla_cores = runtime->getNbDLACores();
  return info;
}

bool executable_exists(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    return false;
  }
  const auto permissions = status.permissions();
  using Perms = std::filesystem::perms;
  return (permissions & (Perms::owner_exec | Perms::group_exec |
                         Perms::others_exec)) != Perms::none;
}

void write_optional_error(const std::optional<std::string>& error) {
  if (error.has_value()) {
    jdg::write_json_string(std::cout, *error);
  } else {
    std::cout << "null";
  }
}

void write_device(const DeviceInfo& device) {
  std::cout << "    {\n"
            << "      \"ordinal\": " << device.ordinal << ",\n"
            << "      \"name\": ";
  jdg::write_json_string(std::cout, device.name);
  std::cout << ",\n"
            << "      \"compute_capability\": ";
  jdg::write_json_string(
      std::cout, std::to_string(device.compute_major) + "." +
                     std::to_string(device.compute_minor));
  std::cout << ",\n"
            << "      \"multiprocessors\": " << device.multiprocessors
            << ",\n"
            << "      \"memory_bytes\": " << device.memory_bytes << ",\n"
            << "      \"unified_addressing\": "
            << (device.unified_addressing ? "true" : "false") << ",\n"
            << "      \"concurrent_kernels\": "
            << (device.concurrent_kernels ? "true" : "false") << ",\n"
            << "      \"green_context\": {\n"
            << "        \"resource_query_supported\": "
            << (device.green_context.resource_query_supported ? "true"
                                                              : "false")
            << ",\n"
            << "        \"context_creation_supported\": "
            << (device.green_context.context_creation_supported ? "true"
                                                                : "false")
            << ",\n"
            << "        \"sm_count\": " << device.green_context.sm_count
            << ",\n"
            << "        \"minimum_partition_size\": "
            << device.green_context.minimum_partition_size << ",\n"
            << "        \"coscheduled_alignment\": "
            << device.green_context.coscheduled_alignment << ",\n"
            << "        \"resource_query_error\": ";
  write_optional_error(device.green_context.resource_query_error);
  std::cout << ",\n        \"context_creation_error\": ";
  write_optional_error(device.green_context.context_creation_error);
  std::cout << "\n      }\n"
            << "    }";
}

}  // namespace

int main() {
  const CudaInfo cuda = probe_cuda();
  const TensorRtInfo tensorrt = probe_tensorrt();
  const std::string board_model = read_file("/proc/device-tree/model");
  const std::string l4t_release = read_file("/etc/nv_tegra_release");
  const bool mps_control = executable_exists("/usr/bin/nvidia-cuda-mps-control");
  const bool mps_server = executable_exists("/usr/bin/nvidia-cuda-mps-server");

  std::cout << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"board\": {\n"
            << "    \"model\": ";
  jdg::write_json_string(std::cout, board_model);
  std::cout << ",\n    \"l4t_release\": ";
  jdg::write_json_string(std::cout, l4t_release);
  std::cout << "\n  },\n"
            << "  \"cuda\": {\n"
            << "    \"available\": " << (cuda.available ? "true" : "false")
            << ",\n"
            << "    \"compile_api_version\": " << CUDA_VERSION << ",\n"
            << "    \"driver_version\": " << cuda.driver_version << ",\n"
            << "    \"error\": ";
  write_optional_error(cuda.error);
  std::cout << ",\n    \"devices\": [\n";
  for (std::size_t index = 0; index < cuda.devices.size(); ++index) {
    write_device(cuda.devices[index]);
    std::cout << (index + 1U == cuda.devices.size() ? "\n" : ",\n");
  }
  std::cout << "    ]\n"
            << "  },\n"
            << "  \"tensorrt\": {\n"
            << "    \"available\": "
            << (tensorrt.available ? "true" : "false") << ",\n"
            << "    \"version\": ";
  jdg::write_json_string(
      std::cout, std::to_string(NV_TENSORRT_MAJOR) + "." +
                     std::to_string(NV_TENSORRT_MINOR) + "." +
                     std::to_string(NV_TENSORRT_PATCH) + "." +
                     std::to_string(NV_TENSORRT_BUILD));
  std::cout << ",\n"
            << "    \"dla_cores\": " << tensorrt.dla_cores << ",\n"
            << "    \"error\": ";
  write_optional_error(tensorrt.error);
  std::cout << "\n  },\n"
            << "  \"mps\": {\n"
            << "    \"control_binary\": "
            << (mps_control ? "true" : "false") << ",\n"
            << "    \"server_binary\": "
            << (mps_server ? "true" : "false") << "\n"
            << "  }\n"
            << "}\n";

  return cuda.available && tensorrt.available ? EXIT_SUCCESS : EXIT_FAILURE;
}
