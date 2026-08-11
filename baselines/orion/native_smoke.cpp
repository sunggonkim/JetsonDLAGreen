#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "driver_capture/scheduler.hpp"
#include "jetson_dla_green/json.hpp"
#include "jetson_dla_green/trt_benchmark.hpp"

namespace {

struct Options {
  std::filesystem::path engine;
  std::filesystem::path best_effort_engine;
  std::filesystem::path output_dir;
  std::string model_name{"resnet10-detection"};
  std::string best_effort_model_name{"distilbert-sst2"};
  std::filesystem::path best_effort_profile;
  std::filesystem::path high_priority_profile;
  double max_be_duration_us{1.0};
  int samples{6};
  int warmup{1};
  bool single_client_profile{};
  bool profile_aware{};
};

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view key(argv[i]);
    if (key == "--single-client-profile") {
      options.single_client_profile = true;
      continue;
    }
    if (key == "--profile-aware") {
      options.profile_aware = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string(argv[i]) + " requires a value");
    }
    const std::string value(argv[++i]);
    if (key == "--engine") {
      options.engine = value;
    } else if (key == "--best-effort-engine") {
      options.best_effort_engine = value;
    } else if (key == "--output-dir") {
      options.output_dir = value;
    } else if (key == "--samples") {
      options.samples = std::stoi(value);
    } else if (key == "--warmup") {
      options.warmup = std::stoi(value);
    } else if (key == "--model-name") {
      options.model_name = value;
    } else if (key == "--best-effort-model-name") {
      options.best_effort_model_name = value;
    } else if (key == "--best-effort-profile") {
      options.best_effort_profile = value;
    } else if (key == "--high-priority-profile") {
      options.high_priority_profile = value;
    } else if (key == "--max-be-duration-us") {
      options.max_be_duration_us = std::stod(value);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(key));
    }
  }
  if (options.engine.empty() || !std::filesystem::is_regular_file(options.engine)) {
    throw std::invalid_argument("--engine must name an existing TensorRT engine");
  }
  if (options.output_dir.empty() || options.samples <= 0 || options.warmup < 0 ||
      options.model_name.empty()) {
    throw std::invalid_argument(
        "--output-dir, --model-name, positive --samples, and nonnegative --warmup are required");
  }
  if (options.profile_aware) {
    if (options.single_client_profile ||
        !std::filesystem::is_regular_file(options.best_effort_engine) ||
        !std::filesystem::is_regular_file(options.best_effort_profile) ||
        !std::filesystem::is_regular_file(options.high_priority_profile) ||
        options.best_effort_model_name.empty() ||
        !(options.max_be_duration_us > 0.0)) {
      throw std::invalid_argument(
          "--profile-aware requires BE engine/model, both profiles, and a "
          "positive --max-be-duration-us");
    }
  }
  return options;
}

struct ClientResult {
  int status{1};
  std::string output;
  std::string error;
};

ClientResult run_client(const int client_id, const bool high_priority,
                        const Options& options) {
  ClientResult result;
  const int registration =
      orion_trt_register_client(client_id, high_priority ? 1 : 0);
  if (registration != 0) {
    result.error = "Orion client registration failed: " +
                   std::to_string(registration);
    return result;
  }

  const auto& engine = high_priority || !options.profile_aware
                           ? options.engine
                           : options.best_effort_engine;
  const auto& model = high_priority || !options.profile_aware
                          ? options.model_name
                          : options.best_effort_model_name;
  const std::string trace =
      (options.output_dir /
       (high_priority ? "high-requests.csv" : "best-effort-requests.csv"))
          .string();
  std::vector<std::string> storage{
      "jdg-trt-bench", "--engine", engine.string(), "--model-name",
      model, "--role", "benchmark", "--samples",
      std::to_string(options.samples), "--warmup", std::to_string(options.warmup),
      "--burst-size", "1",
      "--period-ms", "0", "--deadline-ms", "0", "--priority",
      high_priority ? "high" : "low", "--include-transfers", "true",
      "--trace", trace};
  std::vector<char*> arguments;
  arguments.reserve(storage.size());
  for (auto& argument : storage) {
    arguments.push_back(argument.data());
  }
  std::ostringstream output;
  std::ostringstream error;
  result.status = jdg::run_trt_benchmark(
      static_cast<int>(arguments.size()), arguments.data(), output, error);
  result.output = output.str();
  result.error = error.str();
  return result;
}

void write_profile_result(const Options& options, const ClientResult& client,
                          const orion::thor::SchedulerStats& stats) {
  std::ofstream output(options.output_dir / "result.json",
                       std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create Orion profile result.json");
  }
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"kind\": \"orion-thor-operation-profile-raw\",\n"
         << "  \"upstream_commit\": "
            "\"20f9469764fb96d94ce23a8e70615196e9ce4ba1\",\n"
         << "  \"model\": ";
  jdg::write_json_string(output, options.model_name);
  output << ",\n  \"warmup\": " << options.warmup
         << ",\n  \"samples\": " << options.samples
         << ",\n  \"numeric_comparison_allowed\": false,\n"
         << "  \"scheduler\": {\"algorithm\":"
            "\"orion-profile-recording\",\"initial_gate_clients\":0,"
            "\"arrivals\":"
         << stats.arrivals << ",\"decisions\":" << stats.decisions
         << "},\n  \"client\": " << client.output << ",\n  \"error\": ";
  jdg::write_json_string(output, client.error);
  output << "\n}\n";
  if (!output) {
    throw std::runtime_error("failed to write Orion profile result.json");
  }
}

void write_result(const Options& options, const ClientResult& best_effort,
                  const ClientResult& high,
                  const orion::thor::SchedulerStats& stats) {
  std::ofstream output(options.output_dir / "result.json",
                       std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create Orion result.json");
  }
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"kind\": \""
         << (options.profile_aware
                 ? "orion-thor-profile-aware-positive-control"
                 : "orion-thor-native-positive-control")
         << "\",\n"
         << "  \"upstream_commit\": "
            "\"20f9469764fb96d94ce23a8e70615196e9ce4ba1\",\n"
         << "  \"port_stage\": \""
         << (options.profile_aware ? "profile-aware-admission"
                                   : "driver-operation-software-queue")
         << "\",\n"
         << "  \"numeric_comparison_allowed\": false,\n"
         << "  \"next_gate\": \"profile-aware Orion pairing with the "
            "positive-control gate disabled\",\n"
         << "  \"scheduler\": {\"algorithm\":\""
         << (options.profile_aware ? "orion-profile-aware"
                                   : "orion-hp-first-software-queue")
         << "\","
            "\"operation\":\"cuLaunchKernelEx\","
            "\"initial_gate_clients\":"
         << (options.profile_aware ? 0 : 2) << ",\"arrivals\":"
         << stats.arrivals << ",\"decisions\":" << stats.decisions
         << ",\"reordered_decisions\":" << stats.reordered_decisions
         << ",\"high_priority_decisions\":"
         << stats.high_priority_decisions
         << ",\"profiled_best_effort_admissions\":"
         << stats.profiled_best_effort_admissions
         << ",\"complementary_admissions\":"
         << stats.complementary_admissions
         << ",\"profile_blocked_polls\":"
         << stats.profile_blocked_polls
         << ",\"max_be_duration_us\":" << options.max_be_duration_us
         << "},\n"
         << "  \"best_effort\": " << best_effort.output << ",\n"
         << "  \"high_priority\": " << high.output << ",\n"
         << "  \"errors\": {\"best_effort\":";
  jdg::write_json_string(output, best_effort.error);
  output << ",\"high_priority\":";
  jdg::write_json_string(output, high.error);
  output << "}\n}\n";
  if (!output) {
    throw std::runtime_error("failed to write Orion result.json");
  }
}

int run(const Options& options) {
  if (std::filesystem::exists(options.output_dir)) {
    throw std::runtime_error("refusing an existing Orion output directory");
  }
  std::filesystem::create_directories(options.output_dir);
  const std::string decisions =
      (options.output_dir / "scheduler-decisions.jsonl").string();
  if (options.single_client_profile) {
    const char* profile_trace = std::getenv("ORION_TRT_PROFILE_TRACE");
    if (profile_trace == nullptr || profile_trace[0] == '\0') {
      throw std::runtime_error(
          "--single-client-profile requires ORION_TRT_PROFILE_TRACE");
    }
  }
  const int initial_gate = options.single_client_profile ? 0 : 2;
  const int start = options.profile_aware
                        ? orion_trt_scheduler_start_profiled(
                              decisions.c_str(),
                              options.best_effort_profile.c_str(),
                              options.high_priority_profile.c_str(),
                              options.max_be_duration_us)
                        : orion_trt_scheduler_start(decisions.c_str(),
                                                   initial_gate);
  if (start != 0) {
    throw std::runtime_error("failed to start Orion scheduler: " +
                             std::to_string(start));
  }

  ClientResult high;
  if (options.single_client_profile) {
    high = run_client(1, true, options);
    orion::thor::SchedulerStats stats;
    const int stats_status = orion_trt_scheduler_stats(&stats);
    const int stop_status = orion_trt_scheduler_stop();
    if (stats_status != 0 || stop_status != 0 || high.status != 0) {
      throw std::runtime_error("Orion single-client profiling failed: " +
                               high.error);
    }
    write_profile_result(options, high, stats);
    return 0;
  }

  ClientResult best_effort;
  std::thread best_effort_thread(
      [&] { best_effort = run_client(0, false, options); });
  bool best_effort_arrived = false;
  const auto arrival_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < arrival_deadline) {
    orion::thor::SchedulerStats observed;
    if (orion_trt_scheduler_stats(&observed) == 0 && observed.arrivals >= 1U) {
      best_effort_arrived = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::thread high_thread([&] { high = run_client(1, true, options); });
  best_effort_thread.join();
  high_thread.join();

  orion::thor::SchedulerStats stats;
  const int stats_status = orion_trt_scheduler_stats(&stats);
  const int stop_status = orion_trt_scheduler_stop();
  if (stats_status != 0 || stop_status != 0) {
    throw std::runtime_error("failed to finalize Orion scheduler");
  }
  if (best_effort.status != 0 || high.status != 0) {
    throw std::runtime_error("a TensorRT Orion client failed: " +
                             best_effort.error + high.error);
  }
  if (!best_effort_arrived) {
    throw std::runtime_error(
        "best-effort client did not reach the Orion queue before HP start");
  }
  write_result(options, best_effort, high, stats);
  return 0;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
