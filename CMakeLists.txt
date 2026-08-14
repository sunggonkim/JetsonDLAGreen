cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING
      "CUDA architectures to compile for (110 is Jetson AGX Thor)")
endif()

project(
  JetsonDLAGreen
  VERSION 0.1.0
  DESCRIPTION "Predictable mixed-critical inference experiments on NVIDIA Jetson"
  LANGUAGES CXX CUDA)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.2)
  message(
    FATAL_ERROR
      "CUDA 13.2 or newer is required for the Green Context runtime API. "
      "Configure with -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc.")
endif()

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit 13.2 REQUIRED)

find_path(
  TENSORRT_INCLUDE_DIR
  NAMES NvInferRuntime.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include)
find_library(
  TENSORRT_LIBRARY
  NAMES nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT development headers and libnvinfer are required")
endif()

add_library(jdg_project_options INTERFACE)
target_include_directories(
  jdg_project_options INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_options(
  jdg_project_options
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
    # NVCC-generated #line directives trigger GCC -Wpedantic before user code.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)

add_executable(jdg-probe src/platform_probe.cpp)
target_include_directories(jdg-probe PRIVATE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(
  jdg-probe PRIVATE jdg_project_options CUDA::cuda_driver "${TENSORRT_LIBRARY}")
set_target_properties(
  jdg-probe PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                       INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(jdg-bench benchmarks/cuda_interference.cu)
target_link_libraries(jdg-bench PRIVATE jdg_project_options CUDA::cudart)
set_target_properties(
  jdg-bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                       BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                       INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_library(jdg-trt-benchmark-core STATIC benchmarks/trt_inference.cpp)
target_include_directories(jdg-trt-benchmark-core PRIVATE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(
  jdg-trt-benchmark-core PUBLIC jdg_project_options CUDA::cudart
                                "${TENSORRT_LIBRARY}")
set_target_properties(
  jdg-trt-benchmark-core PROPERTIES POSITION_INDEPENDENT_CODE ON
                                     BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                     INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(jdg-trt-bench benchmarks/trt_inference_main.cpp)
target_link_libraries(jdg-trt-bench PRIVATE jdg-trt-benchmark-core)

add_executable(jdg-cuda-init-probe benchmarks/cuda_init_probe.cpp)
target_link_libraries(jdg-cuda-init-probe PRIVATE jdg_project_options CUDA::cudart)
set_target_properties(
  jdg-cuda-init-probe PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                 INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(jdg-mig-sysmem-handoff benchmarks/mig_sysmem_handoff.cu)
target_link_libraries(jdg-mig-sysmem-handoff PRIVATE jdg_project_options CUDA::cudart)
set_target_properties(
  jdg-mig-sysmem-handoff
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(jdg-mig-sysmem-ring benchmarks/mig_sysmem_ring.cu)
target_link_libraries(jdg-mig-sysmem-ring PRIVATE jdg_project_options CUDA::cudart)
set_target_properties(
  jdg-mig-sysmem-ring
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(jdg-mig-trt-pipeline benchmarks/mig_trt_pipeline.cpp)
target_include_directories(jdg-mig-trt-pipeline PRIVATE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(
  jdg-mig-trt-pipeline PRIVATE jdg_project_options CUDA::cudart
                               "${TENSORRT_LIBRARY}")
set_target_properties(
  jdg-mig-trt-pipeline PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_library(orion-trt-driver-capture SHARED
            baselines/orion/driver_capture/intercept_driver.cpp
            baselines/orion/driver_capture/scheduler.cpp)
target_link_libraries(orion-trt-driver-capture PRIVATE jdg_project_options
                                                        CUDA::cuda_driver
                                                        CUDA::cudart dl)
set_target_properties(orion-trt-driver-capture PROPERTIES
                      OUTPUT_NAME "orion-trt-driver-capture")

add_executable(jdg-orion-mig-trt-pipeline benchmarks/mig_trt_pipeline.cpp)
target_compile_definitions(jdg-orion-mig-trt-pipeline PRIVATE JDG_WITH_ORION=1)
target_include_directories(jdg-orion-mig-trt-pipeline PRIVATE
                           "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(
  jdg-orion-mig-trt-pipeline PRIVATE jdg_project_options CUDA::cudart
                                      "${TENSORRT_LIBRARY}"
                                      orion-trt-driver-capture)
set_target_properties(
  jdg-orion-mig-trt-pipeline
  PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(jdg-mig-whisper-asr benchmarks/mig_whisper_asr.cpp)
target_include_directories(jdg-mig-whisper-asr PRIVATE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(
  jdg-mig-whisper-asr PRIVATE jdg_project_options CUDA::cudart
                              "${TENSORRT_LIBRARY}")
set_target_properties(
  jdg-mig-whisper-asr PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                 INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(orion-trt-native-smoke baselines/orion/native_smoke.cpp)
target_include_directories(orion-trt-native-smoke PRIVATE
                           "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(orion-trt-native-smoke PRIVATE
                      jdg-trt-benchmark-core orion-trt-driver-capture)
set_target_properties(
  orion-trt-native-smoke PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                     INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(orion-profile-pressure baselines/orion/profile_pressure.cu)
target_link_libraries(orion-profile-pressure PRIVATE jdg_project_options
                                                     CUDA::cudart)
set_target_properties(
  orion-profile-pressure
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(bless-context-probe baselines/bless/context_probe.cpp)
target_link_libraries(bless-context-probe PRIVATE jdg_project_options
                                                  CUDA::cuda_driver)
set_target_properties(
  bless-context-probe PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                                 INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(bless-trt-affinity-smoke
               baselines/bless/trt_affinity_smoke.cpp)
target_link_libraries(bless-trt-affinity-smoke PRIVATE
                      jdg-trt-benchmark-core CUDA::cuda_driver)
set_target_properties(
  bless-trt-affinity-smoke
  PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(bless-trt-context-replica-smoke
               baselines/bless/trt_context_replica_smoke.cpp)
target_link_libraries(bless-trt-context-replica-smoke PRIVATE
                      jdg-trt-benchmark-core CUDA::cuda_driver)
set_target_properties(
  bless-trt-context-replica-smoke
  PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(bless-trt-activation-replica-smoke
               baselines/bless/trt_activation_replica_smoke.cpp)
target_include_directories(bless-trt-activation-replica-smoke PRIVATE
                           "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(bless-trt-activation-replica-smoke PRIVATE
                      jdg_project_options CUDA::cuda_driver CUDA::cudart
                      "${TENSORRT_LIBRARY}")
set_target_properties(
  bless-trt-activation-replica-smoke
  PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_library(bless-trt-squad-intercept SHARED
            baselines/bless/trt_squad_intercept.cpp)
target_include_directories(bless-trt-squad-intercept PRIVATE
                           baselines/bless)
target_link_libraries(bless-trt-squad-intercept PRIVATE jdg_project_options
                                                      CUDA::cuda_driver dl)
set_target_properties(bless-trt-squad-intercept PROPERTIES
                      OUTPUT_NAME "bless-trt-squad-intercept")

add_executable(bless-trt-squad-replica-smoke
               baselines/bless/trt_activation_replica_smoke.cpp)
target_compile_definitions(bless-trt-squad-replica-smoke PRIVATE
                           JDG_BLESS_SQUAD=1)
target_include_directories(bless-trt-squad-replica-smoke PRIVATE
                           "${TENSORRT_INCLUDE_DIR}" baselines/bless)
target_link_libraries(bless-trt-squad-replica-smoke PRIVATE
                      jdg_project_options CUDA::cuda_driver CUDA::cudart
                      "${TENSORRT_LIBRARY}" bless-trt-squad-intercept)
set_target_properties(
  bless-trt-squad-replica-smoke
  PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(bless-context-switch-smoke
               baselines/bless/context_switch_smoke.cu)
target_link_libraries(bless-context-switch-smoke PRIVATE jdg_project_options
                                                        CUDA::cuda_driver
                                                        CUDA::cudart)
set_target_properties(
  bless-context-switch-smoke
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

add_executable(bless-native-squad-smoke
               baselines/bless/native_squad_smoke.cu)
target_link_libraries(bless-native-squad-smoke PRIVATE jdg_project_options
                                                     CUDA::cuda_driver
                                                     CUDA::cudart)
set_target_properties(
  bless-native-squad-smoke
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
             BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
             INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(jdg-stats-test tests/stats_test.cpp)
  target_link_libraries(jdg-stats-test PRIVATE jdg_project_options)
  add_test(NAME stats-unit COMMAND jdg-stats-test)
  add_test(NAME platform-probe-smoke COMMAND jdg-probe)
  add_test(NAME mig-sysmem-handoff-cli COMMAND jdg-mig-sysmem-handoff --help)
  add_test(NAME mig-sysmem-ring-cli COMMAND jdg-mig-sysmem-ring --help)
  add_test(
    NAME mig-sysmem-ring-contract-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_mig_sysmem_ring_contract.py"
            -v)
  add_test(
    NAME mig-sysmem-handoff-transport-contract-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_mig_sysmem_handoff_contract.py"
            -v)
  add_test(
    NAME p1-mig-sysmem-characterization-contract-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_p1_mig_sysmem_characterization.py"
            -v)
  add_test(
    NAME quiet-dag-contract-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_quiet_dag_contract.py"
            -v)
  add_test(NAME mig-trt-pipeline-cli COMMAND jdg-mig-trt-pipeline --help)
  add_test(NAME mig-whisper-asr-cli COMMAND jdg-mig-whisper-asr --help)
  add_test(
    NAME mig-whisper-asr-contract-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_mig_whisper_asr_contract.py"
            -v)
  add_test(
    NAME repeat-jdgint-trace-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_repeat_jdgint_trace.py"
            -v)
  add_test(
    NAME p9-whisper-asr-mig-crossover-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_p9_whisper_asr_mig_crossover.py"
            -v)
  add_test(
    NAME generate-p9-whisper-asr-crossover-figure-unit
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/test_generate_p9_whisper_asr_crossover_figure.py"
            -v)
  add_test(NAME read-application-output-trace-unit
           COMMAND "${Python3_EXECUTABLE}"
                   "${PROJECT_SOURCE_DIR}/tests/test_read_application_output_trace.py"
                   -v)
  add_test(NAME decode-whisper-token-trace-unit
           COMMAND "${Python3_EXECUTABLE}"
                   "${PROJECT_SOURCE_DIR}/tests/test_decode_whisper_token_trace.py"
                   -v)
  add_test(
    NAME cuda-benchmark-smoke
    COMMAND jdg-bench --background none --samples 3 --warmup 1)
  add_test(NAME mig-governor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_mig_governor.py" -v)
  add_test(NAME multimodal-governor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_multimodal_governor.py" -v)
  add_test(NAME full-gpu-governor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_full_gpu_governor.py" -v)
  add_test(NAME mig-slack-governor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_mig_slack_governor.py" -v)
  add_test(NAME tegrastats-telemetry-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_tegrastats_telemetry.py" -v)
  add_test(NAME mig-slack-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_mig_slack_summary.py" -v)
  add_test(NAME compare-sota-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_compare_sota.py" -v)
  add_test(NAME verify-orion-differential-gate-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_orion_differential_gate.py" -v)
  add_test(NAME verify-pantheon-common-workload-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_pantheon_common_workload.py" -v)
  add_test(NAME verify-application-accuracy-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_application_accuracy.py" -v)
  add_test(NAME boer-thor-adapter-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_boer_thor_adapter.py" -v)
  add_test(NAME gslice-thor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_gslice_thor.py" -v)
  add_test(NAME gpulet-thor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_gpulet_thor.py" -v)
  add_test(NAME gpulet-resnet-control-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_gpulet_resnet_control_smoke.py" -v)
  add_test(NAME deepplan-thor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_deepplan_thor.py" -v)
  add_test(NAME boer-candidate-evaluator-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_boer_candidate_evaluator.py" -v)
  add_test(NAME boer-finalize-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_boer_finalize.py" -v)
  add_test(NAME boer-dependent-pipeline-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_boer_dependent_pipeline.py" -v)
  add_test(NAME boer-independent-pair-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_boer_independent_pair.py" -v)
  add_test(NAME parvagpu-thor-adapter-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_parvagpu_thor_adapter.py" -v)
  add_test(NAME parvagpu-allocation-evaluator-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_parvagpu_allocation_evaluator.py" -v)
  add_test(NAME orion-probe-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_probe.py" -v)
  add_test(NAME orion-tensorrt-surface-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_tensorrt_surface.py" -v)
  add_test(NAME orion-driver-capture-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_driver_capture.py" -v)
  add_test(NAME orion-native-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_native_smoke.py" -v)
  add_test(NAME orion-profile-thor-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_profile_thor.py" -v)
  add_test(NAME orion-profiled-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_profiled_smoke.py" -v)
  add_test(NAME orion-dependent-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_dependent_smoke.py" -v)
  add_test(NAME orion-dependent-runner-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_orion_dependent_runner.py" -v)
  add_test(NAME orion-resnet-control-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_resnet_control_smoke.py" -v)
  add_test(NAME xsched-port-contract-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_xsched_port_contract.py" -v)
  add_test(NAME xsched-native-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_xsched_native_smoke.py" -v)
  add_test(NAME xsched-dependent-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_xsched_dependent_smoke.py" -v)
  add_test(NAME xsched-resnet-control-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_xsched_resnet_control_smoke.py" -v)
  add_test(NAME pantheon-training-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_pantheon_training.py" -v)
  add_test(NAME pantheon-native-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_pantheon_native_smoke.py" -v)
  add_test(NAME bless-scheduler-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_scheduler.py" -v)
  add_test(NAME bless-affinity-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_affinity_smoke.py" -v)
  add_test(NAME bless-native-squad-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_native_squad_smoke.py" -v)
  add_test(NAME bless-trt-context-replica-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_trt_context_replica_smoke.py" -v)
  add_test(NAME bless-thor-fidelity-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_thor_fidelity.py" -v)
  add_test(NAME bless-trt-activation-replica-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_trt_activation_replica_smoke.py" -v)
  add_test(NAME bless-trt-squad-replica-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bless_trt_squad_replica_smoke.py" -v)
  add_test(NAME freeze-bless-trt-safe-boundaries-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_freeze_bless_trt_safe_boundaries.py" -v)
  add_test(NAME build-bless-common-schedule-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_build_bless_common_schedule.py" -v)
  add_test(NAME summarize-bless-resnet-compatibility-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_bless_resnet_compatibility.py" -v)
  add_test(NAME p9-published-sota-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_published_sota_smoke.py" -v)
  add_test(NAME p9-sota-positive-controls-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_sota_positive_controls.py" -v)
  add_test(NAME orion-thor-policy-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_orion_thor.py" -v)
  add_test(NAME p9-sota-campaign-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_sota_campaign_summary.py" -v)
  add_test(NAME p9-dependent-stress-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_dependent_stress_smoke.py" -v)
  add_test(NAME p9-comparator-manifest-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_comparator_manifest.py" -v)
  add_test(NAME p9-active-comparator-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_active_comparator_smoke.py" -v)
  add_test(NAME p9-active-williams-repeats-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_active_williams_repeats.py" -v)
  add_test(NAME mig-trt-pipeline-contract-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_mig_trt_pipeline_contract.py" -v)
  add_test(NAME verify-activation-replay-trace-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_activation_replay_trace.py" -v)
  add_test(NAME verify-operational-arrival-trace-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_operational_arrival_trace.py" -v)
  add_test(NAME mig-trt-causal-contract-shell-unit
           COMMAND bash -n "${PROJECT_SOURCE_DIR}/scripts/run_p9_mig_trt_causal_contract_smoke.sh")
  add_test(NAME p9-numeric-mechanism-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_numeric_sota_smoke.py" -v)
  add_test(NAME p9-numeric-mechanism-repetitions-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_numeric_sota_repetitions.py" -v)
  add_test(NAME p9-structural-limits-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_structural_limits.py" -v)
  add_test(NAME p9-resnet-structural-limits-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_resnet_structural_limits.py" -v)
  add_test(NAME p9-orion-resnet-runner-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_orion_resnet_runner.py" -v)
  add_test(NAME p9-common-sota-repeats-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_common_sota_repeats.py" -v)
  add_test(NAME p9-common-sota-williams-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_common_sota_williams.py" -v)
  add_test(NAME p9-common-sota-williams-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_common_sota_williams_summary.py" -v)
  add_test(NAME p9-bind-current-dependent-spec-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_bind_p9_current_dependent_spec.py" -v)
  add_test(NAME p9-current-whisper-evidence-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_current_whisper_evidence.py" -v)
  add_test(NAME p9-sota-tables-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_generate_p9_sota_tables.py" -v)
  add_test(NAME p9-whisper-tables-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_generate_p9_whisper_tables.py" -v)
  add_test(NAME p9-quiet-heldout-load-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_quiet_heldout_load.py" -v)
  add_test(NAME p9-transport-williams-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_transport_williams.py" -v)
  add_test(NAME p9-transport-williams-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_transport_williams_summary.py" -v)
  add_test(NAME p9-paper-tables-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_generate_p9_paper_tables.py" -v)
  add_test(NAME p9-paper-contract-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_paper_contract.py" -v)
  add_test(NAME p9-goal-completion-audit-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_audit_p9_goal_completion.py" -v)
  add_test(NAME p9-campaign-preflight-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_campaign_preflight.py" -v)
  add_test(NAME verify-split-dag-equivalence-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_split_dag_equivalence.py" -v)
  add_test(NAME p9-learned-head-fast-pair-shell-unit
           COMMAND bash -n "${PROJECT_SOURCE_DIR}/scripts/run_p9_learned_head_fast_pair.sh")
  add_test(NAME quiet-stage-dag-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_quiet_stage_dag.py" -v)
  add_test(NAME p9-quiet-candidate-spec-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_quiet_candidate_spec.py" -v)
  add_test(NAME p9-placement-comparison-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_compare_p9_placement_runs.py" -v)
  add_test(NAME p9-placement-repeat-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_placement_repeats.py" -v)
  add_test(NAME p9-common-frontier-repeat-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_common_frontier_repeats.py" -v)
  add_test(NAME p9-common-frontier-load-sweep-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_common_frontier_load_sweep.py" -v)
  add_test(NAME p9-checksum-mode-probe-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_checksum_mode_probe.py" -v)
  add_test(NAME p9-causal-pair-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_validate_p9_causal_pair.py" -v)
  add_test(NAME p9-sota-frontier-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_summarize_p9_sota_frontier.py" -v)
  add_test(NAME p9-dependent-frontier-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_dependent_frontier.py" -v)
  add_test(NAME p9-payload-comparison-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_payload_comparison.py" -v)
  add_test(NAME p9-resnet-control-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_verify_p9_resnet_control_smoke.py" -v)
  add_test(NAME p9-dependent-repetitions-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_dependent_repetitions.py" -v)
  add_test(NAME freeze-p9-pipeline-deadline-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_freeze_p9_pipeline_deadline.py" -v)
  add_test(NAME freeze-p9-common-placement-deadline-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_freeze_p9_common_placement_deadline.py" -v)
  add_test(NAME p9-sota-scope-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_sota_scope.py" -v)
  add_test(NAME p9-transport-ablation-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_transport_ablation.py" -v)
  add_test(NAME williams-design-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_williams_design.py" -v)
  add_test(NAME freeze-p9-deadline-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_freeze_p9_deadline.py" -v)
  add_test(NAME freeze-p9-thermal-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_freeze_p9_thermal.py" -v)
  add_test(NAME profile-p9-guard-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_profile_p9_guard.py" -v)
  add_test(NAME freeze-p9-guard-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_freeze_p9_guard.py" -v)
  add_test(NAME sensitivity-summary-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_sensitivity_summary.py" -v)
  add_test(NAME drain-profile-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_drain_profile.py" -v)
  add_test(NAME prepare-resnet10-real-dag-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_prepare_resnet10_real_dag.py" -v)
  add_test(NAME prepare-kitti-coco-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_convert_kitti_to_coco.py" -v)
  add_test(NAME build-application-prediction-trace-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_build_application_prediction_trace.py" -v)
  add_test(NAME p9-active-frontier-campaign-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_active_frontier_campaign.py" -v)
  add_test(NAME p9-static-full-gate-smoke-unit
           COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/test_p9_static_full_gate_smoke.py" -v)
  set_tests_properties(platform-probe-smoke cuda-benchmark-smoke
                       PROPERTIES TIMEOUT 30
                       ENVIRONMENT
                         "CUDA_VISIBLE_DEVICES=0;LD_LIBRARY_PATH=${CUDAToolkit_LIBRARY_DIR}")
endif()

install(TARGETS jdg-probe jdg-bench jdg-trt-bench jdg-cuda-init-probe
                jdg-mig-sysmem-handoff jdg-mig-sysmem-ring jdg-mig-trt-pipeline
                jdg-mig-whisper-asr
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
