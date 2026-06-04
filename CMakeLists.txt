cmake_minimum_required(VERSION 3.20)

project(hdkx_aicompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KXC_ENABLE_CUDA "Enable CUDA support when toolkit is available" ON)
option(KXC_ENABLE_LLVM "Enable LLVM backend for codegen" ON)
option(KXC_BUILD_RESNET18_IR_DUMP "Build test/resnet18_ir_dump.cpp" ON)
option(KXC_BUILD_DEBUG_EXAMPLES "Build temporary debug examples in test/" OFF)
option(KXC_BUILD_PASS_TESTS "Build pass tests in test/" ON)
option(KXC_BUILD_CODEGEN_TESTS "Build codegen tests" ON)

if(MSVC)
  add_compile_options(/utf-8 /EHsc)
endif()

set(KXC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(KXC_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
endif()

if(KXC_ENABLE_CUDA AND CUDAToolkit_FOUND)
  set(KXC_USE_CUDA 1)
  set(KXC_CUDA_LIBS CUDA::cudart)
  message(STATUS "CUDA support: enabled")
elseif(KXC_ENABLE_CUDA AND NOT CUDAToolkit_FOUND)
  set(KXC_USE_CUDA 0)
  message(WARNING "KXC_ENABLE_CUDA=ON but CUDAToolkit was not found. Falling back to CPU-only build.")
else()
  set(KXC_USE_CUDA 0)
  message(STATUS "CUDA support: disabled by KXC_ENABLE_CUDA=OFF")
endif()

# ==================== LLVM Support ====================
if(KXC_ENABLE_LLVM)
  find_package(LLVM QUIET CONFIG)
endif()

if(KXC_ENABLE_LLVM AND LLVM_FOUND)
  set(KXC_USE_LLVM 1)
  message(STATUS "LLVM support: enabled (version ${LLVM_PACKAGE_VERSION})")

  # Use llvm-config to get proper link flags
  execute_process(
    COMMAND llvm-config --link-static --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND llvm-config --link-static --libs core support irreader executionengine orcjit aarch64asmparser aarch64codegen aarch64desc aarch64info passes
    OUTPUT_VARIABLE LLVM_LIBS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND llvm-config --system-libs
    OUTPUT_VARIABLE LLVM_SYSTEM_LIBS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Convert to list
  string(REPLACE " " ";" LLVM_LIBS ${LLVM_LIBS_STR})
  string(REPLACE " " ";" LLVM_LDFLAGS_LIST ${LLVM_LDFLAGS})
  string(REPLACE " " ";" LLVM_SYSTEM_LIBS ${LLVM_SYSTEM_LIBS_STR})
  # Remove -lxml2 if not available (optional LLVM dependency)
  list(REMOVE_ITEM LLVM_SYSTEM_LIBS "-lxml2")

  set(KXC_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
  set(KXC_LLVM_DEFINITIONS ${LLVM_DEFINITIONS})
  set(KXC_LLVM_LINK_FLAGS "${LLVM_LDFLAGS}")
elseif(KXC_ENABLE_LLVM AND NOT LLVM_FOUND)
  set(KXC_USE_LLVM 0)
  message(WARNING "KXC_ENABLE_LLVM=ON but LLVM was not found. Codegen will be disabled.")
else()
  set(KXC_USE_LLVM 0)
  message(STATUS "LLVM support: disabled by KXC_ENABLE_LLVM=OFF")
endif()

set(KXC_RUNTIME_SOURCES
  src/base/expr.cc
  src/base/device.cc
  src/base/device_api.cc
  src/base/profiling.cc
  src/base/target.cc
  src/base/virtual_device.cc
  src/base/disco_placement.cc
  src/base/execution_plan.cc
  src/base/ndarray.cc
  src/base/tensor.cc
  src/base/pass.cc
  src/base/op.cc
  src/base/disco/dref.cc
  src/base/disco/threaded_session.cc
  src/base/disco/worker.cc
  src/base/disco/ccl_cpu.cc
  src/base/disco/ccl_nccl.cc
  src/base/disco/executor.cc
  src/base/device/cpu_device_api.cc
  src/base/device/cuda_device_api.cc
  src/relay/relay.cc
  src/relay/pass_utils.cc
  src/relay/op_attrs.cc
  src/relay/op_macros.cc
  src/relay/common_ops.cc
  src/relay/backend/lower.cc
  src/relay/pass/print_ir.cc
  src/relay/transforms/pipeline.cc
  src/relay/transforms/fold_constant.cc
  src/relay/transforms/simplify_expr.cc
  src/relay/transforms/canonicalize_cast.cc
  src/relay/transforms/remove_standalone_reshapes.cc
  src/relay/transforms/fold_tuple_get_item.cc
  src/relay/transforms/eliminate_common_subexpr.cc
  src/relay/transforms/eliminate_dead_let.cc
  src/relay/transforms/capture_post_dfs_index_in_spans.cc
  src/relay/transforms/annotate_memory_scope.cc
  src/relay/transforms/multi_device.cc
  src/relay/op/tensor/math.cc
  src/relay/op/tensor/reduce.cc
  src/relay/op/tensor/transform.cc
  src/relay/op/nn/activation.cc
  src/relay/op/nn/convolution.cc
  src/relay/op/nn/pooling.cc
  src/relay/op/nn/dense.cc
  src/relay/op/nn/softmax.cc
  src/relay/op/op_ffi.cc
  src/tir/expr.cc
  src/tir/stmt.cc
  src/tir/pass_utils.cc
  src/tir/pass/print_ir.cc
  src/tir/transforms/pipeline.cc
  src/tir/transforms/fold_constant.cc
  src/tir/transforms/force_narrow_index_to_i32.cc
  src/tir/transforms/simplify_expr.cc
  src/tir/transforms/loop_partition.cc
  src/tir/transforms/unroll_loop.cc
  src/tir/transforms/vectorize_loop.cc
  src/tir/transforms/remove_no_op.cc
  src/tir/transforms/convert_for_loops_serial.cc
  src/te/te.cc
)

# ==================== Codegen Sources (conditional) ====================
# C codegen and runtime are always available
list(APPEND KXC_RUNTIME_SOURCES
  src/codegen/compiled_kernel.cc
  src/codegen/codegen_c.cc
  src/api/compile_config.cc
  src/api/compiler.cc
  src/runtime/shape_predictor.cc
  src/runtime/kernel_cache.cc
  src/runtime/kernel_runner.cc
  src/runtime/background_compiler.cc
  src/runtime/runtime_session.cc
)
if(KXC_USE_LLVM)
  list(APPEND KXC_RUNTIME_SOURCES
    src/codegen/codegen_llvm.cc
    src/codegen/llvm_jit.cc
  )
endif()

add_library(kxc_runtime_obj OBJECT ${KXC_RUNTIME_SOURCES})
target_include_directories(kxc_runtime_obj PUBLIC "${KXC_INCLUDE_DIR}")
target_compile_definitions(kxc_runtime_obj PUBLIC KXC_USE_CUDA=${KXC_USE_CUDA} KXC_USE_LLVM=${KXC_USE_LLVM})
if(KXC_USE_CUDA)
  target_link_libraries(kxc_runtime_obj PUBLIC ${KXC_CUDA_LIBS})
endif()
if(KXC_USE_LLVM)
  target_include_directories(kxc_runtime_obj SYSTEM PUBLIC ${KXC_LLVM_INCLUDE_DIRS})
  target_compile_definitions(kxc_runtime_obj PUBLIC ${KXC_LLVM_DEFINITIONS})
endif()

add_library(kxc_runtime STATIC $<TARGET_OBJECTS:kxc_runtime_obj>)
target_include_directories(kxc_runtime PUBLIC "${KXC_INCLUDE_DIR}")
target_compile_definitions(kxc_runtime PUBLIC KXC_USE_CUDA=${KXC_USE_CUDA} KXC_USE_LLVM=${KXC_USE_LLVM})
if(KXC_USE_CUDA)
  target_link_libraries(kxc_runtime PUBLIC ${KXC_CUDA_LIBS})
endif()
if(KXC_USE_LLVM)
  target_include_directories(kxc_runtime SYSTEM PUBLIC ${KXC_LLVM_INCLUDE_DIRS})
  target_compile_definitions(kxc_runtime PUBLIC ${KXC_LLVM_DEFINITIONS})
  target_link_libraries(kxc_runtime PUBLIC ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})
  target_link_options(kxc_runtime PUBLIC ${LLVM_LDFLAGS_LIST})
  # LLVM archives contain bitcode (LTO), use lld to handle them
  target_link_options(kxc_runtime PUBLIC -fuse-ld=lld)
endif()

function(kxc_add_runtime_exe target source_file)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
    message(FATAL_ERROR "Missing source file: ${source_file}")
  endif()

  add_executable(${target}
    "${source_file}"
    $<TARGET_OBJECTS:kxc_runtime_obj>
  )
  target_include_directories(${target} PRIVATE "${KXC_INCLUDE_DIR}")
  target_compile_definitions(${target} PRIVATE KXC_USE_CUDA=${KXC_USE_CUDA} KXC_USE_LLVM=${KXC_USE_LLVM})
  if(KXC_USE_CUDA)
    target_link_libraries(${target} PRIVATE ${KXC_CUDA_LIBS})
  endif()
  if(KXC_USE_LLVM)
    target_include_directories(${target} SYSTEM PRIVATE ${KXC_LLVM_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE ${KXC_LLVM_DEFINITIONS})
    target_link_libraries(${target} PRIVATE ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})
    target_link_options(${target} PRIVATE ${LLVM_LDFLAGS_LIST})
    target_link_options(${target} PRIVATE -fuse-ld=lld)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8 /EHsc)
  endif()
endfunction()

if(KXC_BUILD_RESNET18_IR_DUMP)
  kxc_add_runtime_exe(resnet18_ir_dump test/resnet18_ir_dump.cpp)

  add_custom_target(run_resnet18_ir_dump
    COMMAND "$<TARGET_FILE:resnet18_ir_dump>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS resnet18_ir_dump
    USES_TERMINAL
    COMMENT "Run resnet18_ir_dump and emit test/resnet18_ir_dump.txt"
  )
endif()

if(KXC_BUILD_DEBUG_EXAMPLES)
  kxc_add_runtime_exe(tmp_conv_lower test/tmp_conv_lower.cpp)
  kxc_add_runtime_exe(tmp_maxpool_compute test/tmp_maxpool_compute.cpp)
endif()

if(KXC_BUILD_PASS_TESTS)
  kxc_add_runtime_exe(pass_pipeline_test test/pass_pipeline_test.cpp)
  kxc_add_runtime_exe(profile_bundle_test test/profile_bundle_test.cpp)
  if(KXC_USE_CUDA)
    kxc_add_runtime_exe(cupti_smoke_test test/cupti_smoke_test.cpp)
    if(TARGET CUDA::cuda_driver)
      target_link_libraries(cupti_smoke_test PRIVATE CUDA::cuda_driver)
    endif()
    if(TARGET CUDA::nvrtc)
      target_link_libraries(cupti_smoke_test PRIVATE CUDA::nvrtc)
    endif()
  endif()

  add_custom_target(run_pass_pipeline_test
    COMMAND "$<TARGET_FILE:pass_pipeline_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS pass_pipeline_test
    USES_TERMINAL
    COMMENT "Run pass_pipeline_test"
  )

  add_custom_target(run_profile_bundle_test
    COMMAND "$<TARGET_FILE:profile_bundle_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS profile_bundle_test
    USES_TERMINAL
    COMMENT "Run profile_bundle_test"
  )
  if(KXC_USE_CUDA)
    add_custom_target(run_cupti_smoke_test
      COMMAND "$<TARGET_FILE:cupti_smoke_test>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
      DEPENDS cupti_smoke_test
      USES_TERMINAL
      COMMENT "Run cupti_smoke_test"
    )
  endif()
endif()

if(KXC_BUILD_CODEGEN_TESTS AND KXC_USE_LLVM)
  kxc_add_runtime_exe(codegen_llvm_test test/codegen_llvm_test.cpp)

  add_custom_target(run_codegen_llvm_test
    COMMAND "$<TARGET_FILE:codegen_llvm_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS codegen_llvm_test
    USES_TERMINAL
    COMMENT "Run codegen_llvm_test"
  )
endif()
