cmake_minimum_required(VERSION 3.20)

project(hdkx_aicompiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KXC_ENABLE_CUDA "Enable CUDA support when toolkit is available" ON)
option(KXC_ENABLE_LLVM "Enable LLVM backend for codegen" ON)
option(KXC_BUILD_RESNET18_IR_DUMP "Build the diagnostic ResNet18 IR dump tool" OFF)
option(KXC_BUILD_PASS_TESTS "Build pass tests in test/" ON)
option(KXC_BUILD_CODEGEN_TESTS "Build codegen tests" ON)

find_package(Python3 COMPONENTS Interpreter QUIET)
set(KXC_ONNX_IMPORTER_PY_DEPS_FOUND OFF)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import onnx, numpy"
    RESULT_VARIABLE KXC_ONNX_IMPORTER_PY_DEPS_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(KXC_ONNX_IMPORTER_PY_DEPS_RESULT EQUAL 0)
    set(KXC_ONNX_IMPORTER_PY_DEPS_FOUND ON)
  endif()
endif()

if(MSVC)
  add_compile_options(/utf-8 /EHsc)
endif()

set(KXC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(KXC_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
endif()

if(KXC_ENABLE_CUDA AND CUDAToolkit_FOUND)
  set(KXC_USE_CUDA 1)
  set(KXC_CUDA_LIBS CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
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

  set(KXC_LLVM_LINK_MODE --link-static)
  if(WIN32)
    set(KXC_LLVM_LINK_MODE --link-shared)
  endif()

  # 使用 llvm-config 获取与当前平台匹配的链接参数。
  execute_process(
    COMMAND llvm-config ${KXC_LLVM_LINK_MODE} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND llvm-config ${KXC_LLVM_LINK_MODE} --libs core support irreader executionengine orcjit native passes
    OUTPUT_VARIABLE LLVM_LIBS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND llvm-config ${KXC_LLVM_LINK_MODE} --system-libs
    OUTPUT_VARIABLE LLVM_SYSTEM_LIBS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # 转成 CMake list；Windows 上 llvm-config 某些分组可能返回空字符串。
  set(LLVM_LIBS "")
  if(LLVM_LIBS_STR)
    string(REPLACE " " ";" LLVM_LIBS "${LLVM_LIBS_STR}")
  endif()
  set(LLVM_LDFLAGS_LIST "")
  if(LLVM_LDFLAGS)
    string(REPLACE " " ";" LLVM_LDFLAGS_LIST "${LLVM_LDFLAGS}")
  endif()
  set(LLVM_SYSTEM_LIBS "")
  if(LLVM_SYSTEM_LIBS_STR)
    string(REPLACE " " ";" LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS_STR}")
  endif()
  # Remove -lxml2 if not available (optional LLVM dependency)
  list(REMOVE_ITEM LLVM_SYSTEM_LIBS "-lxml2")

  set(KXC_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
  set(KXC_LLVM_DEFINITIONS "")
  if(LLVM_DEFINITIONS)
    separate_arguments(KXC_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
  endif()
  set(KXC_LLVM_LINK_FLAGS "${LLVM_LDFLAGS}")
  set(KXC_MINGW_LIBGCC_PATH "")
  if(MINGW)
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" -print-libgcc-file-name
      OUTPUT_VARIABLE KXC_MINGW_LIBGCC_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(EXISTS "${KXC_MINGW_LIBGCC_PATH}")
      message(STATUS "MinGW libgcc for LLVM JIT: ${KXC_MINGW_LIBGCC_PATH}")
    else()
      set(KXC_MINGW_LIBGCC_PATH "")
      message(WARNING "MinGW libgcc archive was not found. Windows LLVM JIT may miss runtime symbols.")
    endif()
  endif()
elseif(KXC_ENABLE_LLVM AND NOT LLVM_FOUND)
  set(KXC_USE_LLVM 0)
  message(WARNING "KXC_ENABLE_LLVM=ON but LLVM was not found. Codegen will be disabled.")
else()
  set(KXC_USE_LLVM 0)
  message(STATUS "LLVM support: disabled by KXC_ENABLE_LLVM=OFF")
endif()

# ---------------------------------------------------------------------------
# Phase-1 module object targets (file-architecture plan §9 stage 1).
# Source paths are unchanged; only CMake composition is modularized.
# Registration TUs stay in the aggregate object set so static links keep
# global constructors (RegisterBuiltins bootstrap lands in a later stage).
# ---------------------------------------------------------------------------

set(KXC_BASE_SOURCES
  src/base/expr.cc
  src/base/device.cc
  src/base/device_api.cc
  src/base/device_stream.cc
  src/base/storage.cc
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
  src/base/disco/executor.cc
  src/base/device/cpu_device_api.cc
  src/base/device/cuda_device_api.cc
)

set(KXC_RELAY_SOURCES
  src/relay/relay.cc
  src/relay/pass_utils.cc
  src/relay/type_infer.cc
  src/relay/op_attrs.cc
  src/relay/op_macros.cc
  src/relay/backend/lower.cc
  src/relay/pass/print_ir.cc
  src/relay/transforms/pipeline.cc
  src/relay/transforms/fold_constant.cc
  src/relay/transforms/simplify_expr.cc
  src/relay/transforms/canonicalize_cast.cc
  src/relay/transforms/remove_standalone_reshapes.cc
  src/relay/transforms/fold_tuple_get_item.cc
  src/relay/transforms/infer_type.cc
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
)

set(KXC_TIR_SOURCES
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
  src/tir/transforms/bind_cuda_threads.cc
)

set(KXC_TE_SOURCES
  src/te/te.cc
)

set(KXC_FRONTEND_SOURCES
  src/frontend/onnx_importer.cc
)

set(KXC_CODEGEN_COMMON_SOURCES
  src/codegen/compiled_kernel.cc
  src/codegen/codegen_c.cc
  src/codegen/kernel_signature.cc
)

set(KXC_COMPILER_API_SOURCES
  src/api/compile_config.cc
  src/api/compile_result.cc
  src/api/compiled_module.cc
  src/api/kernel_argument_validation.cc
  src/api/compiler.cc
)

set(KXC_RUNTIME_SESSION_SOURCES
  src/runtime/runtime_session.cc
)

set(KXC_CODEGEN_LLVM_SOURCES "")
if(KXC_USE_LLVM)
  set(KXC_CODEGEN_LLVM_SOURCES
    src/codegen/codegen_llvm.cc
    src/codegen/llvm_jit.cc
  )
endif()

set(KXC_CODEGEN_CUDA_SOURCES "")
if(KXC_USE_CUDA)
  set(KXC_CODEGEN_CUDA_SOURCES
    src/codegen/codegen_cuda.cc
    src/codegen/cuda_module.cc
  )
endif()

# Shared compile interface for every module object (paths still under include/).
# Later stages will narrow PUBLIC deps; phase 1 keeps behavior parity.
function(kxc_configure_module_obj target)
  target_include_directories(${target} PUBLIC "${KXC_INCLUDE_DIR}")
  target_compile_definitions(${target} PUBLIC
    KXC_USE_CUDA=${KXC_USE_CUDA}
    KXC_USE_LLVM=${KXC_USE_LLVM})
  if(KXC_USE_CUDA)
    target_link_libraries(${target} PUBLIC ${KXC_CUDA_LIBS})
  endif()
  if(KXC_USE_LLVM)
    target_include_directories(${target} SYSTEM PUBLIC ${KXC_LLVM_INCLUDE_DIRS})
    target_compile_definitions(${target} PUBLIC ${KXC_LLVM_DEFINITIONS}
      KXC_MINGW_LIBGCC_PATH="${KXC_MINGW_LIBGCC_PATH}")
  endif()
endfunction()

add_library(kxc_base_obj OBJECT ${KXC_BASE_SOURCES})
kxc_configure_module_obj(kxc_base_obj)

add_library(kxc_relay_obj OBJECT ${KXC_RELAY_SOURCES})
kxc_configure_module_obj(kxc_relay_obj)

add_library(kxc_tir_obj OBJECT ${KXC_TIR_SOURCES})
kxc_configure_module_obj(kxc_tir_obj)

add_library(kxc_te_obj OBJECT ${KXC_TE_SOURCES})
kxc_configure_module_obj(kxc_te_obj)

add_library(kxc_frontend_obj OBJECT ${KXC_FRONTEND_SOURCES})
kxc_configure_module_obj(kxc_frontend_obj)

add_library(kxc_codegen_common_obj OBJECT ${KXC_CODEGEN_COMMON_SOURCES})
kxc_configure_module_obj(kxc_codegen_common_obj)

add_library(kxc_compiler_api_obj OBJECT ${KXC_COMPILER_API_SOURCES})
kxc_configure_module_obj(kxc_compiler_api_obj)

add_library(kxc_runtime_session_obj OBJECT ${KXC_RUNTIME_SESSION_SOURCES})
kxc_configure_module_obj(kxc_runtime_session_obj)

set(KXC_MODULE_OBJECT_TARGETS
  kxc_base_obj
  kxc_relay_obj
  kxc_tir_obj
  kxc_te_obj
  kxc_frontend_obj
  kxc_codegen_common_obj
  kxc_compiler_api_obj
  kxc_runtime_session_obj
)

if(KXC_USE_LLVM)
  add_library(kxc_codegen_llvm_obj OBJECT ${KXC_CODEGEN_LLVM_SOURCES})
  kxc_configure_module_obj(kxc_codegen_llvm_obj)
  list(APPEND KXC_MODULE_OBJECT_TARGETS kxc_codegen_llvm_obj)
endif()

if(KXC_USE_CUDA)
  add_library(kxc_codegen_cuda_obj OBJECT ${KXC_CODEGEN_CUDA_SOURCES})
  kxc_configure_module_obj(kxc_codegen_cuda_obj)
  list(APPEND KXC_MODULE_OBJECT_TARGETS kxc_codegen_cuda_obj)
endif()

# Compatibility alias: previous single object library name used by tooling.
add_library(kxc_runtime_obj INTERFACE)
foreach(_kxc_mod IN LISTS KXC_MODULE_OBJECT_TARGETS)
  target_sources(kxc_runtime_obj INTERFACE $<TARGET_OBJECTS:${_kxc_mod}>)
  target_link_libraries(kxc_runtime_obj INTERFACE ${_kxc_mod})
endforeach()

set(KXC_RUNTIME_OBJECT_FILES "")
foreach(_kxc_mod IN LISTS KXC_MODULE_OBJECT_TARGETS)
  list(APPEND KXC_RUNTIME_OBJECT_FILES $<TARGET_OBJECTS:${_kxc_mod}>)
endforeach()

add_library(kxc_runtime STATIC ${KXC_RUNTIME_OBJECT_FILES})
target_include_directories(kxc_runtime PUBLIC "${KXC_INCLUDE_DIR}")
target_compile_definitions(kxc_runtime PUBLIC
  KXC_USE_CUDA=${KXC_USE_CUDA}
  KXC_USE_LLVM=${KXC_USE_LLVM})
if(KXC_USE_CUDA)
  target_link_libraries(kxc_runtime PUBLIC ${KXC_CUDA_LIBS})
endif()
if(KXC_USE_LLVM)
  target_include_directories(kxc_runtime SYSTEM PUBLIC ${KXC_LLVM_INCLUDE_DIRS})
  target_compile_definitions(kxc_runtime PUBLIC ${KXC_LLVM_DEFINITIONS}
    KXC_MINGW_LIBGCC_PATH="${KXC_MINGW_LIBGCC_PATH}")
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
    ${KXC_RUNTIME_OBJECT_FILES}
  )
  # Ensure module objects build before linking executables.
  add_dependencies(${target} ${KXC_MODULE_OBJECT_TARGETS})
  target_include_directories(${target} PRIVATE "${KXC_INCLUDE_DIR}")
  target_compile_definitions(${target} PRIVATE
    KXC_USE_CUDA=${KXC_USE_CUDA}
    KXC_USE_LLVM=${KXC_USE_LLVM})
  if(KXC_USE_CUDA)
    target_link_libraries(${target} PRIVATE ${KXC_CUDA_LIBS})
  endif()
  if(KXC_USE_LLVM)
    target_include_directories(${target} SYSTEM PRIVATE ${KXC_LLVM_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE ${KXC_LLVM_DEFINITIONS}
      KXC_MINGW_LIBGCC_PATH="${KXC_MINGW_LIBGCC_PATH}")
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

if(Python3_Interpreter_FOUND)
  add_custom_target(check_relay_op_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_relay_op_contract.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --matrix "${CMAKE_CURRENT_SOURCE_DIR}/test/relay_op_contract.json"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Check Relay operator implementation contract"
  )
else()
  message(WARNING "Python3 was not found; skipping check_relay_op_contract target.")
endif()

if(KXC_BUILD_PASS_TESTS)
  kxc_add_runtime_exe(object_test test/object_test.cpp)
  kxc_add_runtime_exe(pass_pipeline_test test/pass_pipeline_test.cpp)
  kxc_add_runtime_exe(device_info_test test/device_info_test.cpp)
  kxc_add_runtime_exe(device_runtime_test test/device_runtime_test.cpp)
  kxc_add_runtime_exe(infer_type_test test/infer_type_test.cpp)
  kxc_add_runtime_exe(profile_bundle_test test/profile_bundle_test.cpp)
  kxc_add_runtime_exe(compiler_contract_test test/compiler_contract_test.cpp)
  kxc_add_runtime_exe(kernel_signature_test test/kernel_signature_test.cpp)
  kxc_add_runtime_exe(compiled_module_test test/compiled_module_test.cpp)
  kxc_add_runtime_exe(runtime_session_test test/runtime_session_test.cpp)
  kxc_add_runtime_exe(cuda_schedule_test test/cuda_schedule_test.cpp)
  if(KXC_ONNX_IMPORTER_PY_DEPS_FOUND)
    set(KXC_ONNX_IMPORT_JSON "${CMAKE_CURRENT_BINARY_DIR}/onnx/resnet18.import.json")
    set(KXC_ONNX_IMPORT_PARAMS "${CMAKE_CURRENT_BINARY_DIR}/onnx/resnet18.params.bin")
    add_custom_command(
      OUTPUT "${KXC_ONNX_IMPORT_JSON}" "${KXC_ONNX_IMPORT_PARAMS}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/onnx"
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
              "${Python3_EXECUTABLE}" -m kxc_onnx
              --model "${CMAKE_CURRENT_SOURCE_DIR}/resnet18.onnx"
              --json "${KXC_ONNX_IMPORT_JSON}"
              --params "${KXC_ONNX_IMPORT_PARAMS}"
              --batch 1
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/resnet18.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/__main__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/importer.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/spec.py"
      VERBATIM
      COMMENT "Generate ResNet18 ONNX import fixture")
    add_custom_target(onnx_importer_fixture DEPENDS
      "${KXC_ONNX_IMPORT_JSON}" "${KXC_ONNX_IMPORT_PARAMS}")
    kxc_add_runtime_exe(onnx_importer_test test/onnx_importer_test.cpp)
    add_dependencies(onnx_importer_test onnx_importer_fixture)
    target_compile_definitions(onnx_importer_test PRIVATE
      KXC_ONNX_IMPORT_JSON_PATH="${KXC_ONNX_IMPORT_JSON}"
      KXC_ONNX_IMPORT_PARAMS_PATH="${KXC_ONNX_IMPORT_PARAMS}")
  else()
    message(WARNING "Python3 with onnx and numpy was not found; skipping onnx_importer_test target.")
  endif()
  if(KXC_USE_CUDA)
    kxc_add_runtime_exe(cupti_smoke_test test/cupti_smoke_test.cpp)
    kxc_add_runtime_exe(codegen_cuda_test test/codegen_cuda_test.cpp)
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

  add_custom_target(run_object_test
    COMMAND "$<TARGET_FILE:object_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS object_test
    USES_TERMINAL
    COMMENT "Run object_test"
  )

  add_custom_target(run_device_info_test
    COMMAND "$<TARGET_FILE:device_info_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS device_info_test
    USES_TERMINAL
    COMMENT "Run device_info_test"
  )
  add_custom_target(run_device_runtime_test
    COMMAND "$<TARGET_FILE:device_runtime_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS device_runtime_test
    USES_TERMINAL
    COMMENT "Run device_runtime_test"
  )

  add_custom_target(run_profile_bundle_test
    COMMAND "$<TARGET_FILE:profile_bundle_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS profile_bundle_test
    USES_TERMINAL
    COMMENT "Run profile_bundle_test"
  )
  add_custom_target(run_infer_type_test
    COMMAND "$<TARGET_FILE:infer_type_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS infer_type_test
    USES_TERMINAL
    COMMENT "Run infer_type_test"
  )
  add_custom_target(run_compiler_contract_test
    COMMAND "$<TARGET_FILE:compiler_contract_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS compiler_contract_test
    USES_TERMINAL
    COMMENT "Run compiler_contract_test"
  )
  add_custom_target(run_kernel_signature_test
    COMMAND "$<TARGET_FILE:kernel_signature_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS kernel_signature_test
    USES_TERMINAL
    COMMENT "Run kernel_signature_test"
  )
  add_custom_target(run_compiled_module_test
    COMMAND "$<TARGET_FILE:compiled_module_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS compiled_module_test
    USES_TERMINAL
    COMMENT "Run compiled_module_test"
  )
  add_custom_target(run_runtime_session_test
    COMMAND "$<TARGET_FILE:runtime_session_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS runtime_session_test
    USES_TERMINAL
    COMMENT "Run runtime_session_test"
  )
  add_custom_target(run_cuda_schedule_test
    COMMAND "$<TARGET_FILE:cuda_schedule_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS cuda_schedule_test
    USES_TERMINAL
    COMMENT "Run cuda_schedule_test"
  )
  if(KXC_ONNX_IMPORTER_PY_DEPS_FOUND)
    add_custom_target(run_onnx_importer_test
      COMMAND "$<TARGET_FILE:onnx_importer_test>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      DEPENDS onnx_importer_test
      USES_TERMINAL
      COMMENT "Run onnx_importer_test"
    )
  endif()
  if(KXC_USE_CUDA)
    add_custom_target(run_codegen_cuda_test
      COMMAND "$<TARGET_FILE:codegen_cuda_test>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
      DEPENDS codegen_cuda_test
      USES_TERMINAL
      COMMENT "Run codegen_cuda_test"
    )
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
  kxc_add_runtime_exe(op_numeric_llvm_test test/op_numeric_llvm_test.cpp)

  add_custom_target(run_codegen_llvm_test
    COMMAND "$<TARGET_FILE:codegen_llvm_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS codegen_llvm_test
    USES_TERMINAL
    COMMENT "Run codegen_llvm_test"
  )

  add_custom_target(run_op_numeric_llvm_test
    COMMAND "$<TARGET_FILE:op_numeric_llvm_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS op_numeric_llvm_test
    USES_TERMINAL
    COMMENT "Run per-op LLVM numeric tests"
  )
endif()
