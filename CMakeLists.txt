cmake_minimum_required(VERSION 3.23)

project(hdkx_aicompiler LANGUAGES C CXX)

include(CTest)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

option(KXC_ENABLE_CUDA "Enable CUDA support when toolkit is available" ON)
option(KXC_ENABLE_LLVM "Enable LLVM backend for codegen" ON)
option(KXC_ENABLE_CONTROL_RUNTIME
  "Enable default-OFF static-exact CPU control compilation and execution" OFF)
if(DEFINED KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION)
  message(FATAL_ERROR "KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION was removed; use KXC_ENABLE_CONTROL_RUNTIME")
endif()
if(DEFINED KXC_ENABLE_REGION_TASK_DAG)
  message(FATAL_ERROR "KXC_ENABLE_REGION_TASK_DAG was removed; RuntimeSession uses ExecutablePlan only")
endif()
if(DEFINED KXC_ENABLE_RUNTIME_SHAPE_TASKS OR DEFINED KXC_ENABLE_RUNTIME_SHAPE_CUDA OR DEFINED KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE)
  message(FATAL_ERROR "Legacy RuntimeShape/bridge CMake cache variables were removed by W4-1; use KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI")
endif()
if(DEFINED KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION)
  message(FATAL_ERROR "KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION was removed; use KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2")
endif()
option(KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
  "Enable nonconstant CompiledModule invocation contracts" OFF)
option(KXC_ENABLE_SHAPE_PRODUCTION_EXACT
  "Enable the production exact Relay/compiler/shape adapter" OFF)
option(KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
  "Enable the experimental restricted symbolic Shape decision adapter" OFF)
if(KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE AND NOT KXC_ENABLE_SHAPE_PRODUCTION_EXACT)
  message(STATUS "Restricted symbolic Shape requires the production exact adapter; enabling it")
  set(KXC_ENABLE_SHAPE_PRODUCTION_EXACT ON CACHE BOOL
    "Enable the production exact Relay/compiler/shape adapter" FORCE)
endif()

option(KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
  "Enable default-off bounded v2 whole-plan adaptive control plane" OFF)
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
set(KXC_DLPACK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack/include")

# Installed/exported SDK headers. Repository experiments are tracked separately
# below so a source build can test them without publishing them as package API.
set(KXC_PUBLIC_HEADERS
  include/kxc/compiler/compile_config.h
  include/kxc/compiler/compiler.h
  include/kxc/compiler/artifact.h
  include/kxc/compiler/identity.h
  include/kxc/compiler/pipeline.h

  include/kxc/distributed/ccl_backend.h
  include/kxc/distributed/dref.h
  include/kxc/distributed/execution_plan.h
  include/kxc/distributed/executor.h
  include/kxc/distributed/placement.h
  include/kxc/distributed/session.h
  include/kxc/distributed/worker.h
  include/kxc/ffi/detail/packed_func_inl.h
  include/kxc/ffi/packed_func.h
  include/kxc/ffi/registration.h
  include/kxc/ffi/registry.h
  include/kxc/frontend/onnx_importer.h
  include/kxc/ir/expr.h
  include/kxc/pass/context.h
  include/kxc/pass/pass.h
  include/kxc/profiling/profiling.h
  include/kxc/compiler/distributed/multi_device.h
  include/kxc/relay/op.h
  include/kxc/relay/op_attr_types.h
  include/kxc/relay/op_macros.h
  include/kxc/relay/printer/print_ir.h
  include/kxc/relay/pass_utils.h
  include/kxc/relay/relay.h
  include/kxc/relay/transforms/annotate_memory_scope.h
  include/kxc/relay/transforms/canonicalize_cast.h
  include/kxc/relay/transforms/capture_post_dfs_index_in_spans.h
  include/kxc/relay/transforms/eliminate_common_subexpr.h
  include/kxc/relay/transforms/eliminate_dead_let.h
  include/kxc/relay/transforms/fold_constant.h
  include/kxc/relay/transforms/fold_tuple_get_item.h
  include/kxc/relay/transforms/infer_type.h
  include/kxc/relay/transforms/normalize_to_anf.h
  include/kxc/relay/transforms/pipeline.h
  include/kxc/relay/transforms/remove_standalone_reshapes.h
  include/kxc/relay/transforms/simplify_expr.h
  include/kxc/relay/type_infer.h
  include/kxc/relay/visitor.h
  include/kxc/runtime/compiled_module.h
  include/kxc/runtime/control_execution_plan.h
  include/kxc/runtime/control_session.h
  include/kxc/runtime/device.h
  include/kxc/runtime/device_api.h
  include/kxc/runtime/device_info.h
  include/kxc/runtime/device_stream.h
  include/kxc/runtime/executable_plan.h
  include/kxc/runtime/kernel_abi.h
  include/kxc/runtime/ndarray.h
  include/kxc/runtime/module_invocation.h
  include/kxc/runtime/session.h
  include/kxc/runtime/storage.h
  include/kxc/support/arena.h
  include/kxc/support/container.h
  include/kxc/support/detail/array_inl.h
  include/kxc/support/detail/map_inl.h
  include/kxc/support/object.h
  include/kxc/support/object_registration.h
  include/kxc/target/target.h
  include/kxc/target/virtual_device.h
  include/kxc/te/te.h
  include/kxc/te/topi.h
  include/kxc/te/topi/broadcast.h
  include/kxc/te/topi/elemwise.h
  include/kxc/te/topi/nn.h
  include/kxc/te/topi/reduction.h
  include/kxc/te/topi/tags.h
  include/kxc/te/topi/transform.h
  include/kxc/te/topi/utils.h
  include/kxc/tir/expr.h
  include/kxc/tir/printer/print_ir.h
  include/kxc/tir/pass_utils.h
  include/kxc/tir/stmt.h
  include/kxc/tir/transforms/bind_cuda_threads.h
  include/kxc/tir/transforms/convert_for_loops_serial.h
  include/kxc/tir/transforms/fold_constant.h
  include/kxc/tir/transforms/force_narrow_index_to_i32.h
  include/kxc/tir/transforms/loop_partition.h
  include/kxc/tir/transforms/pipeline.h
  include/kxc/tir/transforms/remove_no_op.h
  include/kxc/tir/transforms/simplify_expr.h
  include/kxc/tir/transforms/unroll_loop.h
  include/kxc/tir/transforms/vectorize_loop.h
  include/kxc/tir/visitor.h
)

set(KXC_EXPERIMENTAL_HEADERS
  include/kxc/compiler/adaptive_hot_swap_v2.h
  include/kxc/compiler/experimental_identity.h
  include/kxc/compiler/adaptive_production_experimental.h
  include/kxc/compiler/restricted_symbolic_shape.h
  include/kxc/compiler/shape_exact.h
  include/kxc/compiler/shape_specialization.h
  include/kxc/shape/shape.h
)

add_library(kxc_dlpack INTERFACE)
add_library(kxc::dlpack ALIAS kxc_dlpack)
set_target_properties(kxc_dlpack PROPERTIES EXPORT_NAME dlpack)
target_include_directories(kxc_dlpack INTERFACE
  $<BUILD_INTERFACE:${KXC_DLPACK_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_sources(kxc_dlpack INTERFACE
  FILE_SET dlpack_headers TYPE HEADERS
  BASE_DIRS "${KXC_DLPACK_INCLUDE_DIR}"
  FILES third_party/dlpack/include/dlpack/dlpack.h)

add_library(kxc_public_api INTERFACE)
add_library(kxc::public_api ALIAS kxc_public_api)
set_target_properties(kxc_public_api PROPERTIES EXPORT_NAME public_api)
target_include_directories(kxc_public_api INTERFACE
  $<BUILD_INTERFACE:${KXC_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(kxc_public_api INTERFACE kxc_dlpack)

set(KXC_PUBLIC_MODULE_TARGETS "")
foreach(_kxc_public_module IN ITEMS
    support ffi runtime profiling target pass ir tir te relay distributed
    frontend compiler)
  set(_kxc_module_headers "")
  foreach(_kxc_header IN LISTS KXC_PUBLIC_HEADERS)
    if(_kxc_header MATCHES "^include/kxc/${_kxc_public_module}/")
      list(APPEND _kxc_module_headers "${_kxc_header}")
    endif()
  endforeach()
  if(NOT _kxc_module_headers)
    message(FATAL_ERROR "No public headers declared for module ${_kxc_public_module}")
  endif()

  set(_kxc_module_target "kxc_${_kxc_public_module}_api")
  add_library(${_kxc_module_target} INTERFACE)
  add_library(kxc::${_kxc_public_module}_api ALIAS ${_kxc_module_target})
  set_target_properties(${_kxc_module_target} PROPERTIES
    EXPORT_NAME "${_kxc_public_module}_api")
  target_include_directories(${_kxc_module_target} INTERFACE
    $<BUILD_INTERFACE:${KXC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(${_kxc_module_target} INTERFACE kxc_dlpack)
  target_sources(${_kxc_module_target} INTERFACE
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS "${KXC_INCLUDE_DIR}"
    FILES ${_kxc_module_headers})
  target_link_libraries(kxc_public_api INTERFACE ${_kxc_module_target})
  list(APPEND KXC_PUBLIC_MODULE_TARGETS ${_kxc_module_target})
endforeach()
target_compile_definitions(kxc_runtime_api INTERFACE
  KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=$<BOOL:${KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI}>)

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

  set(KXC_LLVM_CONFIG_EXECUTABLE "")
  if(WIN32 AND EXISTS "${LLVM_TOOLS_BINARY_DIR}/llvm-config.exe")
    set(KXC_LLVM_CONFIG_EXECUTABLE
      "${LLVM_TOOLS_BINARY_DIR}/llvm-config.exe")
  elseif(EXISTS "${LLVM_TOOLS_BINARY_DIR}/llvm-config")
    set(KXC_LLVM_CONFIG_EXECUTABLE
      "${LLVM_TOOLS_BINARY_DIR}/llvm-config")
  else()
    find_program(KXC_LLVM_CONFIG_EXECUTABLE NAMES llvm-config)
  endif()
  if(NOT KXC_LLVM_CONFIG_EXECUTABLE)
    message(FATAL_ERROR
      "LLVM was found, but llvm-config is required to determine link flags")
  endif()

  # 使用 llvm-config 获取与当前平台匹配的链接参数。
  execute_process(
    COMMAND "${KXC_LLVM_CONFIG_EXECUTABLE}" ${KXC_LLVM_LINK_MODE} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND "${KXC_LLVM_CONFIG_EXECUTABLE}" ${KXC_LLVM_LINK_MODE} --libs core support irreader executionengine orcjit native passes
    OUTPUT_VARIABLE LLVM_LIBS_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND "${KXC_LLVM_CONFIG_EXECUTABLE}" ${KXC_LLVM_LINK_MODE} --system-libs
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
# Module object targets. Each source group owns one architectural layer;
# external SDK usage is attached only to the objects that require it.
# Registration translation units remain in the aggregate object set so static
# links keep their global constructors.
# ---------------------------------------------------------------------------

set(KXC_SUPPORT_SOURCES
  src/support/arena.cc
  src/support/hash.cc
  src/support/object.cc
  src/support/type_registry.cc
  src/support/string.cc
)

set(KXC_FFI_SOURCES
  src/ffi/builtin_registry.cc
  src/ffi/registry.cc
  src/ffi/packed_func.cc
)

set(KXC_IR_SOURCES
  src/ir/expr.cc
)

set(KXC_RUNTIME_CORE_SOURCES
  src/runtime/module_invocation.cc
  src/runtime/device.cc
  src/runtime/device_api.cc
  src/runtime/device_stream.cc
  src/runtime/executable_plan.cc
  src/runtime/memory_plan.cc
  src/runtime/storage.cc
  src/runtime/ndarray.cc
  src/runtime/device/cpu_device_api.cc
  src/runtime/kernel_abi.cc
)

set(KXC_RUNTIME_CUDA_SOURCES
  src/runtime/device/cuda_device_api.cc
)

set(KXC_PROFILING_SOURCES
  src/profiling/options.cc
  src/profiling/profiling.cc)
set(KXC_TARGET_SOURCES
  src/target/target.cc
  src/target/virtual_device.cc)
set(KXC_PASS_SOURCES
  src/pass/context.cc
  src/pass/pass.cc)

set(KXC_DISTRIBUTED_PLACEMENT_SOURCES
  src/distributed/placement.cc)

set(KXC_DISTRIBUTED_PLAN_SOURCES
  src/distributed/execution_plan.cc
  src/distributed/execution_plan_json.cc)

set(KXC_DISTRIBUTED_RUNTIME_SOURCES
  src/distributed/dref.cc
  src/distributed/threaded_session.cc
  src/distributed/worker.cc
  src/distributed/ccl_cpu.cc
  src/distributed/executor.cc)

set(KXC_RELAY_IR_SOURCES
  src/relay/visitor.cc
  src/relay/relay.cc
  src/relay/printer/print_ir.cc
  src/relay/type_infer.cc
  src/relay/op_attrs.cc
  src/relay/op_macros.cc
  src/relay/op_registry.cc
  src/relay/op/tensor/math.cc
  src/relay/op/tensor/reduce.cc
  src/relay/op/tensor/transform.cc
  src/relay/op/nn/activation.cc
  src/relay/op/nn/convolution.cc
  src/relay/op/nn/pooling.cc
  src/relay/op/nn/dense.cc
  src/relay/op/nn/layer_norm.cc
  src/relay/op/nn/softmax.cc
  src/relay/op/op_ffi.cc
)

set(KXC_RELAY_PASS_SOURCES
  src/relay/pass_utils.cc
  src/relay/transforms/pipeline.cc
  src/relay/transforms/fold_constant.cc
  src/relay/transforms/simplify_expr.cc
  src/relay/transforms/canonicalize_cast.cc
  src/relay/transforms/remove_standalone_reshapes.cc
  src/relay/transforms/fold_tuple_get_item.cc
  src/relay/transforms/infer_type.cc
  src/relay/transforms/normalize_to_anf.cc
  src/relay/transforms/eliminate_common_subexpr.cc
  src/relay/transforms/eliminate_dead_let.cc
  src/relay/transforms/capture_post_dfs_index_in_spans.cc
  src/relay/transforms/annotate_memory_scope.cc)

set(KXC_RELAY_DISTRIBUTED_SOURCES
  src/relay/distributed/plan_adapter.cc)

set(KXC_TIR_IR_SOURCES
  src/tir/visitor.cc
  src/tir/expr.cc
  src/tir/stmt.cc
  src/tir/printer/print_ir.cc)

set(KXC_TIR_PASS_SOURCES
  src/tir/pass_utils.cc
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
  src/te/topi/broadcast.cc
  src/te/topi/elemwise.cc
  src/te/topi/nn.cc
  src/te/topi/reduction.cc
  src/te/topi/transform.cc
  src/te/topi/utils.cc
)

set(KXC_FRONTEND_SOURCES
  src/frontend/onnx_importer.cc
)

set(KXC_CODEGEN_COMMON_SOURCES src/codegen/common/compiled_kernel.cc)
set(KXC_CODEGEN_C_SOURCES src/codegen/c/codegen_c.cc)

set(KXC_RUNTIME_EXECUTABLE_SOURCES
  src/runtime/compiled_module.cc
  src/runtime/control_execution_plan.cc
  src/runtime/kernel_argument_validation.cc
)

set(KXC_COMPILER_SOURCES
  src/compiler/analysis/logical_value.cc
  src/compiler/analysis/primitive_unit.cc
  src/compiler/analysis/relay_program.cc
  src/compiler/analysis/resolved_relay_call.cc
  src/compiler/cache/primitive_cache.cc
  src/compiler/compile_config.cc
  src/compiler/control_flow/control_plan.cc
  src/compiler/control_flow/control_plan_adapter.cc
  src/compiler/control_flow/relay_control_plan.cc
  src/compiler/control_flow/production_control_flow.cc
  src/compiler/identity.cc
  src/compiler/pipeline_resolver.cc
  src/compiler/primitive_compiler.cc
  src/compiler/compiler.cc

  src/compiler/graph/partition.cc
  src/compiler/graph/value_graph.cc
  src/compiler/kernel_abi_builder.cc
  src/compiler/lowering/lowered_function.cc
  src/compiler/lowering/lowered_graph.cc
  src/compiler/lowering/te_to_tir.cc
  src/compiler/distributed/multi_device.cc
)

set(KXC_RUNTIME_SESSION_SOURCES
  src/runtime/control_session.cc
  src/runtime/session.cc
)

if(KXC_ENABLE_SHAPE_PRODUCTION_EXACT OR KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE OR KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2)
  list(APPEND KXC_COMPILER_SOURCES src/compiler/experimental_identity.cc)
endif()
if(KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2)
  list(APPEND KXC_COMPILER_SOURCES
    src/compiler/adaptive/production_path_experimental.cc
    src/compiler/adaptive/adaptive_hot_swap_v2.cc)
endif()
if(KXC_ENABLE_SHAPE_PRODUCTION_EXACT OR KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE)
  list(APPEND KXC_COMPILER_SOURCES
    src/compiler/shape_exact.cc
    src/compiler/shape_specialization.cc
    src/compiler/restricted_symbolic_shape.cc)
endif()

set(KXC_CODEGEN_LLVM_SOURCES "")
if(KXC_USE_LLVM)
  set(KXC_CODEGEN_LLVM_SOURCES
    src/codegen/llvm/codegen_llvm.cc
    src/codegen/llvm/codegen_llvm_stmt.cc
    src/codegen/llvm/llvm_jit.cc
  )
endif()

set(KXC_CODEGEN_CUDA_SOURCES "")
if(KXC_USE_CUDA)
  set(KXC_CODEGEN_CUDA_SOURCES
    src/codegen/cuda/codegen_cuda.cc
    src/codegen/cuda/cuda_module.cc
  )
endif()

function(kxc_configure_module_obj target)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(${target} PRIVATE kxc_public_api)
  target_compile_definitions(${target} PRIVATE
    KXC_USE_CUDA=${KXC_USE_CUDA}
    KXC_USE_LLVM=${KXC_USE_LLVM}
    KXC_ENABLE_CONTROL_RUNTIME=$<BOOL:${KXC_ENABLE_CONTROL_RUNTIME}>
    KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=$<BOOL:${KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI}>
    KXC_ENABLE_SHAPE_PRODUCTION_EXACT=$<BOOL:${KXC_ENABLE_SHAPE_PRODUCTION_EXACT}>
    KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=$<BOOL:${KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE}>
    KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=$<BOOL:${KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2}>)
endfunction()

function(kxc_enable_cuda target)
  if(KXC_USE_CUDA)
    target_link_libraries(${target} PRIVATE ${KXC_CUDA_LIBS})
  endif()
endfunction()

function(kxc_enable_llvm target)
  if(KXC_USE_LLVM)
    target_include_directories(${target} SYSTEM PRIVATE ${KXC_LLVM_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE ${KXC_LLVM_DEFINITIONS}
      KXC_MINGW_LIBGCC_PATH="${KXC_MINGW_LIBGCC_PATH}")
  endif()
endfunction()

add_library(kxc_support_obj OBJECT ${KXC_SUPPORT_SOURCES})
kxc_configure_module_obj(kxc_support_obj)

if(KXC_ENABLE_SHAPE_PRODUCTION_EXACT OR KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE)
  add_library(kxc_shape_obj OBJECT src/shape/shape.cc)
  kxc_configure_module_obj(kxc_shape_obj)
else()
  add_library(kxc_shape_test_obj OBJECT src/shape/shape.cc)
  kxc_configure_module_obj(kxc_shape_test_obj)
  if(NOT KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2)
    add_library(kxc_experimental_identity_test_obj OBJECT
      src/compiler/experimental_identity.cc)
    kxc_configure_module_obj(kxc_experimental_identity_test_obj)
  endif()
  add_library(kxc_shape_experimental_test_obj OBJECT
    src/compiler/shape_exact.cc
    src/compiler/shape_specialization.cc
    src/compiler/restricted_symbolic_shape.cc)
  kxc_configure_module_obj(kxc_shape_experimental_test_obj)
endif()

add_library(kxc_ffi_obj OBJECT ${KXC_FFI_SOURCES})
kxc_configure_module_obj(kxc_ffi_obj)

add_library(kxc_ir_obj OBJECT ${KXC_IR_SOURCES})
kxc_configure_module_obj(kxc_ir_obj)

add_library(kxc_runtime_core_obj OBJECT ${KXC_RUNTIME_CORE_SOURCES})
kxc_configure_module_obj(kxc_runtime_core_obj)
kxc_enable_cuda(kxc_runtime_core_obj)

add_library(kxc_runtime_cuda_obj OBJECT ${KXC_RUNTIME_CUDA_SOURCES})
kxc_configure_module_obj(kxc_runtime_cuda_obj)
kxc_enable_cuda(kxc_runtime_cuda_obj)

add_library(kxc_profiling_obj OBJECT ${KXC_PROFILING_SOURCES})
kxc_configure_module_obj(kxc_profiling_obj)
kxc_enable_cuda(kxc_profiling_obj)

add_library(kxc_target_obj OBJECT ${KXC_TARGET_SOURCES})
kxc_configure_module_obj(kxc_target_obj)

add_library(kxc_pass_core_obj OBJECT ${KXC_PASS_SOURCES})
kxc_configure_module_obj(kxc_pass_core_obj)

add_library(kxc_distributed_placement_obj OBJECT
  ${KXC_DISTRIBUTED_PLACEMENT_SOURCES})
kxc_configure_module_obj(kxc_distributed_placement_obj)

add_library(kxc_distributed_plan_obj OBJECT ${KXC_DISTRIBUTED_PLAN_SOURCES})
kxc_configure_module_obj(kxc_distributed_plan_obj)

add_library(kxc_distributed_runtime_obj OBJECT
  ${KXC_DISTRIBUTED_RUNTIME_SOURCES})
kxc_configure_module_obj(kxc_distributed_runtime_obj)

add_library(kxc_relay_obj OBJECT ${KXC_RELAY_IR_SOURCES})
kxc_configure_module_obj(kxc_relay_obj)

add_library(kxc_relay_passes_obj OBJECT ${KXC_RELAY_PASS_SOURCES})
kxc_configure_module_obj(kxc_relay_passes_obj)

add_library(kxc_relay_distributed_obj OBJECT
  ${KXC_RELAY_DISTRIBUTED_SOURCES})
kxc_configure_module_obj(kxc_relay_distributed_obj)

add_library(kxc_tir_obj OBJECT ${KXC_TIR_IR_SOURCES})
kxc_configure_module_obj(kxc_tir_obj)

add_library(kxc_tir_passes_obj OBJECT ${KXC_TIR_PASS_SOURCES})
kxc_configure_module_obj(kxc_tir_passes_obj)

add_library(kxc_te_obj OBJECT ${KXC_TE_SOURCES})
kxc_configure_module_obj(kxc_te_obj)

add_library(kxc_frontend_obj OBJECT ${KXC_FRONTEND_SOURCES})
kxc_configure_module_obj(kxc_frontend_obj)

add_library(kxc_codegen_common_obj OBJECT ${KXC_CODEGEN_COMMON_SOURCES})
kxc_configure_module_obj(kxc_codegen_common_obj)

add_library(kxc_codegen_c_obj OBJECT ${KXC_CODEGEN_C_SOURCES})
kxc_configure_module_obj(kxc_codegen_c_obj)

add_library(kxc_runtime_executable_obj OBJECT ${KXC_RUNTIME_EXECUTABLE_SOURCES})
kxc_configure_module_obj(kxc_runtime_executable_obj)

add_library(kxc_compiler_obj OBJECT ${KXC_COMPILER_SOURCES})
kxc_configure_module_obj(kxc_compiler_obj)
target_link_libraries(kxc_compiler_obj PRIVATE Threads::Threads)
kxc_enable_cuda(kxc_compiler_obj)
kxc_enable_llvm(kxc_compiler_obj)

add_library(kxc_runtime_session_obj OBJECT ${KXC_RUNTIME_SESSION_SOURCES})
kxc_configure_module_obj(kxc_runtime_session_obj)

set(KXC_MODULE_OBJECT_TARGETS
  kxc_support_obj
  kxc_ffi_obj
  kxc_ir_obj
  kxc_runtime_core_obj
  kxc_runtime_cuda_obj
  kxc_profiling_obj
  kxc_target_obj
  kxc_pass_core_obj
  kxc_distributed_placement_obj
  kxc_distributed_plan_obj
  kxc_distributed_runtime_obj
  kxc_relay_obj
  kxc_relay_passes_obj
  kxc_relay_distributed_obj
  kxc_tir_obj
  kxc_tir_passes_obj
  kxc_te_obj
  kxc_frontend_obj
  kxc_codegen_common_obj
  kxc_codegen_c_obj
  kxc_runtime_executable_obj
  kxc_compiler_obj
  kxc_runtime_session_obj
)

if(TARGET kxc_shape_obj)
  list(APPEND KXC_MODULE_OBJECT_TARGETS kxc_shape_obj)
endif()

if(KXC_USE_LLVM)
  add_library(kxc_codegen_llvm_obj OBJECT ${KXC_CODEGEN_LLVM_SOURCES})
  kxc_configure_module_obj(kxc_codegen_llvm_obj)
  kxc_enable_llvm(kxc_codegen_llvm_obj)
  list(APPEND KXC_MODULE_OBJECT_TARGETS kxc_codegen_llvm_obj)
endif()

if(KXC_USE_CUDA)
  add_library(kxc_codegen_cuda_obj OBJECT ${KXC_CODEGEN_CUDA_SOURCES})
  kxc_configure_module_obj(kxc_codegen_cuda_obj)
  kxc_enable_cuda(kxc_codegen_cuda_obj)
  list(APPEND KXC_MODULE_OBJECT_TARGETS kxc_codegen_cuda_obj)
endif()

set(KXC_RUNTIME_OBJECT_FILES "")
foreach(_kxc_mod IN LISTS KXC_MODULE_OBJECT_TARGETS)
  list(APPEND KXC_RUNTIME_OBJECT_FILES $<TARGET_OBJECTS:${_kxc_mod}>)
endforeach()

add_library(kxc_runtime STATIC ${KXC_RUNTIME_OBJECT_FILES})
add_library(kxc::runtime ALIAS kxc_runtime)
set_target_properties(kxc_runtime PROPERTIES EXPORT_NAME runtime)
target_link_libraries(kxc_runtime PUBLIC kxc_public_api Threads::Threads)
if(KXC_USE_CUDA)
  target_link_libraries(kxc_runtime PRIVATE ${KXC_CUDA_LIBS})
endif()
if(KXC_USE_LLVM)
  target_link_libraries(kxc_runtime PRIVATE ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})
  target_link_options(kxc_runtime PRIVATE ${LLVM_LDFLAGS_LIST})
  # LLVM archives contain bitcode (LTO), use lld to handle them
  target_link_options(kxc_runtime PRIVATE -fuse-ld=lld)
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
  target_link_libraries(${target} PRIVATE kxc_public_api Threads::Threads)
  target_compile_definitions(${target} PRIVATE
    KXC_USE_CUDA=${KXC_USE_CUDA}
    KXC_USE_LLVM=${KXC_USE_LLVM}
    KXC_ENABLE_CONTROL_RUNTIME=$<BOOL:${KXC_ENABLE_CONTROL_RUNTIME}>
    KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=$<BOOL:${KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI}>
    KXC_ENABLE_SHAPE_PRODUCTION_EXACT=$<BOOL:${KXC_ENABLE_SHAPE_PRODUCTION_EXACT}>
    KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=$<BOOL:${KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE}>
    KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=$<BOOL:${KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2}>)
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


add_executable(shape_system_test test/shape_system_test.cpp)
if(TARGET kxc_shape_obj)
  target_sources(shape_system_test PRIVATE $<TARGET_OBJECTS:kxc_shape_obj>)
  add_dependencies(shape_system_test kxc_shape_obj)
elseif(TARGET kxc_shape_test_obj)
  target_sources(shape_system_test PRIVATE
    $<TARGET_OBJECTS:kxc_shape_test_obj>)
  add_dependencies(shape_system_test kxc_shape_test_obj)
endif()
target_link_libraries(shape_system_test PRIVATE kxc_public_api)
if(MSVC)
  target_compile_options(shape_system_test PRIVATE /utf-8 /EHsc)
endif()
add_custom_target(run_shape_system_test
  COMMAND "$<TARGET_FILE:shape_system_test>"
  DEPENDS shape_system_test
  USES_TERMINAL
  COMMENT "Run shape_system_test")

kxc_add_runtime_exe(shape_specialization_test
  test/shape_specialization_test.cpp)
if(TARGET kxc_shape_experimental_test_obj)
  target_sources(shape_specialization_test PRIVATE
    $<TARGET_OBJECTS:kxc_shape_test_obj>
    $<TARGET_OBJECTS:kxc_shape_experimental_test_obj>)
  add_dependencies(shape_specialization_test kxc_shape_test_obj
    kxc_shape_experimental_test_obj)
  if(TARGET kxc_experimental_identity_test_obj)
    target_sources(shape_specialization_test PRIVATE
      $<TARGET_OBJECTS:kxc_experimental_identity_test_obj>)
    add_dependencies(shape_specialization_test
      kxc_experimental_identity_test_obj)
  endif()
endif()
add_custom_target(run_shape_specialization_test
  COMMAND "$<TARGET_FILE:shape_specialization_test>"
  DEPENDS shape_specialization_test
  USES_TERMINAL
  COMMENT "Run shape_specialization_test")

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
  add_custom_target(check_nlp_gpu_validation
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_nlp_gpu_validation.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Check deterministic NLP reference and capability gates"
  )
  add_custom_target(check_relay_op_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_relay_op_contract.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --matrix "${CMAKE_CURRENT_SOURCE_DIR}/contracts/relay_op_contract.json"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Check Relay operator implementation contract"
  )
  add_custom_target(check_pass_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_pass_contract.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --matrix "${CMAKE_CURRENT_SOURCE_DIR}/contracts/pass_contract.json"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Check Relay and TIR pass contracts"
  )
  add_custom_target(check_include_layers
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/architecture/check_include_layers.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Check module include directions"
  )
  add_custom_target(check_public_headers
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/architecture/check_public_headers.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --compile
            --compiler "${CMAKE_CXX_COMPILER}"
            --define "KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=$<BOOL:${KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2}>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Check and compile public headers"
  )
else()
  message(WARNING "Python3 was not found; skipping check_relay_op_contract target.")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/kxcConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/kxcConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kxc"
  NO_SET_AND_CHECK_MACRO)
install(TARGETS kxc_runtime
  EXPORT kxcTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(TARGETS kxc_public_api
  EXPORT kxcTargets)
foreach(_kxc_module_target IN LISTS KXC_PUBLIC_MODULE_TARGETS)
  install(TARGETS ${_kxc_module_target}
    EXPORT kxcTargets
    FILE_SET public_headers DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endforeach()
install(TARGETS kxc_dlpack
  EXPORT kxcTargets
  FILE_SET dlpack_headers DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT kxcTargets
  NAMESPACE kxc::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kxc")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/kxcConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/kxc")

if(KXC_BUILD_PASS_TESTS)
  kxc_add_runtime_exe(shape_production_exact_test test/shape_production_exact_test.cpp)
  kxc_add_runtime_exe(restricted_symbolic_shape_test test/restricted_symbolic_shape_test.cpp)
  if(TARGET kxc_shape_experimental_test_obj)
    foreach(_kxc_shape_test IN ITEMS shape_production_exact_test
        restricted_symbolic_shape_test)
      target_sources(${_kxc_shape_test} PRIVATE
        $<TARGET_OBJECTS:kxc_shape_test_obj>
        $<TARGET_OBJECTS:kxc_shape_experimental_test_obj>)
      add_dependencies(${_kxc_shape_test} kxc_shape_test_obj
        kxc_shape_experimental_test_obj)
      if(TARGET kxc_experimental_identity_test_obj)
        target_sources(${_kxc_shape_test} PRIVATE
          $<TARGET_OBJECTS:kxc_experimental_identity_test_obj>)
        add_dependencies(${_kxc_shape_test}
          kxc_experimental_identity_test_obj)
      endif()
    endforeach()
  endif()
  if(KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2)
    kxc_add_runtime_exe(adaptive_hot_swap_v2_test
      test/adaptive_preparation_v2_test.cpp)
  endif()
  if(BUILD_TESTING)
    add_custom_target(run_restricted_symbolic_shape_test
      COMMAND "$<TARGET_FILE:restricted_symbolic_shape_test>"
      DEPENDS restricted_symbolic_shape_test
      USES_TERMINAL
      COMMENT "Run restricted symbolic Shape decision adapter test")
    add_custom_target(run_shape_production_exact_test
      COMMAND "$<TARGET_FILE:shape_production_exact_test>"
      DEPENDS shape_production_exact_test
      USES_TERMINAL
      COMMENT "Run shape production exact adapter test")
    if(KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2)
      add_test(NAME adaptive_hot_swap_v2_test
        COMMAND "$<TARGET_FILE:adaptive_hot_swap_v2_test>")
      set_tests_properties(adaptive_hot_swap_v2_test PROPERTIES
        LABELS "adaptive;adaptive-hot-swap-v2;cpu" TIMEOUT 60)
      add_custom_target(run_adaptive_hot_swap_v2_tests
        COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
                -L adaptive-hot-swap-v2 -L cpu
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        DEPENDS adaptive_hot_swap_v2_test
        USES_TERMINAL
        COMMENT "Run bounded adaptive v2 CPU CTest suite")
    endif()

    if(KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2)
      add_custom_target(run_adaptive_tests
        COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
                -L adaptive -L cpu
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        DEPENDS adaptive_hot_swap_v2_test
        USES_TERMINAL
        COMMENT "Run adaptive v2 CPU CTest suite")
    endif()
  endif()
  kxc_add_runtime_exe(object_test test/object_test.cpp)
  kxc_add_runtime_exe(packed_func_test test/packed_func_test.cpp)
  kxc_add_runtime_exe(registry_test test/registry_test.cpp)
  kxc_add_runtime_exe(type_registration_test test/type_registration_test.cpp)
  kxc_add_runtime_exe(pass_pipeline_test test/pass_pipeline_test.cpp)
  kxc_add_runtime_exe(pipeline_resolver_test test/pipeline_resolver_test.cpp)
  kxc_add_runtime_exe(device_info_test test/device_info_test.cpp)
  kxc_add_runtime_exe(device_runtime_test test/device_runtime_test.cpp)
  kxc_add_runtime_exe(executable_plan_test test/executable_plan_test.cpp)
  kxc_add_runtime_exe(control_plan_test test/control_plan_test.cpp)
  kxc_add_runtime_exe(control_plan_reference_executor_test test/control_plan_reference_executor_test.cpp)
  kxc_add_runtime_exe(control_runtime_integration_test test/control_runtime_integration_test.cpp)
  kxc_add_runtime_exe(graph_partition_test test/graph_partition_test.cpp)
  kxc_add_runtime_exe(relay_anf_test test/relay_anf_test.cpp)
  kxc_add_runtime_exe(executable_capability_test test/executable_capability_test.cpp)
  kxc_add_runtime_exe(relay_control_plan_test test/relay_control_plan_test.cpp)
  kxc_add_runtime_exe(infer_type_test test/infer_type_test.cpp)
  kxc_add_runtime_exe(onnx_import_spec_contract_test test/onnx_import_spec_contract_test.cpp)
  kxc_add_runtime_exe(profile_bundle_test test/profile_bundle_test.cpp)
  kxc_add_runtime_exe(compiler_contract_test test/compiler_contract_test.cpp)
  kxc_add_runtime_exe(compiler_identity_test test/compiler_identity_test.cpp)
  if(TARGET kxc_experimental_identity_test_obj)
    target_sources(compiler_identity_test PRIVATE
      $<TARGET_OBJECTS:kxc_experimental_identity_test_obj>)
    add_dependencies(compiler_identity_test kxc_experimental_identity_test_obj)
  endif()
  kxc_add_runtime_exe(operator_compilation_test test/operator_compilation_test.cpp)
  kxc_add_runtime_exe(primitive_cache_test test/primitive_cache_test.cpp)
  kxc_add_runtime_exe(compiler_extension_contract_test test/compiler_extension_contract_test.cpp)
  kxc_add_runtime_exe(kernel_signature_test test/kernel_signature_test.cpp)
  kxc_add_runtime_exe(compiled_module_test test/compiled_module_test.cpp)
  kxc_add_runtime_exe(runtime_session_test test/runtime_session_test.cpp)
  kxc_add_runtime_exe(cuda_schedule_test test/cuda_schedule_test.cpp)

  set(KXC_CONTROL_FLOW_TESTS
    relay_anf_test
    executable_capability_test
    control_plan_test
    control_plan_reference_executor_test
    relay_control_plan_test)
  foreach(_kxc_control_flow_test IN LISTS KXC_CONTROL_FLOW_TESTS)
    add_test(NAME ${_kxc_control_flow_test}
      COMMAND "$<TARGET_FILE:${_kxc_control_flow_test}>")
    set_tests_properties(${_kxc_control_flow_test} PROPERTIES
      LABELS "control-flow;cpu"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
  endforeach()
  add_test(NAME control_runtime_integration_test
    COMMAND "$<TARGET_FILE:control_runtime_integration_test>")
  set_tests_properties(control_runtime_integration_test PROPERTIES
    LABELS "control-flow;control-runtime;runtime-plan;cpu"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
  add_custom_target(run_control_flow_tests
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
            -L control-flow
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS ${KXC_CONTROL_FLOW_TESTS} control_runtime_integration_test
    USES_TERMINAL
    COMMENT "Run control-flow CTest suite")
  add_custom_target(run_control_runtime_tests
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
            -L control-runtime
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS control_runtime_integration_test
    USES_TERMINAL
    COMMENT "Run resolved control-runtime CTest suite")

  foreach(_kxc_runtime_plan_test IN ITEMS runtime_session_test)
    add_test(NAME ${_kxc_runtime_plan_test}
      COMMAND "$<TARGET_FILE:${_kxc_runtime_plan_test}>")
    set_tests_properties(${_kxc_runtime_plan_test} PROPERTIES
      LABELS "runtime-plan;cpu;core"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
  endforeach()

  if(KXC_ONNX_IMPORTER_PY_DEPS_FOUND)
    set(KXC_ONNX_IMPORT_JSON "${CMAKE_CURRENT_BINARY_DIR}/onnx/resnet18.import.json")
    set(KXC_ONNX_IMPORT_PARAMS "${CMAKE_CURRENT_BINARY_DIR}/onnx/resnet18.params.bin")
    set(KXC_ONNX_TRANSFORMER_PROTO "${CMAKE_CURRENT_BINARY_DIR}/onnx/exact_transformer.onnx")
    set(KXC_ONNX_TRANSFORMER_JSON "${CMAKE_CURRENT_BINARY_DIR}/onnx/exact_transformer.import.json")
    set(KXC_ONNX_TRANSFORMER_PARAMS "${CMAKE_CURRENT_BINARY_DIR}/onnx/exact_transformer.params.bin")
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
    add_custom_command(
      OUTPUT
        "${KXC_ONNX_TRANSFORMER_PROTO}"
        "${KXC_ONNX_TRANSFORMER_JSON}"
        "${KXC_ONNX_TRANSFORMER_PARAMS}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/onnx"
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
              "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/test/generate_onnx_transformer_fixture.py"
              --onnx "${KXC_ONNX_TRANSFORMER_PROTO}"
              --json "${KXC_ONNX_TRANSFORMER_JSON}"
              --params "${KXC_ONNX_TRANSFORMER_PARAMS}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/test/generate_onnx_transformer_fixture.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/importer.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/kxc_onnx/spec.py"
      VERBATIM
      COMMENT "Generate exact Transformer protobuf/import fixtures")
    add_custom_target(onnx_importer_fixture DEPENDS
      "${KXC_ONNX_IMPORT_JSON}" "${KXC_ONNX_IMPORT_PARAMS}"
      "${KXC_ONNX_TRANSFORMER_PROTO}" "${KXC_ONNX_TRANSFORMER_JSON}"
      "${KXC_ONNX_TRANSFORMER_PARAMS}")
    kxc_add_runtime_exe(onnx_importer_test test/onnx_importer_test.cpp)
    add_dependencies(onnx_importer_test onnx_importer_fixture)
    target_compile_definitions(onnx_importer_test PRIVATE
      KXC_ONNX_IMPORT_JSON_PATH="${KXC_ONNX_IMPORT_JSON}"
      KXC_ONNX_IMPORT_PARAMS_PATH="${KXC_ONNX_IMPORT_PARAMS}"
      KXC_ONNX_TRANSFORMER_JSON_PATH="${KXC_ONNX_TRANSFORMER_JSON}"
      KXC_ONNX_TRANSFORMER_PARAMS_PATH="${KXC_ONNX_TRANSFORMER_PARAMS}")
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
  add_custom_target(run_onnx_import_spec_contract_test
    COMMAND "$<TARGET_FILE:onnx_import_spec_contract_test>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS onnx_import_spec_contract_test
    USES_TERMINAL
    COMMENT "Run onnx_import_spec_contract_test"
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
  add_custom_target(run_runtime_plan_tests
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L runtime-plan
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS runtime_session_test control_runtime_integration_test
    USES_TERMINAL
    COMMENT "Run runtime plan CTest suite"
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
  if(KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI)
    kxc_add_runtime_exe(compiled_module_dynamic_llvm_test
      test/compiled_module_dynamic_llvm_test.cpp)
  endif()

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
  if(TARGET compiled_module_dynamic_llvm_test)
    add_custom_target(run_compiled_module_dynamic_llvm_test
      COMMAND "$<TARGET_FILE:compiled_module_dynamic_llvm_test>"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
      DEPENDS compiled_module_dynamic_llvm_test
      USES_TERMINAL
      COMMENT "Run real LLVM dynamic CompiledModule Invoke E2E"
    )
  endif()
endif()

# CTest is the CI execution surface. Keep the existing run_* custom targets for
# developer compatibility, but make every CPU/core executable independently
# runnable and keep backend/hardware tests out of the CPU label.
function(kxc_add_ctest_if_target target labels)
  if(TARGET ${target})
    add_test(NAME ${target} COMMAND "$<TARGET_FILE:${target}>")
    set_tests_properties(${target} PROPERTIES LABELS "${labels}")
  endif()
endfunction()

kxc_add_ctest_if_target(shape_system_test "shape;cpu")
kxc_add_ctest_if_target(shape_specialization_test "shape;cpu")
kxc_add_ctest_if_target(shape_production_exact_test "shape;shape-production;cpu")
kxc_add_ctest_if_target(restricted_symbolic_shape_test "shape;shape-restricted-symbolic;cpu")

foreach(_kxc_cpu_core_test IN ITEMS
    object_test packed_func_test registry_test type_registration_test
    pass_pipeline_test pipeline_resolver_test device_info_test device_runtime_test
    executable_plan_test graph_partition_test infer_type_test
    onnx_import_spec_contract_test profile_bundle_test
    compiler_contract_test compiler_identity_test
    operator_compilation_test primitive_cache_test
    compiler_extension_contract_test kernel_signature_test compiled_module_test
    cuda_schedule_test)
  # cuda_schedule_test exercises CPU-only scheduling contracts with synthetic
  # Target snapshots; it neither requires nor launches CUDA hardware.
  kxc_add_ctest_if_target(${_kxc_cpu_core_test} "cpu;core")
endforeach()

kxc_add_ctest_if_target(codegen_llvm_test "llvm;codegen")
kxc_add_ctest_if_target(op_numeric_llvm_test "llvm;numeric")
kxc_add_ctest_if_target(compiled_module_dynamic_llvm_test "llvm;dynamic-module;e2e;numeric")
if(KXC_USE_LLVM)
  kxc_add_ctest_if_target(onnx_importer_test "llvm;onnx;protobuf;e2e;integration")
else()
  kxc_add_ctest_if_target(onnx_importer_test "onnx;integration")
endif()
kxc_add_ctest_if_target(codegen_cuda_test "cuda;hardware;codegen;local-evidence")
kxc_add_ctest_if_target(cupti_smoke_test "cuda;hardware")

if(Python3_Interpreter_FOUND)
  add_test(NAME relay_op_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_relay_op_contract.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --matrix "${CMAKE_CURRENT_SOURCE_DIR}/contracts/relay_op_contract.json")
  add_test(NAME pass_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_pass_contract.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --matrix "${CMAKE_CURRENT_SOURCE_DIR}/contracts/pass_contract.json")
  add_test(NAME include_layers
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/architecture/check_include_layers.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME public_headers
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/architecture/check_public_headers.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --compile
            --compiler "${CMAKE_CXX_COMPILER}"
            --define "KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=$<BOOL:${KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2}>")
  set_tests_properties(relay_op_contract pass_contract include_layers public_headers
    PROPERTIES LABELS "cpu;contract")
endif()
