cmake_minimum_required(VERSION 3.20)

project(hdkx_aicompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KXC_ENABLE_CUDA "Enable CUDA support when toolkit is available" ON)
option(KXC_BUILD_RESNET18_IR_DUMP "Build test/resnet18_ir_dump.cpp" ON)
option(KXC_BUILD_DEBUG_EXAMPLES "Build temporary debug examples in test/" OFF)

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

set(KXC_RUNTIME_SOURCES
  src/base/expr.cc
  src/base/device.cc
  src/base/device_api.cc
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
  src/relay/op_attrs.cc
  src/relay/op_macros.cc
  src/relay/common_ops.cc
  src/relay/backend/lower.cc
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
  src/tir/pass/print_ir.cc
  src/te/te.cc
)

add_library(kxc_runtime_obj OBJECT ${KXC_RUNTIME_SOURCES})
target_include_directories(kxc_runtime_obj PUBLIC "${KXC_INCLUDE_DIR}")
target_compile_definitions(kxc_runtime_obj PUBLIC KXC_USE_CUDA=${KXC_USE_CUDA})
if(KXC_USE_CUDA)
  target_link_libraries(kxc_runtime_obj PUBLIC ${KXC_CUDA_LIBS})
endif()

add_library(kxc_runtime STATIC $<TARGET_OBJECTS:kxc_runtime_obj>)
target_include_directories(kxc_runtime PUBLIC "${KXC_INCLUDE_DIR}")
target_compile_definitions(kxc_runtime PUBLIC KXC_USE_CUDA=${KXC_USE_CUDA})
if(KXC_USE_CUDA)
  target_link_libraries(kxc_runtime PUBLIC ${KXC_CUDA_LIBS})
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
  target_compile_definitions(${target} PRIVATE KXC_USE_CUDA=${KXC_USE_CUDA})
  if(KXC_USE_CUDA)
    target_link_libraries(${target} PRIVATE ${KXC_CUDA_LIBS})
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
