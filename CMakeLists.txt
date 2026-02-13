cmake_minimum_required(VERSION 3.20)

project(hdkx_aicompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KXC_BUILD_TEST_LOWER "Build test/test_lower.cpp" ON)
option(KXC_BUILD_RESNET18_IR_DUMP "Build test/resnet18_ir_dump.cpp" ON)
option(KXC_BUILD_GENERATED_NETWORK "Build test/generated_network.cpp" OFF)
option(KXC_BUILD_TESTS "Build test targets from test/build_*.bat" ON)
option(KXC_BUILD_DEBUG_TARGETS "Build temporary debug targets in test/" OFF)
option(KXC_BUILD_LEGACY_TESTS "Build legacy tests that may require API refresh" OFF)

if(MSVC)
  add_compile_options(/utf-8 /EHsc)
endif()

find_package(CUDAToolkit REQUIRED)

set(KXC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(KXC_RUNTIME_SOURCES
  src/base/op.cc
  src/base/device_api.cc
  src/base/device/cpu_device_api.cc
  src/base/device/cuda_device_api.cc
  src/tir/pass/print_ir.cc
  src/relay/backend/lower.cc
  src/relay/common_ops.cc
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

set(KXC_DEVICE_API_SOURCES
  src/base/device_api.cc
  src/base/device/cpu_device_api.cc
  src/base/device/cuda_device_api.cc
)

set(KXC_RELAY_OP_SOURCES
  src/relay/common_ops.cc
  src/relay/op/tensor/math.cc
  src/relay/op/tensor/reduce.cc
  src/relay/op/tensor/transform.cc
  src/relay/op/nn/activation.cc
  src/relay/op/nn/convolution.cc
  src/relay/op/nn/pooling.cc
  src/relay/op/nn/dense.cc
  src/relay/op/nn/softmax.cc
)

add_library(kxc_runtime_obj OBJECT ${KXC_RUNTIME_SOURCES})
target_include_directories(kxc_runtime_obj PUBLIC "${KXC_INCLUDE_DIR}")
target_link_libraries(kxc_runtime_obj PUBLIC CUDA::cudart)

# Optional archive target for IDE browsing and external linkage.
add_library(kxc_runtime STATIC $<TARGET_OBJECTS:kxc_runtime_obj>)
target_include_directories(kxc_runtime PUBLIC "${KXC_INCLUDE_DIR}")
target_link_libraries(kxc_runtime PUBLIC CUDA::cudart)

function(kxc_add_runtime_exe target source_file)
  add_executable(${target}
    "${source_file}"
    $<TARGET_OBJECTS:kxc_runtime_obj>
  )
  target_include_directories(${target} PRIVATE "${KXC_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8 /EHsc)
  endif()
endfunction()

function(kxc_add_exe target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE "${KXC_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8 /EHsc)
  endif()
endfunction()

if(KXC_BUILD_TEST_LOWER)
  kxc_add_runtime_exe(test_lower test/test_lower.cpp)
endif()

if(KXC_BUILD_GENERATED_NETWORK)
  kxc_add_exe(generated_network
    test/generated_network.cpp
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
    src/relay/op/tensor/math.cc
    src/relay/op/tensor/reduce.cc
    src/relay/op/tensor/transform.cc
    src/relay/op/nn/softmax.cc
    src/relay/common_ops.cc
  )
endif()

if(KXC_BUILD_RESNET18_IR_DUMP)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/resnet18_ir_dump.cpp")
    message(FATAL_ERROR
      "Missing test/resnet18_ir_dump.cpp. "
      "Generate it first with: python python/gen_resnet18_ir_dump_cpp.py --model resnet18.onnx --out test/resnet18_ir_dump.cpp"
    )
  endif()

  kxc_add_runtime_exe(resnet18_ir_dump test/resnet18_ir_dump.cpp)

  add_custom_target(run_resnet18_ir_dump
    COMMAND "$<TARGET_FILE:resnet18_ir_dump>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
    DEPENDS resnet18_ir_dump
    USES_TERMINAL
    COMMENT "Run resnet18_ir_dump and emit test/resnet18_ir_dump.txt"
  )
endif()

if(KXC_BUILD_TESTS)
  # Matches test/build_test_attr.bat
  kxc_add_exe(test_attr
    test/test_attr_registry.cpp
    src/relay/common_ops.cc
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )

  # Matches test/build_test_control.bat
  kxc_add_exe(test_control
    test/test_control_flow.cpp
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )

  # Matches test/build_test_graph.bat
  kxc_add_exe(test_graph
    test/test_graph_construction.cpp
    src/relay/common_ops.cc
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )

  # Matches test/build_test_op.bat
  kxc_add_exe(test_op
    test/test_op_structure.cpp
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )

  # Matches test/build_test_op_params.bat
  kxc_add_exe(test_op_params
    test/test_op_params.cpp
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )

  # Matches test/build_test_relay.bat
  kxc_add_exe(test_relay
    test/test_relay_structure.cpp
    ${KXC_DEVICE_API_SOURCES}
  )

  # Matches test/build_test_relay_ffi.bat
  kxc_add_exe(test_relay_ffi
    test/test_relay_op_ffi.cpp
    ${KXC_DEVICE_API_SOURCES}
    src/base/op.cc
    src/relay/op/op_ffi.cc
    src/relay/op/tensor/math.cc
    src/relay/op/tensor/reduce.cc
    src/relay/op/tensor/transform.cc
    src/relay/op/nn/convolution.cc
    src/relay/op/nn/pooling.cc
    src/relay/op/nn/softmax.cc
    src/relay/op/nn/activation.cc
    src/relay/op/nn/dense.cc
  )

  # Header-only/lightweight tests (same spirit as their .bat files).
  kxc_add_exe(test_packedfunc_tensor test/test_packedfunc_tensor.cpp)
  kxc_add_exe(test_pass test/test_pass.cpp)
  kxc_add_exe(test_te test/test_te.cpp)
  kxc_add_exe(test_tir test/test_tir_structure.cpp)
  kxc_add_exe(test_topi test/test_topi.cpp)

  # Add commonly used utility tests to CMake.
  kxc_add_exe(test_base test/test_base.cpp src/base/capi.cpp)
  kxc_add_exe(test_file_op test/test_file_op.cpp src/base/capi.cpp)
  kxc_add_exe(test_graph_construction test/test_graph_construction.cpp)
  kxc_add_exe(test_new_ops test/test_new_ops.cpp src/base/op.cc ${KXC_DEVICE_API_SOURCES})
endif()

if(KXC_BUILD_LEGACY_TESTS)
  kxc_add_exe(test_schedule_api test/test_schedule_api.cpp)
  kxc_add_exe(test_te_comprehensive test/test_te_comprehensive.cpp)
endif()

if(KXC_BUILD_DEBUG_TARGETS)
  kxc_add_exe(tmp_conv_lower
    test/tmp_conv_lower.cpp
    src/relay/backend/lower.cc
    src/relay/op/tensor/math.cc
    src/relay/op/tensor/transform.cc
    src/relay/op/nn/activation.cc
    src/relay/op/nn/convolution.cc
    src/relay/op/nn/pooling.cc
    src/relay/op/nn/dense.cc
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )

  kxc_add_exe(tmp_maxpool_compute
    test/tmp_maxpool_compute.cpp
    src/relay/op/nn/pooling.cc
    src/base/op.cc
    ${KXC_DEVICE_API_SOURCES}
  )
endif()
