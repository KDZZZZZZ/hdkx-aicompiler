/*! \file test/kernel_signature_test.cpp
 * \brief 验证后端无关 KernelSignature 和启动元数据的不变量。
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "codegen/kernel_signature.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

// 捕获所有标准异常，负例只关心契约是否在对象边界被拒绝。
bool Throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// 构造测试统一使用的 float32 dtype，避免每个用例重复裸字段。
DLDataType Float32() {
    return DLDataType{kDLFloat, 32, 1};
}

// 验证合法 input/constant/output 签名保留顺序、类型和稳定类型 key。
bool TestValidSignature() {
    using namespace kxc;
    using namespace kxc::codegen;

    KernelArgSpec input("input", KernelArgRole::kInput, Float32(), {2, 3},
                        Device::CPU(), 16, false);
    KernelArgSpec constant("weight", KernelArgRole::kConstant, Float32(), {3},
                           Device::CPU(), 16, false, "relay.constant.0");
    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {2, 3},
                         Device::CPU(), 16, true);
    KernelSignature signature("kxc_kernel", {input, constant, output});
    const Array<KernelArgSpec> arguments = signature.arguments();

    TEST_CHECK(input.rank() == 2 && constant.rank() == 1 && output.rank() == 2,
               "rank must be derived from shape");
    TEST_CHECK(arguments.size() == 3,
               "signature argument count mismatch");
    TEST_CHECK(arguments[0]->role == KernelArgRole::kInput &&
                   arguments[1]->role == KernelArgRole::kConstant &&
                   arguments[2]->role == KernelArgRole::kOutput,
               "signature argument order mismatch");
    TEST_CHECK(!signature.has_dynamic_input_shape(),
               "static signature reported a dynamic input");
    TEST_CHECK(input.get()->GetTypeKey() == "kxc.codegen.KernelArgSpecNode" &&
                   signature.get()->GetTypeKey() == "kxc.codegen.KernelSignatureNode",
               "kernel contract type keys must be stable");

    const std::string first = signature.ToString();
    const std::string second = signature.ToString();
    TEST_CHECK(first == second && first.find("relay.constant.0") != std::string::npos,
               "signature diagnostic text must be deterministic and include constant key");
    return true;
}

// 动态维只允许出现在输入；标量和零尺寸静态 shape 都是合法张量契约。
bool TestShapeVariants() {
    using namespace kxc;
    using namespace kxc::codegen;

    KernelArgSpec dynamic_input("input", KernelArgRole::kInput, Float32(), {-1, 4},
                                Device::CPU());
    KernelArgSpec scalar_output("scalar", KernelArgRole::kOutput, Float32(), {},
                                Device::CPU(), 4, true);
    KernelSignature dynamic_signature("dynamic_kernel", {dynamic_input, scalar_output});
    TEST_CHECK(dynamic_signature.has_dynamic_input_shape(),
               "-1 input dimension must be reported as dynamic");
    TEST_CHECK(scalar_output.rank() == 0, "empty shape must represent a scalar");

    KernelArgSpec zero_input("empty", KernelArgRole::kInput, Float32(), {2, 0, 3},
                             Device::CPU());
    KernelArgSpec zero_output("empty_out", KernelArgRole::kOutput, Float32(), {2, 0, 3},
                              Device::CPU(), 4, true);
    KernelSignature zero_signature("zero_kernel", {zero_input, zero_output});
    TEST_CHECK(zero_signature.arguments()[0].shape()[1] == 0,
               "zero-size dimension must be preserved");
    return true;
}

// 构造输入和访问器返回值都必须与节点内部 Array 隔离，防止绕过 Validate。
bool TestArrayImmutability() {
    using namespace kxc;
    using namespace kxc::codegen;

    Array<int64_t> source_shape = {2, 3};
    KernelArgSpec input("input", KernelArgRole::kInput, Float32(), source_shape,
                        Device::CPU());
    source_shape.push_back(-2);
    TEST_CHECK(input.rank() == 2, "mutating constructor shape changed KernelArgSpec");
    Array<int64_t> returned_shape = input.shape();
    returned_shape.push_back(-2);
    TEST_CHECK(input.rank() == 2, "mutating returned shape changed KernelArgSpec");

    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {2, 3},
                         Device::CPU(), 4, true);
    Array<KernelArgSpec> source_arguments = {input, output};
    KernelSignature signature("immutable_kernel", source_arguments);
    source_arguments.push_back(output);
    TEST_CHECK(signature.arguments().size() == 2,
               "mutating constructor arguments changed KernelSignature");
    Array<KernelArgSpec> returned_arguments = signature.arguments();
    returned_arguments.push_back(output);
    TEST_CHECK(signature.arguments().size() == 2,
               "mutating returned arguments changed KernelSignature");
    return true;
}

// 合法整数、布尔和向量 lanes 必须原样保存在后端无关签名中。
bool TestDTypeCoverage() {
    using namespace kxc;
    using namespace kxc::codegen;

    KernelArgSpec integer("integer", KernelArgRole::kInput,
                          DLDataType{kDLInt, 32, 1}, {4}, Device::CPU());
    KernelArgSpec boolean("boolean", KernelArgRole::kInput,
                          DLDataType{kDLBool, 8, 1}, {4}, Device::CPU());
    KernelArgSpec vector_output("vector_output", KernelArgRole::kOutput,
                                DLDataType{kDLFloat, 32, 4}, {4}, Device::CPU(),
                                16, true);
    KernelSignature signature("dtype_kernel", {integer, boolean, vector_output});
    TEST_CHECK(signature.arguments()[0]->dtype.code == kDLInt &&
                   signature.arguments()[1]->dtype.code == kDLBool &&
                   signature.arguments()[2]->dtype.lanes == 4,
               "valid dtype code, bits, or lanes were not preserved");
    return true;
}

// 单参数校验必须拒绝所有会让 Launch 无法安全解释 NDArray 的状态。
bool TestInvalidArgumentSpecs() {
    using namespace kxc;
    using namespace kxc::codegen;

    TEST_CHECK(Throws([] {
                   KernelArgSpec("", KernelArgRole::kInput, Float32(), {1}, Device::CPU());
               }),
               "empty argument name should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput,
                                 DLDataType{static_cast<uint8_t>(250), 32, 1}, {1},
                                 Device::CPU());
               }),
               "unsupported dtype code should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput,
                                 DLDataType{kDLFloat, 7, 1}, {1}, Device::CPU());
               }),
               "non-byte-aligned dtype should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput,
                                 DLDataType{kDLFloat, 32, 0}, {1}, Device::CPU());
               }),
               "zero dtype lanes should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput,
                                 DLDataType{kDLFloat, 0, 1}, {1}, Device::CPU());
               }),
               "zero dtype bits should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput, Float32(), {-2},
                                 Device::CPU());
               }),
               "dimension below the dynamic sentinel should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("out", KernelArgRole::kOutput, Float32(), {-1},
                                 Device::CPU(), 4, true);
               }),
               "dynamic output without a shape function should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), {1},
                                 Device::CPU());
               }),
               "constant without key should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput, Float32(), {1},
                                 Device::CPU(), 4, false, "unexpected");
               }),
               "non-constant key should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput, Float32(), {1},
                                 Device::CPU(), 3);
               }),
               "non-power-of-two alignment should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput, Float32(), {1},
                                 Device::CPU(), 0);
               }),
               "zero alignment should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("out", KernelArgRole::kOutput, Float32(), {1},
                                 Device::CPU(), 4, false);
               }),
               "immutable output should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput, Float32(), {1},
                                 Device(), 4, false);
               }),
               "undefined device should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", KernelArgRole::kInput, Float32(), {1},
                                 Device::CPU(), 4, true);
               }),
               "mutable input should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), {1},
                                 Device::CPU(), 4, true, "relay.constant.0");
               }),
               "mutable constant should fail");
    TEST_CHECK(Throws([] {
                   KernelArgSpec("x", static_cast<KernelArgRole>(99), Float32(), {1},
                                 Device::CPU());
               }),
               "unknown argument role should fail");
    return true;
}

// 跨参数校验必须拒绝角色回退、重复名称/key 和无输出签名。
bool TestInvalidSignatures() {
    using namespace kxc;
    using namespace kxc::codegen;

    KernelArgSpec input("input", KernelArgRole::kInput, Float32(), {1}, Device::CPU());
    KernelArgSpec input_two("input_two", KernelArgRole::kInput, Float32(), {1},
                            Device::CPU());
    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {1},
                         Device::CPU(), 4, true);
    KernelArgSpec constant("weight", KernelArgRole::kConstant, Float32(), {1},
                           Device::CPU(), 4, false, "relay.constant.0");
    KernelArgSpec constant_duplicate_key(
        "bias", KernelArgRole::kConstant, Float32(), {1}, Device::CPU(), 4, false,
        "relay.constant.0");
    KernelArgSpec duplicate_name("input", KernelArgRole::kOutput, Float32(), {1},
                                 Device::CPU(), 4, true);

    TEST_CHECK(Throws([&] { KernelSignature("", {input, output}); }),
               "empty symbol should fail");
    TEST_CHECK(Throws([&] { KernelSignature("no_output", {input, input_two}); }),
               "signature without output should fail");
    TEST_CHECK(Throws([&] { KernelSignature("role_backtrack", {output, input}); }),
               "role order backtracking should fail");
    TEST_CHECK(Throws([&] { KernelSignature("duplicate_name", {input, duplicate_name}); }),
               "duplicate argument name should fail");
    TEST_CHECK(Throws([&] {
                   KernelSignature("duplicate_key",
                                   {input, constant, constant_duplicate_key, output});
               }),
               "duplicate constant key should fail");
    KernelArgSpec cuda_output("cuda_output", KernelArgRole::kOutput, Float32(), {1},
                              Device::CUDA(), 4, true);
    TEST_CHECK(Throws([&] { KernelSignature("mixed_device", {input, cuda_output}); }),
               "mixed physical devices should fail");
    return true;
}

// ObjectRef 恢复必须检查节点类型，防止错误静态转换穿透公共对象系统。
bool TestObjectRefTypeChecks() {
    using namespace kxc;
    using namespace kxc::codegen;

    ObjectRef device_ref = Device::CPU();
    TEST_CHECK(Throws([&] { KernelArgSpec invalid(device_ref); }),
               "KernelArgSpec should reject DeviceNode");
    TEST_CHECK(Throws([&] { KernelSignature invalid(device_ref); }),
               "KernelSignature should reject DeviceNode");
    TEST_CHECK(Throws([&] { KernelLaunchMetadata invalid(device_ref); }),
               "KernelLaunchMetadata should reject DeviceNode");

    ObjectRef raw_argument(new KernelArgSpecNode());
    ObjectRef raw_signature(new KernelSignatureNode());
    ObjectRef raw_metadata(new KernelLaunchMetadataNode());
    TEST_CHECK(Throws([&] { KernelArgSpec invalid(raw_argument); }),
               "same-type invalid KernelArgSpecNode should be revalidated");
    TEST_CHECK(Throws([&] { KernelSignature invalid(raw_signature); }),
               "same-type invalid KernelSignatureNode should be revalidated");
    TEST_CHECK(Throws([&] { KernelLaunchMetadata invalid(raw_metadata); }),
               "same-type invalid metadata node should be revalidated");
    return true;
}

// 启动元数据必须严格保持 LLVM/CPU 与 CUDA/GPU 的后端边界。
bool TestLaunchMetadata() {
    using namespace kxc;
    using namespace kxc::codegen;

    KernelLaunchMetadata llvm(Device::CPU(), CodeGenBackend::kLLVM);
    TEST_CHECK(llvm->grid.x == 1 && llvm->block.x == 1,
               "LLVM launch dimensions must default to 1x1x1");
    TEST_CHECK(llvm.get()->GetTypeKey() == "kxc.codegen.KernelLaunchMetadataNode",
               "launch metadata type key must be stable");

    KernelLaunchMetadata cuda(Device::CUDA(), CodeGenBackend::kCUDA,
                              Dim3{4, 1, 1}, Dim3{128, 1, 1}, 256);
    TEST_CHECK(cuda->grid.x == 4 && cuda->block.x == 128 &&
                   cuda->dynamic_shared_memory_bytes == 256,
               "CUDA launch metadata fields mismatch");
    TEST_CHECK(cuda.ToString().find("backend=cuda") != std::string::npos,
               "launch metadata text must name the backend");

    TEST_CHECK(Throws([] {
                   KernelLaunchMetadata(Device::CUDA(), CodeGenBackend::kLLVM);
               }),
               "LLVM metadata on CUDA device should fail");
    TEST_CHECK(Throws([] {
                   KernelLaunchMetadata(Device::CPU(), CodeGenBackend::kCUDA);
               }),
               "CUDA metadata on CPU device should fail");
    TEST_CHECK(Throws([] {
                   KernelLaunchMetadata(Device::CUDA(), CodeGenBackend::kCUDA,
                                        Dim3{0, 1, 1}, Dim3{1, 1, 1});
               }),
               "zero grid dimension should fail");
    TEST_CHECK(Throws([] {
                   KernelLaunchMetadata(Device::CPU(), CodeGenBackend::kLLVM,
                                        Dim3{2, 1, 1}, Dim3{1, 1, 1});
               }),
               "LLVM metadata should reject CUDA dimensions");
    TEST_CHECK(Throws([] {
                   KernelLaunchMetadata(Device::CPU(), CodeGenBackend::kC);
               }),
               "non-executable C backend metadata should fail");
    return true;
}

}  // namespace

// 顺序运行全部签名测试并汇总失败，使单次 CI 输出保留所有契约问题。
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"valid_signature", TestValidSignature},
        {"shape_variants", TestShapeVariants},
        {"array_immutability", TestArrayImmutability},
        {"dtype_coverage", TestDTypeCoverage},
        {"invalid_argument_specs", TestInvalidArgumentSpecs},
        {"invalid_signatures", TestInvalidSignatures},
        {"object_ref_type_checks", TestObjectRefTypeChecks},
        {"launch_metadata", TestLaunchMetadata},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
