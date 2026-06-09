/*! \file test/device_info_test.cpp
 * \brief 验证 Device/Target 信息查询契约。
 */

#include "base/device.h"
#include "base/device_api.h"
#include "base/packedfunc.h"
#include "base/registry.h"
#include "base/target.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

#define TEST_CHECK(cond, msg)                                                     \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (msg) << "\n";     \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool TestCPUDeviceInfo() {
    class kxc::Device cpu(kxc::kCPU, 0);
    kxc::DeviceAttributes attrs = kxc::CollectDeviceAttributes(cpu);

    TEST_CHECK(attrs.exists == 1, "CPU device should exist");
    TEST_CHECK(attrs.max_threads_per_block >= 1, "CPU thread count should be populated");
    TEST_CHECK(attrs.multi_processor_count >= 1, "CPU processor count should be populated");
    TEST_CHECK(attrs.max_threads_per_multiprocessor >= 1,
               "CPU threads per multiprocessor should be populated");
    TEST_CHECK(!attrs.device_name.empty(), "CPU device name should be populated");
    TEST_CHECK(!attrs.arch.empty(), "CPU arch should be populated");
    TEST_CHECK(attrs.total_global_memory > 0, "CPU total memory should be populated");
    TEST_CHECK(attrs.available_global_memory >= 0, "CPU available memory should be non-negative");
    TEST_CHECK(attrs.total_global_memory >= attrs.available_global_memory,
               "CPU total memory should be >= available memory");

    kxc::RetValue name_rv;
    kxc::GetDeviceAttr(cpu, kxc::DeviceAttrKind::kDeviceName, &name_rv);
    TEST_CHECK(name_rv.As<std::string>() == attrs.device_name,
               "DeviceAttrKind::kDeviceName should match DeviceAttributes");
    return true;
}

bool TestTargetBuildMatchesDeviceInfo() {
    kxc::Target target = kxc::BuildTarget(kxc::kCPU, 0);
    TEST_CHECK(target.defined(), "CPU target should be defined");
    TEST_CHECK(target->kind == "llvm", "CPU target kind should be llvm");
    TEST_CHECK(target->device_type == kxc::kCPU, "CPU target device type should match");
    TEST_CHECK(target->device_id == 0, "CPU target device id should match");
    TEST_CHECK(target->attrs.exists == 1, "CPU target attrs should mark device existing");
    TEST_CHECK(!target->attrs.arch.empty(), "CPU target attrs should include arch");
    TEST_CHECK(!target->attrs.device_name.empty(), "CPU target attrs should include device name");
    return true;
}

bool TestUnifiedDeviceInfo() {
    std::vector<kxc::DeviceInfo> infos = kxc::GetAllDeviceInfo();
    TEST_CHECK(!infos.empty(), "GetAllDeviceInfo should return at least CPU info");

    bool saw_cpu = false;
    bool saw_cuda = false;
    for (const auto& info : infos) {
        if (info.device_type == kxc::kCPU) {
            saw_cpu = true;
            TEST_CHECK(info.device_id == 0, "CPU device id should be 0");
            TEST_CHECK(info.device_type_name == "cpu", "CPU type name should be cpu");
            TEST_CHECK(info.target_kind == "llvm", "CPU target kind should be llvm");
            TEST_CHECK(info.available, "CPU info should be available");
            TEST_CHECK(info.status == "ok", "CPU status should be ok");
            TEST_CHECK(info.attrs.exists == 1, "CPU attrs should mark existence");
        }
        if (info.device_type == kxc::kGPU) {
            saw_cuda = true;
            TEST_CHECK(info.device_type_name == "cuda", "GPU type name should be cuda");
            TEST_CHECK(info.target_kind == "cuda", "GPU target kind should be cuda");
            if (info.available) {
                TEST_CHECK(info.attrs.exists == 1, "Available CUDA info should have exists=1");
                TEST_CHECK(!info.attrs.device_name.empty(),
                           "Available CUDA info should include device name");
                TEST_CHECK(!info.attrs.arch.empty(), "Available CUDA info should include arch");
            } else {
                TEST_CHECK(info.attrs.exists == 0, "Unavailable CUDA info should have exists=0");
                TEST_CHECK(!info.status.empty(), "Unavailable CUDA info should explain status");
            }
        }
    }

    TEST_CHECK(saw_cpu, "GetAllDeviceInfo should include CPU");
    TEST_CHECK(saw_cuda, "GetAllDeviceInfo should include CUDA availability status");

    std::vector<class kxc::Device> devices = kxc::ListDevices();
    TEST_CHECK(!devices.empty(), "ListDevices should return available devices");
    bool listed_cpu = false;
    for (const auto& device : devices) {
        TEST_CHECK(device.device_type() != kxc::kUnknown,
                   "ListDevices should not return unknown devices");
        if (device.device_type() == kxc::kCPU && device.device_id() == 0) {
            listed_cpu = true;
        }
    }
    TEST_CHECK(listed_cpu, "ListDevices should include CPU:0");
    return true;
}

bool TestCUDAAttributesDoNotRequireCUDARuntimeAvailability() {
    class kxc::Device cuda(kxc::kGPU, 0);
    kxc::DeviceAPI* api = kxc::GetDeviceAPI(kxc::kGPU);
    kxc::DeviceAttributes attrs = api->GetDeviceAttributes(cuda);

    if (attrs.exists) {
        TEST_CHECK(!attrs.device_name.empty(), "Available CUDA device should have a name");
        TEST_CHECK(attrs.compute_version_major >= 0,
                   "Available CUDA device should have compute major");
        TEST_CHECK(attrs.compute_version_minor >= 0,
                   "Available CUDA device should have compute minor");
    } else {
        TEST_CHECK(!attrs.device_name.empty(), "Unavailable CUDA should carry status text");
    }
    return true;
}

bool TestRegistryJSONEntrypoints() {
    kxc::PackedFunc get_all = kxc::Registry::Global().Get("device_api.GetAllDeviceInfo");
    TEST_CHECK(get_all, "device_api.GetAllDeviceInfo should be registered");
    std::string all_json = get_all().As<std::string>();
    TEST_CHECK(all_json.find("\"devices\"") != std::string::npos,
               "GetAllDeviceInfo JSON should contain devices key");
    TEST_CHECK(all_json.find("\"device_type_name\":\"cpu\"") != std::string::npos,
               "GetAllDeviceInfo JSON should contain CPU info");
    TEST_CHECK(all_json.find("\"device_type_name\":\"cuda\"") != std::string::npos,
               "GetAllDeviceInfo JSON should contain CUDA status");

    kxc::PackedFunc list_devices = kxc::Registry::Global().Get("device_api.ListDevices");
    TEST_CHECK(list_devices, "device_api.ListDevices should be registered");
    std::string list_json = list_devices().As<std::string>();
    TEST_CHECK(list_json.find("\"device_type_name\":\"cpu\"") != std::string::npos,
               "ListDevices JSON should contain available CPU");
    TEST_CHECK(list_json.find("\"available\":false") == std::string::npos,
               "ListDevices JSON should only include available devices");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, bool (*)()>> tests = {
        {"cpu_device_info", TestCPUDeviceInfo},
        {"target_build_matches_device_info", TestTargetBuildMatchesDeviceInfo},
        {"unified_device_info", TestUnifiedDeviceInfo},
        {"cuda_attrs_do_not_require_runtime_availability",
         TestCUDAAttributesDoNotRequireCUDARuntimeAvailability},
        {"registry_json_entrypoints", TestRegistryJSONEntrypoints},
    };

    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                std::cerr << "Test failed: " << test.first << "\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << test.first << ": " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "All device info tests passed.\n";
    return 0;
}

