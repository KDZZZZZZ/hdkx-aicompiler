/*! \file test/device_runtime_test.cpp
 * \brief 验证 Device、Storage、NDArray、stream 及 CPU/CUDA 复制契约。
 */

#include "kxc/runtime/device.h"
#include "kxc/runtime/device_api.h"
#include "kxc/runtime/device_stream.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/runtime/storage.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

#define TEST_CHECK(condition, message)                                         \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << '\n'; \
            return false;                                                       \
        }                                                                       \
    } while (0)

template <typename F>
// 执行失败用例并判断其是否按契约抛出标准异常。
bool Throws(F&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// 验证 Device 值语义、DLPack 映射和多线程规范驻留。
bool TestDeviceIdentity() {
    const kxc::Device cpu = kxc::Device::CPU();
    const kxc::Device cuda = kxc::Device::CUDA();
    TEST_CHECK(cpu.ToString() == "cpu:0", "CPU canonical string");
    TEST_CHECK(cuda.ToString() == "cuda:0", "CUDA canonical string");
    TEST_CHECK(cpu == kxc::Device(kxc::kCPU, 0), "Device value equality");
    TEST_CHECK(cpu != cuda, "different device types must differ");
    TEST_CHECK(Throws([] { kxc::Device(kxc::kCPU, 1); }), "cpu:1 must be rejected");
    TEST_CHECK(Throws([] { kxc::Device(kxc::kCUDA, -1); }), "negative id must be rejected");

    DLDevice dl = kxc::ToDLDevice(cuda);
    TEST_CHECK(dl.device_type == kDLCUDA && dl.device_id == 0, "DLPack mapping");
    TEST_CHECK(kxc::FromDLDevice(dl) == cuda, "DLPack round trip");

    std::vector<kxc::Device> devices(16);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < devices.size(); ++i) {
        workers.emplace_back([&, i] {
            devices[i] = kxc::DeviceManager::Global()->Get(kxc::kCPU, 0);
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& device : devices) {
        TEST_CHECK(device.get() == devices.front().get(), "manager must intern devices");
    }
    const kxc::Array<kxc::Device> available =
        kxc::DeviceManager::Global()->ListAvailableDevices();
    TEST_CHECK(!available.empty() && available[0] == cpu,
               "available devices must include cpu:0");
    return true;
}

// 验证 CPU 对齐分配、容量边界和外部 deleter 恰好执行一次。
bool TestCPUAllocationAndStorage() {
    const kxc::Device cpu = kxc::Device::CPU();
    TEST_CHECK(kxc::DeviceAlloc(cpu, 0) == nullptr, "zero allocation must be null");
    kxc::DeviceFree(cpu, nullptr);
    for (size_t alignment : {size_t{1}, size_t{16}, size_t{64}, size_t{256}}) {
        void* ptr = kxc::DeviceAlloc(cpu, 257, alignment);
        TEST_CHECK(reinterpret_cast<uintptr_t>(ptr) % alignment == 0,
                   "CPU alignment contract");
        kxc::DeviceFree(cpu, ptr);
    }
    TEST_CHECK(Throws([&] { kxc::DeviceAlloc(cpu, 8, 3); }),
               "non-power-of-two alignment must fail");

    int delete_count = 0;
    void* external = std::malloc(32);
    {
        kxc::Storage storage = kxc::Storage::FromExternal(
            cpu, external, 32,
            [](void* data, void* context) {
                ++*static_cast<int*>(context);
                std::free(data);
            },
            &delete_count);
        kxc::Storage alias = storage;
        storage.ValidateRange(24, 8);
        TEST_CHECK(Throws([&] { storage.ValidateRange(25, 8); }),
                   "Storage range must be bounded");
    }
    TEST_CHECK(delete_count == 1, "external deleter must run exactly once");
    return true;
}

// 验证 CPU offset/重叠复制以及同步完成的异步句柄语义。
bool TestCPUCopyAndAsync() {
    const kxc::Device cpu = kxc::Device::CPU();
    kxc::Storage source = kxc::Storage::Alloc(cpu, 32, 16);
    kxc::Storage destination = kxc::Storage::Alloc(cpu, 32, 16);
    for (int i = 0; i < 32; ++i) static_cast<uint8_t*>(source.data())[i] = i;
    kxc::StorageCopySync(source, 4, destination, 8, 16);
    TEST_CHECK(std::memcmp(static_cast<uint8_t*>(source.data()) + 4,
                           static_cast<uint8_t*>(destination.data()) + 8, 16) == 0,
               "offset copy must preserve bytes");
    std::vector<uint8_t> original(32);
    std::memcpy(original.data(), source.data(), original.size());
    kxc::StorageCopySync(source, 0, source, 4, 16);
    TEST_CHECK(std::memcmp(static_cast<uint8_t*>(source.data()) + 4,
                           original.data(), 16) == 0,
               "overlapping CPU copy must use memmove semantics");

    kxc::DeviceStream stream = kxc::DeviceStream::Create(cpu);
    TEST_CHECK(stream.device() == cpu && stream.is_default() == false,
               "created CPU stream identity");
    kxc::AsyncOperation operation =
        kxc::StorageCopyAsync(source, 0, destination, 0, 32, stream);
    TEST_CHECK(operation.IsReady(), "CPU async operation completes inline");
    operation.Wait();
    operation.Wait();
    return true;
}

// 验证 NDArray 分配、清零、复制、视图、零尺寸和溢出门禁。
bool TestNDArray() {
    static_assert(!std::is_constructible_v<kxc::runtime::NDArray,
                                            kxc::Array<int64_t>, std::string>);
    const kxc::Device cpu = kxc::Device::CPU();
    const DLDataType f32 = kxc::runtime::DataTypeFromString("float32");
    kxc::runtime::NDArray array =
        kxc::runtime::NDArray::Zeros({2, 3}, f32, cpu);
    TEST_CHECK(array.NBytes() == 24, "NBytes");
    TEST_CHECK(array.IsContiguous(), "new arrays are contiguous");
    TEST_CHECK(array.device() == cpu, "array device");
    std::vector<float> zeros(6, 1.0f);
    array.CopyToBytes(zeros.data(), array.NBytes());
    for (float value : zeros) TEST_CHECK(value == 0.0f, "Zeros must initialize");

    const std::vector<float> values{1, 2, 3, 4, 5, 6};
    array.CopyFromBytes(values.data(), array.NBytes());
    std::vector<float> round_trip(6);
    array.CopyToBytes(round_trip.data(), array.NBytes());
    TEST_CHECK(round_trip == values, "byte round trip");

    kxc::runtime::NDArray copy = array.CopyTo(cpu);
    const std::vector<float> replacement(6, 9.0f);
    copy.CopyFromBytes(replacement.data(), copy.NBytes());
    array.CopyToBytes(round_trip.data(), array.NBytes());
    TEST_CHECK(round_trip == values, "CopyTo must allocate independent storage");

    kxc::runtime::NDArray scalar =
        kxc::runtime::NDArray::Empty({}, f32, cpu);
    TEST_CHECK(scalar.NBytes() == sizeof(float), "scalar is one element");
    kxc::runtime::NDArray empty =
        kxc::runtime::NDArray::Empty({2, 0, 3}, f32, cpu);
    TEST_CHECK(empty.NBytes() == 0 && empty.storage().data() == nullptr,
               "zero dimension has zero storage");
    empty.CopyFromBytes(nullptr, 0);
    empty.CopyToBytes(nullptr, 0);

    kxc::runtime::NDArray view = array.CreateView({3, 2}, {2, 1}, 0);
    TEST_CHECK(view.storage().get() == array.storage().get(), "view shares storage");
    TEST_CHECK(Throws([&] { array.CreateView({2, 2}, {3, 1}, 0); }),
               "non-contiguous view must fail");
    TEST_CHECK(Throws([&] { array.CreateView({8}, {1}, 0); }),
               "out-of-capacity view must fail");
    TEST_CHECK(Throws([&] { array.CreateView({2, 3}, {-3, 1}, 0); }),
               "negative strides must fail");
    TEST_CHECK(Throws([&] { kxc::runtime::NDArray::Empty({-1}, f32, cpu); }),
               "negative shapes must fail");
    TEST_CHECK(Throws([&] { kxc::runtime::NDArray::Empty({0, -1}, f32, cpu); }),
               "negative dimensions after zero must fail");
    TEST_CHECK(Throws([&] {
                   kxc::runtime::NDArray::Empty(
                       {std::numeric_limits<int64_t>::max(), 2}, f32, cpu);
               }),
               "shape byte overflow must fail");
    TEST_CHECK(Throws([&] {
                   kxc::runtime::NDArray::Empty({1}, DLDataType{kDLFloat, 1, 1}, cpu);
               }),
               "sub-byte dtype must fail");
    TEST_CHECK(Throws([&] {
                   kxc::runtime::NDArray::Empty({1}, DLDataType{kDLFloat, 32, 0}, cpu);
               }),
               "zero dtype lanes must fail");
    return true;
}

// 验证 CPU-only 明确失败或真实 CUDA 环境中的同步、异步往返复制。
bool TestCUDAPath() {
    const kxc::Device cuda = kxc::Device::CUDA();
#if !KXC_USE_CUDA
    TEST_CHECK(Throws([&] { kxc::Storage::Alloc(cuda, 16); }),
               "CPU-only build must reject CUDA allocation");
    TEST_CHECK(Throws([&] { kxc::DeviceStream::Create(cuda); }),
               "CPU-only build must reject CUDA stream creation");
    return true;
#else
    if (!kxc::CollectDeviceAttributes(cuda).exists) {
        std::cout << "SKIPPED: no CUDA device detected\n";
        return true;
    }
    const DLDataType f32 = kxc::runtime::DataTypeFromString("float32");
    std::vector<float> values(1024);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
    kxc::runtime::NDArray cpu = kxc::runtime::NDArray::Empty(
        {static_cast<int64_t>(values.size())}, f32, kxc::Device::CPU());
    cpu.CopyFromBytes(values.data(), cpu.NBytes());

    kxc::runtime::NDArray cuda_array = cpu.CopyTo(cuda);
    kxc::runtime::NDArray round_trip = cuda_array.CopyTo(kxc::Device::CPU());
    std::vector<float> actual(values.size());
    round_trip.CopyToBytes(actual.data(), round_trip.NBytes());
    TEST_CHECK(actual == values, "CUDA synchronous round trip");

    kxc::DeviceStream stream = kxc::DeviceStream::Create(cuda);
    kxc::runtime::NDArray async_cuda = kxc::runtime::NDArray::Empty(
        cpu.shape(), cpu.dtype(), cuda);
    kxc::AsyncOperation h2d = async_cuda.CopyFromAsync(cpu, stream);
    // 主动释放源 NDArray，验证 AsyncOperation 会保活 DMA 依赖的 Storage。
    cpu = kxc::runtime::NDArray();
    h2d.Wait();
    TEST_CHECK(h2d.IsReady(), "CUDA event must be ready after Wait");
    round_trip.CopyFrom(async_cuda);
    round_trip.CopyToBytes(actual.data(), round_trip.NBytes());
    TEST_CHECK(actual == values, "async operation retains source storage");
    return true;
#endif
}

}  // namespace

// 运行 Device 1-7 的基础契约矩阵，并将首个失败转换为退出码。
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"device_identity", TestDeviceIdentity},
        {"cpu_allocation_storage", TestCPUAllocationAndStorage},
        {"cpu_copy_async", TestCPUCopyAndAsync},
        {"ndarray", TestNDArray},
        {"cuda_path", TestCUDAPath},
    };
    for (const auto& test : tests) {
        try {
            if (!test.second()) return 1;
        } catch (const std::exception& error) {
            std::cerr << "[EXCEPTION] " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "All device runtime tests passed.\n";
    return 0;
}
