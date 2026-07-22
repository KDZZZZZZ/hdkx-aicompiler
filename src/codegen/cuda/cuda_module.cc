/*! \file src/codegen/cuda/cuda_module.cc
 * \brief 实现 NVRTC 编译、CUDA Driver module RAII 和 NDArray 异步内核启动。
 */

#include "internal/cuda_module.h"

#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include "kxc/runtime/device_api.h"

namespace kxc::codegen {
namespace {

/*! \brief 把 NVRTC 错误码转换为带操作名的异常。 */
void CheckNVRTC(nvrtcResult result, const char* operation) {
    if (result != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 nvrtcGetErrorString(result));
    }
}

/*! \brief 把 CUDA Driver 错误码转换为同时包含名称和描述的异常。 */
void CheckDriver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* message = nullptr;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : "CUDA_ERROR_UNKNOWN") +
                             (message ? std::string(" (") + message + ")" : ""));
}

/*! \brief 在 Driver API 操作前选择 module 所属设备并建立对应 primary context。 */
void SelectDevice(const Device& device) {
    if (!device.defined() || device.device_type() != kCUDA) {
        throw std::invalid_argument("CUDA module requires a cuda:N device");
    }
    GetDeviceAPI(kCUDA)->SetDevice(device);
    // Runtime API 的 cudaSetDevice 会令该设备 primary context 成为当前上下文；
    // 显式检查可避免 Driver module 被意外加载到调用线程遗留的其他 context。
    CUcontext context = nullptr;
    CheckDriver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (context == nullptr) {
        throw std::runtime_error("CUDA device selection did not establish a current context");
    }
}

/*! \brief nvrtcProgram 的局部 RAII 包装，保证所有异常路径都销毁编译器句柄。 */
class NVRTCProgram final {
public:
    NVRTCProgram(const std::string& source, const std::string& source_name) {
        CheckNVRTC(nvrtcCreateProgram(&program_, source.c_str(), source_name.c_str(),
                                      0, nullptr, nullptr),
                   "nvrtcCreateProgram");
    }

    ~NVRTCProgram() {
        if (program_ != nullptr) (void)nvrtcDestroyProgram(&program_);
    }

    NVRTCProgram(const NVRTCProgram&) = delete;
    NVRTCProgram& operator=(const NVRTCProgram&) = delete;

    /*! \brief 返回仅在当前对象生命周期内有效的 NVRTC 原生句柄。 */
    nvrtcProgram get() const noexcept { return program_; }

    /*! \brief 提取完整编译日志；NVRTC 的结尾 NUL 不进入返回字符串。 */
    std::string Log() const {
        size_t size = 0;
        CheckNVRTC(nvrtcGetProgramLogSize(program_, &size),
                   "nvrtcGetProgramLogSize");
        if (size == 0) return {};
        std::string log(size, '\0');
        CheckNVRTC(nvrtcGetProgramLog(program_, log.data()),
                   "nvrtcGetProgramLog");
        if (!log.empty() && log.back() == '\0') log.pop_back();
        return log;
    }

private:
    nvrtcProgram program_{nullptr};
};

/*!
 * \brief 独占 CUmodule，并实现 CompiledKernel 的 CUDA 启动协议。
 *
 * CUfunction 的有效期从属于 CUmodule，因此二者由同一个不可复制 launcher 保存。
 */
class CudaModuleLauncher final : public KernelLauncher {
public:
    CudaModuleLauncher(Device device, CUmodule module, CUfunction function,
                       KernelLaunchMetadata metadata, size_t argument_count)
        : device_(std::move(device)),
          module_(module),
          function_(function),
          metadata_(std::move(metadata)),
          argument_count_(argument_count) {}

    ~CudaModuleLauncher() override {
        if (module_ == nullptr) return;
        try {
            SelectDevice(device_);
            CheckDriver(cuModuleUnload(module_), "cuModuleUnload");
        } catch (const std::exception& error) {
            // 析构边界不能抛异常；驱动回收失败仅保留可诊断信息。
            std::cerr << "CUDA module release failed for " << device_.ToString()
                      << ": " << error.what() << '\n';
        }
    }

    CudaModuleLauncher(const CudaModuleLauncher&) = delete;
    CudaModuleLauncher& operator=(const CudaModuleLauncher&) = delete;

    /*! \brief module 和入口函数均成功解析后 launcher 才视为可启动。 */
    bool IsReady() const noexcept override {
        return module_ != nullptr && function_ != nullptr;
    }

    /*! \brief 按 CUDA Driver 参数 ABI 提交内核并返回拥有完成 event 的操作。 */
    AsyncOperation Launch(const Array<runtime::NDArray>& arguments,
                          const DeviceStream& stream,
                          const ObjectRef& executable_owner) const override {
        if (!IsReady()) throw std::runtime_error("CUDA module is not ready");
        if (arguments.size() != argument_count_) {
            throw std::invalid_argument("CUDA launch argument count mismatch");
        }
        if (stream.device() != device_) {
            throw std::invalid_argument("CUDA launch stream device mismatch");
        }
        if (!executable_owner.defined()) {
            throw std::invalid_argument("CUDA launch requires executable ownership");
        }

        SelectDevice(device_);
        const auto* launch = metadata_.operator->();
        if (launch->dynamic_shared_memory_bytes >
            static_cast<uint64_t>(std::numeric_limits<unsigned int>::max())) {
            throw std::overflow_error("CUDA dynamic shared memory exceeds Driver ABI range");
        }

        Array<Storage> retained;
        std::vector<void*> pointer_values;
        std::vector<void*> kernel_params;
        pointer_values.reserve(arguments.size());
        kernel_params.reserve(arguments.size());
        for (const auto& argument : arguments) {
            Storage storage = argument.storage();
            retained.push_back(storage);
            // NDArray 可以是 Storage 的偏移视图；内核 ABI 接收逻辑首元素地址，而不是
            // allocation 基址。公共 CompiledModule 已校验连续性和范围。
            auto* base = static_cast<unsigned char*>(storage.data());
            if (base == nullptr) {
                if (argument->byte_offset != 0) {
                    throw std::invalid_argument(
                        "CUDA argument has an offset from null Storage");
                }
                pointer_values.push_back(nullptr);
            } else {
                pointer_values.push_back(base + argument->byte_offset);
            }
        }
        for (void*& value : pointer_values) kernel_params.push_back(&value);

        DeviceAPI* api = GetDeviceAPI(kCUDA);
        void* event = api->CreateEvent(device_);
        AsyncOperation operation = AsyncOperation::Pending(
            stream, event, std::move(retained), executable_owner);
        try {
            CheckDriver(
                cuLaunchKernel(function_, launch->grid.x, launch->grid.y,
                               launch->grid.z, launch->block.x, launch->block.y,
                               launch->block.z,
                               static_cast<unsigned int>(
                                   launch->dynamic_shared_memory_bytes),
                               reinterpret_cast<CUstream>(stream->backend_handle),
                               kernel_params.data(), nullptr),
                "cuLaunchKernel");
            // cuLaunchKernel 返回前已把 kernelParams 指向的参数值复制进 launch command，
            // 因而两个宿主 vector 只需活到本次调用返回；设备内存和 module 则由 operation 保活。
            api->RecordEvent(device_, event, stream->backend_handle);
        } catch (...) {
            std::exception_ptr original = std::current_exception();
            try {
                stream.Sync();
            } catch (...) {
                // 无法证明已提交工作结束时，故意把操作保留到进程结束，禁止释放
                // event、Storage 或 CUmodule 后形成异步 use-after-free。
                try {
                    (void)new AsyncOperation(operation);
                } catch (...) {
                    std::terminate();
                }
                std::rethrow_exception(original);
            }
            try {
                operation.Wait();
            } catch (...) {
                try {
                    (void)new AsyncOperation(operation);
                } catch (...) {
                    std::terminate();
                }
            }
            std::rethrow_exception(original);
        }
        return operation;
    }

private:
    Device device_;
    CUmodule module_{nullptr};
    CUfunction function_{nullptr};
    KernelLaunchMetadata metadata_;
    size_t argument_count_{0};
};

}  // namespace

std::string CUDAModule::CompileToPTX(const std::string& source,
                                     const CUDACompileOptions& options) {
    if (source.empty()) throw std::invalid_argument("CUDA source must not be empty");
    if (options.architecture.empty()) {
        throw std::invalid_argument("CUDA NVRTC architecture must not be empty");
    }
    if (options.source_name.empty()) {
        throw std::invalid_argument("CUDA source name must not be empty");
    }

    NVRTCProgram program(source, options.source_name);
    std::vector<std::string> option_storage;
    option_storage.reserve(options.additional_options.size() + 2);
    option_storage.push_back("--std=c++17");
    option_storage.push_back("--gpu-architecture=" + options.architecture);
    for (const auto& option : options.additional_options) {
        if (option.empty()) {
            throw std::invalid_argument("CUDA NVRTC option must not be empty");
        }
        option_storage.push_back(option);
    }
    std::vector<const char*> option_pointers;
    option_pointers.reserve(option_storage.size());
    for (const auto& option : option_storage) option_pointers.push_back(option.c_str());

    const nvrtcResult compile_result = nvrtcCompileProgram(
        program.get(), static_cast<int>(option_pointers.size()),
        option_pointers.data());
    if (compile_result != NVRTC_SUCCESS) {
        std::string message = std::string("nvrtcCompileProgram: ") +
                              nvrtcGetErrorString(compile_result);
        const std::string log = program.Log();
        if (!log.empty()) message += "\n" + log;
        throw std::runtime_error(message);
    }

    size_t ptx_size = 0;
    CheckNVRTC(nvrtcGetPTXSize(program.get(), &ptx_size), "nvrtcGetPTXSize");
    if (ptx_size == 0) throw std::runtime_error("NVRTC produced empty PTX");
    std::string ptx(ptx_size, '\0');
    CheckNVRTC(nvrtcGetPTX(program.get(), ptx.data()), "nvrtcGetPTX");
    return ptx;
}

CompiledKernel CUDAModule::Compile(const std::string& source,
                                   KernelSignature signature,
                                   KernelLaunchMetadata metadata,
                                   const CUDACompileOptions& options) {
    signature.Validate();
    metadata.Validate();
    const auto* launch = metadata.operator->();
    if (launch->backend != CodeGenBackend::kCUDA ||
        launch->device.device_type() != kCUDA) {
        throw std::invalid_argument("CUDAModule requires CUDA launch metadata");
    }

    const std::string ptx = CompileToPTX(source, options);
    SelectDevice(launch->device);
    CUmodule module = nullptr;
    CheckDriver(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr),
                "cuModuleLoadDataEx");
    try {
        CUfunction function = nullptr;
        const std::string symbol = signature.operator->()->symbol;
        CheckDriver(cuModuleGetFunction(&function, module, symbol.c_str()),
                    "cuModuleGetFunction");
        auto launcher = std::make_shared<CudaModuleLauncher>(
            launch->device, module, function, metadata,
            signature.arguments().size());
        return CompiledKernel(std::move(signature), std::move(metadata),
                              std::move(launcher));
    } catch (...) {
        std::exception_ptr original = std::current_exception();
        try {
            CheckDriver(cuModuleUnload(module), "cuModuleUnload");
        } catch (...) {
        }
        std::rethrow_exception(original);
    }
}

}  // namespace kxc::codegen
