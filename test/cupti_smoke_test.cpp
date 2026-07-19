/*! \file test/cupti_smoke_test.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "base/device.h"
#include "base/device_api.h"
#include "base/profiling.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if KXC_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif

// 在真实 CUDA 设备上执行内存和 NVRTC 内核操作，并验证 CUPTI 事件输出。
int main() {
#if !KXC_USE_CUDA
    std::cout << "SKIPPED: KXC_USE_CUDA not enabled\n";
    return 0;
#else
    namespace fs = std::filesystem;
    using namespace kxc;

    const fs::path bundle_dir = fs::current_path() / "cupti_smoke_test_output";
    std::error_code ec;
    fs::remove_all(bundle_dir, ec);

    const auto fail = [](const std::string& message) {
        std::cerr << message << "\n";
        return 1;
    };

    const auto check_cuda = [&](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            return fail(std::string(what) + ": " + cudaGetErrorString(err));
        }
        return 0;
    };

    const auto check_driver = [&](CUresult err, const char* what) {
        if (err != CUDA_SUCCESS) {
            const char* name = nullptr;
            const char* text = nullptr;
            cuGetErrorName(err, &name);
            cuGetErrorString(err, &text);
            return fail(std::string(what) + ": " + (name ? name : "CUDA_ERROR") + " / " +
                        (text ? text : "unknown"));
        }
        return 0;
    };

    const auto check_nvrtc = [&](nvrtcResult err, const char* what, nvrtcProgram prog) {
        if (err != NVRTC_SUCCESS) {
            std::string log;
            if (prog != nullptr) {
                size_t log_size = 0;
                nvrtcGetProgramLogSize(prog, &log_size);
                log.assign(log_size, '\0');
                if (log_size > 1) {
                    nvrtcGetProgramLog(prog, log.data());
                }
            }
            return fail(std::string(what) + ": " + nvrtcGetErrorString(err) +
                        (!log.empty() ? "\n" + log : ""));
        }
        return 0;
    };

    // 使用强类型 Device 身份统一 CUPTI、CUDA Runtime 和 DeviceAPI 的设备归属。
    Device gpu = Device::CUDA();
    Device cpu = Device::CPU();
    DeviceAPI* api = GetDeviceAPI(kCUDA);
    DeviceAttributes attrs = api->GetDeviceAttributes(gpu);
    if (!attrs.exists) {
        std::cout << "SKIPPED: no CUDA device detected\n";
        return 0;
    }

    profiling::ProfileOptions options;
    options.enabled = true;
    options.enable_cupti = true;
    options.bundle_dir = bundle_dir.string();

    auto ctx = profiling::ProfileContext::Create(options);
    const std::string run_id = ctx->NextRunId("cupti");
    profiling::ActivationScope activation(ctx, run_id);

    {
        profiling::EventSpec root_spec;
        root_spec.component = "test";
        root_spec.event_type = "cupti_smoke";
        profiling::ScopedSpan root_span(ctx, std::move(root_spec), run_id);

        std::vector<float> host_in(64, 3.5f);
        std::vector<float> host_out(64, -1.0f);
        const size_t nbytes = host_in.size() * sizeof(float);

        api->SetDevice(gpu);
        void* gpu_buf = api->AllocDataSpace(gpu, nbytes, 64);
        void* gpu_out = api->AllocDataSpace(gpu, nbytes, 64);
        StreamHandle stream = api->CreateStream(gpu);

        api->CopyDataSync(cpu, host_in.data(), 0, gpu, gpu_buf, 0, nbytes);
        cudaError_t err =
            cudaMemsetAsync(gpu_buf, 0, nbytes, reinterpret_cast<cudaStream_t>(stream));
        if (check_cuda(err, "cudaMemsetAsync") != 0) {
            return 1;
        }

        err = cudaMemcpyAsync(gpu_buf, host_in.data(), nbytes, cudaMemcpyHostToDevice,
                              reinterpret_cast<cudaStream_t>(stream));
        if (check_cuda(err, "cudaMemcpyAsync H2D") != 0) {
            return 1;
        }

        if (check_driver(cuInit(0), "cuInit") != 0) {
            return 1;
        }
        CUcontext cu_ctx = nullptr;
        if (check_driver(cuCtxGetCurrent(&cu_ctx), "cuCtxGetCurrent") != 0) {
            return 1;
        }
        if (cu_ctx == nullptr) {
            return fail("cuCtxGetCurrent returned null context");
        }

        std::string gpu_arch = "--gpu-architecture=compute_" +
                               std::to_string(attrs.compute_version_major) +
                               std::to_string(attrs.compute_version_minor);
        std::string include_dir = "--include-path=/usr/local/cuda/targets/sbsa-linux/include";

        const char* source = R"(
extern "C" __global__ void scale_add(const float* x, float* y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * alpha + 1.0f;
  }
}
)";

        nvrtcProgram program{};
        if (check_nvrtc(nvrtcCreateProgram(&program, source, "scale_add.cu", 0, nullptr, nullptr),
                        "nvrtcCreateProgram", nullptr) != 0) {
            return 1;
        }
        std::vector<const char*> options_vec = {
            gpu_arch.c_str(),
            "--device-as-default-execution-space",
            "--fmad=false",
            "--std=c++14",
            include_dir.c_str(),
        };
        if (check_nvrtc(nvrtcCompileProgram(program, static_cast<int>(options_vec.size()),
                                            options_vec.data()),
                        "nvrtcCompileProgram", program) != 0) {
            nvrtcDestroyProgram(&program);
            return 1;
        }

        size_t ptx_size = 0;
        if (check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize", program) != 0) {
            nvrtcDestroyProgram(&program);
            return 1;
        }
        std::string ptx(ptx_size, '\0');
        if (check_nvrtc(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX", program) != 0) {
            nvrtcDestroyProgram(&program);
            return 1;
        }
        nvrtcDestroyProgram(&program);

        CUmodule module = nullptr;
        if (check_driver(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr),
                         "cuModuleLoadDataEx") != 0) {
            return 1;
        }
        CUfunction kernel = nullptr;
        if (check_driver(cuModuleGetFunction(&kernel, module, "scale_add"),
                         "cuModuleGetFunction") != 0) {
            cuModuleUnload(module);
            return 1;
        }

        float alpha = 2.0f;
        int n = static_cast<int>(host_in.size());
        void* kernel_args[] = {&gpu_buf, &gpu_out, &alpha, &n};
        const unsigned int block_x = 128;
        const unsigned int grid_x = static_cast<unsigned int>((n + block_x - 1) / block_x);
        if (check_driver(cuLaunchKernel(kernel, grid_x, 1, 1, block_x, 1, 1, 0,
                                        reinterpret_cast<CUstream>(stream), kernel_args, nullptr),
                         "cuLaunchKernel") != 0) {
            cuModuleUnload(module);
            return 1;
        }

        err = cudaMemcpyAsync(host_out.data(), gpu_out, nbytes, cudaMemcpyDeviceToHost,
                              reinterpret_cast<cudaStream_t>(stream));
        if (check_cuda(err, "cudaMemcpyAsync D2H") != 0) {
            cuModuleUnload(module);
            return 1;
        }

        api->StreamSync(gpu, stream);
        cuModuleUnload(module);

        for (float value : host_out) {
            if (value != 8.0f) {
                return fail("Kernel output verification failed");
            }
        }

        api->FreeStream(gpu, stream);
        api->FreeDataSpace(gpu, gpu_out);
        api->FreeDataSpace(gpu, gpu_buf);
    }

    ctx->Flush();

    const fs::path manifest = bundle_dir / "manifest.json";
    const fs::path events = bundle_dir / "events.jsonl";
    if (!fs::exists(manifest) || !fs::exists(events)) {
        std::cerr << "Missing expected bundle files\n";
        return 1;
    }

    std::ifstream manifest_ifs(manifest);
    std::string manifest_text((std::istreambuf_iterator<char>(manifest_ifs)),
                              std::istreambuf_iterator<char>());
    if (manifest_text.find("\"cupti_available\":true") == std::string::npos) {
        std::cerr << "Manifest did not mark CUPTI as available\n";
        return 1;
    }

    bool saw_runtime = false;
    bool saw_driver = false;
    bool saw_memcpy = false;
    bool saw_memset = false;
    bool saw_kernel = false;
    std::ifstream events_ifs(events);
    std::string line;
    while (std::getline(events_ifs, line)) {
        if (line.find("\"component\":\"backend.cuda\"") == std::string::npos) {
            continue;
        }
        saw_runtime = saw_runtime || line.find("\"event_type\":\"cuda_runtime_api\"") != std::string::npos;
        saw_driver = saw_driver || line.find("\"event_type\":\"cuda_driver_api\"") != std::string::npos;
        saw_memcpy = saw_memcpy || line.find("\"event_type\":\"cuda_memcpy\"") != std::string::npos;
        saw_memset = saw_memset || line.find("\"event_type\":\"cuda_memset\"") != std::string::npos;
        saw_kernel = saw_kernel || line.find("\"event_type\":\"cuda_kernel\"") != std::string::npos;
    }

    if (!saw_runtime || !saw_driver || !saw_memcpy || !saw_memset || !saw_kernel) {
        std::cerr << "Missing expected CUPTI activity events\n";
        return 1;
    }

    std::cout << "CUPTI bundle generated at: " << bundle_dir.string() << "\n";
    return 0;
#endif
}
