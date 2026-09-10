# Windows GPU 接入与 CUDA/CUPTI 验证报告

2026-09-09。已通过 Tailscale 和 SSH 使用 Windows 的 **RTX 4070 Ti SUPER**，在独立目录完成原生编译，五项专项测试全部通过。实际 NVRTC 编译、模块加载、kernel 数值、异步资源保活和 CUPTI 设备活动都有执行证据。补充的 Compute Sanitizer 检查未能进入插桩，不能算内存检查通过。

## 环境与方法

| 项目 | 实测值 |
|---|---|
| 主机与系统 | BF-202408261826，Windows 11 Pro，10.0.26200 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER，16376 MiB |
| 驱动 | 576.57 |
| CUDA | 显式选择 12.9 Update 1，nvcc 12.9.86，NVRTC 来自同一 SDK |
| CUDA Driver API | `cuInit(0)=0`，版本 12090，设备数 1 |
| 构建工具 | CMake 4.3.3；VS 2019、MSVC 19.29.30156；x64 Release |
| 验证目录 | `F:\kxc-gpu\20260909-joint` |
| 编译选项 | `KXC_ENABLE_CUDA=ON`、`KXC_ENABLE_LLVM=OFF` |

机器还安装了 CUDA 13.1，默认 PATH 会找到它。此次在构建和测试进程中固定 `CUDAToolkit_ROOT`、`CUDA_PATH` 和 PATH 为 12.9：576.57 正好满足 [CUDA 12.9 Update 1 的 Windows 驱动要求](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)。没有通过更新显示驱动来满足另一套 SDK。

已注册的 Ubuntu WSL 缺少其配置路径下的 `ext4.vhdx`，启动返回 `ERROR_FILE_NOT_FOUND`。本轮采用现成的 Windows 原生编译器；工作目录放在空间充足的 F 盘。Windows 未发现 LLVM 开发包，LLVM 数值回归由原有 Linux 环境执行。

SSH 初次失败的原因也已定位：该机器有效的 `AuthorizedKeysFile` 是用户目录 `.ssh/authorized_keys`，初始脚本写入的管理员公共文件没有被这份配置使用。修复脚本保留原有公钥，备份文件与权限，修复用户公钥文件的编码、所有者和 ACL 后认证成功。客户端使用独立密钥和经 Taildrop 回执核对、固定的主机公钥；详情见 [环境修复报告](GPU_DRIVER_REPAIR_REPORT.md)。

## 源码与两处修复

上传的是包含当时未提交工作的源码快照，含 682 份文件。压缩包 `out/windows-gpu/kxc-gpu-source-joint-20260909.tar.gz` 的 SHA256 为 `258f88dc906e44d5e0b9cb7e2c5f776cf0a7c7aae5c736f62517e725a14f14bd`。解包前验证整包，解包后按清单验证全部文件，差异为 0。

第一次专项运行得到 4/5：kernel 数值和运行时通过，但 CUPTI manifest 报不可用。定位并修复了两层问题：

1. [CMakeLists.txt](../../CMakeLists.txt) 没有给 profiling 模块提供 CUPTI 的独立 include 路径。Windows SDK 把头文件放在 `extras/CUPTI/include`，原来的 `__has_include` 因而关闭了采集实现。现在复用 CMake 已发现的 `CUDA::cupti` 的 include 属性，仅作用于 profiling 模块。库仍由现有 adapter 动态加载；最终可执行文件的导入表没有 CUPTI DLL。此方法沿用 [FindCUDAToolkit 的导入目标](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#cupti)。
2. 接通头文件后，编译暴露出 [profiling.cc](../../src/profiling/profiling.cc) 固定使用 `CUpti_ActivityKernel10`。根据 [NVIDIA 的 CUDA 13.0 变更说明](https://docs.nvidia.com/cupti/release-notes/release-notes.html#updates-in-cuda-13-0)，Kernel10 在该版本引入；12.9 使用 [Kernel9](https://docs.nvidia.com/cupti/12.9.1/api/structCUpti__ActivityKernel9.html)。现在根据 `CUPTI_API_VERSION` 选择对应类型，复用相同的事件写出代码。

没有增加新的 IR、采集器或运行时状态系统。两处改动只恢复已有可选 CUPTI 路径，不改变 kernel ABI、调度或编译产物 identity。已有硬件测试直接复现了修改前的失败，并验证修改后的真实消费者。

最终再次核验 682 份远端源码，只有下面两份采用修复后的哈希，其余与原包完全一致：

| 文件 | 修复后 SHA256 |
|---|---|
| `CMakeLists.txt` | `235191de14d368c190ff92e74fe17a359f353ecad3ab141f460477f0c3ea1cc4` |
| `src/profiling/profiling.cc` | `d13510b66b88662828613329e1cb717fca9437705b32075367421428ff2884b8` |

## 验证结果与效果

| 检查 | 结果与实际覆盖 |
|---|---|
| `device_runtime_test` | 通过；真实 CPU/GPU 同步往返、异步 H2D、源 Storage 保活，设备路径未跳过 |
| `codegen_cuda_test` | 13 个子项通过；含 NVRTC 错误日志、符号负例、异步 CUmodule 保活、批量模块、byte offset，以及经 Compiler/RuntimeSession 执行的 add、constant、relu、非空一维 Where/Slice/Concatenate 数值 |
| `cuda_schedule_test` | 通过；synthetic Target 的调度、launch authority、归约/间接加载等拒绝合同，属于结构验证 |
| `runtime_profiling_test` | 通过；通用执行观测合同；其中 LLVM Where 子项按 `KXC_USE_LLVM=0` 跳过，不计 Windows LLVM 数值证据 |
| `cupti_smoke_test` | 通过；实际 `scale_add` 输出全部为 8，manifest 标记 CUPTI 可用，五类 CUDA 活动均存在 |
| Windows 专项合计 | **5/5，1.19 秒**；时间是本次测试总耗时，不是模型性能指标 |
| Linux CPU/LLVM 完整回归 | **55/55，146.33 秒**；实际设置完整视觉、三组图文联合、原有 L1 prefill/state/decode fixture |
| Linux CUDA 12.9 | runtime archive、CUDA codegen 与 CUPTI 测试编译/链接通过；本机驱动仍未重启恢复，不作为设备执行证据 |
| CUDA 13.1 类型分支 | Windows 独立构建 `kxc_profiling_obj` 通过；只验证编译，没有在现有驱动上运行 CUDA 13.1 |
| 强定义审计 | Linux CPU/CUDA archive 分别 1603/1614 个强定义，重复为 0 |
| 集成检查 | Relay/Pass 生成物 freshness、NLP gate、284 文件 include 层级、64 份文档链接和 `git diff --check` 通过；公共头文件独立编译随 CPU 全量回归通过 |

CUPTI bundle 有 33 条事件：20 条 runtime API、7 条 driver API、3 条 memcpy、1 条 memset、1 条 kernel 和 1 条测试根 span。五条设备活动全部关联到 `cupti-1` 及其根 span，设备为 `cuda:0`，持续时间为正；拷贝与清零均为 256 字节，kernel 的 grid 为 `[1,1,1]`、block 为 `[128,1,1]`。没有错误、警告或 dropped-record 事件。

这一结果补上了真实设备执行和设备活动采集的证据。它没有证明归约、Gather、完整 Transformer、MiniMind-V 或设备端 KV/bounded 执行已经支持。NLP 能力矩阵仍按各行的完整 gate 判定，不能由这组非空一维局部测试整体解锁。

## 内存检查的未完成项

CUDA 12.9 的 Compute Sanitizer 2025.2.1 在 `codegen_cuda_test --memcheck` 和 `device_runtime_test` 上均报告：`Target application terminated before first instrumented API call`。独立的最小原生程序只调用 `cudaGetDeviceCount`，直接运行返回成功、设备数 1；经同一工具运行时进入了 `main`，随后复现相同失败。原始 KXC 程序直接运行也仍通过。

已确认 [NVIDIA 要求的 Windows 调试接口注册表项](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior) `GPUDebugger\EnableInterface` 已为 1。当前尚未定位插桩失败的最终原因，没有得到零错误或零泄漏报告。该工具问题不应写成 kernel 数值失败，也不能把直接运行通过写成 memcheck 通过。

本次查询 NVIDIA 技能目录首页成功，分页 2 返回 HTTP 202/WAF challenge 空正文，目录读取未完成。因此接口兼容修复以 SDK 头文件和上述官方文档为依据，没有安装外部技能。

## 复现与回执

以下命令在 Windows PowerShell 中运行，`$root` 指向本次已核验的源码和构建目录：

```powershell
$root = 'F:\kxc-gpu\20260909-joint'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
cmake -S "$root\source" -B "$root\build" -G 'Visual Studio 16 2019' -A x64 `
  -DKXC_ENABLE_CUDA=ON -DKXC_ENABLE_LLVM=OFF "-DCUDAToolkit_ROOT=$env:CUDA_PATH" `
  -DPython3_EXECUTABLE=D:/Anaconda/python.exe
cmake --build "$root\build" --config Release --parallel 6 --target `
  codegen_cuda_test device_runtime_test cuda_schedule_test runtime_profiling_test cupti_smoke_test
ctest --test-dir "$root\build" -C Release -V --no-tests=error -j1 `
  -R '^(codegen_cuda_test|device_runtime_test|cuda_schedule_test|runtime_profiling_test|cupti_smoke_test)$'
```

SSH 客户端配置在 Linux 的 `/home/oops/.ssh/config_kxc_windows_gpu`，别名为 `kxc-windows-gpu`；本地 `out/windows-gpu/remote_ps.py` 可通过固定的主机公钥执行 PowerShell。私钥仅保存在 Linux。

远端 `logs/` 保留初次 4/5 失败、头文件接线后的类型编译失败、最终 5/5、源码校验、DLL 导入表、二进制哈希及 sanitizer 最小复现。回传记录位于本地 `out/windows-gpu/logs/`，CUPTI 原始 bundle 在 `out/windows-gpu/cupti_smoke_test_output/`，审计摘要为 `out/windows-gpu/cupti-bundle-audit.json`。最终 `events.jsonl` 的 SHA256 是 `1fc33a59027b8b7bdf6ff6436779cb3d7788b361a51d8d876d0e38a0403d53eb`。

本地 CPU/LLVM 构建和全量回归日志为 `/tmp/kxc-windows-gpu-default-{build,ctest}.log`，Linux CUDA 编译日志为 `/tmp/kxc-windows-gpu-linux-cuda-build.log`。这些回执记录本次运行；后续复现仍需固定工具版本、源码及 fixture。
