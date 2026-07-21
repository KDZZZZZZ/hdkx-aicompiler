# RuntimeSession 强类型执行层实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `CompiledModule` 之上实现只接收输入 NDArray、自动绑定常量和分配静态输出的同步/异步 RuntimeSession。

**Architecture:** RuntimeSession 只持有一个已经编译完成的 `CompiledModule`，不参与 Relay 编译、shape specialization 或 module cache。每次运行先依据 `KernelSignature` 校验输入，再按签名顺序组装 input、constant、output，最后调用唯一的 `CompiledModule::Launch`；异步完成仍由 `AsyncOperation` 保活 Storage 和后端 executable。

**Tech Stack:** C++17、项目 Object/ObjectRef 与 Array/Map 容器、NDArray/DeviceStream/AsyncOperation、LLVM ORC JIT、CUDA NVRTC/Driver API、CMake/Ninja。

---

### Task 1: 锁定 RuntimeSession 参数装配契约

**Files:**
- Create: `test/runtime_session_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 使用记录型 `KernelLauncher` 构造包含 input、constant、output 的 `CompiledModule` fixture。
2. 写入失败测试，覆盖 undefined/not-ready module、输入数量、dtype、shape、device 和 stream 不匹配。
3. 写入输出 shape/dtype/device 自动分配、常量对象身份和最终参数顺序断言。
4. 注册 `runtime_session_test` 与 `run_runtime_session_test` CMake target。
5. 构建测试，确认因 `runtime/runtime_session.h` 尚不存在而失败。

**Verification:**

```powershell
cmake --build out/build/dev-mingw-cpu --target runtime_session_test -j 4
```

Expected: compile failure names the missing RuntimeSession API.

### Task 2: 实现 RuntimeSession 对象模型和静态输出分配

**Files:**
- Create: `include/runtime/runtime_session.h`
- Create: `src/runtime/runtime_session.cc`
- Modify: `CMakeLists.txt`

**Steps:**

1. 定义 `RunAsyncResult`，只包含 `Array<NDArray> outputs` 和 `AsyncOperation completion`。
2. 定义 `RuntimeSessionNode`，只强持有一个 ready `api::CompiledModule`。
3. 定义 `RuntimeSession : ObjectRef`，提供构造器、ObjectRef 恢复、`Run` 和 `RunAsync`。
4. 在构造阶段验证 module、Signature、metadata 和 executable readiness。
5. 按 Signature 的 output specs 使用 `NDArray::Empty` 分配静态输出；发现动态输出维度时明确拒绝。
6. 全部新增结构体、类和函数使用中文 Doxygen 注释，关键装配语句解释 ABI 顺序来源。

**Verification:**

```powershell
cmake --build out/build/dev-mingw-cpu --target run_runtime_session_test -j 4
```

Expected: basic construction, allocation and argument order tests pass.

### Task 3: 完成输入验证、常量绑定和同步/异步语义

**Files:**
- Modify: `src/runtime/runtime_session.cc`
- Modify: `test/runtime_session_test.cpp`

**Steps:**

1. 在任何输出分配前统计 input specs 并校验输入数量。
2. 校验每个输入的定义状态、dtype、rank、静态维度、Device 和连续布局；动态输入维度接受任意非负实际值。
3. 通过 `constant_key` 从 `CompiledModule::constants()` 取回唯一绑定对象，不接受调用方常量参数。
4. 严格按 Signature 遍历组装 ordered arguments，不按类别另行猜测顺序。
5. `RunAsync` 校验显式 stream 后调用 `CompiledModule::Launch`，返回 outputs 与 completion。
6. `Run` 使用模块 Device 的默认 stream，等待 completion 后返回 outputs。
7. 验证调用方丢弃 inputs 后 completion 仍能保活异步依赖。

**Verification:**

```powershell
cmake --build out/build/dev-mingw-cpu --target run_runtime_session_test run_compiled_module_test -j 4
```

Expected: all RuntimeSession contract and existing CompiledModule tests pass.

### Task 4: 增加 LLVM RuntimeSession 数值闭环

**Files:**
- Modify: `test/codegen_llvm_test.cpp`

**Steps:**

1. 编译 Relay add，构造 RuntimeSession，只传两个 input NDArray。
2. 使用 `Run` 验证输出自动分配和数值正确。
3. 编译带 Relay Constant 的 add，验证 session 自动绑定 module constant。
4. 使用 `RunAsync` 和显式 CPU stream 验证 completion 与 outputs 返回契约。

**Verification:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target run_codegen_llvm_test -j 4
```

Expected: direct LLVM、Compiler 和 RuntimeSession numeric tests all pass.

### Task 5: 增加 CUDA RuntimeSession 数值和异步生命周期测试

**Files:**
- Modify: `test/codegen_cuda_test.cpp`

**Steps:**

1. 将 Compiler add/relu 测试改为通过 RuntimeSession 只传 inputs。
2. 验证 CPU Relay Constant 在 Compiler 放置到 CUDA 后由 session 自动绑定。
3. 使用拥有型 CUDA stream 调用 `RunAsync`，释放外部 module/session/input 引用后等待 completion。
4. 回读自动分配 output 并验证 add、constant add 和 relu 数值。
5. Compute Sanitizer 模式继续跳过预期 Driver missing-symbol 负例，其余 session 路径全部执行。

**Verification:**

```bash
cmake --build out/build/omen-cuda --target codegen_cuda_test -j 4
/usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  out/build/omen-cuda/codegen_cuda_test --memcheck
```

Expected: RuntimeSession CUDA tests pass and memcheck reports 0 errors.

### Task 6: 更新模型文档并证明旧运行路径没有复活

**Files:**
- Modify: `docs/DEVICE_MODEL.md`

**Steps:**

1. 将第 10 节状态改为已实现的静态 RuntimeSession 核心。
2. 明确动态输出、shape function、exact cache、ExecutionPlan registry 和后台编译仍未实现。
3. 更新模块边界表和最小完成标准。
4. 搜索旧 Adaptive、裸参数 runner 和伪 cache 接口，要求公共层零命中。

**Verification:**

```bash
rg -n "CompileMode|CompileConfig::(AOT|JIT|Adaptive)|BackgroundCompiler|KernelRunner|ShapePredictor|CompiledModule::Run|func_ptr|module_ptr" include src test
```

Expected: 0 matches.

### Task 7: 完整验证与提交

**Files:**
- Modify only files required by test failures.

**Steps:**

1. 运行 Windows CPU Object/Device/Signature/CompiledModule/RuntimeSession/Compiler/Pass tests。
2. 运行 Windows LLVM codegen 和完整数值矩阵。
3. 在 omen CUDA Debug 构建运行 RuntimeSession、CUDA codegen、CUPTI 和 Relay contract。
4. 对 Object、RuntimeSession、CompiledModule、AsyncOperation 运行 ASan/UBSan；对 CUDA session 运行 Compute Sanitizer。
5. 执行 `git diff --check`，删除仅由测试生成的 profile/CUPTI 输出并保留用户既有工作区改动。
6. 形成可审查提交并推送当前分支；GitHub Workflow 继续保持关闭。

**Verification:**

```bash
git diff --check
git status --short
```

Expected: task-owned changes committed, local/omen preserved changes explicitly reported.
