# CUDA 有界索引、形状重排与完整 MiniMind prefill 接入报告

2026-09-10。**状态：完整 CUDA 数值、状态前置和 CUPTI 设备活动验收已完成。** 本报告服务 [PROJECT_GOAL](../PROJECT_GOAL.md) 的变长 Transformer 与观测层目标，复用现有 IR、编译和运行时所有者。

报告中关于“尚未执行”的段落是 2026-09-09 Windows 主机离线时保留的编译快照；本报告末尾的 2026-09-10 Windows 证据是当前结论。

## 已达到的效果

实际八层 MiniMind bounded prefill 的 **742/742 个原语**已通过生产 lowering、CUDA 地址/所有权证明、源码生成和 NVCC PTX 编译。此前补齐静态表 Gather 后，仍有 100 个形状原语无法通过证明：48 个 Reshape、18 个 Unsqueeze、16 个 Concatenate、16 个 Expand 和 2 个位置表 Slice；本次已闭合这些编译缺口。

新增边界测试还复现并修复了一个错误：在 `c<8` 分支内证明 `c/4<2` 后，原缓存错误地把这个较小范围用于分支外的读取。现在兄弟分支与后续表达式只能使用各自有效的事实。整模型的 742 个原语在修复后仍全部通过，生成源码逐字节不变。

上述结果证明完整图已能通过 CUDA 编译链，**不证明 GPU 数值正确、性能达标或完整变长推理已完成**。Windows RTX 4070 Ti SUPER 主机在 2026-09-09 22:20（上海时间）后离线；本次源代码验证在 Linux、CUDA 12.9.86 与显式 synthetic `sm_89` Target 上完成，没有 GPU 执行或 CPU fallback。

## 采用的方法

### 复用实际导出与已有生产链

输入沿用 [CPU 完整 prefill 报告](M3_FULL_PREFILL_REPORT.md) 的 fixture：八层、hidden 768、Q heads 8、KV heads 4、head dimension 96、FFN 2432、词表 6400。权重为 seed 0 初始化的 float32，未缩小层数或宽度。模型结果用于编译与数值对齐，不用于证明预训练语言质量。

显式合同为 `1≤B≤3`、`1≤S≤8`，四组输入为 `(1,1)、(1,4)、(2,3)、(3,8)`；输出包括全部 logits `[B,S,6400]` 和八层交错排列的 16 份 K/V `[B,S,4,96]`。位置表容量为 128。动态 ONNX 导出 receipt：

```text
c52ef7d36dc5d37611341adf7d3e4ba0cc9255e665199acc1ec4712dd473a1d6
```

硬件 consumer 复用 [minimind_bounded_prefill_llvm_test.cpp](../../test/minimind_bounded_prefill_llvm_test.cpp)，通过 `--cuda` 选择设备，走 `LoadONNXShapeSource → RestrictedSymbolicShapeAdapter → Compiler::CompileBounded → RuntimeSession`。CUDA 使用已有的优化等级 3，以执行唯一的线程绑定 Pass。CPU 仍使用优化等级 2。

无设备时的 [codegen consumer](../../test/minimind_bounded_cuda_codegen_test.cpp) 只执行同一 bounded preparation、逐 primitive lowering 和已解析的 TIR pipeline，不发布运行时产物，也不调用 GPU。其 [Python wrapper](../../test/minimind_cuda_codegen_test.py) 再用 NVCC 编译源码，并核对 PTX 保留全部 742 个不同入口。这个入口专门提供可重复的编译证据。

### 静态只读表的有界间接读取

扩展仍位于 `BindCudaThreads`。原有有界多项式证明先核验索引张量的实际紧凑地址；遇到数据索引时，只有静态、只读的表可交给已有的分支区间证明。两份证明通过局部回调交换“直接读取已安全”“两个地址是否为同一多项式”“整数范围”，不新增参数 ABI、launch owner 或运行时检查器。

例如 `ids[i*S+j]` 的读取必须针对本次实际 `B*S` 证明，而不是分配上界。它的范围保护不能被套用到另一个同样在界内、却表示不同元素的 `ids[j*B+i]`。负索引归一化、整数宽度、每个中间表达式的溢出和最终表地址仍须逐项通过原保护规则；写入表、动态表和未初始化输出的间接读取继续拒绝。

参考 [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html) 的 int32/int64 索引与 `[-capacity, capacity-1]` 有效范围。KXC 已有的非法索引零填充属于显式扩展；ONNX 对越界索引要求报错，本报告不把两者等同。

### 商、余数与分支内的坐标范围

Reshape/Unsqueeze 的实际 TE 索引会先把坐标展平，再用除法与余数拆回各轴。参考 [MLIR Affine](https://mlir.llvm.org/docs/Dialects/Affine/) 对正除数、商/余数和符号表达式的约束，本实现只采用所需不变量，不引入 MLIR 或新 IR。

- 只有证明除数为正，才处理除法和余数。进入 extent 为单项式的循环体可以证明其各因子为正；进入 `P+1` 循环不能推出 `P>0`。
- 先按多项式系数和因子分离整除部分。若剩余值已证明小于除数，直接消去相应商；否则保留具有范围的内部索引项。
- 位置表 `0:S` 前缀使用 `S` 的上界证明容量充足，运行时地址仍使用实际 `S`。
- `select(S==1,0,i)` 只有在已知 `i<S`、两分支确实等价时才按 `i` 证明。原始运行时表达式不变。
- Concatenate 的 `c<48` 只在对应分支内限制静态循环坐标，因而可以分别证明 `c` 与 `c-48` 的输入地址。分支内创建的商/余数缓存项不向外传播。

原有整数宽度、线程输出独占、生产顺序和初始化检查保持有效。上界用于证明及固定 launch，不能代替实际 extent/stride。私有 scratch 仍受每线程 64 KiB 上限约束；本次不增加跨线程协作或模型专用内核。

### 身份与回归边界

Pass schema 更新为 **7**，backend identity 为 **`cuda-nvrtc-driver-v7`**，通过原有 pipeline/artifact canonical identity 生效。既有 runtime extent 的 `uint64[1]` 设备参数 ABI 和 bounded schedule identity 保持不变。v6 的硬件结果保留在 [上一模块报告](GPU_BOUNDED_REDUCTION_REPORT.md)，不能把那些结果当作 v7 新索引规则的硬件验收。

三轮 Ponytail QA：A，复用既有 shape、多项式、静态保护证明和同一个模型 fixture；B，没有第二份 launch、状态或缓存身份权威，语义变化进入既有版本；C，生产编译 consumer、独立 PTX consumer、实际反例与明确的硬件 gate 均已接入，GPU 数值未执行时保持未验收。

## 已执行的验证

| 检查 | 当前结果 |
|---|---|
| 有界 Gather 结构测试 | 6 个正例、14 个拒绝例通过，含 int32/int64、负索引、等价地址和错误保护对象 |
| 有界形状地址测试 | 12 个正例、16 个拒绝例通过，含 reshape、广播、位置前缀、拼接、零除数、整数溢出、尾部越界与分支范围泄漏 |
| CUDA schedule / pipeline resolver | 2/2；schedule 内共 18 组测试通过 |
| 完整八层源码与 PTX CTest | 742 个原语、742 个 PTX 入口；52.84 秒 |
| Linux 混合后端专项 | 7 个实际通过、2 个硬件测试返回 77 跳过；总计 53.34 秒 |
| 既有 GPU bundle 的检查器回归 | 原四组有界图的 62 次运行、286 个 kernel、638 次 extent DMA 仍通过；仅读取既有证据，没有新增硬件运行 |
| 默认 CPU/LLVM | 55 项最终全部通过：首轮 54/55、147.73 秒；报告链接和索引补齐后，文档项单独重跑 1/1 |
| adaptive CPU/LLVM | 55/55，218.16 秒；含实际 MiniMind 两代执行与替换 |
| bounded CPU/LLVM | 70/70，228.19 秒；含完整变长 prefill/decode、状态交接与真实请求批处理 |
| 契约与架构 | Relay 40/40、Pass 20/20、生成物 freshness、NLP 矩阵、284 文件 include-layer、90 个安装头及 10 个实验头独立编译均通过；诊断引擎 4/4 |

共享 prefill consumer 的 CPU 回归实际执行 3,710 个 LLVM call，核对四组 B/S 的全部 17 个输出及未来 token 扰动，最大绝对误差仍为 `8.46386e-6`。首组输出在后续形状运行及 session 销毁后再次通过参考核验。完整 bounded decode/state/batching 也保持通过，不能把这些 CPU 结果替代新 GPU gate。

默认构建的第一轮文档检查发生在新报告写入前，因此报告链接尚无目标；报告与总索引补齐后仅重跑失败的文档项，保留原始失败和修复后的日志。最终 CPU 审计分别记录原始 run 与 docs retry，未把它伪装成一次 55/55 的运行。

整模型编译证据位于 `out/build/bounded-cuda/bounded-cuda-codegen-evidence/bounded-prefill-_lvtbv12/`，含 `lowering.log`、`nvcc.log`、源码、PTX 和 `codegen-audit.json`。对应文件：

| 产物 | 字节数 | SHA256 |
|---|---:|---|
| CUDA source | 2,009,294 | `10572c2f007774cffec19cbe6b2cc522916f672550d4fb4910b1e48dff5e619d` |
| PTX | 2,512,940 | `cb120fccd15c6557895b7bc2b1a9ef6433e7eb83d76746fe7e4b389571bacabb` |

原始过程与回归日志在 `out/windows-gpu/logs/bounded-prefill-*.log`。`bounded-prefill-shape-regression-before-fix.log` 保留分支缓存反例确实失败的记录，`bounded-prefill-shape-regression-after-fix.log` 保留修复后的通过结果。

本地可分发源码为 `out/windows-gpu/bounded-prefill-source.tar.gz`，逐文件清单为同目录的 `bounded-prefill-source-manifest.json`；编译产物与本轮日志归档为 `bounded-prefill-local-evidence.tar.gz`。最终 Windows 复核使用隔离目录 `F:\kxc-gpu\20260910-goal`，通过 SSH 上传后重新构建，没有使用旧 bundle 冒充本轮结果。

## 已实现、尚待硬件执行的验收入口

[CUPTI wrapper](../../test/cuda_profile_bundle_test.py) 新增 `bounded-prefill` 模式，必须同时获得完整 C++ 数值结束标记与实际设备事件。普通退出码 0、空 bundle 或缺失 fixture 均不能使硬件 gate 通过。

下表是**入口要求检查的数量，尚不是测得的 GPU 结果**：

| 项目 | 验收要求 |
|---|---|
| 正常运行 | 四组 B/S 各两次，再对最后一组进行两次未来 token 扰动，共 10 次 |
| 数值 | 独立 ONNX 参考的 878,080 个值，误差 `<5e-5`；另比较 602,112 个因果扰动值，前缀逐位一致 |
| 生命周期 | 两个 session、两条非默认流；输入句柄在等待前释放，首组输出在后续形状运行及 session 销毁后再次检查 |
| 不隐式编译 | 每次运行和拒绝后，primitive cache 十项统计保持不变 |
| 设备关联 | 7,420 个 kernel 与 run/call_index 一一关联；同一 call 的 grid/block 不随实际形状改变 |
| extent ABI | 10 个 S-only call，其余 732 个使用 B/S；共 14,740 次 8 字节 H2D、117,920 字节；每份 scalar DMA 必须在所属 kernel 开始前完成 |
| 拒绝 | 超出 B/S、空轴、rank/dtype、实参数量、输入设备与 stream 设备错误，共 9 项，均不得分配或提交设备工作 |
| 证据反例 | 空事件、shape、ABI、receipt、grid、scalar 字节数、复制顺序及拒绝提交数被篡改时，检查器必须拒绝 |

辅助 embedding/FFN 的 int32/int64 和 sigmoid 数值测试也保留在 CUDA 模式，使用独立 `auxiliary` bundle，避免把小图 kernel 混入整模型的 742-call 关联。

## 复现

沿用 [有界 CUDA 构建配置](GPU_BOUNDED_CORE_REPORT.md) 并准备原有完整 prefill fixture：

```sh
export KXC_MINIMIND_BOUNDED_PREFILL_DIR="$PWD/out/fx_minimind_bounded_prefill"
cmake --build out/build/bounded-cuda --parallel 2 \
  --target cuda_schedule_test minimind_bounded_cuda_codegen_test minimind_bounded_prefill_cuda_test
ctest --test-dir out/build/bounded-cuda --output-on-failure \
  -R '^(cuda_schedule_test|minimind_bounded_cuda_codegen_test)$'
ctest --test-dir out/build/bounded-cuda -V -R '^minimind_bounded_prefill_cuda_test$'
```

Windows 使用相同目标、`--config Release` 和 `ctest -C Release`，fixture 目录为 `F:\kxc-gpu\20260910-goal\fixtures\bounded-prefill`。设备复核结果：完整模型 10 次运行、7420 个 CUDA kernel、878080 个参考值和 602112 个因果值，最大绝对误差 `1.06096e-5`；两条 stream；9 个拒绝均为零提交；14740 次 8-byte extent DMA；CUPTI checker 的 8 个篡改反例全部拒绝。辅助 proof 为 17 项拒绝，int32/int64 embedding/FFN 各 40 calls，sigmoid 三种形状，全数通过。最终标记为 `[PASS] bounded_minimind_prefill_profile_verified` 和 `[PASS] cuda_profile_correlation_verified`。

本轮还修复了一个跨平台 fail-closed 缺陷：字面量 slice 控制只保留 payload，后续证明访问空的 `proof.elements` 会在 MSVC 抛出 `invalid vector subscript`。解析器现在为非负字面量建立并行 `DimExpr` 视图，对负控制值先给出明确的 `start 0` 拒绝；没有放宽动态 shape 合同，也没有跳过 CPU-device/stream 负例。
