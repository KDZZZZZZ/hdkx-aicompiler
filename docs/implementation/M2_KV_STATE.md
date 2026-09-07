# M2：KV cache 与动态有效状态

MiniMind 是当前 L1 纯文本 decoder，MiniMind-O 的 Thinker 复用同一 backbone。第一波已经证明 RuntimeSession 可以重复运行并记录执行事件，也有 state/alias/persistent buffer 的通用机制；但这些机制还没有表达 Transformer 的 past/present、每层 KV 布局、cursor 和有效长度。当前导出目录有静态 prefill/decode 图，decode 的 past/present 只是外部张量，不能据此宣称 session 自己维护了 cache。

本模块要先让 MiniMind-L1 在同一个 RuntimeSession 中完成 prefill 后的多步 decode：固定物理容量，追加新 K/V，更新 valid extent，注意力只读取有效区。MiniMind-O 的 Mimi 流式卷积 ring buffer 是更强的状态，留到 L3，不把它混入这份 KV 合同。

> 状态：待实施，第二波 B 线优先模块。先通过 M9 的 decode 签名门禁，再与 M3 的 extent/shape 合同协同；观测沿用第一波 M1。M10 的控制流运行时当前明确拒绝 state 和 runtime extent；若后续要让 bounded `While` 承载真实 decode，本模块的 `ExecutablePlan`/`RuntimeSession` state owner 与 extent ABI 必须先成为唯一交接点。当前目标模型以 PROJECT_GOAL.md §2.2 为准，第一波证据见 G1_RECORD.md。

## 本模块要做的模块

| 模块 | 需要定义的内容 | 首个可验收结果 |
|---|---|---|
| KV layout | 层、K/V、batch、sequence、kv_heads、head_dim 的稳定顺序；当前导出记录为 batch, seq, kv_heads, head_dim | past/present 输入输出顺序和字节布局可复现 |
| capacity/extent | 物理容量与当前有效长度分离，total = past + current | 超容量、负值、溢出在 launch 前拒绝 |
| append/read | prefill 写入，decode 追加，attention 只读有效区 | 同一 session 连续三步 decode 与全量参考一致 |
| ownership | session 持有 state，completion 保活，失败后的 session 状态明确 | cache 地址跨 run 稳定，不能跨 session 偷传裸指针 |
| evidence | profile 关联 stage、layer、extent 和 state version | bundle 能解释每步读写，不改变 kernel ABI |

## 当前支持与缺口

| 已有机制 | 真实缺口 |
|---|---|
| ValueSpec 的 is_state、alias_source、kInPlace | 没有动态 cursor/valid extent 的读写协议 |
| RuntimeSession 独占持久 state storage | 没有生产编译路径生成 Transformer 缓存更新计划 |
| 静态 valid_bytes | 不能把一个会变化的序列长度塞进这个静态字段 |
| pending 有状态 RunAsync 会被拒绝 | 需要保持明确提交顺序，不能假设不同 stream 自动同步 |
| 新 bounded 分支运行 fresh-output 图 | 此模式明确拒绝 state、alias、donation 和 reuse |
| 已有 external-KV decode fixture | 外部传入 K/V 不是会话自己维护和更新的 KV cache |

另外，一个 RuntimeSession 当前绑定固定 plan。分别创建 prefill 和 decode 会话会得到不同的 state，不能把两个独立的零初始化缓存当成共享状态。

## 首切片的明确合同

- CPU/LLVM，固定 rank、dtype、设备和物理容量；先用小规模确定性张量，例如 `B=1,H=2,C=8,D=4`。
- runtime 持有缓存和通用有效长度 metadata；输入显式描述本次追加的数据和长度关系，kernel ABI 能读取经校验的 extent。
- 初始有效长度为 0；追加 `n` 前检查 `0 <= length`、`0 <= n`、`length + n <= capacity` 及算术溢出。
- 追加后只能读取 `[0, length+n)`；物理容量固定，不能每次重新分配整份缓存。
- 原地写继续由 ValueSpec alias/write-mode 声明。新动态有状态合同独立版本化，不能简单删掉 fresh-output 的拒绝检查。
- 第一版同一会话有未完成写入时继续明确拒绝再次提交，先不增加隐式队列。

固定容量与 past/present 共用内存可参考 ONNX Runtime 的[缓存共享说明](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)。本计划采用容量与有效长度分开的不变量，具体 API 和所有权仍归 KXC 的 ExecutablePlan/RuntimeSession。

## 分阶段实施

### S1：一个真实的缓存追加/读取图

1. 在既有 ValueSpec/ExecutablePlan 中表达动态 metadata、允许更新的字段和容量关系；字段由唯一会话状态机制消费，不增加模型名到缓存的 registry。
2. 在 plan validation 检查状态角色、alias source、使用顺序、容量和设备/dtype/shape。memory planner 保证 state 不被错误复用。
3. 在 ModuleInvocationContract/KernelSignature 的既有边界中传入经过校验的 extent；明确数据张量与形状/有效长度 metadata 的区别。
4. 从明确的计算声明经 production compiler 生成追加/读取 kernel 和状态计划，不能只在测试里手工拼 plan 后声称完整编译链已接通。
5. RuntimeSession 在首次构造时分配，run 时绑定持久 Storage；长度更新在成功完成后提交。在失败前拒绝的调用不能改变长度或缓存。
6. 做 3、1、1 个 token 的连续追加，与独立 CPU 参考逐元素对比，证明有效长度依次为 3、4、5。

S1 可以先用受控的生产 state 图跑通，不等待完整 shape-as-value；但只有在 M9 E2 锁定 MiniMind past/present 签名后，才能把它接到真实 decoder。受控图只证明状态能力，不代表 MiniMind-L1 已通过。

### S2：prefill 和 decode 沿用同一份状态

与 [M3](M3_SHAPE_VALUES.md) 联合实现一个能够处理多 token prefill 和单 token decode 的有界计划，优先让同一个会话持续持有 state。这样不需要跨会话复制 KV 或引入模型专用资源表。

如果需要两个不同计划，必须先在现有 runtime owner 中明确状态绑定/交接合同，并测试容量、布局、ABI 和生命周期；不能把裸指针从一个会话递给另一个会话。两个方案不能同时形成两套缓存权威。

验收使用固定权重的微型因果注意力：一次 prefill 后至少三步 decode，每一步与全量重算参考比较；检查无效容量区填入的哨兵值不会影响结果。真实 MiniMind ONNX 路径由 M4/M9 共同接通；如果 dynamic_axes 导出仍未通过，先使用固定形状 decode 和明确标注的 external-KV 证据。

### S3：扩展目标与执行形态

在 S2 后才扩展 batch、更多层、设备端状态更新和连续批处理。分页、跨会话共享、分布式一致性以及中途迁移 KV 均不属于首版本。

## 代码落点与版本

| 位置 | 职责 |
|---|---|
| [executable_plan.h](../../include/kxc/runtime/executable_plan.h)、[executable_plan.cc](../../src/runtime/executable_plan.cc) | 通用状态、extent 与动态执行合同校验 |
| [memory_plan.cc](../../src/runtime/memory_plan.cc)、[ValueTable](../../src/runtime/internal/value_table.h) | 物理容量、alias 与生命周期 |
| [session.cc](../../src/runtime/session.cc)、[device_stream.h](../../include/kxc/runtime/device_stream.h) | 持久所有权、顺序、完成与失败处理 |
| [compiled_module.h](../../include/kxc/runtime/compiled_module.h)、[kernel_abi.h](../../include/kxc/runtime/kernel_abi.h) | extent 与 kernel 参数的明确对应 |
| [experimental_identity.cc](../../src/compiler/identity/experimental_identity.cc)及 production lowering | 状态/extent 合同进入 plan ABI 与产物身份 |

容量、布局、extent 表达和写入约束进入相应 identity/version；每次变化的 cursor 数值是运行数据，不为每一个 token 重新编译或重建 artifact identity。只有实际内存计划算法改变时才相应提升 memory-plan version。

## 失败与生命周期

容量和参数错误必须在 launch 前失败，原状态保持不变。后端已经开始写入后若失败，第一版不承诺事务回滚：会话应标记为不能继续，要求显式重建，不能沿用可能部分写入的缓存。

完成句柄必须保活所有 Storage、模块和更新上下文；不同 session 的缓存独立。读取最新状态必须发生在上一次写可见之后，不能靠 CPU 同步行为推断 CUDA 安全。

## 验证与退出条件

```bash
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error \
  -R 'kv_state_llvm_test|executable_plan_test|runtime_session_test|codegen_llvm_test|compiler_identity_test|compiled_module.*test|kernel_signature_test'
```

为 S1/S2 新增 `test/kv_state_llvm_test.cpp`（拟用目标 `kv_state_llvm_test`），注册到 CTest 后用 `ctest -N` 确认存在，再运行公共检查和 LLVM 全量。测试至少覆盖：

- [ ] 连续追加、零长度、刚好达到容量、超过容量、负值和整数溢出。
- [ ] 真实 LLVM 原址写入且 buffer 地址跨 run 不变；未更新区域不被误读。
- [ ] 独立 session 隔离、pending async 拒绝、会话销毁后 completion 保活。
- [ ] 所有 contract mismatch 的 launcher 计数为零。
- [ ] 结构性 extent/alias/布局变化改变 identity，运行 cursor 变化不触发编译。
- [ ] S2 的 prefill + 多步 decode 与全量参考一致，生产链使用同一份有效状态。

S1 与 S2 分别标记完成，只有 S1 不得将 kv_cache 全行改为已验证。矩阵中的参考/mock numeric 证据继续与真正编译运行证据区分。
