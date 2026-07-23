# 02：Shape 系统与特化

> **状态：** 规划中；不表示当前已支持 dynamic shape  
> **所属路线：** [编译器基础路线图](README.md)  
> **共同基础：** [01：契约、identity 与 cache](01-core-contracts-identity-cache.md)  
> **并行轨道：** [03：自适应编译与安全热替换](03-adaptive-compilation-hot-swap.md)  
> **权威输入：** [编译器基础架构审查](../../COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)  
> **范围：** Shape 语义、GraphTemplate、profile 与 specialization；热替换控制面属于 03。

## 1. 目标与当前边界
把“输入维度可变”定义为可验证契约，而不是扩散 `-1` 哨兵。
```text
Frontend fidelity -> typed symbolic graph -> GraphTemplate
  -> ShapeProfile -> specialization request -> frozen PlanVariant
  -> static RuntimeSession
```
`RuntimeSession` 继续只执行冻结 module + plan/runtime 数据，不能依赖 Compiler、Relay、frontend registry 或 ShapePredictor。
阶段顺序固定为：**exact -> bucket -> polymorphic -> dynamic output**；未证明适用时必须编译、等待或拒绝，绝不猜测执行。
`kDynamicDimension = -1` 当前只是 legacy 输入 ABI 校验哨兵，不表示符号、约束、输出公式、物理容量、valid extent 或 tail-safe kernel。
当前 lowering 将 shape 固化为 `IntImm`，`ValueSpec` 只有静态 shape/dtype/device，`RuntimeSession::ValidateShape` 仅对 input 忽略 `-1`。
当前 output `ValueSpec`/`KernelSignature` 没有端到端动态输出分配协议，因此 exact 是首个正确性闭环。
未知维不得再被默认 batch 或 `1` 静默具体化后视为正确语义。
`value_id` 是图内路由/诊断 locator，不能进入 unit semantic/artifact key；per-call unit 只是当前 partition policy。
shape 改变不应默认重跑 graph Pass、ValueGraph 与 partition。

## 2. 依赖、非目标、并行
### 硬依赖
- `OperatorSpec`、类型推导与 frontend 必须保留 symbolic dim，或在边界明确拒绝。
- graph Pass 后、partition 前 capability verify；未支持 shape/control-flow fail closed。
- unit semantic key 必须剥离 value id、symbol、对象地址、span。
- 静态 `ExecutablePlan`、`KernelSignature`、`RuntimeSession` exact 路径必须可用。
- type relation、ShapeProgram、TE/lowering shape 公式必须同源或有差分测试。
### 软依赖
- `PipelineResolver`/fingerprint 可后接 key 和 profile；03 coordinator 可先由 fake synchronous resolver 取代。
- KernelSlot、generation、canary、后台编译、region partition、task DAG、多 stream、持久化 cache 都不阻塞 exact。
### 可并行工作包
| 工作包 | 可并行对象 | 汇合边界 |
|---|---|---|
| DimExpr/Constraint/solver | frontend fidelity、key 修复 | 纯 runtime-neutral DTO |
| GraphTemplate skeleton | 静态 executor adapter | 不做 bucket/后台编译 |
| exact instantiation | fake coordinator | 产出 concrete static plan |
| frontend 保真 | type/lowering 对照 | 不扩展全 ONNX opset |
| extent/layout metadata | 03 ABI verifier | exact 时三者可相同 |
并行分支只能经不可变 DTO/序列化契约汇合，任何分支不得把编译依赖倒流进 `RuntimeSession`。
### 非目标
- 不恢复 `cached dims >= query dims` fuzzy cache，也不把 `-1` 升格为 Shape IR。
- 不实现编译队列、singleflight、热替换、canary、rollback；它们属于 03。
- 不承诺完整 ONNX、Transformer/CUDA 支持或 data-dependent/ragged output。
- 不把 allocation capacity 当作 logical shape。

## 3. Shape IR
### DimExpr
最小表达式受限且确定性序列化：
```text
DimExpr ::= Const(n >= 0) | Symbol(name)
          | Add(x,y) | Mul(x,y) | FloorDiv(x,positive_const) | Min(x,y) | Max(x,y)
```
`Const(0)` 合法且不同于 unknown；`Symbol` 在 template scope 唯一，绑定后不可重新解释。
frontend 对无法保留/降低的表达式必须报错，不能近似。
### Constraint
```text
Constraint ::= Eq | Range | DivisibleBy | BroadcastCompatible | SameRank | LayoutCompatible
```
profile instantiate 前统一求解，错误定位到 value、axis、来源；未绑定、矛盾、overflow、非法 rank/layout 与不确定输出均 fail closed。
求解不得猜测 batch、sequence 或 physical capacity。
### ShapeProgram
```text
inputs: named logical bindings
constraints: normalized Constraint[]
outputs: value -> logical DimExpr[]
valid_extents: value -> axis expressions / predicates
```
它由 template 建立，不能由 backend kernel 反向猜测；只计算 shape/extent，不分配、不查 cache、不 launch。
初期可解释执行，不需要 shape VM/JIT。

## 4. logical / physical / valid extent
| 层 | 内容 | 消费者 |
|---|---|---|
| logical | 数学维度、dtype、axis meaning | type、ShapeProgram、输出语义 |
| physical | capacity、stride、padding、alignment、scope | allocator、schedule、ABI |
| valid extent | 实际读写/结果有效域、tail predicate | kernel guard、mask/crop、验证 |
例：logical token 为 `[B=4,S=97,H=768]`；seq-128 bucket 可有 physical `[4,128,768]` 和对应 stride/alignment。
valid extent 必须声明 `0 <= s < 97`，并由 tail predicate/mask 保证；“buffer 更大”不证明读写安全或输出正确。
workspace、alias、layout、alignment 都属于 physical/ABI 合同。
exact 首阶段允许 logical = physical = valid extent，以缩小实现面，但三者语义不能混同。

## 5. GraphTemplate 与 profile 实例化
### 模板
```text
GraphTemplate {
  GraphTemplateKey; normalized typed graph + partition skeleton;
  CompilationUnit[]; ShapeProgram; value routing + boundary contracts;
  pipeline/capability fingerprints;
}
```
`PrepareGraphTemplate` 一次执行 normalize、graph Pass、capability check、partition skeleton；模板保留符号/约束，不是 concrete module。
`GraphTemplateKey`、`UnitSemanticKey`、`KernelArtifactKey`、`ShapeProfileKey`、`PlanVariantKey` 必须分离。
### 接口草案
```cpp
struct ShapeBinding { std::string symbol; int64_t value; };
struct ShapeProfileRequest { GraphTemplateKey key; BindingSet bindings; ProfilePolicy policy; };
struct ShapeProfile { LogicalContracts logical; PhysicalContracts physical; ValidExtents extent; };
Result<ShapeProfile> InstantiateShapeProfile(const GraphTemplate&, const ShapeProfileRequest&);
Result<FrozenPlanVariant> AssembleExactPlan(const GraphTemplate&, const ShapeProfile&, ArtifactResolver&);
```
名称是草案；边界校验 binding 的范围、rank、dtype、layout 和关联约束。
实例化重算受 shape 影响的 unit/output contract，不重跑无关图级工作。
exact 可用 fake/synchronous `ArtifactResolver`；03 可替换其实现而不改变 Shape 语义。
profile 决定 specialization request；artifact key 还含 target、ABI、backend/schedule、pipeline、shape-ABI version。
plan call id、storage id、entry symbol 是运行时/链接身份，不能代替 semantic identity。

## 6. 分阶段 specialization
### Phase 1：exact
全部 ABI 参与的 logical shape、layout、约束精确相等；physical=logical，extent 覆盖全 logical domain。
miss 时编译、同步等待或报 unavailable；不找“较大”版本。exact 复用 static `RuntimeSession(module, plan)`。
### Phase 2：bucket
bucket 将请求映射到有限、显式 physical profile，目标是减少 artifact cardinality。
每 bucket 必须声明 logical 接受域、physical capacity/layout、tail/mask、workspace、output crop；pad/mask/crop 是可审计 plan 步骤。
超出 bucket 时走 exact、另一个已声明 bucket 或拒绝；正确性依赖 valid extent，而非 capacity 足够。
### Phase 3：polymorphic
仅接收 runtime extent/shape scalar 且 guard 已验证的 kernel 可标记 polymorphic。
初期 allowlist 为 elementwise、copy、简单 transform 等 tail-safe unit；applicability 含范围、整除、rank、dtype、layout、target、workspace 上界。
`-1`、未绑定 symbol、“每维更大”均不是 polymorphic 证明。
### Phase 4：dynamic output
仅 ShapeProgram 能确定 output logical extent 时，加入：
```text
ShapeEvalTask -> AllocateTask -> KernelTask
```
allocation 用实际 program 结果，禁止负 extent/猜测容量；data-dependent/ragged output 继续拒绝，等待独立长度/ownership 协议。

## 7. frontend fidelity 与 NLP
symbolic/unknown dim 只能 preserve 为 `Symbol`、要求 bind、或明确 reject；禁止未知 non-batch dim 静默替换成 `1` 并写入 key。
frontend 保留 axis meaning、layout、attention mask、KV capacity/extent；Shape 系统不按 op 名识别“NLP”，只处理维、约束、layout、extent。
### prefill
prefill `[B,S,H]` 使用 sequence profile bucket，选择计入 attention 近似 `S^2` 成本。
报告 requested/physical token、padding、valid token、peak memory、compile/cache、kernel latency；mask extent 与 padded physical tensor 必须一致。
### decode
decode 通常 token=1，不为每个 token position 重新编译；变化项是 past-KV logical context 与 KV capacity/page bucket。
kernel 使用 runtime valid context 或已验证 bucket extent；capacity 不能冒充 context。prefill/decode 使用独立 policy、统计、阈值。

## 8. 实施步骤
1. 冻结术语、DTO、shape-ABI version、legacy `-1` 适配边界和负例。
2. 实现纯 DimExpr/Constraint evaluator 与确定性 ShapeProgram 单测。
3. 修正 frontend 为 preserve/bind/reject，禁止新路径 silent concretization。
4. 实现 `PrepareGraphTemplate`，冻结图级 pipeline/capability/partition skeleton。
5. 实现 exact `InstantiateShapeProfile`，适配现有 static plan/signature。
6. 用 fake coordinator/synchronous resolver 编译 miss 并组装 immutable exact variant。
7. exact 门禁后增加有限 bucket、tail-safe contract、pad/mask/crop。
8. 仅 allowlist 增加 polymorphic guard/runtime extent；最后才动态输出 task。
03 可并行定义 resolver future/result，但绝不阻塞步骤 2--6。

## 9. 测试
| 层次 | 正例 | 反例 |
|---|---|---|
| IR/solver | bind、broadcast、range、zero dim | unbound、overflow、矛盾、负值 |
| program | input 到 output/extent 确定 | output 不确定、缺 binding |
| frontend | symbolic 保留并 bind | unknown 偷换为 `1` |
| exact | 多 profile 复用 template、等于 baseline | 重跑无关图级工作 |
| bucket | edge/tail/mask/reduction/crop 一致 | 容量大但无 extent/guard |
| polymorphic | 声明域内等于 exact | 域外未在 dispatch 拒绝 |
| dynamic output | 按 ShapeEval 分配 | ragged 被错误接受 |
| NLP | prefill/decode 数值与统计 | KV capacity 当 valid context |
bucket/polymorphic CPU 路径配合 ASan 越界测试。

## 10. Done 条件
- template、binding、constraint、ShapeProgram、exact profile、frozen static plan 已闭环。
- 同 template 多 profile 不重跑无关 graph Pass/partition，只有受影响 unit specialization miss。
- 每个 runtime value 可区分 logical/physical/valid extent；exact 可相等但不混同。
- frontend unknown/symbolic 只能 preserve/bind/reject。
- bucket 有 applicability、physical contract、tail/mask/crop 和数值/越界证据。
- polymorphic 仅 allowlist，有 guard、runtime extent ABI、域内差分测试。
- dynamic output 仅接受可确定 ShapeProgram 结果。
- `RuntimeSession` 无 Compiler/Relay 依赖；03 coordinator 可替换 resolver 而不改变 Shape 语义。
