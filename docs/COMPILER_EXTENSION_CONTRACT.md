# 编译器扩展契约

本文件定义新增 Relay 算子与编译器 Pass 的强制扩展契约。该契约具有规范性：运行时元数据、实现绑定、校验、测试与文档全部一致前，扩展都视为未完成。系统边界见 [架构总览](ARCHITECTURE.md)。

## 1. 稳定身份

- 算子身份由规范注册名与 Schema 版本共同确定。
- Pass 身份由 IR 方言、规范名称与 Schema 版本共同确定。
- 编译单元语义身份由规范化后的算子与属性、单元内逻辑到边界的映射、边界张量契约、副作用与别名语义共同推导。图内值、编译单元和存储 ID，对象地址及链接符号都不参与。
- 图值 ID 仍用于执行计划路由。对于冻结的静态计划调用，`PrimFunc global_symbol`、`KernelSignature` 符号、`CompiledModule` 入口与 `ExecutablePlan` 调用符号必须一致。就绪的可复用产物可以在校验后通过符号别名重新绑定，但符号本身不决定缓存等价性。

注册表必须拒绝身份冲突的重复项。注册表枚举与序列化必须具有确定性。

## 2. 新增算子要求

新增算子必须按顺序完成：

1. 在 `contracts/relay_op_contract.json` 中声明机器可读契约。
2. 定义或复用属性 Schema，给出类型字段、默认值、合法范围与稳定序列化顺序。每个包含字段的 `BaseAttrsNode` 必须实现 `SerializeCanonical(CanonicalAttrWriter&)`，并按 Schema 顺序恰好输出每个字段一次；编译单元身份包含该类型及其值的序列化结果。
3. 重新生成 `src/relay/generated/relay_op_contract.inc` 与 `src/relay/generated/relay_op_registration.cc`。生成的 `OperatorSpec` 是关键元数据的权威来源。带 `registration` 对象的条目会把对外可见的类型推导与 Lowering 回调绑定到生成翻译单元；尚未完成迁移的条目只能保留一个手工注册权威。
4. 实现类型校验与输出类型推导。
5. 实现 Lowering 时，只使用当前 `Call` 的显式输入、属性、已检查类型与目标能力。
6. 补充 Schema、类型推导、Lowering、ABI、目标能力与数值测试。
7. 更新 [Relay 算子支持矩阵](OP_SUPPORT_MATRIX.md)；当前端映射或导入器子集发生变化时更新 [ONNX 导入器](ONNX_IMPORTER.md)。
8. 运行算子契约、文档、头文件层级、公有头与目标专用数值检查。

普通计算算子只有在规格完整，而且 Lowering 绑定与声明的输出元数一致时，才能作为编译单元处理。Lowering 不得递归进入生产者 `Call`。不支持的目标必须明确报错；空张量、空 `PrimFunc` 列表、空操作内核和静默后端回退都不是合法实现。

必须声明纯函数性、副作用、确定性、别名与原地更新行为。缺失声明按保守语义解释。运行时代码不得按算子名做条件分支。

## 3. 新增 Pass 要求

新增 Pass 必须按顺序完成：

1. 在 `contracts/pass_contract.json` 中声明机器可读契约与默认成员关系。
2. 重新生成 `src/pass/generated/pass_contract.inc`；C++ 绑定表只能把生成的实现键映射到函数。
3. 执行前校验 IR 方言、作用域、阶段、必需不变量、分析结果与目标能力。
4. 只改写声明范围内的 IR 与作用域。
5. 按声明精确保留或失效元数据与分析结果。
6. 执行后校验产出不变量。
7. 增加命中、不命中、非法输入、元数据与确定性测试。若声明幂等性，还要补充二次执行测试。
8. 更新 [Pass 契约](PASS_CONTRACT.md)。

新增 Pass 不会自动加入默认流水线。默认流水线的变化必须更新显式契约顺序，并接受单独评审。

需要查询算子语义的框架级 Pass 必须使用通用 `OperatorSpec` 字段。禁止硬编码算子名白名单。

## 4. Pass 作用域

| 作用域 | 输入 | 可变更内容 | 禁止操作 |
|---|---|---|---|
| `graph` | 完整 `Relay Function` | 编译单元冻结前的 Relay 拓扑 | 保留过期的已检查类型 |
| `compilation unit` | 单个编译单元及显式边界值 | 单元内 Relay IR | 读取或合并生产者/消费者编译单元 |
| `PrimFunc` | 单个 `PrimFunc` | 单元内 TIR | 修改其他编译单元或符号身份 |
| `module` | 一组已编译入口 | 声明的模块元数据 | 静默更改已冻结的内核 ABI |

图拓扑变更必须在最终值与编译单元身份分配前完成。编译单元冻结后若要修改输入输出，必须回到图阶段重新分区。

## 5. 层级交互

```text
前端 / FFI
  -> 已校验的 Relay Call 与 Attrs
算子注册表 + Relay 类型/图流水线
  -> 类型完备的 Relay Function
值图 + 按算子分区
  -> 每个普通计算 Call 对应一个 CompilationUnit
编译单元 Lowering
  -> 每个 CompilationUnit 对应一个 PrimFunc
PrimFunc Pass 流水线
  -> 带稳定符号的已校验 PrimFunc
内核 ABI + 代码生成
  -> 每个链接符号对应不可变签名与可执行入口
CompiledModule + ExecutablePlan
  -> 冻结的链接符号与图内值路由
RuntimeSession
  -> 仅负责 NDArray 分配、启动与完成
```

后端批处理不会削弱编译单元身份：同一目标上的多个 `PrimFunc` 可以共用一个 LLVM JIT 或 CUDA 模块，但每个编译单元仍保留自己的符号、签名、启动元数据、模块入口与 `KernelCall`。

原语产物键包含完整的 `UnitSemanticKey`、与编译相关的目标能力身份、规范化的 `PipelineResolver` 指纹、内核 ABI 版本、调度版本与后端实现版本。摘要只用于索引；完整的规范字节决定等价性。图内值、编译单元和存储 ID，对象地址、请求热度与链接符号都不参与产物身份。

可执行计划中的存储复用属于物理内存决策，不是值身份或别名机制。只有契约相同且调用区间严格不重叠的中间值才能共享存储 ID。输入、常量、图输出、声明别名与异步存活值始终保留独立存储。运行时保持单流执行顺序，并在最终完成对象可以安全释放前保留被替换的存储。

允许的依赖方向是单向的：

- 前端可以依赖 Relay 契约。
- 编译器可以依赖 Relay、Pass、TE、TIR、代码生成契约，以及运行时可执行计划契约。
- 代码生成层可以依赖 TIR 与运行时内核 ABI 契约。
- 运行时不得依赖 Relay、TE、TIR、算子注册表、Pass 注册表、编译器内部细节或后端私有类型。

## 6. 跨契约重写规则

Relay Pass 创建或替换 `Call` 时：

- 目标算子必须已注册；
- 属性与元数必须通过 `OperatorSpec` 校验；
- 发生变化的类型和形状信息必须失效并重建；
- 副作用与别名契约必须允许删除、复制或重排。

`PrimFunc` Pass 修改函数时：

- 编译单元语义身份保持稳定；任何链接符号别名都必须有独立、可校验的模块或计划契约；
- 声明的参数角色与常量键保持一致；
- 任何会改变 ABI 的改写都必须在 ABI 冻结前完成，并显式声明所属阶段。

流水线运行期间，任何 Pass 都不得修改算子注册表或 Pass 注册表。

## 7. 必需检查

编译准备阶段与选定的拓扑构建器会在实际边界上强制执行失败即拒绝的 Relay 契约。`PipelineResolver` 是生产环境中 Pass 顺序、不变量状态迁移与编译产物指纹的权威来源；直接连接命名流水线的方式只作为兼容和测试入口。

扩展契约变更的最小本地验证为：

```powershell
cmake --build out/build/dev-mingw-cpu --target `
  registry_test pass_pipeline_test compiler_extension_contract_test `
  check_relay_op_contract check_pass_contract check_docs `
  check_include_layers check_public_headers -j 4
```

测试二进制应从 `test/` 目录执行。后端或数值行为变更还需运行对应的 LLVM/CUDA 测试；不支持的后端必须有明确的诊断测试。
