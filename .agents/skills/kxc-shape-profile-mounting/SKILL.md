---
name: kxc-shape-profile-mounting
description: Attach symbolic shape bindings and finite exact-profile routing to KXC without implicit compilation. Use for ShapeProgram, DimExpr constraints, ExactOracle, exact specialization, DispatchKey, profile publication/lookup, restricted symbolic shapes, or dynamic-shape control-plane work.
compatibility: KXC repository with experimental exact shape control and identity builders.
---

# KXC Shape / Profile 挂载

先完整读取 [总则](../kxc-capability-mounting/SKILL.md)。Runtime allocation/state 属于 `kxc-runtime-state-mounting`，不要混入路由器。

## 唯一 owner

```text
include/kxc/compiler/shape_specialization.h       ShapeProgram/DimExpr/ExactOracle
include/kxc/compiler/experimental_identity.h      ShapeProfileKey/DispatchKey/Plan keys
include/kxc/compiler/shape_control.h
src/compiler/shape/shape_control.cc                finite exact publish/lookup
include/kxc/compiler/restricted_symbolic_shape.h
src/compiler/shape/restricted_symbolic_shape.cc   producer-specific materialize/verify
```

`ShapeProgram` 是唯一 evaluator；`BuildStaticExactDispatchKey` 等既有 builder 是唯一 identity authority。Route table 只选择 caller 已发布的 executable，绝不编译、查询 primitive cache 或分配 runtime buffer。

## 现有 exact 路径

```text
named concrete input shapes
  -> BindExactInputShapes
  -> BindingSet / ShapeProgram::Evaluate
  -> ExactOracle
  -> producer materialize + Compiler::Compile + producer verify
  -> ExactProfileRouteTable::Publish
  -> canonical DispatchKey lookup
```

当前 binder 只反解 direct `Const`/`Symbol` input axis。算术表达式不反解；所有 logical/physical/valid contract 必须 exact。

## 挂载 exact profile

1. 在 `ShapeProgram` 声明 symbols、named inputs/outputs 和 range/divisibility constraints。
2. 使用 `BindExactInputShapes` 或 producer 的明确 binder；重复 symbol 必须一致。
3. 用 `InstantiateExactProfile`/`ExactOracle` 证明完整求值结果。
4. Producer 显式 materialize/compile，并调用自身 validator 绑定 graph semantics、dtype、边界和 call structure。
5. Route table publication 校验 target fingerprint、expected `PlanAbiFingerprint`、重复和输入重叠。
6. Lookup 只按 canonical dispatch key 命中或抛错；没有默认 profile/fallback/compile-on-miss。

不要让通用 route table复制 producer-specific lowering verifier；但也不能跳过 producer 验证步骤。

## 新 bounded/bucket 能力

不要放宽 exact API 的含义。若确有范围 profile：

- 新建明确 versioned applicability contract，定义 bounds、selection precedence 和 capacity/valid extent。
- 继续复用 graph/profile/plan identity builders，必要时扩展 builder，而非拼接新 key。
- 编译仍显式；miss 行为必须定义且默认 fail closed。
- overlap 必须有确定规则，不能依赖 vector 插入顺序。
- runtime physical capacity 与动态 valid extent 交给 Runtime owner。

先用一个有限 bucket 贯通，不做服务化 cache、eviction 或异步 compile，除非有独立需求。

## Identity 与 ownership

- Template key、profile key、dispatch key、plan variant 和 plan ABI 各司其职，不互相替代。
- Materialized exact graph 的 semantic key 可能与 symbolic template key 不同；由 producer validator 证明对应关系。
- Published value 持有 `CompiledGraph` copy，从而保活 artifact pins；不能返回会因 vector growth 失效的引用。
- 一个 route table 固定一个 template 和 target capability fingerprint。

## 测试

- Binder：乱序 named inputs、static axis、shared symbol、range/divisibility。
- Negative：缺失/重复/未知名、负 extent、rank、算术反解、non-exact input/output。
- Route：foreign oracle、duplicate key、相同 concrete input overlap、target/ABI mismatch、miss。
- Side effect：publish/lookup/miss 前后 primitive cache stats 不变。
- LLVM：显式编译两个 exact profile，lookup 后经 `RuntimeSession` 数值执行。
- Lifetime：table/cache 销毁后 looked-up value 仍保活 executable。

```bash
ctest --test-dir <build> --output-on-failure \
  -R 'shape_control_test|shape_specialization_test|shape_production_exact_test|shape_exact_dispatch_test|restricted_symbolic_shape_test'
```

## 禁止

- 第二个 shape evaluator 或 route key。
- lookup 中隐式编译、cache fill、bucket widening。
- 用一个“dynamic” bool 代替 applicability contract。
- route table 直接验证 backend-specific lowering 细节。
- 重开已关闭 exact key-space 问题而没有新反例。

完成标准：concrete inputs 经唯一 evaluator 和 identity builders 命中 caller-published variant；miss 无副作用且明确失败。
