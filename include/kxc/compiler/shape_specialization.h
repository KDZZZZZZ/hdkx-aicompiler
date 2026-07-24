#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kxc/compiler/identity.h"
#include "kxc/shape/shape.h"

// =============================================================================
// 轨 02 — exact 特化契约（experimental-v1）
// -----------------------------------------------------------------------------
// 在纯 ShapeProgram 之上：图模板 → exact profile → unit 特化请求。
// 所有 graph/unit/profile/artifact/plan identity 均来自 compiler/identity.h。
// 典型用法：
//   GraphTemplate tmpl(key, shape_program, ordered_units);
//   tmpl.Verify();
//   ExactOracle oracle = InstantiateExactProfile(tmpl, bindings);
//   auto reqs = MakeExactSpecializationRequests(tmpl, oracle);
//   // reqs[i].unit_semantic_key → compiler；call_locator 只做 plan 路由
// 生产桥见 shape_exact.h；本头仍不编译、不 launch。
// =============================================================================
namespace kxc::api::experimental::shape_specialization::v1 {

using BindingSet = kxc::shape::experimental::v1::BindingSet;
using Binding = kxc::shape::experimental::v1::Binding;
using ConcreteTensorShapeContract =
    kxc::shape::experimental::v1::ConcreteTensorShapeContract;
using Constraint = kxc::shape::experimental::v1::Constraint;
using DimExpr = kxc::shape::experimental::v1::DimExpr;
using ExactConstraintSolver =
    kxc::shape::experimental::v1::ExactConstraintSolver;
using EvaluatedShapeProgram =
    kxc::shape::experimental::v1::EvaluatedShapeProgram;
using LogicalShape = kxc::shape::experimental::v1::LogicalShape;
using NamedConcreteTensorContract =
    kxc::shape::experimental::v1::NamedConcreteTensorContract;
using NamedTensorContract =
    kxc::shape::experimental::v1::NamedTensorContract;
using PhysicalCapacity = kxc::shape::experimental::v1::PhysicalCapacity;
using ShapeProgram = kxc::shape::experimental::v1::ShapeProgram;
using TensorShapeContract =
    kxc::shape::experimental::v1::TensorShapeContract;
using ValidExtent = kxc::shape::experimental::v1::ValidExtent;

class ExactOracle;
struct UnitSpecializationRequest;

inline constexpr uint32_t kExactSpecializationContractVersion = 1;
inline constexpr uint32_t kShapeProfileAbiVersion = 1;

// ---------------------------------------------------------------------------
// GraphLocalCallLocator — 图内 call 定位（路由/诊断 only）
// 用法：GraphLocalCallLocator loc("call.3"); 或编译器生成的稳定字符串
// 禁止：写入 PrimitiveArtifactKey / 跨图共享 artifact 身份
// ---------------------------------------------------------------------------
class GraphLocalCallLocator {
 public:
  explicit GraphLocalCallLocator(std::string value);

  [[nodiscard]] const std::string& value() const noexcept;
  [[nodiscard]] bool operator==(const GraphLocalCallLocator& other) const noexcept;

 private:
  std::string value_;
};

// ---------------------------------------------------------------------------
// UnitSkeleton — 模板中一个有序 unit 的骨架
// 用法：填入 GraphTemplate 的 ordered_units
//   UnitSkeleton{loc, semantic_key, {"x","w"}, {"y"}}
// input/output_value_names 必须对应 ShapeProgram 中的 named 契约
// ---------------------------------------------------------------------------
struct UnitSkeleton {
  GraphLocalCallLocator call_locator;
  UnitSemanticKey semantic_key;
  std::vector<std::string> input_value_names;
  std::vector<std::string> output_value_names;
};

// ---------------------------------------------------------------------------
// GraphTemplate — compiler graph semantics + ShapeProgram + 有序 unit
// 用法：
//   GraphTemplate tmpl(key, program, units);
//   tmpl.Verify();                      // producer/consumer、ABI 一致
// ---------------------------------------------------------------------------
class GraphTemplate {
 public:
  GraphTemplate(GraphSemanticKey key, ShapeProgram shape_program,
                std::vector<UnitSkeleton> ordered_units);

  [[nodiscard]] const GraphSemanticKey& key() const noexcept;
  [[nodiscard]] const ShapeProgram& shape_program() const noexcept;
  [[nodiscard]] const std::vector<UnitSkeleton>& ordered_units() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  void Verify() const;  // 结构非法则抛错

 private:
  GraphSemanticKey key_;
  ShapeProgram shape_program_;
  std::vector<UnitSkeleton> ordered_units_;
};

// ---------------------------------------------------------------------------
// ExactShapeProfile — 绑定后的全部 named concrete 值
// 用法：不要直接构造；从 ExactOracle::profile() 读取
// 语义：exact 下 logical == physical == valid
// ---------------------------------------------------------------------------
class ExactShapeProfile {
 public:
  [[nodiscard]] const ShapeProfileKey& key() const noexcept;
  [[nodiscard]] const BindingSet& bindings() const noexcept;
  [[nodiscard]] const std::string& policy_id() const noexcept;
  [[nodiscard]] uint32_t shape_abi_version() const noexcept;
  [[nodiscard]] const std::vector<NamedConcreteTensorContract>& values() const noexcept;
  // 按 name 取单个值；不存在抛错
  [[nodiscard]] const NamedConcreteTensorContract& Value(const std::string& name) const;

 private:
  ExactShapeProfile(ShapeProfileKey key, BindingSet bindings,
                    std::string policy_id, uint32_t shape_abi_version,
                    std::vector<NamedConcreteTensorContract> values);

  ShapeProfileKey key_;
  BindingSet bindings_;
  std::string policy_id_;
  uint32_t shape_abi_version_;
  std::vector<NamedConcreteTensorContract> values_;

  friend class ExactOracle;
  friend ExactOracle InstantiateExactProfile(const GraphTemplate&, const BindingSet&);
};

// ---------------------------------------------------------------------------
// ExactOracle — exact 适用性证明（唯一合法 proof 句柄）
// 用法：
//   ExactOracle o = InstantiateExactProfile(tmpl, bindings);
//   const auto& profile = o.profile();
// 消费方（MakeExactSpecializationRequests / production adapter）必须持有 oracle，
// 不能只传裸 BindingSet 冒充已证明
// ---------------------------------------------------------------------------
class ExactOracle {
 public:
  [[nodiscard]] const ExactShapeProfile& profile() const noexcept;

 private:
  explicit ExactOracle(ExactShapeProfile profile);

  ExactShapeProfile profile_;

  friend ExactOracle InstantiateExactProfile(const GraphTemplate&, const BindingSet&);
};

// 绑定实例化 exact profile；失败 fail closed（未绑定、不等式、layout 等）
[[nodiscard]] ExactOracle InstantiateExactProfile(const GraphTemplate& graph_template,
                                                   const BindingSet& bindings);

// ---------------------------------------------------------------------------
// UnitSignatureDigest — 有序 I/O boundary 摘要
// 用法：
//   auto d = UnitSignatureDigest::ForExactContracts(ins, outs);
//   MatchesExactSignatureDigest(d, ins2, outs2);
// ---------------------------------------------------------------------------
class UnitSignatureDigest {
 public:
  [[nodiscard]] static UnitSignatureDigest ForExactContracts(
      const std::vector<ConcreteTensorShapeContract>& ordered_inputs,
      const std::vector<ConcreteTensorShapeContract>& ordered_outputs);
  [[nodiscard]] const std::string& value() const noexcept;
  [[nodiscard]] bool operator==(const UnitSignatureDigest& other) const noexcept;

 private:
  explicit UnitSignatureDigest(std::string value);

  std::string value_;

  friend std::vector<UnitSpecializationRequest> MakeExactSpecializationRequests(
      const GraphTemplate&, const ExactOracle&);
};

// ---------------------------------------------------------------------------
// UnitSpecializationRequest — 一次 unit 特化请求
// 用法：遍历 MakeExactSpecializationRequests 的返回值
//   req.call_locator / ordered_call_index → plan 路由
//   req.unit_semantic_key → compiler identity / primitive cache
// ---------------------------------------------------------------------------
struct UnitSpecializationRequest {
  size_t ordered_call_index;
  GraphLocalCallLocator call_locator;
  ShapeProfileKey shape_profile_key;
  UnitSemanticKey unit_semantic_key;
  UnitSignatureDigest signature_digest;
  std::vector<ConcreteTensorShapeContract> ordered_inputs;
  std::vector<ConcreteTensorShapeContract> ordered_outputs;
};

// 从模板 + oracle 生成有序 exact 请求（每个 unit 一条）
[[nodiscard]] std::vector<UnitSpecializationRequest> MakeExactSpecializationRequests(
    const GraphTemplate& graph_template, const ExactOracle& oracle);
// 对照 digest 与当前 I/O 是否仍 exact 匹配
[[nodiscard]] bool MatchesExactSignatureDigest(
    const UnitSignatureDigest& digest,
    const std::vector<ConcreteTensorShapeContract>& ordered_inputs,
    const std::vector<ConcreteTensorShapeContract>& ordered_outputs);

}  // namespace kxc::api::experimental::shape_specialization::v1
