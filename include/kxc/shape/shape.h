#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kxc::shape::experimental::v1 {

// This isolated contract is intentionally not a stable kxc::shape public ABI.
inline constexpr uint32_t kShapeContractVersion = 1;
inline constexpr uint32_t kShapeAbiVersion = 1;

class BindingSet;
class GraphTemplate;

enum class DataType : uint8_t { kFloat16, kFloat32 };
enum class DeviceKind : uint8_t { kCpu, kCuda };
enum class TargetKind : uint8_t { kX86_64, kAArch64, kNvptx64 };
enum class BackendKind : uint8_t { kNative, kLlvm, kCuda };

class DeviceDescriptor {
 public:
  DeviceDescriptor(DeviceKind kind, uint32_t id);

  [[nodiscard]] DeviceKind kind() const noexcept;
  [[nodiscard]] uint32_t id() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] bool operator==(const DeviceDescriptor& other) const noexcept;

 private:
  DeviceKind kind_;
  uint32_t id_;
};

class TargetBackendAbiDescriptor {
 public:
  TargetBackendAbiDescriptor(TargetKind target, BackendKind backend,
                             uint32_t abi_version);

  [[nodiscard]] TargetKind target() const noexcept;
  [[nodiscard]] BackendKind backend() const noexcept;
  [[nodiscard]] uint32_t abi_version() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] bool operator==(const TargetBackendAbiDescriptor& other) const noexcept;

 private:
  TargetKind target_;
  BackendKind backend_;
  uint32_t abi_version_;
};

class TensorAbiDescriptor {
 public:
  TensorAbiDescriptor(DataType dtype, DeviceDescriptor device,
                      TargetBackendAbiDescriptor target_backend_abi);

  [[nodiscard]] DataType dtype() const noexcept;
  [[nodiscard]] const DeviceDescriptor& device() const noexcept;
  [[nodiscard]] const TargetBackendAbiDescriptor& target_backend_abi() const noexcept;
  [[nodiscard]] uint32_t element_bytes() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] bool operator==(const TensorAbiDescriptor& other) const noexcept;

 private:
  DataType dtype_;
  DeviceDescriptor device_;
  TargetBackendAbiDescriptor target_backend_abi_;
};

class DimExpr {
 public:
  enum class Kind { kConst, kSymbol, kAdd, kMul, kFloorDiv, kMin, kMax };

  static DimExpr Const(int64_t value);
  static DimExpr Symbol(std::string name);
  static DimExpr Add(std::vector<DimExpr> terms);
  static DimExpr Mul(std::vector<DimExpr> terms);
  static DimExpr FloorDiv(DimExpr dividend, int64_t positive_divisor);
  static DimExpr Min(std::vector<DimExpr> terms);
  static DimExpr Max(std::vector<DimExpr> terms);

  [[nodiscard]] Kind kind() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] int64_t Evaluate(const BindingSet& bindings) const;
  [[nodiscard]] std::vector<std::string> Symbols() const;
  [[nodiscard]] bool operator==(const DimExpr& other) const;
  [[nodiscard]] bool operator!=(const DimExpr& other) const { return !(*this == other); }

 private:
  struct Node;
  explicit DimExpr(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_;

  friend class ExactConstraintSolver;
};

struct Binding {
  std::string symbol;
  int64_t value;
};

class BindingSet {
 public:
  explicit BindingSet(std::vector<Binding> bindings = {});

  [[nodiscard]] const std::vector<Binding>& bindings() const noexcept;
  [[nodiscard]] std::optional<int64_t> Find(const std::string& symbol) const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const BindingSet& other) const noexcept;
  [[nodiscard]] bool operator!=(const BindingSet& other) const noexcept {
    return !(*this == other);
  }

 private:
  std::vector<Binding> bindings_;
};

enum class DeferredConstraintKind { kSameRank, kLayoutCompatible };

[[nodiscard]] bool SupportsConstraint(DeferredConstraintKind kind) noexcept;
void RequireConstraintSupport(DeferredConstraintKind kind);

class Constraint {
 public:
  enum class Kind { kEq, kRange, kDivisibleBy, kBroadcastCompatible };

  static Constraint Eq(DimExpr left, DimExpr right);
  static Constraint Range(DimExpr expression, int64_t lower, int64_t upper);
  static Constraint DivisibleBy(DimExpr expression, int64_t divisor);
  static Constraint BroadcastCompatible(DimExpr left, DimExpr right);

  [[nodiscard]] Kind kind() const noexcept;
  [[nodiscard]] const DimExpr& left() const noexcept;
  [[nodiscard]] const std::optional<DimExpr>& right() const noexcept;
  [[nodiscard]] int64_t lower() const noexcept;
  [[nodiscard]] int64_t upper() const noexcept;
  [[nodiscard]] int64_t divisor() const noexcept;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] std::vector<std::string> Symbols() const;

 private:
  Constraint(Kind kind, DimExpr left, std::optional<DimExpr> right,
             int64_t lower, int64_t upper, int64_t divisor);

  Kind kind_;
  DimExpr left_;
  std::optional<DimExpr> right_;
  int64_t lower_;
  int64_t upper_;
  int64_t divisor_;
};

class ExactConstraintSolver {
 public:
  [[nodiscard]] static BindingSet Solve(const std::vector<std::string>& declared_symbols,
                                        const BindingSet& initial_bindings,
                                        const std::vector<Constraint>& constraints);
};

class LogicalShape {
 public:
  explicit LogicalShape(std::vector<DimExpr> dimensions,
                        std::vector<std::optional<std::string>> axis_names = {});

  [[nodiscard]] const std::vector<DimExpr>& dimensions() const noexcept;
  [[nodiscard]] const std::vector<std::optional<std::string>>& axis_names() const noexcept;

 private:
  std::vector<DimExpr> dimensions_;
  std::vector<std::optional<std::string>> axis_names_;
};

class PhysicalShape {
 public:
  explicit PhysicalShape(std::vector<DimExpr> capacity,
                         std::optional<std::vector<DimExpr>> strides = std::nullopt,
                         std::string layout = "contiguous.row_major",
                         int64_t alignment = 1,
                         std::string memory_scope = "default");

  [[nodiscard]] const std::vector<DimExpr>& capacity() const noexcept;
  [[nodiscard]] const std::optional<std::vector<DimExpr>>& strides() const noexcept;
  [[nodiscard]] const std::string& layout() const noexcept;
  [[nodiscard]] int64_t alignment() const noexcept;
  [[nodiscard]] const std::string& memory_scope() const noexcept;

 private:
  std::vector<DimExpr> capacity_;
  std::optional<std::vector<DimExpr>> strides_;
  std::string layout_;
  int64_t alignment_;
  std::string memory_scope_;
};

class ValidExtent {
 public:
  explicit ValidExtent(std::vector<DimExpr> dimensions);
  [[nodiscard]] const std::vector<DimExpr>& dimensions() const noexcept;

 private:
  std::vector<DimExpr> dimensions_;
};

struct ConcreteTensorShapeContract {
  std::vector<int64_t> logical;
  std::vector<int64_t> physical;
  std::vector<int64_t> valid;
  std::vector<int64_t> strides;
  std::vector<std::optional<std::string>> axis_names;
  std::string layout;
  int64_t alignment;
  std::string memory_scope;
  TensorAbiDescriptor abi;

  [[nodiscard]] bool operator==(const ConcreteTensorShapeContract& other) const noexcept;
};

class TensorShapeContract {
 public:
  TensorShapeContract(LogicalShape logical, PhysicalShape physical, ValidExtent valid,
                      TensorAbiDescriptor abi);

  [[nodiscard]] const LogicalShape& logical() const noexcept;
  [[nodiscard]] const PhysicalShape& physical() const noexcept;
  [[nodiscard]] const ValidExtent& valid() const noexcept;
  [[nodiscard]] const TensorAbiDescriptor& abi() const noexcept;
  [[nodiscard]] ConcreteTensorShapeContract Evaluate(const BindingSet& bindings) const;
  [[nodiscard]] std::vector<std::string> Symbols() const;

 private:
  LogicalShape logical_;
  PhysicalShape physical_;
  ValidExtent valid_;
  TensorAbiDescriptor abi_;
};

struct NamedTensorContract {
  std::string name;
  TensorShapeContract contract;
};

struct NamedConcreteTensorContract {
  std::string name;
  ConcreteTensorShapeContract contract;
};

struct EvaluatedShapeProgram {
  BindingSet bindings;
  std::vector<NamedConcreteTensorContract> inputs;
  std::vector<NamedConcreteTensorContract> outputs;
};

class GraphTemplateKey {
 public:
  GraphTemplateKey(uint32_t version, std::string graph_semantic_fingerprint,
                   std::string pipeline_fingerprint,
                   std::string capability_fingerprint,
                   std::string partition_fingerprint,
                   TargetBackendAbiDescriptor target_backend_abi);

  [[nodiscard]] uint32_t version() const noexcept;
  [[nodiscard]] const std::string& graph_semantic_fingerprint() const noexcept;
  [[nodiscard]] const std::string& pipeline_fingerprint() const noexcept;
  [[nodiscard]] const std::string& capability_fingerprint() const noexcept;
  [[nodiscard]] const std::string& partition_fingerprint() const noexcept;
  [[nodiscard]] const TargetBackendAbiDescriptor& target_backend_abi() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const GraphTemplateKey& other) const noexcept;

 private:
  uint32_t version_;
  std::string graph_semantic_fingerprint_;
  std::string pipeline_fingerprint_;
  std::string capability_fingerprint_;
  std::string partition_fingerprint_;
  TargetBackendAbiDescriptor target_backend_abi_;
};

// Opaque, full-content identity minted only from an immutable GraphTemplate.
class GraphTemplateContentKey {
 public:
  [[nodiscard]] const std::string& CanonicalBytes() const noexcept;
  [[nodiscard]] bool operator==(const GraphTemplateContentKey& other) const noexcept;

 private:
  explicit GraphTemplateContentKey(std::string canonical_bytes);

  std::string canonical_bytes_;
  friend class GraphTemplate;
};

class ShapeProfileKey {
 public:
  ShapeProfileKey(GraphTemplateKey graph_template,
                  GraphTemplateContentKey graph_template_content,
                  BindingSet bindings, std::string policy_id,
                  uint32_t shape_abi_version);

  [[nodiscard]] const GraphTemplateKey& graph_template() const noexcept;
  [[nodiscard]] const GraphTemplateContentKey& graph_template_content() const noexcept;
  [[nodiscard]] const BindingSet& bindings() const noexcept;
  [[nodiscard]] const std::string& policy_id() const noexcept;
  [[nodiscard]] uint32_t shape_abi_version() const noexcept;
  [[nodiscard]] std::string CanonicalBytes() const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const ShapeProfileKey& other) const noexcept;

 private:
  GraphTemplateKey graph_template_;
  GraphTemplateContentKey graph_template_content_;
  BindingSet bindings_;
  std::string policy_id_;
  uint32_t shape_abi_version_;
};

class ShapeProgram {
 public:
  ShapeProgram(std::vector<std::string> declared_symbols,
               std::vector<NamedTensorContract> inputs,
               std::vector<NamedTensorContract> outputs,
               std::vector<Constraint> constraints = {});

  [[nodiscard]] const std::vector<std::string>& declared_symbols() const noexcept;
  [[nodiscard]] const std::vector<NamedTensorContract>& inputs() const noexcept;
  [[nodiscard]] const std::vector<NamedTensorContract>& outputs() const noexcept;
  [[nodiscard]] const std::vector<Constraint>& constraints() const noexcept;
  void Verify() const;
  [[nodiscard]] EvaluatedShapeProgram Evaluate(const BindingSet& bindings) const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const ShapeProgram& other) const;

 private:
  std::vector<std::string> declared_symbols_;
  std::vector<NamedTensorContract> inputs_;
  std::vector<NamedTensorContract> outputs_;
  std::vector<Constraint> constraints_;
};

}  // namespace kxc::shape::experimental::v1
