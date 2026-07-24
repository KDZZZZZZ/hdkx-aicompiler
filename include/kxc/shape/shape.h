#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// =============================================================================
// experimental-v1 形状契约 / 规范格式
// -----------------------------------------------------------------------------
// - 非稳定 kxc::shape API：不承诺跨版本源兼容或 C++ 二进制 ABI。
// - 本头只描述“形状怎么表达、约束、求值”，不分配、不编译、不 launch。
// - 典型用法管线：
//     1) 用 DimExpr + Constraint + BindingSet 描述符号维与约束
//     2) 用 TensorShapeContract / ShapeProgram 描述 I/O 契约
//     3) ShapeProgram::Verify() 检查自洽
//     4) ShapeProgram::Evaluate(bindings) 得到 Concrete*（全 int64）
// - dtype、device、target、backend、kernel ABI 和 artifact/plan identity
//   均由 compiler/runtime 权威合同拥有，不属于 Shape。
// =============================================================================
namespace kxc::shape::experimental::v1 {

// 契约格式版本号：区分 ShapeProgram 规范格式代际。
inline constexpr uint32_t kShapeContractVersion = 1;

class BindingSet;

// ---------------------------------------------------------------------------
// DimExpr — 维度表达式 IR（常量 / 符号 / 算术）
// 用法（构造）：
//   auto N = DimExpr::Symbol("N");
//   auto c4 = DimExpr::Const(4);
//   auto n_plus_4 = DimExpr::Add({N, c4});          // 自动展平/排序/折叠
//   auto half = DimExpr::FloorDiv(N, 2);            // 除数必须 > 0
// 用法（求值）：
//   int64_t v = n_plus_4.Evaluate(BindingSet({{"N", 8}}));  // → 12
// 规则：常量维必须 >= 0（0 合法）；缺绑定、溢出 → 抛错 fail closed。
// ---------------------------------------------------------------------------
class DimExpr {
 public:
  enum class Kind { kConst, kSymbol, kAdd, kMul, kFloorDiv, kMin, kMax };

  // 非负常量维；负数构造即失败。
  static DimExpr Const(int64_t value);
  // 符号维；name 非空，在 BindingSet 中按名查找。
  static DimExpr Symbol(std::string name);
  // 多项加法：内部 flatten + 排序 + 常量折叠；传空/单元素也合法。
  static DimExpr Add(std::vector<DimExpr> terms);
  // 多项乘法：同上。
  static DimExpr Mul(std::vector<DimExpr> terms);
  // 向下取整除：positive_divisor 必须 > 0。
  static DimExpr FloorDiv(DimExpr dividend, int64_t positive_divisor);
  // 多项最小值 / 最大值。
  static DimExpr Min(std::vector<DimExpr> terms);
  static DimExpr Max(std::vector<DimExpr> terms);

  [[nodiscard]] Kind kind() const;  // 当前根节点种类
  // 稳定文本，便于诊断与测试断言；不要当跨版本持久化格式依赖。
  [[nodiscard]] std::string CanonicalString() const;
  // 用绑定求值为 int64；符号未绑定或中间溢出 → 失败。
  [[nodiscard]] int64_t Evaluate(const BindingSet& bindings) const;
  // 收集表达式中出现的符号名（去重后，顺序由实现决定/可测）。
  [[nodiscard]] std::vector<std::string> Symbols() const;
  [[nodiscard]] bool operator==(const DimExpr& other) const;
  [[nodiscard]] bool operator!=(const DimExpr& other) const { return !(*this == other); }

 private:
  struct Node;
  explicit DimExpr(std::shared_ptr<const Node> node);
  std::shared_ptr<const Node> node_;

  friend class ExactConstraintSolver;
};

// ---------------------------------------------------------------------------
// Binding — 单个符号赋值
// 用法：Binding{"N", 128}；value 必须非负（由 BindingSet 构造校验）。
// ---------------------------------------------------------------------------
struct Binding {
  std::string symbol;  // 符号名，对应 DimExpr::Symbol
  int64_t value;       // 非负具体值
};

// ---------------------------------------------------------------------------
// BindingSet — 有序、规范化后的符号绑定表
// 用法：
//   BindingSet b({{"N", 8}, {"C", 3}});
//   auto n = b.Find("N");           // optional<int64_t>
//   program.Evaluate(b);            // 交给 ShapeProgram / DimExpr
// 构造时会：去重校验、按 symbol 排序、拒绝负值/重复符号（失败则抛错）。
// 相等性比较完整结构，不只比 hash。
// ---------------------------------------------------------------------------
class BindingSet {
 public:
  // 默认空集；传入列表会规范化。重复 symbol 或 value<0 → 失败。
  explicit BindingSet(std::vector<Binding> bindings = {});

  // 只读遍历规范化后的绑定（已排序）。
  [[nodiscard]] const std::vector<Binding>& bindings() const noexcept;
  // 查找单个符号；不存在返回 nullopt（不抛错）。
  [[nodiscard]] std::optional<int64_t> Find(const std::string& symbol) const;
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const BindingSet& other) const noexcept;
  [[nodiscard]] bool operator!=(const BindingSet& other) const noexcept {
    return !(*this == other);
  }

 private:
  std::vector<Binding> bindings_;
};

// ---------------------------------------------------------------------------
// DeferredConstraintKind — 已规划但未实现的约束
// 用法：
//   if (!SupportsConstraint(DeferredConstraintKind::kSameRank)) { ... }
//   RequireConstraintSupport(...);  // 直接抛错，禁止静默当“已支持”
// ---------------------------------------------------------------------------
enum class DeferredConstraintKind { kSameRank, kLayoutCompatible };

// 查询该约束是否已实现；当前一律 false。
[[nodiscard]] bool SupportsConstraint(DeferredConstraintKind kind) noexcept;
// 要求必须支持；未实现则抛错（用于调用方显式声明依赖时 fail closed）。
void RequireConstraintSupport(DeferredConstraintKind kind);

// ---------------------------------------------------------------------------
// Constraint — 形状约束
// 用法：
//   auto c1 = Constraint::Eq(DimExpr::Symbol("N"), DimExpr::Const(8));
//   auto c2 = Constraint::Range(DimExpr::Symbol("N"), 1, 1024);
//   auto c3 = Constraint::DivisibleBy(DimExpr::Symbol("N"), 8);
//   auto c4 = Constraint::BroadcastCompatible(a, b);
// 放入 ShapeProgram 的 constraints；ExactConstraintSolver 主要从 Eq 推绑定。
// Range/DivisibleBy/Broadcast 用于校验，不用于“猜”未知符号值。
// ---------------------------------------------------------------------------
class Constraint {
 public:
  enum class Kind { kEq, kRange, kDivisibleBy, kBroadcastCompatible };

  // left == right（可推出 Symbol = 可求值表达式）。
  static Constraint Eq(DimExpr left, DimExpr right);
  // lower <= expression <= upper（含端点；用于校验）。
  static Constraint Range(DimExpr expression, int64_t lower, int64_t upper);
  // expression % divisor == 0；divisor 规则由实现校验。
  static Constraint DivisibleBy(DimExpr expression, int64_t divisor);
  // 广播兼容性检查（不推导具体绑定值）。
  static Constraint BroadcastCompatible(DimExpr left, DimExpr right);

  [[nodiscard]] Kind kind() const noexcept;
  [[nodiscard]] const DimExpr& left() const noexcept;
  // Eq/Broadcast 有 right；Range/DivisibleBy 可能为空。
  [[nodiscard]] const std::optional<DimExpr>& right() const noexcept;
  [[nodiscard]] int64_t lower() const noexcept;   // Range 用
  [[nodiscard]] int64_t upper() const noexcept;   // Range 用
  [[nodiscard]] int64_t divisor() const noexcept; // DivisibleBy 用
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

// ---------------------------------------------------------------------------
// ExactConstraintSolver — exact 绑定求解
// 用法：
//   BindingSet solved = ExactConstraintSolver::Solve(
//       {"N", "C"},                          // 声明符号全集
//       BindingSet({{"N", 8}}),              // 已知绑定
//       {Constraint::Eq(DimExpr::Symbol("C"), DimExpr::Const(3))});
// 行为：只从 Eq 且右侧（或可推侧）完全可求值时写入 Symbol；
//       未绑定、矛盾、多余绑定、负值、溢出 → 失败。
// 不要指望它根据 Range 上界“猜”一个 N。
// ---------------------------------------------------------------------------
class ExactConstraintSolver {
 public:
  // declared_symbols: ShapeProgram 声明的符号；initial_bindings: 调用方已知值；
  // constraints: 待应用约束。返回补全后的 BindingSet。
  [[nodiscard]] static BindingSet Solve(const std::vector<std::string>& declared_symbols,
                                        const BindingSet& initial_bindings,
                                        const std::vector<Constraint>& constraints);
};

// ---------------------------------------------------------------------------
// LogicalShape — 数学逻辑形状
// 用法：
//   LogicalShape log({DimExpr::Symbol("N"), DimExpr::Const(128)});
//   // 可选轴名，数量必须与 dimensions 一致（或默认全空）：
//   LogicalShape log2(dims, {std::string("batch"), std::string("hidden")});
// ---------------------------------------------------------------------------
class LogicalShape {
 public:
  // dimensions: 各轴 DimExpr；axis_names: 可选，大小为 0 或与 dimensions 相同。
  explicit LogicalShape(std::vector<DimExpr> dimensions,
                        std::vector<std::optional<std::string>> axis_names = {});

  [[nodiscard]] const std::vector<DimExpr>& dimensions() const noexcept;
  [[nodiscard]] const std::vector<std::optional<std::string>>& axis_names() const noexcept;

 private:
  std::vector<DimExpr> dimensions_;
  std::vector<std::optional<std::string>> axis_names_;
};

// ---------------------------------------------------------------------------
// PhysicalCapacity — 可容纳的物理元素范围。
// 布局、stride、alignment 和 memory scope 属于 Kernel/Runtime ABI。
// ---------------------------------------------------------------------------
class PhysicalCapacity {
 public:
  explicit PhysicalCapacity(std::vector<DimExpr> dimensions);

  [[nodiscard]] const std::vector<DimExpr>& dimensions() const noexcept;

 private:
  std::vector<DimExpr> dimensions_;
};

// ---------------------------------------------------------------------------
// ValidExtent — 本次调用有效范围
// 用法：ValidExtent valid(dims);
// 语义：运行时要求 valid <= logical <= physical（各轴对应比较）。
// exact specialization 进一步要求三者完全相等。
// ---------------------------------------------------------------------------
class ValidExtent {
 public:
  explicit ValidExtent(std::vector<DimExpr> dimensions);
  [[nodiscard]] const std::vector<DimExpr>& dimensions() const noexcept;

 private:
  std::vector<DimExpr> dimensions_;
};

// ---------------------------------------------------------------------------
// ConcreteTensorShapeContract — 已全部求值为 int64 的张量契约
// 用法：由 TensorShapeContract::Evaluate 或 ShapeProgram::Evaluate 得到；
//       读 .logical/.physical/.valid 做形状验证。
// ---------------------------------------------------------------------------
struct ConcreteTensorShapeContract {
  std::vector<int64_t> logical;
  std::vector<int64_t> physical;
  std::vector<int64_t> valid;
  std::vector<std::optional<std::string>> axis_names;

  [[nodiscard]] bool operator==(const ConcreteTensorShapeContract& other) const noexcept;
};

// ---------------------------------------------------------------------------
// TensorShapeContract — 符号化 logical / physical capacity / valid extent
// 用法：
//   TensorShapeContract tsc(logical, physical, valid);
//   auto concrete = tsc.Evaluate(bindings);  // → ConcreteTensorShapeContract
//   auto syms = tsc.Symbols();               // 收集用到的符号
// Evaluate 会检查 valid <= logical <= physical。
// ---------------------------------------------------------------------------
class TensorShapeContract {
 public:
  TensorShapeContract(LogicalShape logical, PhysicalCapacity physical,
                      ValidExtent valid);

  [[nodiscard]] const LogicalShape& logical() const noexcept;
  [[nodiscard]] const PhysicalCapacity& physical() const noexcept;
  [[nodiscard]] const ValidExtent& valid() const noexcept;
  // 用 bindings 求值；失败抛错（缺符号、约束冲突或容量不足）。
  [[nodiscard]] ConcreteTensorShapeContract Evaluate(const BindingSet& bindings) const;
  [[nodiscard]] std::vector<std::string> Symbols() const;

 private:
  LogicalShape logical_;
  PhysicalCapacity physical_;
  ValidExtent valid_;
};

// ---------------------------------------------------------------------------
// NamedTensorContract — 带名字的符号化 I/O 槽
// 用法：NamedTensorContract{"x", tsc}；name 在 ShapeProgram 内唯一。
// ---------------------------------------------------------------------------
struct NamedTensorContract {
  std::string name;
  TensorShapeContract contract;
};

// ---------------------------------------------------------------------------
// NamedConcreteTensorContract — 带名字的已求值 I/O 槽
// 用法：读 EvaluatedShapeProgram::inputs/outputs 中的元素。
// ---------------------------------------------------------------------------
struct NamedConcreteTensorContract {
  std::string name;
  ConcreteTensorShapeContract contract;
};

// ---------------------------------------------------------------------------
// EvaluatedShapeProgram — ShapeProgram::Evaluate 的完整结果
// 用法：
//   auto ev = program.Evaluate(bindings);
//   // ev.bindings  — 可能被 solver 补全后的绑定
//   // ev.inputs/outputs — 与 program 声明同序的具体契约
// ---------------------------------------------------------------------------
struct EvaluatedShapeProgram {
  BindingSet bindings;
  std::vector<NamedConcreteTensorContract> inputs;
  std::vector<NamedConcreteTensorContract> outputs;
};

// ---------------------------------------------------------------------------
// ShapeProgram — 纯形状程序（本头最顶层“可执行”入口，但仍不 launch）
// 用法：
//   ShapeProgram prog(
//       {"N"},                                    // 声明符号
//       {NamedTensorContract{"x", tsc_in}},       // 输入
//       {NamedTensorContract{"y", tsc_out}},      // 输出
//       {Constraint::Eq(...)});                   // 可选约束
//   prog.Verify();                                // 结构自洽检查
//   auto ev = prog.Evaluate(BindingSet({{"N", 8}}));
// 规则：
//   - Verify: 名称唯一、rank 一致、符号引用 ⊆ 声明集、排序确定性等
//   - Evaluate: 只算 logical/physical/valid，不分配内存、不编译、不 launch
//   - 失败一律 fail closed（抛错），无 silent 默认值
// ---------------------------------------------------------------------------
class ShapeProgram {
 public:
  // declared_symbols: 本程序允许出现的符号全集（可空，表示无符号、全常量）；
  // inputs/outputs: 命名契约，name 不得冲突；
  // constraints: 可选，参与求解与校验。
  ShapeProgram(std::vector<std::string> declared_symbols,
               std::vector<NamedTensorContract> inputs,
               std::vector<NamedTensorContract> outputs,
               std::vector<Constraint> constraints = {});

  [[nodiscard]] const std::vector<std::string>& declared_symbols() const noexcept;
  [[nodiscard]] const std::vector<NamedTensorContract>& inputs() const noexcept;
  [[nodiscard]] const std::vector<NamedTensorContract>& outputs() const noexcept;
  [[nodiscard]] const std::vector<Constraint>& constraints() const noexcept;
  // 构造后、Evaluate 前建议调用；也可由 Evaluate 路径内部再验。
  void Verify() const;
  // 求值得到具体 I/O；bindings 可只含部分符号，其余由 Eq 约束推导。
  [[nodiscard]] EvaluatedShapeProgram Evaluate(const BindingSet& bindings) const;
  // 稳定字符串，用于模板 content 身份等。
  [[nodiscard]] std::string CanonicalString() const;
  [[nodiscard]] bool operator==(const ShapeProgram& other) const;

 private:
  std::vector<std::string> declared_symbols_;
  std::vector<NamedTensorContract> inputs_;
  std::vector<NamedTensorContract> outputs_;
  std::vector<Constraint> constraints_;
};

}  // namespace kxc::shape::experimental::v1
