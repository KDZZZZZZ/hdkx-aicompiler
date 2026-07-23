#include "kxc/shape/shape.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace kxc::shape {
namespace {

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("shape: " + message);
}

int64_t CheckedAdd(int64_t left, int64_t right) {
  if (right > std::numeric_limits<int64_t>::max() - left) {
    Invalid("dimension addition overflow");
  }
  return left + right;
}

int64_t CheckedMul(int64_t left, int64_t right) {
  if (left != 0 && right > std::numeric_limits<int64_t>::max() / left) {
    Invalid("dimension multiplication overflow");
  }
  return left * right;
}

std::string Field(std::string_view value) {
  return std::to_string(value.size()) + ":" + std::string(value);
}

void AppendU64(std::string* bytes, uint64_t value) {
  for (int shift = 0; shift != 64; shift += 8) {
    bytes->push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

void AppendField(std::string* bytes, std::string_view value) {
  AppendU64(bytes, value.size());
  bytes->append(value.data(), value.size());
}

std::string Hex(const std::string& bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    result.push_back(kDigits[byte >> 4U]);
    result.push_back(kDigits[byte & 0xfU]);
  }
  return result;
}

void CheckName(const std::string& value, const char* label) {
  if (value.empty()) {
    Invalid(std::string(label) + " must not be empty");
  }
}

std::vector<std::string> SortedUnique(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

}  // namespace

struct DimExpr::Node {
  Kind kind;
  int64_t value = 0;
  std::string symbol;
  std::vector<DimExpr> children;
};

DimExpr::DimExpr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

DimExpr DimExpr::Const(int64_t value) {
  if (value < 0) {
    Invalid("dimension constants must be nonnegative");
  }
  auto node = std::make_shared<Node>();
  node->kind = Kind::kConst;
  node->value = value;
  return DimExpr(std::move(node));
}

DimExpr DimExpr::Symbol(std::string name) {
  CheckName(name, "symbol name");
  auto node = std::make_shared<Node>();
  node->kind = Kind::kSymbol;
  node->symbol = std::move(name);
  return DimExpr(std::move(node));
}

DimExpr DimExpr::Add(std::vector<DimExpr> terms) {
  std::vector<DimExpr> flattened;
  for (const DimExpr& term : terms) {
    if (term.kind() == Kind::kAdd) {
      flattened.insert(flattened.end(), term.node_->children.begin(), term.node_->children.end());
    } else {
      flattened.push_back(term);
    }
  }
  int64_t constant = 0;
  std::vector<DimExpr> result;
  for (const DimExpr& term : flattened) {
    if (term.kind() == Kind::kConst) {
      constant = CheckedAdd(constant, term.node_->value);
    } else {
      result.push_back(term);
    }
  }
  if (constant != 0) {
    result.push_back(Const(constant));
  }
  if (result.empty()) {
    return Const(0);
  }
  std::sort(result.begin(), result.end(), [](const DimExpr& a, const DimExpr& b) {
    return a.CanonicalString() < b.CanonicalString();
  });
  if (result.size() == 1) {
    return result.front();
  }
  auto node = std::make_shared<Node>();
  node->kind = Kind::kAdd;
  node->children = std::move(result);
  return DimExpr(std::move(node));
}

DimExpr DimExpr::Mul(std::vector<DimExpr> terms) {
  std::vector<DimExpr> flattened;
  for (const DimExpr& term : terms) {
    if (term.kind() == Kind::kMul) {
      flattened.insert(flattened.end(), term.node_->children.begin(), term.node_->children.end());
    } else {
      flattened.push_back(term);
    }
  }
  for (const DimExpr& term : flattened) {
    if (term.kind() == Kind::kConst && term.node_->value == 0) {
      return Const(0);
    }
  }
  int64_t constant = 1;
  std::vector<DimExpr> result;
  for (const DimExpr& term : flattened) {
    if (term.kind() == Kind::kConst) {
      constant = CheckedMul(constant, term.node_->value);
    } else {
      result.push_back(term);
    }
  }
  if (constant != 1 || result.empty()) {
    result.push_back(Const(constant));
  }
  std::sort(result.begin(), result.end(), [](const DimExpr& a, const DimExpr& b) {
    return a.CanonicalString() < b.CanonicalString();
  });
  if (result.size() == 1) {
    return result.front();
  }
  auto node = std::make_shared<Node>();
  node->kind = Kind::kMul;
  node->children = std::move(result);
  return DimExpr(std::move(node));
}

DimExpr DimExpr::FloorDiv(DimExpr dividend, int64_t positive_divisor) {
  if (positive_divisor <= 0) {
    Invalid("FloorDiv divisor must be positive");
  }
  if (dividend.kind() == Kind::kConst) {
    return Const(dividend.node_->value / positive_divisor);
  }
  if (positive_divisor == 1) {
    return dividend;
  }
  auto node = std::make_shared<Node>();
  node->kind = Kind::kFloorDiv;
  node->value = positive_divisor;
  node->children = {std::move(dividend)};
  return DimExpr(std::move(node));
}

DimExpr DimExpr::Min(std::vector<DimExpr> terms) {
  if (terms.empty()) {
    Invalid("Min requires at least one term");
  }
  std::vector<DimExpr> flattened;
  for (const DimExpr& term : terms) {
    if (term.kind() == Kind::kMin) {
      flattened.insert(flattened.end(), term.node_->children.begin(), term.node_->children.end());
    } else {
      flattened.push_back(term);
    }
  }
  int64_t constant = std::numeric_limits<int64_t>::max();
  bool has_constant = false;
  std::vector<DimExpr> result;
  for (const DimExpr& term : flattened) {
    if (term.kind() == Kind::kConst) {
      constant = has_constant ? std::min(constant, term.node_->value) : term.node_->value;
      has_constant = true;
    } else {
      result.push_back(term);
    }
  }
  if (has_constant && constant != std::numeric_limits<int64_t>::max()) {
    result.push_back(Const(constant));
  }
  if (result.empty()) {
    result.push_back(Const(constant));
  }
  std::sort(result.begin(), result.end(), [](const DimExpr& a, const DimExpr& b) {
    return a.CanonicalString() < b.CanonicalString();
  });
  result.erase(std::unique(result.begin(), result.end(), [](const DimExpr& a, const DimExpr& b) {
                 return a == b;
               }), result.end());
  if (result.size() == 1) {
    return result.front();
  }
  auto node = std::make_shared<Node>();
  node->kind = Kind::kMin;
  node->children = std::move(result);
  return DimExpr(std::move(node));
}

DimExpr DimExpr::Max(std::vector<DimExpr> terms) {
  if (terms.empty()) {
    Invalid("Max requires at least one term");
  }
  std::vector<DimExpr> flattened;
  for (const DimExpr& term : terms) {
    if (term.kind() == Kind::kMax) {
      flattened.insert(flattened.end(), term.node_->children.begin(), term.node_->children.end());
    } else {
      flattened.push_back(term);
    }
  }
  int64_t constant = 0;
  bool has_constant = false;
  std::vector<DimExpr> result;
  for (const DimExpr& term : flattened) {
    if (term.kind() == Kind::kConst) {
      constant = has_constant ? std::max(constant, term.node_->value) : term.node_->value;
      has_constant = true;
    } else {
      result.push_back(term);
    }
  }
  if (has_constant && constant != 0) {
    result.push_back(Const(constant));
  }
  if (result.empty()) {
    result.push_back(Const(constant));
  }
  std::sort(result.begin(), result.end(), [](const DimExpr& a, const DimExpr& b) {
    return a.CanonicalString() < b.CanonicalString();
  });
  result.erase(std::unique(result.begin(), result.end(), [](const DimExpr& a, const DimExpr& b) {
                 return a == b;
               }), result.end());
  if (result.size() == 1) {
    return result.front();
  }
  auto node = std::make_shared<Node>();
  node->kind = Kind::kMax;
  node->children = std::move(result);
  return DimExpr(std::move(node));
}

DimExpr::Kind DimExpr::kind() const { return node_->kind; }

std::string DimExpr::CanonicalString() const {
  switch (node_->kind) {
    case Kind::kConst:
      return "C(" + std::to_string(node_->value) + ")";
    case Kind::kSymbol:
      return "S(" + Field(node_->symbol) + ")";
    case Kind::kFloorDiv:
      return "D(" + node_->children.front().CanonicalString() + "," +
             std::to_string(node_->value) + ")";
    default:
      break;
  }
  char tag = 'A';
  if (node_->kind == Kind::kMul) tag = 'M';
  if (node_->kind == Kind::kMin) tag = 'N';
  if (node_->kind == Kind::kMax) tag = 'X';
  std::string result(1, tag);
  result += "(";
  for (const DimExpr& child : node_->children) {
    result += child.CanonicalString();
    result += ";";
  }
  return result + ")";
}

int64_t DimExpr::Evaluate(const BindingSet& bindings) const {
  std::function<int64_t(const DimExpr&)> evaluate = [&evaluate, &bindings](const DimExpr& expr) {
    const Node& node = *expr.node_;
    switch (node.kind) {
      case Kind::kConst:
        return node.value;
      case Kind::kSymbol: {
        const std::optional<int64_t> value = bindings.Find(node.symbol);
        if (!value.has_value()) {
          Invalid("unbound symbol '" + node.symbol + "'");
        }
        return *value;
      }
      case Kind::kAdd: {
        int64_t result = 0;
        for (const DimExpr& child : node.children) result = CheckedAdd(result, evaluate(child));
        return result;
      }
      case Kind::kMul: {
        int64_t result = 1;
        for (const DimExpr& child : node.children) result = CheckedMul(result, evaluate(child));
        return result;
      }
      case Kind::kFloorDiv:
        return evaluate(node.children.front()) / node.value;
      case Kind::kMin: {
        int64_t result = evaluate(node.children.front());
        for (size_t i = 1; i < node.children.size(); ++i) result = std::min(result, evaluate(node.children[i]));
        return result;
      }
      case Kind::kMax: {
        int64_t result = evaluate(node.children.front());
        for (size_t i = 1; i < node.children.size(); ++i) result = std::max(result, evaluate(node.children[i]));
        return result;
      }
    }
    Invalid("invalid dimension expression");
  };
  return evaluate(*this);
}

std::vector<std::string> DimExpr::Symbols() const {
  std::vector<std::string> result;
  std::function<void(const DimExpr&)> collect = [&collect, &result](const DimExpr& expr) {
    if (expr.node_->kind == Kind::kSymbol) result.push_back(expr.node_->symbol);
    for (const DimExpr& child : expr.node_->children) collect(child);
  };
  collect(*this);
  return SortedUnique(std::move(result));
}

bool DimExpr::operator==(const DimExpr& other) const {
  return CanonicalString() == other.CanonicalString();
}

BindingSet::BindingSet(std::vector<Binding> bindings) : bindings_(std::move(bindings)) {
  for (const Binding& binding : bindings_) {
    CheckName(binding.symbol, "binding symbol");
    if (binding.value < 0) Invalid("binding values must be nonnegative");
  }
  std::sort(bindings_.begin(), bindings_.end(), [](const Binding& a, const Binding& b) {
    return a.symbol < b.symbol;
  });
  for (size_t i = 1; i < bindings_.size(); ++i) {
    if (bindings_[i - 1].symbol == bindings_[i].symbol) Invalid("duplicate binding for '" + bindings_[i].symbol + "'");
  }
}

const std::vector<Binding>& BindingSet::bindings() const noexcept { return bindings_; }

std::optional<int64_t> BindingSet::Find(const std::string& symbol) const {
  const auto it = std::lower_bound(bindings_.begin(), bindings_.end(), symbol,
                                   [](const Binding& binding, const std::string& key) {
                                     return binding.symbol < key;
                                   });
  if (it == bindings_.end() || it->symbol != symbol) return std::nullopt;
  return it->value;
}

std::string BindingSet::CanonicalString() const {
  std::string result = "B(";
  for (const Binding& binding : bindings_) {
    result += Field(binding.symbol) + "=" + std::to_string(binding.value) + ";";
  }
  return result + ")";
}

bool BindingSet::operator==(const BindingSet& other) const noexcept {
  if (bindings_.size() != other.bindings_.size()) return false;
  for (size_t i = 0; i < bindings_.size(); ++i) {
    if (bindings_[i].symbol != other.bindings_[i].symbol ||
        bindings_[i].value != other.bindings_[i].value) return false;
  }
  return true;
}

Constraint::Constraint(Kind kind, DimExpr left, std::optional<DimExpr> right,
                       int64_t lower, int64_t upper, int64_t divisor)
    : kind_(kind), left_(std::move(left)), right_(std::move(right)), lower_(lower),
      upper_(upper), divisor_(divisor) {}

Constraint Constraint::Eq(DimExpr left, DimExpr right) {
  if (right.CanonicalString() < left.CanonicalString()) std::swap(left, right);
  return Constraint(Kind::kEq, std::move(left), std::move(right), 0, 0, 0);
}

Constraint Constraint::Range(DimExpr expression, int64_t lower, int64_t upper) {
  if (lower < 0 || upper < lower) Invalid("range must be nonnegative and ordered");
  return Constraint(Kind::kRange, std::move(expression), std::nullopt, lower, upper, 0);
}

Constraint Constraint::DivisibleBy(DimExpr expression, int64_t divisor) {
  if (divisor <= 0) Invalid("divisibility divisor must be positive");
  return Constraint(Kind::kDivisibleBy, std::move(expression), std::nullopt, 0, 0, divisor);
}

Constraint Constraint::BroadcastCompatible(DimExpr left, DimExpr right) {
  if (right.CanonicalString() < left.CanonicalString()) std::swap(left, right);
  return Constraint(Kind::kBroadcastCompatible, std::move(left), std::move(right), 0, 0, 0);
}

Constraint::Kind Constraint::kind() const noexcept { return kind_; }
const DimExpr& Constraint::left() const noexcept { return left_; }
const std::optional<DimExpr>& Constraint::right() const noexcept { return right_; }
int64_t Constraint::lower() const noexcept { return lower_; }
int64_t Constraint::upper() const noexcept { return upper_; }
int64_t Constraint::divisor() const noexcept { return divisor_; }

std::string Constraint::CanonicalString() const {
  switch (kind_) {
    case Kind::kEq: return "Eq(" + left_.CanonicalString() + "," + right_->CanonicalString() + ")";
    case Kind::kRange: return "Range(" + left_.CanonicalString() + "," + std::to_string(lower_) + "," + std::to_string(upper_) + ")";
    case Kind::kDivisibleBy: return "Div(" + left_.CanonicalString() + "," + std::to_string(divisor_) + ")";
    case Kind::kBroadcastCompatible: return "Broadcast(" + left_.CanonicalString() + "," + right_->CanonicalString() + ")";
  }
  Invalid("invalid constraint kind");
}

std::vector<std::string> Constraint::Symbols() const {
  std::vector<std::string> result = left_.Symbols();
  if (right_) {
    const std::vector<std::string> right_symbols = right_->Symbols();
    result.insert(result.end(), right_symbols.begin(), right_symbols.end());
  }
  return SortedUnique(std::move(result));
}

BindingSet ExactConstraintSolver::Solve(const std::vector<std::string>& declared_symbols,
                                        const BindingSet& initial_bindings,
                                        const std::vector<Constraint>& constraints) {
  std::vector<std::string> declared = declared_symbols;
  for (const std::string& symbol : declared) CheckName(symbol, "declared symbol");
  std::sort(declared.begin(), declared.end());
  if (std::adjacent_find(declared.begin(), declared.end()) != declared.end()) Invalid("duplicate declared symbol");
  const std::set<std::string> declared_set(declared.begin(), declared.end());
  for (const Binding& binding : initial_bindings.bindings()) {
    if (declared_set.count(binding.symbol) == 0) Invalid("binding references undeclared symbol '" + binding.symbol + "'");
  }
  for (const Constraint& constraint : constraints) {
    for (const std::string& symbol : constraint.Symbols()) {
      if (declared_set.count(symbol) == 0) Invalid("constraint references undeclared symbol '" + symbol + "'");
    }
  }

  std::vector<Binding> bindings = initial_bindings.bindings();
  auto lookup = [&bindings](const std::string& symbol) -> std::optional<int64_t> {
    for (const Binding& binding : bindings) if (binding.symbol == symbol) return binding.value;
    return std::nullopt;
  };
  auto add_binding = [&bindings, &lookup](const std::string& symbol, int64_t value) -> bool {
    if (value < 0) Invalid("inferred negative value for '" + symbol + "'");
    const std::optional<int64_t> previous = lookup(symbol);
    if (previous.has_value()) {
      if (*previous != value) Invalid("contradictory binding for '" + symbol + "'");
      return false;
    }
    bindings.push_back(Binding{symbol, value});
    return true;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    const BindingSet current(bindings);
    for (const Constraint& constraint : constraints) {
      if (constraint.kind() != Constraint::Kind::kEq) continue;
      const DimExpr& left = constraint.left();
      const DimExpr& right = *constraint.right();
      const bool left_symbol = left.node_->kind == DimExpr::Kind::kSymbol;
      const bool right_symbol = right.node_->kind == DimExpr::Kind::kSymbol;
      try {
        if (left_symbol) changed = add_binding(left.node_->symbol, right.Evaluate(current)) || changed;
        if (right_symbol) changed = add_binding(right.node_->symbol, left.Evaluate(current)) || changed;
      } catch (const std::invalid_argument&) {
        // A non-evaluable equation remains for the exact final verification.
      }
    }
  }

  const BindingSet solved(bindings);
  for (const std::string& symbol : declared) {
    if (!solved.Find(symbol).has_value()) Invalid("unbound declared symbol '" + symbol + "'");
  }
  for (const Constraint& constraint : constraints) {
    const int64_t left = constraint.left().Evaluate(solved);
    switch (constraint.kind()) {
      case Constraint::Kind::kEq:
        if (left != constraint.right()->Evaluate(solved)) Invalid("contradictory equality constraint");
        break;
      case Constraint::Kind::kRange:
        if (left < constraint.lower() || left > constraint.upper()) Invalid("range constraint violated");
        break;
      case Constraint::Kind::kDivisibleBy:
        if (left % constraint.divisor() != 0) Invalid("divisibility constraint violated");
        break;
      case Constraint::Kind::kBroadcastCompatible: {
        const int64_t right = constraint.right()->Evaluate(solved);
        if (!(left == right || left == 1 || right == 1)) Invalid("broadcast compatibility constraint violated");
        break;
      }
    }
  }
  return solved;
}

LogicalShape::LogicalShape(std::vector<DimExpr> dimensions,
                           std::vector<std::optional<std::string>> axis_names)
    : dimensions_(std::move(dimensions)), axis_names_(std::move(axis_names)) {
  if (!axis_names_.empty() && axis_names_.size() != dimensions_.size()) Invalid("logical axis-name rank mismatch");
  std::set<std::string> names;
  for (const auto& name : axis_names_) {
    if (name.has_value()) {
      CheckName(*name, "axis name");
      if (!names.insert(*name).second) Invalid("duplicate axis name '" + *name + "'");
    }
  }
}
const std::vector<DimExpr>& LogicalShape::dimensions() const noexcept { return dimensions_; }
const std::vector<std::optional<std::string>>& LogicalShape::axis_names() const noexcept { return axis_names_; }

PhysicalShape::PhysicalShape(std::vector<DimExpr> capacity,
                             std::optional<std::vector<DimExpr>> strides,
                             std::string layout, int64_t alignment, std::string memory_scope)
    : capacity_(std::move(capacity)), strides_(std::move(strides)), layout_(std::move(layout)),
      alignment_(alignment), memory_scope_(std::move(memory_scope)) {
  if (alignment_ <= 0 || (alignment_ & (alignment_ - 1)) != 0) {
    Invalid("physical alignment must be a positive power of two");
  }
  CheckName(layout_, "physical layout");
  CheckName(memory_scope_, "physical memory scope");
  if (strides_ && strides_->size() != capacity_.size()) Invalid("physical stride rank mismatch");
}
const std::vector<DimExpr>& PhysicalShape::capacity() const noexcept { return capacity_; }
const std::optional<std::vector<DimExpr>>& PhysicalShape::strides() const noexcept { return strides_; }
const std::string& PhysicalShape::layout() const noexcept { return layout_; }
int64_t PhysicalShape::alignment() const noexcept { return alignment_; }
const std::string& PhysicalShape::memory_scope() const noexcept { return memory_scope_; }

ValidExtent::ValidExtent(std::vector<DimExpr> dimensions) : dimensions_(std::move(dimensions)) {}
const std::vector<DimExpr>& ValidExtent::dimensions() const noexcept { return dimensions_; }

bool ConcreteTensorShapeContract::operator==(const ConcreteTensorShapeContract& other) const noexcept {
  return logical == other.logical && physical == other.physical && valid == other.valid &&
         strides == other.strides && axis_names == other.axis_names && layout == other.layout &&
         alignment == other.alignment && memory_scope == other.memory_scope;
}

TensorShapeContract::TensorShapeContract(LogicalShape logical, PhysicalShape physical, ValidExtent valid)
    : logical_(std::move(logical)), physical_(std::move(physical)), valid_(std::move(valid)) {
  if (logical_.dimensions().size() != physical_.capacity().size() ||
      logical_.dimensions().size() != valid_.dimensions().size()) {
    Invalid("logical, physical, and valid ranks must match");
  }
}
const LogicalShape& TensorShapeContract::logical() const noexcept { return logical_; }
const PhysicalShape& TensorShapeContract::physical() const noexcept { return physical_; }
const ValidExtent& TensorShapeContract::valid() const noexcept { return valid_; }

ConcreteTensorShapeContract TensorShapeContract::Evaluate(const BindingSet& bindings) const {
  ConcreteTensorShapeContract result;
  auto evaluate_dimensions = [&bindings](const std::vector<DimExpr>& dimensions) {
    std::vector<int64_t> values;
    values.reserve(dimensions.size());
    for (const DimExpr& dimension : dimensions) values.push_back(dimension.Evaluate(bindings));
    return values;
  };
  result.logical = evaluate_dimensions(logical_.dimensions());
  result.physical = evaluate_dimensions(physical_.capacity());
  result.valid = evaluate_dimensions(valid_.dimensions());
  for (size_t i = 0; i < result.logical.size(); ++i) {
    if (result.valid[i] > result.logical[i] || result.logical[i] > result.physical[i]) {
      Invalid("valid <= logical <= physical contract violated at axis " + std::to_string(i));
    }
  }
  if (physical_.strides()) {
    result.strides = evaluate_dimensions(*physical_.strides());
  } else {
    result.strides.assign(result.physical.size(), 1);
    int64_t stride = 1;
    for (size_t i = result.physical.size(); i > 0; --i) {
      result.strides[i - 1] = stride;
      stride = CheckedMul(stride, result.physical[i - 1]);
    }
  }
  result.axis_names = logical_.axis_names();
  result.layout = physical_.layout();
  result.alignment = physical_.alignment();
  result.memory_scope = physical_.memory_scope();
  return result;
}

std::vector<std::string> TensorShapeContract::Symbols() const {
  std::vector<std::string> result;
  const auto collect = [&result](const std::vector<DimExpr>& dimensions) {
    for (const DimExpr& dimension : dimensions) {
      const std::vector<std::string> symbols = dimension.Symbols();
      result.insert(result.end(), symbols.begin(), symbols.end());
    }
  };
  collect(logical_.dimensions());
  collect(physical_.capacity());
  collect(valid_.dimensions());
  if (physical_.strides()) collect(*physical_.strides());
  return SortedUnique(std::move(result));
}

GraphTemplateKey::GraphTemplateKey(uint32_t version,
                                   std::string graph_semantic_fingerprint,
                                   std::string pipeline_fingerprint,
                                   std::string capability_fingerprint,
                                   std::string partition_fingerprint)
    : version_(version),
      graph_semantic_fingerprint_(std::move(graph_semantic_fingerprint)),
      pipeline_fingerprint_(std::move(pipeline_fingerprint)),
      capability_fingerprint_(std::move(capability_fingerprint)),
      partition_fingerprint_(std::move(partition_fingerprint)) {
  if (version_ == 0) Invalid("graph template key version must be nonzero");
  CheckName(graph_semantic_fingerprint_, "graph semantic fingerprint");
  CheckName(pipeline_fingerprint_, "pipeline fingerprint");
  CheckName(capability_fingerprint_, "capability fingerprint");
  CheckName(partition_fingerprint_, "partition fingerprint");
}
uint32_t GraphTemplateKey::version() const noexcept { return version_; }
const std::string& GraphTemplateKey::graph_semantic_fingerprint() const noexcept {
  return graph_semantic_fingerprint_;
}
const std::string& GraphTemplateKey::pipeline_fingerprint() const noexcept {
  return pipeline_fingerprint_;
}
const std::string& GraphTemplateKey::capability_fingerprint() const noexcept {
  return capability_fingerprint_;
}
const std::string& GraphTemplateKey::partition_fingerprint() const noexcept {
  return partition_fingerprint_;
}
std::string GraphTemplateKey::CanonicalBytes() const {
  std::string bytes("kxc.shape.graph-template.v1", 27);
  AppendU64(&bytes, version_);
  AppendField(&bytes, graph_semantic_fingerprint_);
  AppendField(&bytes, pipeline_fingerprint_);
  AppendField(&bytes, capability_fingerprint_);
  AppendField(&bytes, partition_fingerprint_);
  return bytes;
}
std::string GraphTemplateKey::CanonicalString() const { return "GraphTemplateKey(" + Hex(CanonicalBytes()) + ")"; }
bool GraphTemplateKey::operator==(const GraphTemplateKey& other) const noexcept {
  return version_ == other.version_ &&
         graph_semantic_fingerprint_ == other.graph_semantic_fingerprint_ &&
         pipeline_fingerprint_ == other.pipeline_fingerprint_ &&
         capability_fingerprint_ == other.capability_fingerprint_ &&
         partition_fingerprint_ == other.partition_fingerprint_;
}

ShapeProfileKey::ShapeProfileKey(GraphTemplateKey graph_template, BindingSet bindings,
                                 std::string policy_id, uint32_t shape_abi_version)
    : graph_template_(std::move(graph_template)), bindings_(std::move(bindings)),
      policy_id_(std::move(policy_id)), shape_abi_version_(shape_abi_version) {
  CheckName(policy_id_, "shape profile policy id");
  if (shape_abi_version_ == 0) Invalid("shape ABI version must be nonzero");
}
const GraphTemplateKey& ShapeProfileKey::graph_template() const noexcept { return graph_template_; }
const BindingSet& ShapeProfileKey::bindings() const noexcept { return bindings_; }
const std::string& ShapeProfileKey::policy_id() const noexcept { return policy_id_; }
uint32_t ShapeProfileKey::shape_abi_version() const noexcept { return shape_abi_version_; }
std::string ShapeProfileKey::CanonicalBytes() const {
  std::string bytes("kxc.shape.profile.v1", 20);
  const std::string graph = graph_template_.CanonicalBytes();
  AppendField(&bytes, graph);
  AppendField(&bytes, bindings_.CanonicalString());
  AppendField(&bytes, policy_id_);
  AppendU64(&bytes, shape_abi_version_);
  return bytes;
}
std::string ShapeProfileKey::CanonicalString() const { return "ShapeProfileKey(" + Hex(CanonicalBytes()) + ")"; }
bool ShapeProfileKey::operator==(const ShapeProfileKey& other) const noexcept {
  return graph_template_ == other.graph_template_ && bindings_ == other.bindings_ &&
         policy_id_ == other.policy_id_ && shape_abi_version_ == other.shape_abi_version_;
}

ShapeProgram::ShapeProgram(std::vector<std::string> declared_symbols,
                           std::vector<NamedTensorContract> inputs,
                           std::vector<NamedTensorContract> outputs,
                           std::vector<Constraint> constraints)
    : declared_symbols_(std::move(declared_symbols)), inputs_(std::move(inputs)),
      outputs_(std::move(outputs)), constraints_(std::move(constraints)) {
  std::sort(declared_symbols_.begin(), declared_symbols_.end());
  const auto sort_named = [](std::vector<NamedTensorContract>* values) {
    std::sort(values->begin(), values->end(), [](const NamedTensorContract& a, const NamedTensorContract& b) {
      return a.name < b.name;
    });
  };
  sort_named(&inputs_);
  sort_named(&outputs_);
  std::sort(constraints_.begin(), constraints_.end(), [](const Constraint& a, const Constraint& b) {
    return a.CanonicalString() < b.CanonicalString();
  });
  constraints_.erase(std::unique(constraints_.begin(), constraints_.end(),
                                 [](const Constraint& a, const Constraint& b) {
                                   return a.CanonicalString() == b.CanonicalString();
                                 }), constraints_.end());
  Verify();
}
const std::vector<std::string>& ShapeProgram::declared_symbols() const noexcept { return declared_symbols_; }
const std::vector<NamedTensorContract>& ShapeProgram::inputs() const noexcept { return inputs_; }
const std::vector<NamedTensorContract>& ShapeProgram::outputs() const noexcept { return outputs_; }
const std::vector<Constraint>& ShapeProgram::constraints() const noexcept { return constraints_; }

void ShapeProgram::Verify() const {
  for (const std::string& symbol : declared_symbols_) CheckName(symbol, "declared symbol");
  if (std::adjacent_find(declared_symbols_.begin(), declared_symbols_.end()) != declared_symbols_.end()) Invalid("duplicate declared symbol");
  std::set<std::string> names;
  const auto verify_named = [&names, this](const std::vector<NamedTensorContract>& contracts) {
    for (const NamedTensorContract& named : contracts) {
      CheckName(named.name, "tensor contract name");
      if (!names.insert(named.name).second) Invalid("duplicate tensor contract name '" + named.name + "'");
      for (const std::string& symbol : named.contract.Symbols()) {
        if (!std::binary_search(declared_symbols_.begin(), declared_symbols_.end(), symbol)) {
          Invalid("tensor contract references undeclared symbol '" + symbol + "'");
        }
      }
    }
  };
  verify_named(inputs_);
  verify_named(outputs_);
  for (const Constraint& constraint : constraints_) {
    for (const std::string& symbol : constraint.Symbols()) {
      if (!std::binary_search(declared_symbols_.begin(), declared_symbols_.end(), symbol)) {
        Invalid("constraint references undeclared symbol '" + symbol + "'");
      }
    }
  }
}

EvaluatedShapeProgram ShapeProgram::Evaluate(const BindingSet& bindings) const {
  Verify();
  EvaluatedShapeProgram result{ExactConstraintSolver::Solve(declared_symbols_, bindings, constraints_), {}, {}};
  const auto evaluate_named = [&result](const std::vector<NamedTensorContract>& source,
                                        std::vector<NamedConcreteTensorContract>* destination) {
    destination->reserve(source.size());
    for (const NamedTensorContract& named : source) {
      destination->push_back(NamedConcreteTensorContract{named.name, named.contract.Evaluate(result.bindings)});
    }
  };
  evaluate_named(inputs_, &result.inputs);
  evaluate_named(outputs_, &result.outputs);
  return result;
}

std::string ShapeProgram::CanonicalString() const {
  Verify();
  const auto contract_string = [](const NamedTensorContract& named) {
    const TensorShapeContract& contract = named.contract;
    std::string value = Field(named.name) + "{";
    const auto append_dims = [&value](const std::vector<DimExpr>& dimensions) {
      value += "[";
      for (const DimExpr& dimension : dimensions) value += dimension.CanonicalString() + ";";
      value += "]";
    };
    append_dims(contract.logical().dimensions());
    value += "axes[";
    for (const auto& axis : contract.logical().axis_names()) {
      value += axis ? Field(*axis) : "-";
    }
    value += "]";
    append_dims(contract.physical().capacity());
    append_dims(contract.valid().dimensions());
    if (contract.physical().strides()) append_dims(*contract.physical().strides());
    else value += "derived";
    value += Field(contract.physical().layout()) + Field(contract.physical().memory_scope()) +
             std::to_string(contract.physical().alignment()) + "}";
    return value;
  };
  std::string result = "ShapeProgram(";
  for (const std::string& symbol : declared_symbols_) result += Field(symbol);
  result += "|I";
  for (const NamedTensorContract& input : inputs_) result += contract_string(input);
  result += "|O";
  for (const NamedTensorContract& output : outputs_) result += contract_string(output);
  result += "|C";
  for (const Constraint& constraint : constraints_) result += constraint.CanonicalString() + ";";
  return result + ")";
}

bool ShapeProgram::operator==(const ShapeProgram& other) const {
  return CanonicalString() == other.CanonicalString();
}

}  // namespace kxc::shape
