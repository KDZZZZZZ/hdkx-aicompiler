/*! \file src/runtime/runtime_shape_session.cc */

#include "kxc/runtime/runtime_shape_session.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <set>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kxc::runtime {
namespace {

#ifndef KXC_ENABLE_RUNTIME_SHAPE_TASKS
#define KXC_ENABLE_RUNTIME_SHAPE_TASKS 0
#endif

constexpr std::size_t kMaxExpressionDepth = 64;
constexpr std::size_t kMaxExpressionNodes = 4096;
std::atomic<bool> fail_next_owner_transfer{false};

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("RuntimeShape: " + detail);
}

bool IsPowerOfTwo(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t DTypeBytes(const std::string& dtype) {
    if (dtype == "bool" || dtype == "int8" || dtype == "uint8") return 1;
    if (dtype == "int16" || dtype == "uint16" || dtype == "float16" ||
        dtype == "bfloat16") return 2;
    if (dtype == "int32" || dtype == "uint32" || dtype == "float32") return 4;
    if (dtype == "int64" || dtype == "uint64" || dtype == "float64") return 8;
    Fail("unknown dtype '" + dtype + "'");
}

RuntimeShapeExtent CheckedAdd(RuntimeShapeExtent lhs, RuntimeShapeExtent rhs) {
    if (rhs > std::numeric_limits<RuntimeShapeExtent>::max() - lhs) {
        Fail("shape expression addition overflow");
    }
    return lhs + rhs;
}

RuntimeShapeExtent CheckedMul(RuntimeShapeExtent lhs, RuntimeShapeExtent rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<RuntimeShapeExtent>::max() / lhs) {
        Fail("shape expression multiplication overflow");
    }
    return lhs * rhs;
}

std::size_t CheckedBytes(const std::vector<RuntimeShapeExtent>& shape,
                         std::size_t dtype_bytes) {
    RuntimeShapeExtent elements = 1;
    for (const auto extent : shape) elements = CheckedMul(elements, extent);
    if (elements > std::numeric_limits<std::size_t>::max() / dtype_bytes) {
        Fail("output byte size overflow");
    }
    return static_cast<std::size_t>(elements) * dtype_bytes;
}

std::vector<RuntimeShapeExtent> EvaluateShape(
    const std::vector<RuntimeShapeExpr>& expressions,
    const std::vector<std::vector<RuntimeShapeExtent>>& input_shapes) {
    std::vector<RuntimeShapeExtent> result;
    result.reserve(expressions.size());
    for (const auto& expression : expressions) result.push_back(expression.Evaluate(input_shapes));
    return result;
}

void ValidateInputContract(const RuntimeShapeInputContract& contract) {
    if (contract.dtype.empty()) Fail("input dtype is empty");
    (void)DTypeBytes(contract.dtype);
    if (contract.device != "CPU:0") Fail("v1 requires input device CPU:0");
    if (contract.abi_version != RuntimeShapePlan::kAbiVersion) {
        Fail("input ABI version does not match");
    }
    std::size_t previous_axis = 0;
    bool first = true;
    for (const auto& guard : contract.axis_guards) {
        if (guard.axis >= contract.rank || (!first && guard.axis <= previous_axis) ||
            guard.upper < guard.lower || guard.divisible_by == 0) {
            Fail("input axis guard is invalid");
        }
        if (guard.exact && (*guard.exact < guard.lower || *guard.exact > guard.upper ||
                            *guard.exact % guard.divisible_by != 0)) {
            Fail("input axis exact guard is outside its domain");
        }
        previous_axis = guard.axis;
        first = false;
    }
}

void ValidateContract(const RuntimeShapeTensorContract& contract) {
    if (contract.dtype.empty()) Fail("output dtype is empty");
    (void)DTypeBytes(contract.dtype);
    if (contract.logical.empty() || contract.logical.size() != contract.physical.size() ||
        contract.logical.size() != contract.valid.size()) {
        Fail("logical, physical, and valid ranks must be equal and nonzero");
    }
    for (const auto& expression : contract.logical) {
        if (!expression.defined()) Fail("logical shape contains undefined expression");
    }
    for (const auto& expression : contract.physical) {
        if (!expression.defined()) Fail("physical shape contains undefined expression");
    }
    for (const auto& expression : contract.valid) {
        if (!expression.defined()) Fail("valid shape contains undefined expression");
    }
    if (!IsPowerOfTwo(contract.alignment)) Fail("alignment must be a power of two");
    if (contract.layout != "contiguous.row_major") {
        Fail("v1 requires layout contiguous.row_major");
    }
    if (contract.scope != "global") Fail("v1 requires global scope");
    if (contract.device != "CPU:0") Fail("v1 requires output device CPU:0");
    if (contract.abi_version != RuntimeShapePlan::kAbiVersion) {
        Fail("output ABI version does not match");
    }
}

struct OutputEvaluation {
    RuntimeShapeOutput output;
};

struct AllocationDeleter {
    std::size_t alignment{0};

    void operator()(void* ptr) const noexcept {
        if (!ptr) return;
        if (alignment > alignof(std::max_align_t)) {
            ::operator delete(ptr, std::align_val_t(alignment));
        } else {
            ::operator delete(ptr);
        }
    }
};

using AllocationOwner = std::unique_ptr<void, AllocationDeleter>;

AllocationOwner AllocateOutput(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) return AllocationOwner(nullptr, AllocationDeleter{alignment});
    if (alignment > alignof(std::max_align_t)) {
        return AllocationOwner(::operator new(bytes, std::align_val_t(alignment)),
                               AllocationDeleter{alignment});
    }
    return AllocationOwner(::operator new(bytes), AllocationDeleter{alignment});
}

RuntimeShapeEvent FailureEvent(const std::string& detail) {
    return RuntimeShapeEvent{RuntimeShapeEventKind::kFailure,
                             static_cast<std::size_t>(-1), 0, detail};
}

void AppendU64(std::string& bytes, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        bytes.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void AppendString(std::string& bytes, const std::string& value) {
    AppendU64(bytes, value.size());
    bytes.append(value);
}

}  // namespace

struct RuntimeShapeExpr::Node {
    Kind kind{Kind::kConst};
    RuntimeShapeExtent value{0};
    std::size_t input_index{0};
    std::size_t axis{0};
    std::shared_ptr<const Node> lhs;
    std::shared_ptr<const Node> rhs;
};

RuntimeShapeExpr::RuntimeShapeExpr(std::shared_ptr<const Node> node)
    : node_(std::move(node)) {}

std::size_t RuntimeShapeExpr::Depth(const std::shared_ptr<const Node>& node) {
    if (!node) Fail("undefined shape expression");
    std::size_t maximum = 0;
    std::vector<std::pair<std::shared_ptr<const Node>, std::size_t>> pending;
    std::unordered_set<const Node*> visited;
    pending.emplace_back(node, 1);
    while (!pending.empty()) {
        const auto current = std::move(pending.back());
        pending.pop_back();
        if (current.second > kMaxExpressionDepth) {
            Fail("shape expression exceeds maximum depth");
        }
        maximum = current.second > maximum ? current.second : maximum;
        if (!visited.insert(current.first.get()).second) continue;
        if (visited.size() > kMaxExpressionNodes) {
            Fail("shape expression exceeds maximum node count");
        }
        switch (current.first->kind) {
            case Kind::kConst:
            case Kind::kInputAxis:
                break;
            case Kind::kAdd:
            case Kind::kMul:
            case Kind::kFloorDiv:
            case Kind::kMin:
            case Kind::kMax:
                if (!current.first->lhs || !current.first->rhs) {
                    Fail("binary shape expression contains undefined operand");
                }
                pending.emplace_back(current.first->lhs, current.second + 1);
                pending.emplace_back(current.first->rhs, current.second + 1);
                break;
        }
    }
    return maximum;
}

RuntimeShapeExpr RuntimeShapeExpr::Const(RuntimeShapeExtent value) {
    auto node = std::make_shared<Node>();
    node->value = value;
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::InputAxis(std::size_t input_index, std::size_t axis) {
    auto node = std::make_shared<Node>();
    node->kind = Kind::kInputAxis;
    node->input_index = input_index;
    node->axis = axis;
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::Binary(Kind kind, RuntimeShapeExpr lhs,
                                          RuntimeShapeExpr rhs) {
    if (!lhs.node_ || !rhs.node_) Fail("binary shape expression requires defined operands");
    auto node = std::make_shared<Node>();
    node->kind = kind;
    node->lhs = std::move(lhs.node_);
    node->rhs = std::move(rhs.node_);
    (void)Depth(node);
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::Add(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    return Binary(Kind::kAdd, std::move(lhs), std::move(rhs));
}

RuntimeShapeExpr RuntimeShapeExpr::Mul(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    return Binary(Kind::kMul, std::move(lhs), std::move(rhs));
}

RuntimeShapeExpr RuntimeShapeExpr::FloorDiv(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    return Binary(Kind::kFloorDiv, std::move(lhs), std::move(rhs));
}

RuntimeShapeExpr RuntimeShapeExpr::Min(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    return Binary(Kind::kMin, std::move(lhs), std::move(rhs));
}

RuntimeShapeExpr RuntimeShapeExpr::Max(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    return Binary(Kind::kMax, std::move(lhs), std::move(rhs));
}

RuntimeShapeExtent RuntimeShapeExpr::Evaluate(
    const std::vector<std::vector<RuntimeShapeExtent>>& input_shapes) const {
    (void)Depth(node_);
    std::unordered_map<const Node*, RuntimeShapeExtent> values;
    const auto evaluate = [&](const auto& self, const std::shared_ptr<const Node>& node)
        -> RuntimeShapeExtent {
        const auto existing = values.find(node.get());
        if (existing != values.end()) return existing->second;
        RuntimeShapeExtent value = 0;
        switch (node->kind) {
            case Kind::kConst:
                value = node->value;
                break;
            case Kind::kInputAxis:
                if (node->input_index >= input_shapes.size() ||
                    node->axis >= input_shapes[node->input_index].size()) {
                    Fail("input-axis expression is out of range");
                }
                value = input_shapes[node->input_index][node->axis];
                break;
            case Kind::kAdd:
                value = CheckedAdd(self(self, node->lhs), self(self, node->rhs));
                break;
            case Kind::kMul:
                value = CheckedMul(self(self, node->lhs), self(self, node->rhs));
                break;
            case Kind::kFloorDiv: {
                const auto divisor = self(self, node->rhs);
                if (divisor == 0) Fail("floor division by zero");
                value = self(self, node->lhs) / divisor;
                break;
            }
            case Kind::kMin: {
                const auto lhs = self(self, node->lhs);
                const auto rhs = self(self, node->rhs);
                value = lhs < rhs ? lhs : rhs;
                break;
            }
            case Kind::kMax: {
                const auto lhs = self(self, node->lhs);
                const auto rhs = self(self, node->rhs);
                value = lhs > rhs ? lhs : rhs;
                break;
            }
        }
        values.emplace(node.get(), value);
        return value;
    };
    return evaluate(evaluate, node_);
}

void RuntimeShapeExpr::AppendCanonical(std::string& bytes) const {
    (void)Depth(node_);
    const auto append = [&](const auto& self, const std::shared_ptr<const Node>& node) -> void {
        bytes.push_back(static_cast<char>(node->kind));
        switch (node->kind) {
            case Kind::kConst:
                AppendU64(bytes, node->value);
                break;
            case Kind::kInputAxis:
                AppendU64(bytes, node->input_index);
                AppendU64(bytes, node->axis);
                break;
            case Kind::kAdd:
            case Kind::kMul:
            case Kind::kFloorDiv:
            case Kind::kMin:
            case Kind::kMax:
                self(self, node->lhs);
                self(self, node->rhs);
                break;
        }
    };
    append(append, node_);
}

RuntimeShapeExpr::Kind RuntimeShapeExpr::kind() const {
    if (!node_) Fail("undefined shape expression");
    return node_->kind;
}

bool RuntimeShapeExpr::defined() const noexcept { return static_cast<bool>(node_); }

struct FakeRuntimeShapeCompletion::State {
    std::atomic<bool> ready{false};
};

FakeRuntimeShapeCompletion::FakeRuntimeShapeCompletion(bool ready) noexcept
    : state_(std::make_shared<State>()) {
    state_->ready.store(ready, std::memory_order_relaxed);
}

std::shared_ptr<FakeRuntimeShapeCompletion> FakeRuntimeShapeCompletion::Pending() {
    return std::shared_ptr<FakeRuntimeShapeCompletion>(new FakeRuntimeShapeCompletion(false));
}

std::shared_ptr<FakeRuntimeShapeCompletion> FakeRuntimeShapeCompletion::Completed() {
    return std::shared_ptr<FakeRuntimeShapeCompletion>(new FakeRuntimeShapeCompletion(true));
}

void FakeRuntimeShapeCompletion::Complete() noexcept {
    state_->ready.store(true, std::memory_order_release);
}

bool FakeRuntimeShapeCompletion::IsReady() const noexcept {
    return state_->ready.load(std::memory_order_acquire);
}

namespace detail {

void FailNextRuntimeShapeOwnerTransferForTest() noexcept {
    fail_next_owner_transfer.store(true, std::memory_order_release);
}

}  // namespace detail

struct RuntimeShapePlan::Impl {
    explicit Impl(RuntimeShapePlanSpec value) : spec(std::move(value)) {}
    const RuntimeShapePlanSpec spec;
    mutable std::mutex launcher_mutex;
};

RuntimeShapePlan::RuntimeShapePlan(RuntimeShapePlanSpec spec)
    : impl_(std::make_shared<Impl>(std::move(spec))) {
    Validate();
}

std::string RuntimeShapePlan::ExactAbiFingerprint(
    const std::vector<RuntimeShapeInputContract>& inputs,
    const std::vector<RuntimeShapeTensorContract>& outputs,
    const std::vector<RuntimeShapeExtentScalar>& runtime_extent_abi,
    const std::string& artifact_identity,
    const std::string& tail_policy_identity) {
    std::string bytes;
    AppendString(bytes, "kxc.runtime_shape.trusted_sync_abi.v1");
    AppendU64(bytes, inputs.size());
    for (const auto& input : inputs) {
        AppendString(bytes, input.dtype);
        AppendU64(bytes, input.rank);
        AppendString(bytes, input.device);
        AppendU64(bytes, input.abi_version);
        AppendU64(bytes, input.requires_data ? 1 : 0);
        AppendU64(bytes, input.axis_guards.size());
        for (const auto& guard : input.axis_guards) {
            AppendU64(bytes, guard.axis);
            AppendU64(bytes, guard.lower);
            AppendU64(bytes, guard.upper);
            AppendU64(bytes, guard.divisible_by);
            AppendU64(bytes, guard.exact.has_value() ? 1 : 0);
            if (guard.exact) AppendU64(bytes, *guard.exact);
            AppendU64(bytes, guard.equal_to.has_value() ? 1 : 0);
            if (guard.equal_to) {
                AppendU64(bytes, guard.equal_to->input_index);
                AppendU64(bytes, guard.equal_to->axis);
            }
        }
    }
    AppendU64(bytes, outputs.size());
    const auto append_expressions = [&bytes](const std::vector<RuntimeShapeExpr>& expressions) {
        AppendU64(bytes, expressions.size());
        for (const auto& expression : expressions) expression.AppendCanonical(bytes);
    };
    for (const auto& output : outputs) {
        AppendString(bytes, output.dtype);
        append_expressions(output.logical);
        append_expressions(output.physical);
        append_expressions(output.valid);
        AppendU64(bytes, output.alignment);
        AppendString(bytes, output.layout);
        AppendString(bytes, output.scope);
        AppendU64(bytes, output.max_bytes);
        AppendString(bytes, output.device);
        AppendU64(bytes, output.abi_version);
    }
    AppendU64(bytes, runtime_extent_abi.size());
    for (const auto& scalar : runtime_extent_abi) {
        AppendU64(bytes, scalar.ordinal);
        AppendString(bytes, scalar.name);
        AppendString(bytes, scalar.symbol);
        AppendU64(bytes, scalar.input_index);
        AppendU64(bytes, scalar.axis);
        AppendU64(bytes, scalar.lower);
        AppendU64(bytes, scalar.upper);
        AppendU64(bytes, scalar.divisible_by);
    }
    AppendString(bytes, artifact_identity);
    AppendString(bytes, tail_policy_identity);
    return bytes;
}

bool RuntimeShapePlan::defined() const noexcept { return static_cast<bool>(impl_); }

void RuntimeShapePlan::Validate() const {
    if (!impl_) Fail("plan is undefined");
    const auto& spec = impl_->spec;
    if (spec.abi_version != kAbiVersion) Fail("plan ABI version does not match");
    if (spec.inputs.empty() || spec.outputs.empty()) Fail("plan requires inputs and outputs");
    for (const auto& input : spec.inputs) ValidateInputContract(input);
    for (const auto& input : spec.inputs) for (const auto& guard : input.axis_guards) {
        if (guard.equal_to && (guard.equal_to->input_index >= spec.inputs.size() ||
                               guard.equal_to->axis >= spec.inputs[guard.equal_to->input_index].rank)) {
            Fail("input axis equality guard is out of range");
        }
    }
    for (const auto& output : spec.outputs) ValidateContract(output);
    std::unordered_set<std::string> scalar_names;
    std::unordered_set<std::string> scalar_symbols;
    std::set<std::pair<std::size_t, std::size_t>> scalar_axes;
    for (std::size_t index = 0; index < spec.runtime_extent_abi.size(); ++index) {
        const auto& scalar = spec.runtime_extent_abi[index];
        if (scalar.ordinal != index || scalar.name.empty() || scalar.symbol.empty() ||
            scalar.upper < scalar.lower || scalar.divisible_by == 0 ||
            scalar.input_index >= spec.inputs.size() ||
            scalar.axis >= spec.inputs[scalar.input_index].rank ||
            !scalar_names.insert(scalar.name).second || !scalar_symbols.insert(scalar.symbol).second ||
            !scalar_axes.insert({scalar.input_index, scalar.axis}).second) {
            Fail("runtime extent ABI has duplicate, gap, or invalid scalar mapping");
        }
        const auto& guards = spec.inputs[scalar.input_index].axis_guards;
        const auto guard = std::find_if(guards.begin(), guards.end(), [&scalar](const auto& value) {
            return value.axis == scalar.axis;
        });
        if (guard == guards.end() || guard->lower != scalar.lower || guard->upper != scalar.upper ||
            guard->divisible_by != scalar.divisible_by || guard->exact) {
            Fail("runtime extent scalar does not exactly match its input-axis guard");
        }
    }
    if (!spec.entry.tail_policy_identity.empty() && spec.entry.artifact_identity.empty()) {
        Fail("tail policy identity requires an artifact identity");
    }
    (void)ExactAbiFingerprint(spec.inputs, spec.outputs, spec.runtime_extent_abi,
                              spec.entry.artifact_identity, spec.entry.tail_policy_identity);
    if (!spec.entry.ready || !spec.entry.launcher || spec.entry.module_label.empty() ||
        spec.entry.entry_symbol.empty()) {
        Fail("plan requires a selected ready bound launcher entry");
    }
    if (spec.entry.abi_version != kAbiVersion) Fail("entry ABI version does not match");
    if (spec.entry.exact_abi_fingerprint !=
        ExactAbiFingerprint(spec.inputs, spec.outputs, spec.runtime_extent_abi,
                            spec.entry.artifact_identity, spec.entry.tail_policy_identity)) {
        Fail("entry exact ABI fingerprint does not match plan contracts");
    }
}

const RuntimeShapePlanSpec& RuntimeShapePlan::spec() const {
    if (!impl_) Fail("plan is undefined");
    return impl_->spec;
}

std::mutex& RuntimeShapePlan::launcher_mutex() const {
    if (!impl_) Fail("plan is undefined");
    return impl_->launcher_mutex;
}

struct RuntimeShapeAsyncResult::State {
    RuntimeShapePlan plan;
    std::shared_ptr<void> caller_lease;
    std::vector<std::shared_ptr<void>> input_owners;
    bool ok{false};
    std::string failure_reason;
    std::vector<RuntimeShapeOutput> outputs;
    std::vector<RuntimeShapeEvent> events;
    std::shared_ptr<FakeRuntimeShapeCompletion> fake_completion;
};

RuntimeShapeAsyncResult::RuntimeShapeAsyncResult(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

bool RuntimeShapeAsyncResult::ok() const noexcept { return state_ && state_->ok; }

const std::string& RuntimeShapeAsyncResult::failure_reason() const noexcept {
    static const std::string empty;
    return state_ ? state_->failure_reason : empty;
}

const std::vector<RuntimeShapeOutput>& RuntimeShapeAsyncResult::outputs() const noexcept {
    static const std::vector<RuntimeShapeOutput> empty;
    return state_ ? state_->outputs : empty;
}

const std::vector<RuntimeShapeEvent>& RuntimeShapeAsyncResult::events() const noexcept {
    static const std::vector<RuntimeShapeEvent> empty;
    return state_ ? state_->events : empty;
}

bool RuntimeShapeAsyncResult::IsReady() const noexcept {
    return !state_ || !state_->fake_completion || state_->fake_completion->IsReady();
}

void RuntimeShapeAsyncResult::Wait() const noexcept {
    if (state_ && state_->fake_completion) state_->fake_completion->Complete();
}

RuntimeShapeSession::RuntimeShapeSession(RuntimeShapePlan plan) : plan_(std::move(plan)) {}

RuntimeShapeAsyncResult RuntimeShapeSession::RunAsync(
    const std::vector<RuntimeShapeInput>& inputs, std::shared_ptr<void> caller_lease) const {
    auto state = std::make_shared<RuntimeShapeAsyncResult::State>();
    state->plan = plan_;  // Retains the trusted-entry module lease with this result.
    state->caller_lease = std::move(caller_lease);
    const auto fail = [&](const std::string& reason) {
        state->ok = false;
        state->failure_reason = reason;
        state->outputs.clear();  // Failed launches never publish or reuse allocations.
        state->events.push_back(FailureEvent(reason));
        return RuntimeShapeAsyncResult(state);
    };
#if !KXC_ENABLE_RUNTIME_SHAPE_TASKS
    (void)inputs;
    return fail("disabled by KXC_ENABLE_RUNTIME_SHAPE_TASKS");
#else
    try {
        plan_.Validate();
        const auto& spec = plan_.spec();
        if (inputs.size() != spec.inputs.size()) Fail("input count does not match");
        std::vector<std::vector<RuntimeShapeExtent>> input_shapes;
        input_shapes.reserve(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& input = inputs[index];
            const auto& contract = spec.inputs[index];
            if (input.shape.size() != contract.rank) Fail("input rank does not match");
            if (input.dtype != contract.dtype) Fail("input dtype does not match");
            if (input.device != contract.device) Fail("input device does not match");
            if (input.abi_version != contract.abi_version) Fail("input ABI version does not match");
            const std::size_t expected_bytes = CheckedBytes(input.shape, DTypeBytes(input.dtype));
            if (contract.requires_data || input.data || input.bytes != 0 || input.owner) {
                if (input.bytes != expected_bytes || (expected_bytes != 0 && input.data == nullptr)) {
                    Fail("input data pointer or byte size does not match shape contract");
                }
                state->input_owners.push_back(input.owner);
            }
            for (const auto& guard : contract.axis_guards) {
                const RuntimeShapeExtent extent = input.shape[guard.axis];
                if (extent < guard.lower || extent > guard.upper ||
                    extent % guard.divisible_by != 0 ||
                    (guard.exact && extent != *guard.exact)) {
                    Fail("input axis guard mismatch");
                }
            }
            input_shapes.push_back(input.shape);
        }
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            for (const auto& guard : spec.inputs[index].axis_guards) {
                if (guard.equal_to &&
                    inputs[index].shape[guard.axis] !=
                        inputs[guard.equal_to->input_index].shape[guard.equal_to->axis]) {
                    Fail("input axis equality guard mismatch");
                }
            }
        }
        std::vector<RuntimeShapeExtent> runtime_extent_values;
        runtime_extent_values.reserve(spec.runtime_extent_abi.size());
        for (const auto& scalar : spec.runtime_extent_abi) {
            const auto value = inputs[scalar.input_index].shape[scalar.axis];
            if (value < scalar.lower || value > scalar.upper || value % scalar.divisible_by != 0) {
                Fail("runtime extent scalar value is outside its ABI domain");
            }
            runtime_extent_values.push_back(value);
        }
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kShapeEval,
                                                   static_cast<std::size_t>(-1), 0,
                                                   "CPU:0/default"});
        std::vector<OutputEvaluation> evaluated;
        evaluated.reserve(spec.outputs.size());
        std::size_t total_bytes = 0;
        for (const auto& contract : spec.outputs) {
            RuntimeShapeOutput output;
            output.contract = contract;
            output.logical = EvaluateShape(contract.logical, input_shapes);
            output.physical = EvaluateShape(contract.physical, input_shapes);
            output.valid = EvaluateShape(contract.valid, input_shapes);
            for (std::size_t axis = 0; axis < output.logical.size(); ++axis) {
                if (output.valid[axis] > output.logical[axis] ||
                    output.logical[axis] > output.physical[axis]) {
                    Fail("valid <= logical <= physical is violated");
                }
            }
            output.bytes = CheckedBytes(output.physical, DTypeBytes(contract.dtype));
            if (output.bytes > contract.max_bytes) Fail("output exceeds max_bytes");
            if (output.bytes > std::numeric_limits<std::size_t>::max() - total_bytes) {
                Fail("run byte budget overflow");
            }
            total_bytes += output.bytes;
            evaluated.push_back(OutputEvaluation{std::move(output)});
        }
        if (total_bytes > spec.run_byte_budget) Fail("run byte budget exceeded");

        for (std::size_t index = 0; index < evaluated.size(); ++index) {
            auto& output = evaluated[index].output;
            AllocationOwner allocation = AllocateOutput(output.bytes, output.contract.alignment);
            if (fail_next_owner_transfer.exchange(false, std::memory_order_acq_rel)) {
                throw std::bad_alloc();  // Tests RAII cleanup; not a shared_ptr control-block injector.
            }
            output.owner_ = std::shared_ptr<void>(std::move(allocation));
            output.data = output.owner_.get();
            state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kAllocate, index,
                                                       output.bytes, "CPU:0/default"});
            state->outputs.push_back(std::move(output));
        }

        RuntimeShapeLaunchResult launch;
        try {
            std::lock_guard<std::mutex> lock(plan_.launcher_mutex());
            launch = spec.entry.launcher(RuntimeShapeLaunchArgs{
                inputs, state->outputs, runtime_extent_values, spec.entry.exact_abi_fingerprint});
        } catch (const std::exception& error) {
            return fail(std::string("bound launcher threw: ") + error.what());
        } catch (...) {
            return fail("bound launcher threw a non-standard exception");
        }
        if (!launch.accepted) {
            return fail(launch.failure_reason.empty() ? "bound launcher rejected launch"
                                                       : launch.failure_reason);
        }
        state->fake_completion = std::move(launch.fake_completion);
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kKernel,
                                                   static_cast<std::size_t>(-1), 0,
                                                   spec.entry.entry_symbol});
        state->ok = true;
        return RuntimeShapeAsyncResult(std::move(state));
    } catch (const std::bad_alloc&) {
        return fail("output allocation failed: out of memory");
    } catch (const std::exception& error) {
        return fail(error.what());
    } catch (...) {
        return fail("runtime shape execution threw a non-standard exception");
    }
#endif
}

RuntimeShapeAsyncResult RuntimeShapeSession::Run(
    const std::vector<RuntimeShapeInput>& inputs, std::shared_ptr<void> caller_lease) const {
    RuntimeShapeAsyncResult result = RunAsync(inputs, std::move(caller_lease));
    result.Wait();
    return result;
}

const RuntimeShapePlan& RuntimeShapeSession::plan() const noexcept { return plan_; }

}  // namespace kxc::runtime
