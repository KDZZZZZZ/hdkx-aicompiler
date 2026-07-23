/*! \file src/runtime/runtime_shape_session.cc */

#include "kxc/runtime/runtime_shape_session.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <set>
#include <cctype>
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
#ifndef KXC_ENABLE_RUNTIME_SHAPE_CUDA
#define KXC_ENABLE_RUNTIME_SHAPE_CUDA 0
#endif
#ifndef KXC_USE_CUDA
#define KXC_USE_CUDA 0
#endif

constexpr std::size_t kMaxExpressionDepth = 64;
constexpr std::size_t kMaxExpressionNodes = 4096;
std::atomic<bool> fail_next_owner_transfer{false};
std::atomic<bool> fail_next_cuda_retention{false};

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

bool IsCudaDeviceContract(const std::string& value) {
    if (value.size() <= 5 || value.compare(0, 5, "CUDA:") != 0) return false;
    for (std::size_t index = 5; index < value.size(); ++index) {
        if (!std::isdigit(static_cast<unsigned char>(value[index]))) return false;
    }
    try {
        return std::stoll(value.substr(5)) >= 0;
    } catch (...) {
        return false;
    }
}

::kxc::Device DeviceForContract(const std::string& value) {
    if (value == "CPU:0") return ::kxc::Device::CPU();
    if (!IsCudaDeviceContract(value)) Fail("invalid device contract '" + value + "'");
    try {
        return ::kxc::Device::CUDA(std::stoi(value.substr(5)));
    } catch (...) {
        Fail("invalid CUDA device contract '" + value + "'");
    }
}

void ValidateInputContract(const RuntimeShapeInputContract& contract,
                           RuntimeShapeExecutionKind execution_kind) {
    if (contract.dtype.empty()) Fail("input dtype is empty");
    (void)DTypeBytes(contract.dtype);
    if ((execution_kind == RuntimeShapeExecutionKind::kSynchronousCpu &&
         contract.device != "CPU:0") ||
        (execution_kind == RuntimeShapeExecutionKind::kCudaAsync &&
         !IsCudaDeviceContract(contract.device))) {
        Fail("input device does not match execution kind");
    }
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

void ValidateContract(const RuntimeShapeTensorContract& contract,
                      RuntimeShapeExecutionKind execution_kind) {
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
    if ((execution_kind == RuntimeShapeExecutionKind::kSynchronousCpu &&
         contract.device != "CPU:0") ||
        (execution_kind == RuntimeShapeExecutionKind::kCudaAsync &&
         !IsCudaDeviceContract(contract.device))) {
        Fail("output device does not match execution kind");
    }
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

void FailNextRuntimeShapeCudaRetentionForTest() noexcept {
    fail_next_cuda_retention.store(true, std::memory_order_release);
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
    const std::string& tail_policy_identity,
    RuntimeShapeExecutionKind execution_kind,
    const std::string& device_contract) {
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
    // Preserve the established CPU v1 bytes; async ABI adds an explicit contract.
    if (execution_kind != RuntimeShapeExecutionKind::kSynchronousCpu ||
        !device_contract.empty()) {
        AppendString(bytes, "kxc.runtime_shape.execution.v1");
        AppendU64(bytes, static_cast<std::uint64_t>(execution_kind));
        AppendString(bytes, device_contract);
    }
    return bytes;
}

bool RuntimeShapePlan::defined() const noexcept { return static_cast<bool>(impl_); }

void RuntimeShapePlan::Validate() const {
    if (!impl_) Fail("plan is undefined");
    const auto& spec = impl_->spec;
    if (spec.abi_version != kAbiVersion) Fail("plan ABI version does not match");
    if (spec.inputs.empty() || spec.outputs.empty()) Fail("plan requires inputs and outputs");
    for (const auto& input : spec.inputs) ValidateInputContract(input, spec.entry.execution_kind);
    for (const auto& input : spec.inputs) for (const auto& guard : input.axis_guards) {
        if (guard.equal_to && (guard.equal_to->input_index >= spec.inputs.size() ||
                               guard.equal_to->axis >= spec.inputs[guard.equal_to->input_index].rank)) {
            Fail("input axis equality guard is out of range");
        }
    }
    for (const auto& output : spec.outputs) ValidateContract(output, spec.entry.execution_kind);
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
    const bool cuda_async = spec.entry.execution_kind == RuntimeShapeExecutionKind::kCudaAsync;
    if (cuda_async) {
        if (!IsCudaDeviceContract(spec.entry.device_contract)) {
            Fail("CUDA async entry requires a CUDA:N device contract");
        }
        for (const auto& input : spec.inputs) {
            if (input.device != spec.entry.device_contract) {
                Fail("CUDA async entry has mixed input devices");
            }
        }
        for (const auto& output : spec.outputs) {
            if (output.device != spec.entry.device_contract) {
                Fail("CUDA async entry has mixed output devices");
            }
        }
    } else if (spec.entry.execution_kind != RuntimeShapeExecutionKind::kSynchronousCpu) {
        Fail("unknown runtime shape execution kind");
    }
    if (!spec.entry.ready || spec.entry.module_label.empty() || spec.entry.entry_symbol.empty() ||
        (cuda_async && !spec.entry.cuda_launcher) || (!cuda_async && !spec.entry.launcher)) {
        Fail("plan requires a selected ready launcher for its execution kind");
    }
    if (spec.entry.abi_version != kAbiVersion) Fail("entry ABI version does not match");
    if (spec.entry.exact_abi_fingerprint !=
        ExactAbiFingerprint(spec.inputs, spec.outputs, spec.runtime_extent_abi,
                            spec.entry.artifact_identity, spec.entry.tail_policy_identity,
                            spec.entry.execution_kind, spec.entry.device_contract)) {
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

struct RuntimeShapeAsyncRetention {
    RuntimeShapePlan plan;
    std::shared_ptr<void> caller_lease;
    std::vector<std::shared_ptr<void>> input_owners;
    std::vector<::kxc::Storage> input_storage;
};

struct RuntimeShapeAsyncResult::State {
    RuntimeShapePlan plan;
    std::shared_ptr<void> caller_lease;
    std::vector<std::shared_ptr<void>> input_owners;
    std::vector<::kxc::Storage> input_storage;
    ::kxc::DeviceStream stream;
    bool ok{false};
    RuntimeShapeFailureKind failure_kind{RuntimeShapeFailureKind::kNone};
    std::string failure_reason;
    std::vector<RuntimeShapeOutput> outputs;
    std::vector<RuntimeShapeEvent> events;
    std::shared_ptr<FakeRuntimeShapeCompletion> fake_completion;
    ::kxc::AsyncOperation completion;
    std::size_t retained_device_bytes{0};
    bool completion_reported{false};
    bool quarantined{false};
    mutable std::mutex mutex;
};

namespace {

// A callback that might have submitted work but provides no usable completion
// leaves no safe release point.  Keep the whole run alive for process lifetime.
template <typename State>
void QuarantineRunState(const std::shared_ptr<State>& state) noexcept {
    try {
        static auto* retained = new std::vector<std::shared_ptr<State>>();
        static auto* mutex = new std::mutex();
        std::lock_guard<std::mutex> lock(*mutex);
        retained->push_back(state);
    } catch (...) {
        std::terminate();
    }
}

}  // namespace

RuntimeShapeAsyncResult::RuntimeShapeAsyncResult(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

bool RuntimeShapeAsyncResult::ok() const noexcept { return state_ && state_->ok; }

const std::string& RuntimeShapeAsyncResult::failure_reason() const noexcept {
    static const std::string empty;
    return state_ ? state_->failure_reason : empty;
}

RuntimeShapeFailureKind RuntimeShapeAsyncResult::failure_kind() const noexcept {
    return state_ ? state_->failure_kind : RuntimeShapeFailureKind::kShapeAbiRejected;
}

const std::vector<RuntimeShapeOutput>& RuntimeShapeAsyncResult::outputs() const noexcept {
    static const std::vector<RuntimeShapeOutput> empty;
    return state_ ? state_->outputs : empty;
}

const std::vector<RuntimeShapeEvent>& RuntimeShapeAsyncResult::events() const noexcept {
    static const std::vector<RuntimeShapeEvent> empty;
    return state_ ? state_->events : empty;
}

namespace {
template <typename State>
void MarkCompletion(const std::shared_ptr<State>& state) noexcept {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->completion_reported) return;
    state->completion_reported = true;
    try {
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kCompletion,
                                                   static_cast<std::size_t>(-1), 0,
                                                   "CUDA completion observed; ownership retained"});
        // Retire means only that result ownership may now retire; it never asserts a free.
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kRetire,
                                                   static_cast<std::size_t>(-1),
                                                   state->retained_device_bytes,
                                                   "completion observed; logical retirement eligible, not physical free"});
    } catch (...) {
        // Completion is still recorded; event telemetry must not terminate polling/waiting.
    }
}

template <typename State>
void MarkCompletionFailure(const std::shared_ptr<State>& state,
                           const char* reason) noexcept {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->ok = false;
    state->failure_kind = RuntimeShapeFailureKind::kCompletionFailed;
    state->failure_reason = reason;
    // A quarantined run has no proven dependency attachment, so its outputs stay alive.
    if (!state->quarantined) state->outputs.clear();
    // The completion still owns CUDA Storage; do not under-report retained bytes.
    try {
        state->events.push_back(FailureEvent(reason));
    } catch (...) {
    }
}

RuntimeShapeFailureKind ClassifyShapeFailure(const std::string& reason) {
    if (reason.find("guard") != std::string::npos ||
        reason.find("outside its ABI domain") != std::string::npos) {
        return RuntimeShapeFailureKind::kApplicabilityMiss;
    }
    if (reason.find("budget") != std::string::npos ||
        reason.find("max_bytes") != std::string::npos ||
        reason.find("out of memory") != std::string::npos) {
        return RuntimeShapeFailureKind::kResourceExhausted;
    }
    return RuntimeShapeFailureKind::kShapeAbiRejected;
}
}  // namespace

bool RuntimeShapeAsyncResult::IsReady() const noexcept {
    if (!state_) return true;
    if (state_->completion.defined()) {
        try {
            if (!state_->completion.IsReady()) return false;
            MarkCompletion(state_);
            return true;
        } catch (...) {
            MarkCompletionFailure(state_, "CUDA completion query failed");
            return false;
        }
    }
    return !state_->fake_completion || state_->fake_completion->IsReady();
}

void RuntimeShapeAsyncResult::Wait() const noexcept {
    if (!state_) return;
    if (state_->completion.defined()) {
        try {
            state_->completion.Wait();
            MarkCompletion(state_);
        } catch (...) {
            MarkCompletionFailure(state_, "CUDA completion wait failed");
        }
        return;
    }
    if (state_->fake_completion) state_->fake_completion->Complete();
}

std::size_t RuntimeShapeAsyncResult::retained_device_bytes() const noexcept {
    return state_ ? state_->retained_device_bytes : 0;
}

RuntimeShapeSession::RuntimeShapeSession(RuntimeShapePlan plan) : plan_(std::move(plan)) {}

RuntimeShapeAsyncResult RuntimeShapeSession::RunAsync(
    const std::vector<RuntimeShapeInput>& inputs, std::shared_ptr<void> caller_lease) const {
    return RunImpl(inputs, ::kxc::DeviceStream(), std::move(caller_lease));
}

RuntimeShapeAsyncResult RuntimeShapeSession::RunAsync(
    const std::vector<RuntimeShapeInput>& inputs, ::kxc::DeviceStream stream,
    std::shared_ptr<void> caller_lease) const {
    return RunImpl(inputs, std::move(stream), std::move(caller_lease));
}

RuntimeShapeAsyncResult RuntimeShapeSession::RunImpl(
    const std::vector<RuntimeShapeInput>& inputs, ::kxc::DeviceStream stream,
    std::shared_ptr<void> caller_lease) const {
    auto state = std::make_shared<RuntimeShapeAsyncResult::State>();
    state->plan = plan_;
    state->caller_lease = std::move(caller_lease);
    const auto fail = [&](RuntimeShapeFailureKind kind, const std::string& reason) {
        state->ok = false;
        state->failure_kind = kind;
        state->failure_reason = reason;
        state->outputs.clear();
        state->retained_device_bytes = 0;
        state->events.push_back(FailureEvent(reason));
        return RuntimeShapeAsyncResult(state);
    };
#if !KXC_ENABLE_RUNTIME_SHAPE_TASKS
    (void)inputs;
    (void)stream;
    return fail(RuntimeShapeFailureKind::kDisabled, "disabled by KXC_ENABLE_RUNTIME_SHAPE_TASKS");
#else
    try {
        plan_.Validate();
        const auto& spec = plan_.spec();
        const bool cuda_async = spec.entry.execution_kind == RuntimeShapeExecutionKind::kCudaAsync;
#if !KXC_ENABLE_RUNTIME_SHAPE_CUDA || !KXC_USE_CUDA
        if (cuda_async) {
            return fail(RuntimeShapeFailureKind::kDisabled,
                        "disabled by KXC_ENABLE_RUNTIME_SHAPE_CUDA or CUDA backend");
        }
#endif
        const ::kxc::Device execution_device = cuda_async
            ? DeviceForContract(spec.entry.device_contract) : ::kxc::Device::CPU();
        if (cuda_async) {
            if (!stream.defined()) stream = ::kxc::DeviceStream::Default(execution_device);
            if (stream.device() != execution_device) {
                return fail(RuntimeShapeFailureKind::kShapeAbiRejected,
                            "CUDA stream device does not match entry device contract");
            }
            // Preserve the stream even when a callback fails to return completion.
            state->stream = stream;
        }
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
            if (cuda_async) {
                if (!input.storage.defined() || input.storage.device() != execution_device ||
                    input.storage.capacity_bytes() < expected_bytes || input.bytes != expected_bytes ||
                    input.data != input.storage.data() ||
                    (expected_bytes != 0 && input.data == nullptr)) {
                    Fail("CUDA input storage, pointer, byte size, or device does not match contract");
                }
            } else if (contract.requires_data || input.data || input.bytes != 0 || input.owner) {
                if (input.bytes != expected_bytes || (expected_bytes != 0 && input.data == nullptr)) {
                    Fail("input data pointer or byte size does not match shape contract");
                }
            }
            state->input_owners.push_back(input.owner);
            if (cuda_async) state->input_storage.push_back(input.storage);
            for (const auto& guard : contract.axis_guards) {
                const RuntimeShapeExtent extent = input.shape[guard.axis];
                if (extent < guard.lower || extent > guard.upper ||
                    extent % guard.divisible_by != 0 ||
                    (guard.exact && extent != *guard.exact)) {
                    return fail(RuntimeShapeFailureKind::kApplicabilityMiss,
                                "input axis guard mismatch");
                }
            }
            input_shapes.push_back(input.shape);
        }
        for (std::size_t index = 0; index < inputs.size(); ++index) for (const auto& guard : spec.inputs[index].axis_guards) {
            if (guard.equal_to && inputs[index].shape[guard.axis] !=
                inputs[guard.equal_to->input_index].shape[guard.equal_to->axis]) {
                return fail(RuntimeShapeFailureKind::kApplicabilityMiss,
                            "input axis equality guard mismatch");
            }
        }
        std::vector<RuntimeShapeExtent> runtime_extent_values;
        runtime_extent_values.reserve(spec.runtime_extent_abi.size());
        for (const auto& scalar : spec.runtime_extent_abi) {
            const auto value = inputs[scalar.input_index].shape[scalar.axis];
            if (value < scalar.lower || value > scalar.upper || value % scalar.divisible_by != 0) {
                return fail(RuntimeShapeFailureKind::kApplicabilityMiss,
                            "runtime extent scalar value is outside its ABI domain");
            }
            runtime_extent_values.push_back(value);
        }
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kShapeEval,
                                                   static_cast<std::size_t>(-1), 0,
                                                   cuda_async ? spec.entry.device_contract : "CPU:0/default"});
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
                if (output.valid[axis] > output.logical[axis] || output.logical[axis] > output.physical[axis]) {
                    Fail("valid <= logical <= physical is violated");
                }
            }
            output.bytes = CheckedBytes(output.physical, DTypeBytes(contract.dtype));
            if (output.bytes > contract.max_bytes) return fail(RuntimeShapeFailureKind::kResourceExhausted,
                                                                "output exceeds max_bytes");
            if (output.bytes > std::numeric_limits<std::size_t>::max() - total_bytes) {
                return fail(RuntimeShapeFailureKind::kResourceExhausted, "run byte budget overflow");
            }
            total_bytes += output.bytes;
            evaluated.push_back(OutputEvaluation{std::move(output)});
        }
        if (total_bytes > spec.run_byte_budget) {
            return fail(RuntimeShapeFailureKind::kResourceExhausted, "run byte budget exceeded");
        }
        for (std::size_t index = 0; index < evaluated.size(); ++index) {
            auto& output = evaluated[index].output;
            if (cuda_async) {
                output.storage_ = ::kxc::Storage::Alloc(execution_device, output.bytes,
                                                         output.contract.alignment);
                output.data = output.storage_.data();
            } else {
                AllocationOwner allocation = AllocateOutput(output.bytes, output.contract.alignment);
                if (fail_next_owner_transfer.exchange(false, std::memory_order_acq_rel)) throw std::bad_alloc();
                output.owner_ = std::shared_ptr<void>(std::move(allocation));
                output.data = output.owner_.get();
            }
            state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kAllocate, index,
                output.bytes, cuda_async ? spec.entry.device_contract : "CPU:0/default"});
            state->outputs.push_back(std::move(output));
        }
        if (!cuda_async) {
            RuntimeShapeLaunchResult launch;
            try {
                std::lock_guard<std::mutex> lock(plan_.launcher_mutex());
                launch = spec.entry.launcher(RuntimeShapeLaunchArgs{inputs, state->outputs,
                    runtime_extent_values, spec.entry.exact_abi_fingerprint});
            } catch (const std::exception& error) {
                return fail(RuntimeShapeFailureKind::kLaunchRejected,
                            std::string("bound launcher threw: ") + error.what());
            } catch (...) {
                return fail(RuntimeShapeFailureKind::kLaunchRejected,
                            "bound launcher threw a non-standard exception");
            }
            if (!launch.accepted) return fail(RuntimeShapeFailureKind::kLaunchRejected,
                launch.failure_reason.empty() ? "bound launcher rejected launch" : launch.failure_reason);
            state->fake_completion = std::move(launch.fake_completion);
            state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kKernel,
                static_cast<std::size_t>(-1), 0, spec.entry.entry_symbol});
            state->ok = true;
            return RuntimeShapeAsyncResult(std::move(state));
        }
        ::kxc::AsyncOperation completion;
        const auto fail_closed = [&](const char* reason) noexcept {
            // No completion proves the callback did not submit.  Quarantine the
            // entire run before reporting failure so no CUDA Storage can be freed.
            state->quarantined = true;
            state->retained_device_bytes = total_bytes;
            QuarantineRunState(state);
            try {
                state->ok = false;
                state->failure_kind = RuntimeShapeFailureKind::kSubmissionFailed;
                state->failure_reason = reason;
                state->events.push_back(FailureEvent(reason));
            } catch (...) {
            }
            return RuntimeShapeAsyncResult(state);
        };
        const auto fail_after_submission = [&](const char* reason) noexcept {
            if (!completion.defined()) return fail_closed(reason);
            try {
                completion.Wait();
                MarkCompletion(state);
            } catch (...) {
                // The returned operation is the only possible completion proof;
                // retain it and the run forever rather than clearing outputs.
                state->completion = std::move(completion);
                return fail_closed(reason);
            }
            return fail(RuntimeShapeFailureKind::kSubmissionFailed, reason);
        };
        try {
            std::lock_guard<std::mutex> lock(plan_.launcher_mutex());
            completion = spec.entry.cuda_launcher(RuntimeShapeCudaLaunchArgs{inputs, state->outputs,
                runtime_extent_values, spec.entry.exact_abi_fingerprint, stream});
        } catch (...) {
            return fail_closed("CUDA launcher threw after possible submission; run state quarantined");
        }
        if (!completion.defined()) {
            return fail_closed("CUDA launcher returned no completion; run state quarantined");
        }
        try {
            if (completion.device() != execution_device) {
                return fail_after_submission("CUDA launcher returned completion on wrong device");
            }
            if (completion->backend_event == nullptr || completion->completed) {
                return fail_after_submission(
                    "CUDA launcher must return a pending backend-event completion");
            }
            auto retention = std::make_shared<RuntimeShapeAsyncRetention>();
            retention->plan = plan_;
            retention->caller_lease = state->caller_lease;
            retention->input_owners = state->input_owners;
            retention->input_storage = state->input_storage;
            ::kxc::Array<::kxc::Storage> output_storage;
            for (const auto& output : state->outputs) output_storage.push_back(output.storage_);
            if (fail_next_cuda_retention.exchange(false, std::memory_order_acq_rel)) {
                throw std::bad_alloc();
            }
            completion.RetainDependencies(std::move(output_storage), retention);
        } catch (...) {
            return fail_after_submission("CUDA completion retention failed");
        }
        state->completion = std::move(completion);
        state->retained_device_bytes = total_bytes;
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kSubmission,
            static_cast<std::size_t>(-1), total_bytes, spec.entry.entry_symbol});
        state->ok = true;
        return RuntimeShapeAsyncResult(std::move(state));
    } catch (const std::bad_alloc&) {
        return fail(RuntimeShapeFailureKind::kResourceExhausted, "output allocation failed: out of memory");
    } catch (const std::exception& error) {
        return fail(ClassifyShapeFailure(error.what()), error.what());
    } catch (...) {
        return fail(RuntimeShapeFailureKind::kShapeAbiRejected,
                    "runtime shape execution threw a non-standard exception");
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
