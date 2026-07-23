/*! \file src/runtime/runtime_shape_session.cc */

#include "kxc/runtime/runtime_shape_session.h"

#include <atomic>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace kxc::runtime {
namespace {

#ifndef KXC_ENABLE_RUNTIME_SHAPE_TASKS
#define KXC_ENABLE_RUNTIME_SHAPE_TASKS 0
#endif

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
    for (const auto extent : shape) {
        if (extent == 0) Fail("zero extent is not supported");
        elements = CheckedMul(elements, extent);
    }
    if (elements > std::numeric_limits<std::size_t>::max() / dtype_bytes) {
        Fail("output byte size overflow");
    }
    return static_cast<std::size_t>(elements) * dtype_bytes;
}

std::vector<RuntimeShapeExtent> EvaluateShape(
    const std::vector<RuntimeShapeExpr>& expressions,
    const std::vector<std::vector<RuntimeShapeExtent>>& input_shapes,
    const char* name) {
    std::vector<RuntimeShapeExtent> result;
    result.reserve(expressions.size());
    for (const auto& expression : expressions) {
        const auto extent = expression.Evaluate(input_shapes);
        if (extent == 0) Fail(std::string(name) + " has a zero extent");
        result.push_back(extent);
    }
    return result;
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
    if (contract.layout.empty() || contract.scope.empty()) {
        Fail("layout and scope must be nonempty");
    }
    if (contract.max_bytes == 0) Fail("output max_bytes must be nonzero");
    if (contract.device != "CPU:0") Fail("v1 requires output device CPU:0");
    if (contract.abi_version != RuntimeShapePlan::kAbiVersion) {
        Fail("output ABI version does not match");
    }
}

struct OutputEvaluation {
    RuntimeShapeOutput output;
};

RuntimeShapeEvent FailureEvent(const std::string& detail) {
    return RuntimeShapeEvent{RuntimeShapeEventKind::kFailure,
                             static_cast<std::size_t>(-1), 0, detail};
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

RuntimeShapeExpr RuntimeShapeExpr::Add(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    auto node = std::make_shared<Node>();
    node->kind = Kind::kAdd;
    node->lhs = std::move(lhs.node_);
    node->rhs = std::move(rhs.node_);
    if (!node->lhs || !node->rhs) Fail("Add requires defined expressions");
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::Mul(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    auto node = std::make_shared<Node>();
    node->kind = Kind::kMul;
    node->lhs = std::move(lhs.node_);
    node->rhs = std::move(rhs.node_);
    if (!node->lhs || !node->rhs) Fail("Mul requires defined expressions");
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::FloorDiv(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    auto node = std::make_shared<Node>();
    node->kind = Kind::kFloorDiv;
    node->lhs = std::move(lhs.node_);
    node->rhs = std::move(rhs.node_);
    if (!node->lhs || !node->rhs) Fail("FloorDiv requires defined expressions");
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::Min(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    auto node = std::make_shared<Node>();
    node->kind = Kind::kMin;
    node->lhs = std::move(lhs.node_);
    node->rhs = std::move(rhs.node_);
    if (!node->lhs || !node->rhs) Fail("Min requires defined expressions");
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExpr RuntimeShapeExpr::Max(RuntimeShapeExpr lhs, RuntimeShapeExpr rhs) {
    auto node = std::make_shared<Node>();
    node->kind = Kind::kMax;
    node->lhs = std::move(lhs.node_);
    node->rhs = std::move(rhs.node_);
    if (!node->lhs || !node->rhs) Fail("Max requires defined expressions");
    return RuntimeShapeExpr(std::move(node));
}

RuntimeShapeExtent RuntimeShapeExpr::Evaluate(
    const std::vector<std::vector<RuntimeShapeExtent>>& input_shapes) const {
    if (!node_) Fail("undefined shape expression");
    const auto evaluate = [&](const auto& self, const std::shared_ptr<const Node>& node)
        -> RuntimeShapeExtent {
        switch (node->kind) {
            case Kind::kConst: return node->value;
            case Kind::kInputAxis:
                if (node->input_index >= input_shapes.size() ||
                    node->axis >= input_shapes[node->input_index].size()) {
                    Fail("input-axis expression is out of range");
                }
                return input_shapes[node->input_index][node->axis];
            case Kind::kAdd: return CheckedAdd(self(self, node->lhs), self(self, node->rhs));
            case Kind::kMul: return CheckedMul(self(self, node->lhs), self(self, node->rhs));
            case Kind::kFloorDiv: {
                const auto divisor = self(self, node->rhs);
                if (divisor == 0) Fail("floor division by zero");
                return self(self, node->lhs) / divisor;
            }
            case Kind::kMin: {
                const auto lhs = self(self, node->lhs);
                const auto rhs = self(self, node->rhs);
                return lhs < rhs ? lhs : rhs;
            }
            case Kind::kMax: {
                const auto lhs = self(self, node->lhs);
                const auto rhs = self(self, node->rhs);
                return lhs > rhs ? lhs : rhs;
            }
        }
        Fail("unknown shape expression kind");
    };
    return evaluate(evaluate, node_);
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

struct RuntimeShapePlan::Impl {
    explicit Impl(RuntimeShapePlanSpec value) : spec(std::move(value)) {}
    const RuntimeShapePlanSpec spec;
};

RuntimeShapePlan::RuntimeShapePlan(RuntimeShapePlanSpec spec)
    : impl_(std::make_shared<Impl>(std::move(spec))) {
    Validate();
}

bool RuntimeShapePlan::defined() const noexcept { return static_cast<bool>(impl_); }

void RuntimeShapePlan::Validate() const {
    if (!impl_) Fail("plan is undefined");
    const auto& spec = impl_->spec;
    if (spec.abi_version != kAbiVersion) Fail("plan ABI version does not match");
    if (spec.inputs.empty() || spec.outputs.empty()) Fail("plan requires inputs and outputs");
    if (spec.run_byte_budget == 0) Fail("run byte budget must be nonzero");
    for (const auto& input : spec.inputs) {
        if (input.dtype.empty()) Fail("input dtype is empty");
        (void)DTypeBytes(input.dtype);
        if (input.device != "CPU:0") Fail("v1 requires input device CPU:0");
        if (input.abi_version != kAbiVersion) Fail("input ABI version does not match");
    }
    for (const auto& output : spec.outputs) ValidateContract(output);
    if (!spec.entry.ready || !spec.entry.launcher || spec.entry.module_label.empty() ||
        spec.entry.entry_symbol.empty()) {
        Fail("plan requires a selected ready bound launcher entry");
    }
    if (spec.entry.abi_version != kAbiVersion) Fail("entry ABI version does not match");
}

const RuntimeShapePlanSpec& RuntimeShapePlan::spec() const {
    if (!impl_) Fail("plan is undefined");
    return impl_->spec;
}

struct RuntimeShapeAsyncResult::State {
    RuntimeShapePlan plan;
    std::shared_ptr<void> caller_lease;
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
    state->plan = plan_;  // Each result owns its frozen plan snapshot and module lease.
    state->caller_lease = std::move(caller_lease);
    const auto fail = [&](const std::string& reason) {
        state->ok = false;
        state->failure_reason = reason;
        state->outputs.clear();  // Launch failures never publish or reuse allocations.
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
            for (const auto extent : input.shape) {
                if (extent == 0) Fail("input has a zero extent");
            }
            input_shapes.push_back(input.shape);
        }
        state->events.push_back(
            RuntimeShapeEvent{RuntimeShapeEventKind::kShapeEval,
                              static_cast<std::size_t>(-1), 0, "CPU:0/default"});
        std::vector<OutputEvaluation> evaluated;
        evaluated.reserve(spec.outputs.size());
        std::size_t total_bytes = 0;
        for (const auto& contract : spec.outputs) {
            RuntimeShapeOutput output;
            output.contract = contract;
            output.logical = EvaluateShape(contract.logical, input_shapes, "logical shape");
            output.physical = EvaluateShape(contract.physical, input_shapes, "physical shape");
            output.valid = EvaluateShape(contract.valid, input_shapes, "valid shape");
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
            void* data = nullptr;
            if (output.contract.alignment > alignof(std::max_align_t)) {
                data = ::operator new(output.bytes, std::align_val_t(output.contract.alignment));
                output.owner_ = std::shared_ptr<void>(data, [alignment = output.contract.alignment](void* ptr) {
                    ::operator delete(ptr, std::align_val_t(alignment));
                });
            } else {
                data = ::operator new(output.bytes);
                output.owner_ = std::shared_ptr<void>(data, [](void* ptr) { ::operator delete(ptr); });
            }
            output.data = data;
            state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kAllocate, index,
                                                       output.bytes, "CPU:0/default"});
            state->outputs.push_back(std::move(output));
        }
        state->events.push_back(RuntimeShapeEvent{RuntimeShapeEventKind::kKernel,
                                                   static_cast<std::size_t>(-1), 0,
                                                   spec.entry.entry_symbol});
        RuntimeShapeLaunchResult launch = spec.entry.launcher(
            RuntimeShapeLaunchArgs{inputs, state->outputs});
        if (!launch.accepted) {
            return fail(launch.failure_reason.empty() ? "bound launcher rejected launch"
                                                       : launch.failure_reason);
        }
        state->fake_completion = std::move(launch.fake_completion);
        state->ok = true;
        return RuntimeShapeAsyncResult(std::move(state));
    } catch (const std::bad_alloc&) {
        return fail("output allocation failed: out of memory");
    } catch (const std::exception& error) {
        return fail(error.what());
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
