/*! \file test/support/control_plan_reference_executor.h
 * \brief Deterministic fake interpreter for ControlPlan tests.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/runtime/control_plan.h"

namespace kxc::runtime::test_support {

struct FakeValue {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string device{"cpu"};
    bool boolean{false};
    std::int64_t integer{0};

    static FakeValue Bool(bool value) { return FakeValue{"bool", {}, "cpu", value, value ? 1 : 0}; }
    static FakeValue I64(std::int64_t value) {
        return FakeValue{"int64", {}, "cpu", false, value};
    }
};

using FakeValueTable = std::unordered_map<ValueId, FakeValue>;
using FakeKernelCallback = std::function<std::vector<FakeValue>(
    const ControlTask&, const std::vector<FakeValue>&)>;

struct ControlTrace {
    std::vector<std::string> events;
};

struct ReferenceExecution {
    FakeValueTable values;
    ControlTrace trace;
};

class ControlPlanReferenceExecutor final {
public:
    explicit ControlPlanReferenceExecutor(FakeKernelCallback kernel)
        : kernel_(std::move(kernel)) {}

    ReferenceExecution Execute(const ControlPlan& plan, const FakeValueTable& inputs) const {
        plan.Validate();
        if (!kernel_) throw std::invalid_argument("reference executor requires a kernel callback");
        ReferenceExecution result;
        for (const ValueId id : plan.graph_inputs) {
            const auto it = inputs.find(id);
            if (it == inputs.end()) throw std::invalid_argument("missing graph input");
            CheckContract(plan, id, it->second);
            result.values.emplace(id, it->second);
        }
        std::unordered_map<RegionId, const ControlRegion*> regions;
        for (const ControlRegion& region : plan.regions) regions.emplace(region.id, &region);
        ExecuteRegion(plan, regions, plan.entry_region, &result);
        for (const ValueId id : plan.graph_outputs) {
            const auto it = result.values.find(id);
            if (it == result.values.end()) throw std::invalid_argument("missing graph output");
            CheckContract(plan, id, it->second);
        }
        return result;
    }

private:
    static const ControlValueSpec& Spec(const ControlPlan& plan, ValueId id) {
        for (const ControlValueSpec& spec : plan.values) if (spec.id == id) return spec;
        throw std::invalid_argument("unknown value");
    }

    static void CheckContract(const ControlPlan& plan, ValueId id, const FakeValue& value) {
        const ControlValueSpec& spec = Spec(plan, id);
        if (value.dtype != spec.dtype || value.shape != spec.shape || value.device != spec.device) {
            throw std::invalid_argument("fake value violates its static contract");
        }
    }

    static const FakeValue& Read(const FakeValueTable& values, ValueId id) {
        const auto it = values.find(id);
        if (it == values.end()) throw std::invalid_argument("task reads a missing value");
        return it->second;
    }

    void ExecuteRegion(const ControlPlan& plan,
                       const std::unordered_map<RegionId, const ControlRegion*>& regions,
                       RegionId id, ReferenceExecution* result) const {
        const auto region = regions.find(id);
        if (region == regions.end()) throw std::invalid_argument("missing region");
        for (const ControlTask& task : region->second->tasks) ExecuteTask(plan, regions, task, result);
    }

    void ExecuteTask(const ControlPlan& plan,
                     const std::unordered_map<RegionId, const ControlRegion*>& regions,
                     const ControlTask& task, ReferenceExecution* result) const {
        std::vector<FakeValue> arguments;
        arguments.reserve(task.inputs.size());
        for (const ValueId input : task.inputs) {
            arguments.push_back(Read(result->values, input));
            result->trace.events.push_back("read:" + std::to_string(task.id) + ":" + std::to_string(input));
        }
        result->trace.events.push_back("task:" + std::to_string(task.id));
        if (task.kind == ControlTaskKind::kKernel) {
            const std::vector<FakeValue> outputs = kernel_(task, arguments);
            if (outputs.size() != task.outputs.size()) throw std::invalid_argument("kernel output arity mismatch");
            for (std::size_t i = 0; i < outputs.size(); ++i) Write(plan, task, task.outputs[i], outputs[i], result);
            return;
        }
        if (task.kind == ControlTaskKind::kBranch) {
            const bool choice = Read(result->values, task.branch.predicate).boolean;
            result->trace.events.push_back("branch:" + std::to_string(task.id) + (choice ? ":then" : ":else"));
            ExecuteRegion(plan, regions, choice ? task.branch.then_region : task.branch.else_region, result);
            for (const PhiBinding& phi : task.branch.phis) {
                const ValueId source = choice ? phi.then_value : phi.else_value;
                Write(plan, task, phi.result, Read(result->values, source), result);
            }
            return;
        }
        std::unordered_map<ValueId, FakeValue> current;
        for (const LoopCarriedBinding& binding : task.loop.carried) current.emplace(binding.body_argument, Read(result->values, binding.initial));
        std::int64_t iteration = 0;
        while (true) {
            for (const auto& binding : task.loop.carried) result->values[binding.body_argument] = current.at(binding.body_argument);
            ExecuteRegion(plan, regions, task.loop.condition_region, result);
            if (!Read(result->values, task.loop.condition_value).boolean) break;
            if (iteration >= task.loop.max_trip_count) throw std::runtime_error("loop max_trip_count exhausted");
            result->trace.events.push_back("loop:" + std::to_string(task.id) + ":iteration:" + std::to_string(iteration));
            ExecuteRegion(plan, regions, task.loop.body_region, result);
            for (const LoopCarriedBinding& binding : task.loop.carried) current[binding.body_argument] = Read(result->values, binding.backedge);
            ++iteration;
        }
        for (const LoopCarriedBinding& binding : task.loop.carried) Write(plan, task, binding.result, current.at(binding.body_argument), result);
    }

    static void Write(const ControlPlan& plan, const ControlTask& task, ValueId id,
                      const FakeValue& value, ReferenceExecution* result) {
        CheckContract(plan, id, value);
        result->values[id] = value;
        result->trace.events.push_back("write:" + std::to_string(task.id) + ":" + std::to_string(id));
    }

    FakeKernelCallback kernel_;
};

}  // namespace kxc::runtime::test_support
