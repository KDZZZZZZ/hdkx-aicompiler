#include <iostream>

#include "kxc/runtime/runtime_shape_session.h"

int main() {
    using namespace kxc::runtime;
    RuntimeShapeTensorContract output;
    output.dtype = "float32";
    output.logical = {RuntimeShapeExpr::Const(1)};
    output.physical = {RuntimeShapeExpr::Const(1)};
    output.valid = {RuntimeShapeExpr::Const(1)};
    output.max_bytes = 4;
    RuntimeShapePlanSpec spec;
    spec.inputs = {{"float32", 1, "CPU:0", 1}};
    spec.outputs = {output};
    spec.entry.module_label = "fake";
    spec.entry.entry_symbol = "entry";
    spec.entry.ready = true;
    spec.entry.exact_abi_fingerprint =
        RuntimeShapePlan::ExactAbiFingerprint(spec.inputs, spec.outputs);
    spec.entry.launcher = [](const RuntimeShapeLaunchArgs&) { return RuntimeShapeLaunchResult{}; };
    spec.run_byte_budget = 4;
    const RuntimeShapeAsyncResult result =
        RuntimeShapeSession(RuntimeShapePlan(std::move(spec))).Run({{{1}, "float32", "CPU:0", 1}});
    if (result.ok() || result.failure_reason().find("disabled") == std::string::npos ||
        result.events().size() != 1 || result.events()[0].kind != RuntimeShapeEventKind::kFailure) {
        std::cerr << "runtime shape gate-off contract failed\n";
        return 1;
    }
    std::cout << "runtime_shape_gate_off_test: PASS\n";
    return 0;
}
