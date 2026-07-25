/*! \file test/type_registration_test.cpp
 * \brief Verify every stable type key remains registered exactly once.
 */

#include "kxc/compiler/compile_config.h"
#include "kxc/support/container.h"
#include "kxc/runtime/device.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/relay/relay.h"
#include "kxc/te/te.h"
#include "kxc/tir/expr.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string_view>

namespace {

#define TEST_CHECK(cond, message)                         \
    do {                                                  \
        if (!(cond)) {                                    \
            std::cerr << "[FAIL] " << (message) << '\n'; \
            return false;                                 \
        }                                                 \
    } while (false)

bool TestStableRegistrations() {
    const std::string_view expected_keys[] = {
        "Object",
        "String",
        "Array",
        "Map",
        "PackedFuncObj",
        "Device",
        "DeviceStreamNode",
        "AsyncOperationNode",
        "DRefNode",
        "DiscoSessionNode",
        "ThreadedDiscoSessionNode",
        "WorkerPlacementNode",
        "DiscoPlacementNode",
        "ExecNodeBaseNode",
        "KernelExecNode",
        "CommExecNode",
        "BarrierExecNode",
        "ExecutionPlanNode",
        "TypeNode",
        "SpanNode",
        "ExprNode",
        "NDArrayNode",
        "StorageNode",
        "TargetNode",
        "VirtualDeviceNode",
        "CompileConfigNode",
        "kxc.api.CompiledModuleNode",
        "kxc.codegen.CompiledKernelNode",
        "kxc.codegen.KernelConstantKeysNode",
        "kxc.codegen.KernelArgSpecNode",
        "kxc.codegen.KernelSignatureNode",
        "kxc.codegen.KernelLaunchMetadataNode",
        "kxc.runtime.ValueSpecNode",
        "kxc.runtime.KernelCallNode",
        "kxc.runtime.ExecutablePlanNode",
        "OpNode",
        "BaseAttrsNode",
        "Conv2DAttrsNode",
        "DenseAttrsNode",
        "MaxPool2DAttrsNode",
        "SoftmaxAttrsNode",
        "LayerNormAttrsNode",
        "AddAttrsNode",
        "CastAttrsNode",
        "ReduceMeanAttrsNode",
        "ReshapeAttrsNode",
        "TransposeAttrsNode",
        "GatherAttrsNode",
        "ConcatenateAttrsNode",
        "SliceAttrsNode",
        "ReluAttrsNode",
        "GlobalAvgPool2DAttrsNode",
        "FlattenAttrsNode",
        "GemmAttrsNode",
        "DeviceCopyAttrsNode",
        "CollectiveAttrsNode",
        "TensorTypeNode",
        "TupleTypeNode",
        "IdNode",
        "kxc.relay.VarNode",
        "ConstantNode",
        "kxc.relay.CallNode",
        "FunctionNode",
        "TupleNode",
        "TupleGetItemNode",
        "IfNode",
        "WhileNode",
        "LetNode",
        "kxc.compiler.internal.ConstantBindingNode",
        "kxc.compiler.internal.LoweredFunctionNode",
        "kxc.runtime.RuntimeSessionNode",
        "OperationNode",
        "TensorNode",
        "ProducerLoadNode",
        "kxc.te.IterVarNode",
        "StageNode",
        "ReduceNode",
        "PlaceholderOpNode",
        "ComputeOpNode",
        "ScheduleNode",
        "PrimExprNode",
        "IntImmNode",
        "FloatImmNode",
        "kxc.tir.VarNode",
        "AddNode",
        "SubNode",
        "MulNode",
        "DivNode",
        "ModNode",
        "MinNode",
        "MaxNode",
        "EQNode",
        "LTNode",
        "AndNode",
        "OrNode",
        "NotNode",
        "LoadNode",
        "kxc.tir.CallNode",
        "SelectNode",
        "StmtNode",
        "LetStmtNode",
        "StoreNode",
        "ForNode",
        "kxc.tir.ThreadBindingNode",
        "IfThenElseNode",
        "AllocateNode",
        "AttrStmtNode",
        "RangeNode",
        "kxc.tir.IterVarNode",
        "BufferNode",
        "BufferRegionNode",
        "BlockNode",
        "SeqStmtNode",
        "EvaluateNode",
        "PrimFuncNode",
        "kxc.tir.CudaScheduleResultNode",
    };

    std::set<kxc::TypeIndex> runtime_indices;
    for (std::string_view key : expected_keys) {
        const kxc::TypeInfo* info = kxc::TypeRegistry::FindByKey(key);
        TEST_CHECK(info != nullptr, "expected stable type key is missing");
        TEST_CHECK(info->type_key() == key, "registry returned a mismatched type key");
        TEST_CHECK(kxc::TypeRegistry::FindByIndex(info->runtime_index()) == info,
                   "runtime index lookup should return the same TypeInfo");
        TEST_CHECK(runtime_indices.insert(info->runtime_index()).second,
                   "each stable type key should have one unique runtime index");
    }
    return true;
}

}  // namespace

int main() {
    if (!TestStableRegistrations()) return EXIT_FAILURE;
    std::cout << "All type registration tests passed.\n";
    return EXIT_SUCCESS;
}
