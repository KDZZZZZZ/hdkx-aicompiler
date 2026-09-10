// Actual MiniMind lowering/source compilation evidence, with no GPU execution.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "codegen/cuda/internal/codegen_cuda.h"
#include "compiler/internal/dynamic_shape_contract.h"
#include "compiler/internal/execution_contract.h"
#include "compiler/internal/lowered_graph.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/tir/printer/print_ir.h"
#include "support/request_batching_graph.h"

int main(int argc, char** argv) {
    using namespace kxc;
    namespace ci = api::internal;
    using Adapter = api::experimental::restricted_symbolic_shape::v1::RestrictedSymbolicShapeAdapter;
    try {
        const std::string graph_name = argc == 3 ? argv[2] : "prefill";
        const bool decode = graph_name == "decode";
        const bool requests = graph_name == "requests-axis1" || graph_name == "requests-axis2";
        if ((argc != 2 && argc != 3) || (!decode && !requests && graph_name != "prefill"))
            throw std::invalid_argument("usage: minimind_bounded_cuda_codegen_test OUTPUT_DIRECTORY [decode|requests-axis1|requests-axis2]");
        const size_t expected_calls = requests ? 3 : decode ? 774 : 742;
        const char* fixture = std::getenv(decode ? "KXC_MINIMIND_BOUNDED_DECODE_DIR" : "KXC_MINIMIND_BOUNDED_PREFILL_DIR");
        if (!requests && (!fixture || !*fixture)) {
            std::cout << "[SKIP] set KXC_MINIMIND_BOUNDED_" << (decode ? "DECODE" : "PREFILL") << "_DIR\n";
            return 77;
        }
        const std::filesystem::path root(fixture ? fixture : ""), out(argv[1]);
        std::filesystem::create_directories(out);
        // An explicit synthetic target tests compiler policy only. No device
        // is queried, allocated, launched or replaced by CPU execution.
        auto* node = new TargetNode();
        node->kind = "cuda"; node->device_type = kCUDA; node->device_id = 0;
        node->attrs.device_name = "synthetic CUDA codegen check"; node->attrs.arch = "sm_89";
        node->attrs.warp_size = 32; node->attrs.multi_processor_count = 1;
        node->attrs.exists = 1; node->attrs.max_threads_per_block = 1024;
        node->attrs.max_shared_memory_per_block = 48 * 1024;
        node->attrs.compute_version_major = 8; node->attrs.compute_version_minor = 9;
        const Target target{ObjectRef(node)};
        const auto config = api::CompileConfig::Create(target, 3);
        const auto request = [&] {
            if (requests) {
                const int64_t axis = graph_name == "requests-axis1" ? 1 : 2;
                return test_support::RequestBatchingGraph(config,axis,axis);
            }
            const auto source = frontend::LoadONNXShapeSource((root/(graph_name+".json")).string(), (root/(graph_name+".params")).string());
            if (source.input_names.size() != (decode ? 17U : 1U) || source.input_names[0] != "input_ids" ||
                source.declared_output_types.size() != 17) throw std::runtime_error("full model boundary drifted");
            std::vector<api::experimental::restricted_symbolic_shape::v1::InputAxisSymbol> axes{{0,0,"B",1,3,1}};
            if (decode) {
                for (size_t i = 1; i < 17; ++i) {
                    if (source.input_names[i] != "past_" + std::string(i%2 ? "k_" : "v_") + std::to_string((i-1)/2))
                        throw std::runtime_error("decode KV input order drifted");
                    axes.push_back({i,0,"B",1,3,1}); axes.push_back({i,1,"P",0,8,1});
                }
            } else axes.push_back({0,1,"S",1,8,1});
            const auto prepared = Adapter::Prepare(source.function, config, axes, source.declared_output_types);
            return Adapter::MintBoundedCompileRequest(prepared);
        }();
        const auto preparation = ci::PrepareBoundedCompile(request);
        const auto contract = ci::ResolveCompilerExecutionContract(config);
        const auto& partition = preparation.partitioned_graph();
        if (partition.units.size() != expected_calls) throw std::runtime_error("full model primitive count changed");
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        codegen::CodeGenCUDA emitter;
        size_t failures = 0;
        for (const auto& unit : partition.units) {
            tir::PrimFunc original;
            try {
                const auto lowered = ci::LowerPrimitiveUnit(partition.value_graph.values, unit, target,
                    preparation.unit_shape_contracts().at(unit.id));
                original = lowered->prim_func;
                const auto result = api::PipelineExecutor::ExecuteTIR(contract.tir_pipeline, original, target);
                (void)emitter.Generate(result, std::string(unit.symbol));
                functions.emplace_back(result, std::string(unit.symbol));
            } catch (const std::exception& error) {
                std::cerr << "[FAIL] unit " << unit.id << ' ' << std::string(unit.symbol) << ": " << error.what() << '\n';
                if (original.defined()) {
                    std::ofstream file(out/("unit" + std::to_string(unit.id) + ".tir"));
                    tir::printer::DumpPrimFunc(original, file);
                }
                ++failures;
            }
        }
        if (failures) throw std::runtime_error(std::to_string(failures) + " model primitives failed CUDA codegen");
        std::ofstream file(out/"full-minimind.cu");
        file << emitter.GenerateModule(functions);
        file.close();
        if (!file) throw std::runtime_error("could not write full model CUDA source");
        std::cout << "[PASS] full_minimind_bounded_cuda_codegen: " << expected_calls << '/' << expected_calls
                  << " primitives; " << graph_name << "; synthetic sm_89; no GPU execution\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
