/*! \file include/kxc/frontend/onnx_importer.h */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "kxc/runtime/ndarray.h"
#include "kxc/relay/relay.h"

namespace kxc {
namespace frontend {

struct ImportedONNXModel {
    Function function;
    std::unordered_map<std::string, runtime::NDArray> params;
    std::vector<std::string> param_order;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<TensorType> declared_output_types;
};

ImportedONNXModel LoadONNXImportSpec(const std::string& json_path,
                                     const std::string& params_path);

// Parses kxc.onnx_shape_source.v1 into a source Function with unresolved shape
// controls. It is not a typed/executable import. Pass function and
// declared_output_types together to RestrictedSymbolicShapeAdapter::Prepare;
// that producer proves the controls and checks the representative outputs.
// The ordinary static loader rejects this format.
ImportedONNXModel LoadONNXShapeSource(const std::string& json_path,
                                     const std::string& params_path);

}  // namespace frontend
}  // namespace kxc
