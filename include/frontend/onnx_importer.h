#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "base/ndarray.h"
#include "relay/relay.h"

namespace kxc {
namespace frontend {

struct ImportedONNXModel {
    Function function;
    std::unordered_map<std::string, runtime::NDArray> params;
    std::vector<std::string> param_order;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

ImportedONNXModel LoadONNXImportSpec(const std::string& json_path,
                                     const std::string& params_path);

}  // namespace frontend
}  // namespace kxc
