/*! \brief Trusted lowering access; mutable TE handles never escape publicly. */
#pragma once

#include "kxc/te/program.h"

namespace kxc::te::internal {

struct ProgramState final {
    Array<Tensor> inputs;
    Array<Tensor> constants;
    Array<Tensor> outputs;
    Array<Tensor> metadata_only_inputs;
    Schedule schedule;
    std::string target_canonical;
    std::string tir_pipeline_canonical;
    std::string canonical;
    std::string digest;
};

struct ProgramAccess final {
    static const ProgramState& Borrow(const Program& program);
};

}  // namespace kxc::te::internal
