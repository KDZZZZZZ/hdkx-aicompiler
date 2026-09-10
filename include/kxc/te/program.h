/*! \file include/kxc/te/program.h
 * \brief Immutable static TE candidate, consumed by the existing TIR lowerer.
 */
#pragma once

#include <memory>
#include <string>

#include "kxc/target/target.h"
#include "kxc/te/te.h"

namespace kxc::te {
namespace internal {
struct ProgramAccess;
struct ProgramState;
}

// A Program freezes one candidate. It does not choose schedules, run passes,
// retain constant payloads, or compile. Boundaries are ordered, disjoint,
// contiguous tensors on the explicit Target; inputs/constants are read-only
// and outputs are fresh. Metadata-only inputs remain part of the callable ABI.
class Program final {
public:
    Program() = default;
    Program(const Array<Tensor>& inputs, const Array<Tensor>& constants,
            const Array<Tensor>& outputs, const Schedule& schedule,
            const Target& target, std::string tir_pipeline_canonical,
            const Array<Tensor>& metadata_only_inputs = {});

    bool defined() const noexcept;
    const std::string& canonical_bytes() const;
    const std::string& digest() const;

private:
    friend struct internal::ProgramAccess;
    std::shared_ptr<const internal::ProgramState> state_;
};

}  // namespace kxc::te
