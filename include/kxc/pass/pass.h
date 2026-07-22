/*! \file include/kxc/pass/pass.h
 * \brief Defines IR-independent pass metadata and registry validation.
 */

#pragma once

#include <string>

#include "kxc/support/container.h"

namespace kxc {

enum class IRDialect {
    kUnknown,
    kRelay,
    kTIR,
};

enum class PassScope {
    kUnknown,
    kGraph,
    kCompilationUnit,
    kPrimFunc,
    kModule,
};

struct PassSpec {
    String name;
    int schema_version{1};
    IRDialect dialect{IRDialect::kUnknown};
    PassScope scope{PassScope::kUnknown};
    String phase;
    int opt_level{0};
    Array<String> required_invariants;
    Array<String> produced_invariants;
    Array<String> preserved_analyses;
    Array<String> invalidated_analyses;
    bool may_change_ir{true};
    bool deterministic{true};
    bool idempotent{false};
    bool thread_safe{false};
    bool target_dependent{false};
    String implementation_key;
};

std::string ToString(IRDialect dialect);
std::string ToString(PassScope scope);
IRDialect ParseIRDialect(const std::string& value);
PassScope ParsePassScope(const std::string& value);

std::string PassSpecKey(IRDialect dialect, const String& name);
void ValidatePassSpec(const PassSpec& spec);
void ValidatePassSpecForPipeline(const PassSpec& spec, IRDialect dialect,
                                 PassScope scope, const std::string& phase);
void ValidatePassSpecs(const Array<PassSpec>& specs);

class PassRegistry {
public:
    static PassRegistry& Global();

    void Register(PassSpec spec);
    bool Contains(IRDialect dialect, const String& name) const;
    const PassSpec& Get(IRDialect dialect, const String& name) const;
    Array<PassSpec> List() const;
    void FreezeAndValidate() const;

private:
    PassRegistry() = default;
};

}  // namespace kxc
