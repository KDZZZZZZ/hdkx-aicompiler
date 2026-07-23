/*! \file src/pass/pass.cc
 * \brief Implements IR-independent pass metadata and registry validation.
 */

#include "kxc/pass/pass.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kxc {
namespace {

std::string AsStdString(const String& value) {
    return value.defined() ? static_cast<std::string>(value) : std::string();
}

struct PassRegistryState {
    mutable std::mutex mutex;
    std::map<std::string, PassSpec> specs;
    bool frozen{false};
};

PassRegistryState& State() {
    static PassRegistryState state;
    return state;
}

void RequireNonEmpty(const String& value, const char* field_name,
                     const std::string& pass_key) {
    if (AsStdString(value).empty()) {
        throw std::invalid_argument("PassSpec " + pass_key + " has empty " + field_name);
    }
}

std::set<std::string> ValidateNames(const Array<String>& values,
                                    const char* field_name,
                                    const std::string& pass_key) {
    std::set<std::string> names;
    for (const String& value : values) {
        const std::string name = AsStdString(value);
        if (name.empty() || !names.insert(name).second) {
            throw std::invalid_argument("PassSpec " + pass_key + " " +
                                        field_name +
                                        " must be unique and non-empty");
        }
    }
    return names;
}

}  // namespace

std::string ToString(IRDialect dialect) {
    switch (dialect) {
        case IRDialect::kUnknown:
            return "unknown";
        case IRDialect::kRelay:
            return "relay";
        case IRDialect::kTIR:
            return "tir";
    }
    throw std::invalid_argument("unknown IRDialect value");
}

std::string ToString(PassScope scope) {
    switch (scope) {
        case PassScope::kUnknown:
            return "unknown";
        case PassScope::kGraph:
            return "graph";
        case PassScope::kCompilationUnit:
            return "compilation_unit";
        case PassScope::kPrimFunc:
            return "prim_func";
        case PassScope::kModule:
            return "module";
    }
    throw std::invalid_argument("unknown PassScope value");
}

IRDialect ParseIRDialect(const std::string& value) {
    if (value == "relay") return IRDialect::kRelay;
    if (value == "tir") return IRDialect::kTIR;
    throw std::invalid_argument("unknown pass dialect: " + value);
}

PassScope ParsePassScope(const std::string& value) {
    if (value == "graph") return PassScope::kGraph;
    if (value == "compilation_unit") return PassScope::kCompilationUnit;
    if (value == "prim_func") return PassScope::kPrimFunc;
    if (value == "module") return PassScope::kModule;
    throw std::invalid_argument("unknown pass scope: " + value);
}

std::string PassSpecKey(IRDialect dialect, const String& name) {
    return ToString(dialect) + "." + AsStdString(name);
}

void ValidatePassSpec(const PassSpec& spec) {
    const std::string pass_key = PassSpecKey(spec.dialect, spec.name);
    RequireNonEmpty(spec.name, "name", pass_key);
    if (spec.dialect == IRDialect::kUnknown) {
        throw std::invalid_argument("PassSpec " + pass_key + " has no IR dialect");
    }
    if (spec.scope == PassScope::kUnknown) {
        throw std::invalid_argument("PassSpec " + pass_key + " has no pass scope");
    }
    if (spec.schema_version <= 0) {
        throw std::invalid_argument("PassSpec " + pass_key +
                                    " requires positive schema_version");
    }
    RequireNonEmpty(spec.phase, "phase", pass_key);
    if (spec.opt_level < 0) {
        throw std::invalid_argument("PassSpec " + pass_key +
                                    " requires non-negative opt_level");
    }
    RequireNonEmpty(spec.implementation_key, "implementation_key", pass_key);
    const std::set<std::string> required =
        ValidateNames(spec.required_invariants, "required_invariants", pass_key);
    const std::set<std::string> produced =
        ValidateNames(spec.produced_invariants, "produced_invariants", pass_key);
    const std::set<std::string> declarative = ValidateNames(
        spec.declarative_only_invariants, "declarative_only_invariants", pass_key);
    for (const std::string& invariant : declarative) {
        if (!required.count(invariant) && !produced.count(invariant)) {
            throw std::invalid_argument(
                "PassSpec " + pass_key + " marks undeclared invariant '" +
                invariant + "' as declarative-only");
        }
    }
    (void)ValidateNames(spec.preserved_analyses, "preserved_analyses", pass_key);
    (void)ValidateNames(spec.invalidated_analyses, "invalidated_analyses", pass_key);
    if (spec.dialect == IRDialect::kRelay && spec.scope == PassScope::kPrimFunc) {
        throw std::invalid_argument("PassSpec " + pass_key +
                                    " cannot use prim_func scope for Relay dialect");
    }
    if (spec.dialect == IRDialect::kTIR && spec.scope == PassScope::kGraph) {
        throw std::invalid_argument("PassSpec " + pass_key +
                                    " cannot use graph scope for TIR dialect");
    }
}

void ValidatePassSpecForPipeline(const PassSpec& spec, IRDialect dialect,
                                 PassScope scope, const std::string& phase) {
    ValidatePassSpec(spec);
    const std::string pass_key = PassSpecKey(spec.dialect, spec.name);
    if (spec.dialect != dialect) {
        throw std::invalid_argument("PassSpec " + pass_key + " has dialect " +
                                    ToString(spec.dialect) + ", expected " +
                                    ToString(dialect));
    }
    if (spec.scope != scope) {
        throw std::invalid_argument("PassSpec " + pass_key + " has scope " +
                                    ToString(spec.scope) + ", expected " +
                                    ToString(scope));
    }
    if (AsStdString(spec.phase) != phase) {
        throw std::invalid_argument("PassSpec " + pass_key + " has phase " +
                                    AsStdString(spec.phase) + ", expected " + phase);
    }
}

void ValidatePassSpecs(const Array<PassSpec>& specs) {
    std::map<std::string, PassSpec> seen;
    for (const PassSpec& spec : specs) {
        ValidatePassSpec(spec);
        const std::string key = PassSpecKey(spec.dialect, spec.name);
        if (seen.count(key)) {
            throw std::invalid_argument("duplicate PassSpec identity: " + key);
        }
        seen.emplace(key, spec);
    }
}

PassRegistry& PassRegistry::Global() {
    static PassRegistry registry;
    return registry;
}

void PassRegistry::Register(PassSpec spec) {
    ValidatePassSpec(spec);
    const std::string key = PassSpecKey(spec.dialect, spec.name);
    PassRegistryState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.frozen) {
        throw std::runtime_error("PassSpec registry is frozen");
    }
    if (state.specs.count(key)) {
        throw std::runtime_error("PassSpec " + key + " is already registered");
    }
    state.specs.emplace(key, std::move(spec));
}

bool PassRegistry::Contains(IRDialect dialect, const String& name) const {
    PassRegistryState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.specs.count(PassSpecKey(dialect, name)) != 0;
}

const PassSpec& PassRegistry::Get(IRDialect dialect, const String& name) const {
    PassRegistryState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const std::string key = PassSpecKey(dialect, name);
    auto it = state.specs.find(key);
    if (it == state.specs.end()) {
        throw std::runtime_error("Unknown PassSpec: " + key);
    }
    return it->second;
}

Array<PassSpec> PassRegistry::List() const {
    PassRegistryState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::vector<PassSpec> specs;
    specs.reserve(state.specs.size());
    for (const auto& entry : state.specs) {
        specs.push_back(entry.second);
    }
    return Array<PassSpec>(std::move(specs));
}

void PassRegistry::FreezeAndValidate() const {
    PassRegistryState& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::vector<PassSpec> specs;
    specs.reserve(state.specs.size());
    for (const auto& entry : state.specs) {
        specs.push_back(entry.second);
    }
    ValidatePassSpecs(Array<PassSpec>(std::move(specs)));
    state.frozen = true;
}

}  // namespace kxc
