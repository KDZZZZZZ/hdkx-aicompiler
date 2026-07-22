#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "../../codegen/internal/compiled_kernel.h"
#include "kxc/target/target.h"

namespace kxc::api::internal {

struct CachedPrimitive final {
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata launch_metadata;
    codegen::CompiledKernel kernel;
};

struct PrimitiveCacheStats final {
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t entries{0};
};

std::string BuildPrimitiveCacheKey(const String& structural_hash,
                                   const Target& target, int opt_level,
                                   const char* backend_version);

std::optional<CachedPrimitive> LookupPrimitiveCache(const std::string& key);
std::optional<CachedPrimitive> PeekPrimitiveCache(const std::string& key);
void StorePrimitiveCache(std::string key, CachedPrimitive entry);
PrimitiveCacheStats GetPrimitiveCacheStats();
void ClearPrimitiveCacheForTesting();

}  // namespace kxc::api::internal
