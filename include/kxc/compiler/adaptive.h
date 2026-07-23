/*! \file include/kxc/compiler/adaptive.h
 * \brief Static-exact adaptive compilation control-plane contracts.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace kxc::api::adaptive {

/*! \brief A validated canonical key with value equality, never hash-only identity. */
template <typename Tag>
class CanonicalKey final {
public:
    explicit CanonicalKey(std::string canonical)
        : canonical_(std::move(canonical)) {
        if (canonical_.empty()) {
            throw std::invalid_argument("canonical key must not be empty");
        }
    }

    const std::string& canonical() const noexcept { return canonical_; }

    friend bool operator==(const CanonicalKey& lhs,
                           const CanonicalKey& rhs) noexcept {
        return lhs.canonical_ == rhs.canonical_;
    }
    friend bool operator!=(const CanonicalKey& lhs,
                           const CanonicalKey& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const CanonicalKey& lhs,
                          const CanonicalKey& rhs) noexcept {
        return lhs.canonical_ < rhs.canonical_;
    }

private:
    std::string canonical_;
};

struct KernelSlotKeyTag final {};
struct PlanAbiFingerprintTag final {};
struct PlanVariantKeyTag final {};

/*! \brief Stable unit/target publication identity supplied by Core Track. */
using KernelSlotKey = CanonicalKey<KernelSlotKeyTag>;
/*! \brief Complete static-exact Plan ABI canonical fingerprint. */
using PlanAbiFingerprint = CanonicalKey<PlanAbiFingerprintTag>;
/*! \brief Frozen plan identity, separate from artifact and graph value identity. */
using PlanVariantKey = CanonicalKey<PlanVariantKeyTag>;

/*! \brief Exact-only dispatch identity; Shape Track may add proven kinds later. */
class DispatchKey final {
public:
    static DispatchKey Exact(std::string canonical);

    const std::string& canonical() const noexcept { return canonical_; }

    friend bool operator==(const DispatchKey& lhs,
                           const DispatchKey& rhs) noexcept {
        return lhs.canonical_ == rhs.canonical_;
    }
    friend bool operator!=(const DispatchKey& lhs,
                           const DispatchKey& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const DispatchKey& lhs,
                          const DispatchKey& rhs) noexcept {
        return lhs.canonical_ < rhs.canonical_;
    }

private:
    explicit DispatchKey(std::string canonical);
    std::string canonical_;
};

/*! \brief Artifact identity includes its slot identity and compiler fingerprint. */
class KernelArtifactKey final {
public:
    KernelArtifactKey(KernelSlotKey slot_key, std::string canonical);

    const KernelSlotKey& slot_key() const noexcept { return slot_key_; }
    const std::string& canonical() const noexcept { return canonical_; }

    friend bool operator==(const KernelArtifactKey& lhs,
                           const KernelArtifactKey& rhs) noexcept {
        return lhs.slot_key_ == rhs.slot_key_ &&
               lhs.canonical_ == rhs.canonical_;
    }
    friend bool operator!=(const KernelArtifactKey& lhs,
                           const KernelArtifactKey& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const KernelArtifactKey& lhs,
                          const KernelArtifactKey& rhs) noexcept {
        if (lhs.slot_key_ != rhs.slot_key_) {
            return lhs.slot_key_ < rhs.slot_key_;
        }
        return lhs.canonical_ < rhs.canonical_;
    }

private:
    KernelSlotKey slot_key_;
    std::string canonical_;
};

/*! \brief Scheduling intent; it never participates in artifact equality. */
enum class RequestKind : std::uint8_t {
    kDemand = 0,
    kCanary = 1,
    kPrewarm = 2,
};

/*! \brief Strong, complete compile request for one exact artifact. */
class CompileRequest final {
public:
    CompileRequest(KernelArtifactKey artifact_key, DispatchKey dispatch_key,
                   PlanAbiFingerprint required_abi,
                   std::string model_revision, RequestKind kind,
                   int priority = 0);

    const KernelArtifactKey& artifact_key() const noexcept {
        return artifact_key_;
    }
    const DispatchKey& dispatch_key() const noexcept { return dispatch_key_; }
    const PlanAbiFingerprint& required_abi() const noexcept {
        return required_abi_;
    }
    const std::string& model_revision() const noexcept {
        return model_revision_;
    }
    RequestKind kind() const noexcept { return kind_; }
    int priority() const noexcept { return priority_; }

private:
    KernelArtifactKey artifact_key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint required_abi_;
    std::string model_revision_;
    RequestKind kind_{RequestKind::kDemand};
    int priority_{0};
};

/*! \brief Typed immutable executable payload; no raw void/function pointer ABI. */
class ArtifactExecutable {
public:
    virtual ~ArtifactExecutable() = default;
    virtual bool IsReady() const noexcept = 0;
    virtual std::string DebugName() const = 0;
};

/*! \brief Immutable validated result of compiling one static-exact request. */
class KernelArtifact final {
public:
    KernelArtifact(KernelArtifactKey key, DispatchKey applicability,
                   PlanAbiFingerprint compatible_abi,
                   std::shared_ptr<const ArtifactExecutable> executable,
                   std::size_t byte_size, std::string provenance);

    const KernelArtifactKey& key() const noexcept { return key_; }
    const DispatchKey& applicability() const noexcept {
        return applicability_;
    }
    const PlanAbiFingerprint& compatible_abi() const noexcept {
        return compatible_abi_;
    }
    const std::shared_ptr<const ArtifactExecutable>& executable() const noexcept {
        return executable_;
    }
    std::size_t byte_size() const noexcept { return byte_size_; }
    const std::string& provenance() const noexcept { return provenance_; }

private:
    KernelArtifactKey key_;
    DispatchKey applicability_;
    PlanAbiFingerprint compatible_abi_;
    std::shared_ptr<const ArtifactExecutable> executable_;
    std::size_t byte_size_{0};
    std::string provenance_;
};

}  // namespace kxc::api::adaptive
