/*! \file src/compiler/foundation_contract.cc
 * \brief Implements compiler-foundation DTO validation and canonical requests.
 */

#include "kxc/compiler/foundation_contract.h"

#include "internal/primitive_cache.h"

#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace kxc::api {
namespace {

void AppendField(std::string* canonical, const std::string& name,
                 const std::string& value) {
    *canonical += std::to_string(name.size()) + ":" + name + "=" +
                  std::to_string(value.size()) + ":" + value + ";";
}

std::string NextStandaloneTicketId() {
    static std::atomic<uint64_t> next{1};
    uint64_t value = next.load(std::memory_order_relaxed);
    do {
        if (value == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error(
                "production transaction ticket space is exhausted");
        }
    } while (!next.compare_exchange_weak(
        value, value + 1, std::memory_order_relaxed));
    return "production-transaction-v1:" + std::to_string(value);
}

CompileFailureCategory ToPublicFailureCategory(
    internal::PrimitiveFailureCategory category) {
    switch (category) {
        case internal::PrimitiveFailureCategory::kCompile:
            return CompileFailureCategory::kCompile;
        case internal::PrimitiveFailureCategory::kValidation:
            return CompileFailureCategory::kValidation;
        case internal::PrimitiveFailureCategory::kUnsupported:
            return CompileFailureCategory::kUnsupported;
        case internal::PrimitiveFailureCategory::kCancelled:
            return CompileFailureCategory::kCancelled;
        case internal::PrimitiveFailureCategory::kBackpressure:
            return CompileFailureCategory::kBackpressure;
    }
    throw std::logic_error("unknown primitive failure category");
}

internal::PrimitiveFailureCategory ToPrimitiveFailureCategory(
    CompileFailureCategory category) {
    switch (category) {
        case CompileFailureCategory::kCompile:
            return internal::PrimitiveFailureCategory::kCompile;
        case CompileFailureCategory::kValidation:
            return internal::PrimitiveFailureCategory::kValidation;
        case CompileFailureCategory::kUnsupported:
            return internal::PrimitiveFailureCategory::kUnsupported;
        case CompileFailureCategory::kCancelled:
            return internal::PrimitiveFailureCategory::kCancelled;
        case CompileFailureCategory::kBackpressure:
            return internal::PrimitiveFailureCategory::kBackpressure;
    }
    throw std::logic_error("unknown public compile failure category");
}

CompileFailure ToPublicFailure(
    const internal::PrimitiveFailureRecord& failure) {
    return CompileFailure{ToPublicFailureCategory(failure.category),
                          failure.retry_after_millis, failure.diagnostic};
}

CompileOutcome FailureOutcome(CompileRequestState state,
                              CompileFailure failure) {
    return CompileOutcome{state, {}, std::move(failure)};
}

}  // namespace

struct ProductionCompileTransaction::Impl final {
    ~Impl() {
        if (lease.access() != internal::PrimitiveCacheAccess::kOwner ||
            ticket.outcome) {
            return;
        }
        try {
            internal::FailPrimitiveCacheLease(
                lease, internal::PrimitiveFailureCategory::kCompile,
                "production transaction owner was abandoned",
                std::chrono::seconds(1));
        } catch (...) {
        }
    }

    mutable std::mutex mutex;
    internal::PrimitiveCacheLease lease;
    CancellationToken cancellation;
    CompileTicket ticket;
};

ArtifactHandle::ArtifactHandle(ArtifactRecord record) {
    if (!record.artifact_key.defined() || record.executable_token.empty() ||
        record.signature_digest.empty() ||
        record.launch_metadata_digest.empty() || record.provenance.empty() ||
        record.byte_size == 0 || record.validation_record.empty()) {
        throw std::invalid_argument(
            "ArtifactHandle requires a complete validated ready artifact");
    }
    record_ = std::make_shared<const ArtifactRecord>(std::move(record));
}

bool ArtifactHandle::defined() const noexcept { return record_ != nullptr; }

const ArtifactRecord& ArtifactHandle::record() const {
    if (!record_) throw std::logic_error("ArtifactHandle is undefined");
    return *record_;
}

ArtifactPin::ArtifactPin(ArtifactHandle handle) : handle_(std::move(handle)) {
    if (!handle_.defined()) {
        throw std::invalid_argument("ArtifactPin requires an ArtifactHandle");
    }
}

ArtifactPin::ArtifactPin(ArtifactHandle handle, std::shared_ptr<const void> owner)
    : handle_(std::move(handle)), owner_(std::move(owner)) {
    if (!handle_.defined() || !owner_) {
        throw std::invalid_argument(
            "production ArtifactPin requires an ArtifactHandle and owner");
    }
}

bool ArtifactPin::defined() const noexcept { return handle_.defined(); }

const ArtifactHandle& ArtifactPin::handle() const {
    if (!handle_.defined()) throw std::logic_error("ArtifactPin is undefined");
    return handle_;
}

ProductionArtifactCandidate::ProductionArtifactCandidate(
    std::shared_ptr<const void> owner)
    : owner_(std::move(owner)) {}

bool ProductionArtifactCandidate::defined() const noexcept {
    return owner_ != nullptr;
}

ProductionCompileTransaction::ProductionCompileTransaction(
    std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

bool ProductionCompileTransaction::defined() const noexcept {
    return impl_ != nullptr;
}

bool ProductionCompileTransaction::owns_compile() const noexcept {
    return impl_ &&
           impl_->lease.access() == internal::PrimitiveCacheAccess::kOwner;
}

CompileTicket ProductionCompileTransaction::ticket() const {
    if (!impl_) {
        throw std::logic_error("production compile transaction is undefined");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    CompileTicket ticket = impl_->ticket;
    ticket.merged_waiter_count = impl_->lease.merged_waiter_count();
    return ticket;
}

ProductionCompileOwnership ProductionArtifactCacheAdapter::ownership() const noexcept {
    return {};
}

ArtifactLookup ProductionArtifactCacheAdapter::Lookup(const ArtifactKey& key) const {
    const internal::PrimitiveArtifactPin pin = internal::LookupPrimitiveCache(key);
    if (!pin.defined()) return {};
    return ArtifactLookup{ArtifactLookupKind::kHit, internal::ToArtifactPin(pin)};
}

ProductionCompileTransaction ProductionArtifactCacheAdapter::Acquire(
    const CompileRequest& request) const {
    const std::string canonical = request.CanonicalSingleflightKey();
    auto impl = std::make_shared<ProductionCompileTransaction::Impl>();
    impl->cancellation = request.cancellation;
    impl->ticket.canonical_request_key = canonical;
    impl->ticket.ticket_id = NextStandaloneTicketId();

    const auto reject = [&impl](CompileRequestState state,
                                CompileFailure failure) {
        impl->ticket.state = state;
        impl->ticket.outcome = FailureOutcome(state, std::move(failure));
    };
    if (request.cancellation.cancelled) {
        reject(CompileRequestState::kCancelled,
               CompileFailure{CompileFailureCategory::kCancelled, 0,
                              "request was cancelled before cache acquisition"});
        return ProductionCompileTransaction(std::move(impl));
    }
    if (request.dispatch_key) {
        reject(CompileRequestState::kRejected,
               CompileFailure{CompileFailureCategory::kUnsupported, 0,
                              "dispatch-aware singleflight belongs to the adaptive coordinator"});
        return ProductionCompileTransaction(std::move(impl));
    }
    if (request.priority != CompilePriority::kNormal ||
        request.budget_class != CompileBudgetClass::kGlobal) {
        reject(CompileRequestState::kRejected,
               CompileFailure{CompileFailureCategory::kUnsupported, 0,
                              "priority and budget scheduling belong to the adaptive coordinator"});
        return ProductionCompileTransaction(std::move(impl));
    }

    impl->lease = internal::AcquirePrimitiveCache(request.artifact_key);
    switch (impl->lease.access()) {
        case internal::PrimitiveCacheAccess::kHit: {
            const ArtifactPin pin = internal::ToArtifactPin(impl->lease.pin());
            impl->ticket.state = CompileRequestState::kReady;
            impl->ticket.outcome =
                CompileOutcome{CompileRequestState::kReady, pin, std::nullopt};
            break;
        }
        case internal::PrimitiveCacheAccess::kOwner:
            impl->ticket.ticket_id = impl->lease.ticket_id();
            impl->ticket.state = CompileRequestState::kCompiling;
            break;
        case internal::PrimitiveCacheAccess::kWait:
            impl->ticket.ticket_id = impl->lease.ticket_id();
            impl->ticket.state = CompileRequestState::kQueued;
            break;
        case internal::PrimitiveCacheAccess::kFailed:
            reject(impl->lease.failure().category ==
                           internal::PrimitiveFailureCategory::kCancelled
                       ? CompileRequestState::kCancelled
                       : CompileRequestState::kFailed,
                   ToPublicFailure(impl->lease.failure()));
            break;
        case internal::PrimitiveCacheAccess::kRejected:
            reject(CompileRequestState::kRejected,
                   ToPublicFailure(impl->lease.failure()));
            break;
    }
    return ProductionCompileTransaction(std::move(impl));
}

CompileOutcome ProductionArtifactCacheAdapter::Wait(
    const ProductionCompileTransaction& transaction) const {
    if (!transaction.impl_) {
        throw std::invalid_argument("Wait requires a production transaction");
    }
    CancellationToken cancellation;
    {
        std::lock_guard<std::mutex> lock(transaction.impl_->mutex);
        cancellation = transaction.impl_->cancellation;
    }
    return Wait(transaction, cancellation);
}

CompileOutcome ProductionArtifactCacheAdapter::Wait(
    const ProductionCompileTransaction& transaction,
    const CancellationToken& cancellation) const {
    if (!transaction.impl_) {
        throw std::invalid_argument("Wait requires a production transaction");
    }
    const auto impl = transaction.impl_;
    std::unique_lock<std::mutex> lock(impl->mutex);
    if (cancellation.id.empty() || cancellation.id != impl->cancellation.id) {
        throw std::invalid_argument(
            "wait cancellation token does not own this transaction");
    }
    if (impl->ticket.outcome) return *impl->ticket.outcome;
    if (impl->lease.access() == internal::PrimitiveCacheAccess::kOwner) {
        throw std::logic_error(
            "compile owner must Publish or Fail instead of waiting");
    }
    if (cancellation.cancelled) {
        impl->ticket.state = CompileRequestState::kCancelled;
        impl->ticket.outcome = FailureOutcome(
            CompileRequestState::kCancelled,
            CompileFailure{CompileFailureCategory::kCancelled, 0,
                           "waiter cancelled without cancelling shared work"});
        return *impl->ticket.outcome;
    }

    const internal::PrimitiveCacheWaitResult result =
        internal::WaitPrimitiveCacheLeaseResult(impl->lease);
    const CompileOutcome outcome = result.succeeded()
                                       ? CompileOutcome{
                                             CompileRequestState::kReady,
                                             internal::ToArtifactPin(result.pin),
                                             std::nullopt}
                                       : FailureOutcome(
                                             result.failure.category ==
                                                     internal::PrimitiveFailureCategory::kCancelled
                                                 ? CompileRequestState::kCancelled
                                                 : CompileRequestState::kFailed,
                                             ToPublicFailure(result.failure));
    impl->ticket.state = outcome.state;
    impl->ticket.outcome = outcome;
    return outcome;
}

CompileOutcome ProductionArtifactCacheAdapter::Publish(
    const ProductionCompileTransaction& transaction,
    ProductionArtifactCandidate candidate) const {
    if (!transaction.impl_) {
        throw std::invalid_argument("Publish requires a production transaction");
    }
    const auto impl = transaction.impl_;
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->ticket.outcome) return *impl->ticket.outcome;
    if (impl->lease.access() != internal::PrimitiveCacheAccess::kOwner) {
        throw std::invalid_argument(
            "only a production transaction owner may publish");
    }
    impl->ticket.state = CompileRequestState::kValidating;

    internal::PrimitiveArtifactPin primitive_pin;
    try {
        primitive_pin = internal::PublishPrimitiveCacheLease(
            impl->lease, internal::ProductionArtifactAccess::Copy(candidate));
    } catch (const std::invalid_argument& error) {
        internal::FailPrimitiveCacheLease(
            impl->lease, internal::PrimitiveFailureCategory::kValidation,
            error.what(), std::chrono::seconds(1));
        const CompileOutcome outcome = FailureOutcome(
            CompileRequestState::kFailed,
            CompileFailure{CompileFailureCategory::kValidation, 1000,
                           error.what()});
        impl->ticket.state = outcome.state;
        impl->ticket.outcome = outcome;
        return outcome;
    }

    const CompileOutcome outcome{CompileRequestState::kReady,
                                 internal::ToArtifactPin(primitive_pin),
                                 std::nullopt};
    impl->ticket.state = outcome.state;
    impl->ticket.outcome = outcome;
    return outcome;
}

CompileOutcome ProductionArtifactCacheAdapter::Fail(
    const ProductionCompileTransaction& transaction,
    CompileFailure failure) const {
    if (!transaction.impl_) {
        throw std::invalid_argument("Fail requires a production transaction");
    }
    if (failure.diagnostic.empty() ||
        failure.retry_after_millis >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument(
            "production failure requires a diagnostic and bounded retry delay");
    }
    const auto impl = transaction.impl_;
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->ticket.outcome) return *impl->ticket.outcome;
    if (impl->lease.access() != internal::PrimitiveCacheAccess::kOwner) {
        throw std::invalid_argument(
            "only a production transaction owner may fail");
    }
    internal::FailPrimitiveCacheLease(
        impl->lease, ToPrimitiveFailureCategory(failure.category),
        failure.diagnostic,
        std::chrono::milliseconds(
            static_cast<int64_t>(failure.retry_after_millis)));
    const CompileRequestState state =
        failure.category == CompileFailureCategory::kCancelled
            ? CompileRequestState::kCancelled
            : CompileRequestState::kFailed;
    const CompileOutcome outcome = FailureOutcome(state, std::move(failure));
    impl->ticket.state = outcome.state;
    impl->ticket.outcome = outcome;
    return outcome;
}

ArtifactCacheStats ProductionArtifactCacheAdapter::stats() const {
    const internal::PrimitiveCacheStats stats = internal::GetPrimitiveCacheStats();
    return ArtifactCacheStats{stats.hits,
                              stats.misses,
                              stats.entries,
                              stats.accounted_bytes,
                              stats.evictions,
                              stats.in_flight,
                              stats.merged_waiters,
                              stats.failures,
                              stats.rejections,
                              stats.active_pins};
}

std::string CompileRequest::CanonicalSingleflightKey() const {
    if (!artifact_key.defined() || request_origin.empty() ||
        cancellation.id.empty()) {
        throw std::invalid_argument(
            "CompileRequest requires artifact identity, origin, and cancellation id");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "compile-request-v1");
    AppendField(&canonical, "artifact", artifact_key.canonical_bytes());
    AppendField(&canonical, "dispatch",
                dispatch_key ? dispatch_key->canonical_bytes() : "static-exact");
    return canonical;
}

void FrozenPlanInput::Validate() const {
    if (graph_template_locator.empty() || selected_artifacts.empty() ||
        logical_physical_valid_extent_contract.empty() ||
        memory_plan_version.empty()) {
        throw std::invalid_argument(
            "FrozenPlanInput requires graph, artifacts, value contract, and memory version");
    }
    if (logical_physical_valid_extent_contract.find("-1") !=
        std::string::npos) {
        throw std::invalid_argument(
            "FrozenPlanInput cannot use legacy -1 as a shape contract");
    }
    for (const GraphValueLocator& locator : value_routing) {
        (void)locator.CanonicalBytes();
    }
    for (const SelectedArtifact& selected : selected_artifacts) {
        if (!selected.artifact_pin.defined() ||
            selected.applicability_proof.empty() ||
            selected.signature_digest.empty() ||
            selected.signature_digest !=
                selected.artifact_pin.handle().record().signature_digest) {
            throw std::invalid_argument(
                "FrozenPlanInput selected artifact proof/signature is invalid");
        }
    }
}

}  // namespace kxc::api
