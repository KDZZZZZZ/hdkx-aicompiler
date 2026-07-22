/*! \file src/profiling/profiling.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

// 职责简介：
// - 实现 profiling 上下文、RAII span、日志和 artifact 写出逻辑。
// - 将事件同时序列化为 events.jsonl 和 Perfetto/Chrome trace 可读的 trace.json。
// - 按需动态接入 NVTX/CUPTI，把 CUDA activity 关联到 KXC 的 run/span。

#include "kxc/profiling/profiling.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#if KXC_USE_CUDA && defined(__has_include)
#if __has_include(<cupti.h>) && __has_include(<cupti_activity.h>) && __has_include(<cupti_result.h>)
#define KXC_HAS_CUPTI 1
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_result.h>
#else
#define KXC_HAS_CUPTI 0
#endif
#else
#define KXC_HAS_CUPTI 0
#endif

namespace kxc {
namespace profiling {

namespace {

// profiling 上下文刻意放在线程本地状态里。后台编译线程会携带复制出的 id，
// 再通过 ActivationScope 回到原来的 run。
thread_local std::shared_ptr<ProfileContext> tls_profile_ctx;
thread_local std::string tls_run_id;
thread_local std::vector<std::string> tls_span_stack;

std::int64_t NowWallNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::int64_t NowSteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string EscapeJson(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (unsigned char c : input) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    std::ostringstream os;
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(c);
                    out += os.str();
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string ToJsonObject(const StringMap& values) {
    std::vector<std::pair<std::string, std::string>> items(values.begin(), values.end());
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::ostringstream os;
    os << "{";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) os << ",";
        os << "\"" << EscapeJson(items[i].first) << "\":\"" << EscapeJson(items[i].second)
           << "\"";
    }
    os << "}";
    return os.str();
}

std::string ToJsonObject(const MetricMap& values) {
    std::vector<std::pair<std::string, double>> items(values.begin(), values.end());
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::ostringstream os;
    os << "{";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) os << ",";
        os << "\"" << EscapeJson(items[i].first) << "\":" << items[i].second;
    }
    os << "}";
    return os.str();
}

std::uint64_t Djb2(const std::string& text) {
    std::uint64_t hash = 5381;
    for (unsigned char c : text) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

std::string MakeId(const char* prefix) {
    static std::atomic<std::uint64_t> counter{0};
    std::ostringstream os;
    os << prefix << "-" << std::hex << NowWallNs() << "-" << ++counter;
    return os.str();
}

std::string ThreadIdString() {
    std::ostringstream os;
    os << std::this_thread::get_id();
    return os.str();
}

std::uint64_t ThreadIdNumeric() { return Djb2(ThreadIdString()); }

std::uint64_t ProcessIdNumeric() {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

// NVTX 运行时懒加载，避免 profiling 构建在链接阶段强依赖 NVIDIA 工具库。
class NvtxAdapter {
public:
    NvtxAdapter() = default;
    explicit NvtxAdapter(bool requested) { TryLoad(requested); }
    ~NvtxAdapter() { Unload(); }

    void Push(const std::string& name) {
        if (push_ != nullptr) {
            push_(name.c_str());
        }
    }

    void Pop() {
        if (pop_ != nullptr) {
            pop_();
        }
    }

    bool requested() const { return requested_; }
    bool available() const { return available_; }

private:
    using RangePushFn = int (*)(const char*);
    using RangePopFn = int (*)();

    void TryLoad(bool requested) {
        requested_ = requested;
        if (!requested) {
            return;
        }
#if defined(_WIN32)
        handle_ = LoadLibraryA("nvToolsExt64_1.dll");
        if (handle_ == nullptr) {
            return;
        }
        push_ = reinterpret_cast<RangePushFn>(GetProcAddress(static_cast<HMODULE>(handle_),
                                                             "nvtxRangePushA"));
        pop_ = reinterpret_cast<RangePopFn>(GetProcAddress(static_cast<HMODULE>(handle_),
                                                           "nvtxRangePop"));
#else
        handle_ = dlopen("libnvToolsExt.so.1", RTLD_LAZY);
        if (handle_ == nullptr) {
            handle_ = dlopen("libnvToolsExt.so", RTLD_LAZY);
        }
        if (handle_ == nullptr) {
            return;
        }
        push_ = reinterpret_cast<RangePushFn>(dlsym(handle_, "nvtxRangePushA"));
        pop_ = reinterpret_cast<RangePopFn>(dlsym(handle_, "nvtxRangePop"));
#endif
        available_ = push_ != nullptr && pop_ != nullptr;
        if (!available_) {
            Unload();
        }
    }

    void Unload() {
#if defined(_WIN32)
        if (handle_ != nullptr) {
            FreeLibrary(static_cast<HMODULE>(handle_));
            handle_ = nullptr;
        }
#else
        if (handle_ != nullptr) {
            dlclose(handle_);
            handle_ = nullptr;
        }
#endif
        push_ = nullptr;
        pop_ = nullptr;
    }

    bool requested_{false};
    bool available_{false};
    void* handle_{nullptr};
    RangePushFn push_{nullptr};
    RangePopFn pop_{nullptr};
};

NvtxAdapter* ResolveNvtxAdapter(bool requested) {
    static std::mutex mu;
    static std::unique_ptr<NvtxAdapter> adapter;
    std::lock_guard<std::mutex> lock(mu);
    if (requested && !adapter) {
        adapter = std::make_unique<NvtxAdapter>(true);
    }
    return adapter.get();
}

#if KXC_HAS_CUPTI
class CuptiAdapter;
CuptiAdapter* ResolveCuptiAdapter(bool requested);

// CUPTI activity 采集是可选能力，运行时动态加载。external correlation id
// 用来把异步 CUDA 记录关联回 KXC 的 run/span id。
class CuptiAdapter {
public:
    CuptiAdapter() = default;
    explicit CuptiAdapter(bool requested) { TryLoad(requested); }

    bool requested() const { return requested_; }
    bool available() const { return available_; }

    void BindContext(ProfileContext* ctx) {
        if (!available_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        // 绑定期间产生的 activity 记录归这个 context 负责 flush 和落盘。
        bound_context_ = ctx;
        correlation_map_.clear();
        external_id_to_text_.clear();
        next_external_id_ = 1;
        std::uint64_t origin = 0;
        if (get_timestamp_ != nullptr && get_timestamp_(&origin) == CUPTI_SUCCESS) {
            cupti_origin_ns_ = origin;
        } else {
            cupti_origin_ns_ = 0;
        }
    }

    void UnbindContext(ProfileContext* ctx) {
        if (!available_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (bound_context_ == ctx) {
            bound_context_ = nullptr;
            correlation_map_.clear();
            external_id_to_text_.clear();
        }
    }

    bool PushRunCorrelation(const std::string& run_id) {
        return PushCorrelation(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, run_id);
    }

    void PopRunCorrelation() {
        PopCorrelation(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0);
    }

    bool PushSpanCorrelation(const std::string& span_id) {
        return PushCorrelation(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, span_id);
    }

    void PopSpanCorrelation() {
        PopCorrelation(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1);
    }

    void Flush() {
        if (!available_ || activity_flush_all_ == nullptr) {
            return;
        }
        activity_flush_all_(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    }

private:
    struct CorrelationContext {
        std::string run_id;
        std::string span_id;
    };

    using ActivityRegisterCallbacksFn = decltype(&cuptiActivityRegisterCallbacks);
    using ActivityEnableFn = decltype(&cuptiActivityEnable);
    using ActivityFlushAllFn = decltype(&cuptiActivityFlushAll);
    using ActivityGetNextRecordFn = decltype(&cuptiActivityGetNextRecord);
    using ActivityGetNumDroppedRecordsFn = decltype(&cuptiActivityGetNumDroppedRecords);
    using ActivityPushExternalCorrelationIdFn =
        decltype(&cuptiActivityPushExternalCorrelationId);
    using ActivityPopExternalCorrelationIdFn =
        decltype(&cuptiActivityPopExternalCorrelationId);
    using GetTimestampFn = decltype(&cuptiGetTimestamp);
    using GetResultStringFn = decltype(&cuptiGetResultString);

    static CuptiAdapter* Current() { return ResolveCuptiAdapter(true); }

    static void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size,
                                         size_t* max_num_records) {
        constexpr size_t kBufferSize = 1 << 20;
        *buffer = static_cast<uint8_t*>(std::malloc(kBufferSize));
        *size = *buffer == nullptr ? 0 : kBufferSize;
        *max_num_records = 0;
    }

    static void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id,
                                         uint8_t* buffer, size_t size,
                                         size_t valid_size) {
        (void)size;
        CuptiAdapter* adapter = Current();
        if (adapter != nullptr) {
            adapter->ConsumeBuffer(context, stream_id, buffer, valid_size);
        }
        std::free(buffer);
    }

    void TryLoad(bool requested) {
        requested_ = requested;
        if (!requested_) {
            return;
        }
#if defined(_WIN32)
        handle_ = LoadLibraryA("cupti64.dll");
        if (handle_ == nullptr) {
            handle_ = LoadLibraryA("cupti.dll");
        }
        if (handle_ == nullptr) {
            return;
        }
        activity_register_callbacks_ = reinterpret_cast<ActivityRegisterCallbacksFn>(
            GetProcAddress(static_cast<HMODULE>(handle_), "cuptiActivityRegisterCallbacks"));
        activity_enable_ = reinterpret_cast<ActivityEnableFn>(
            GetProcAddress(static_cast<HMODULE>(handle_), "cuptiActivityEnable"));
        activity_flush_all_ = reinterpret_cast<ActivityFlushAllFn>(
            GetProcAddress(static_cast<HMODULE>(handle_), "cuptiActivityFlushAll"));
        activity_get_next_record_ = reinterpret_cast<ActivityGetNextRecordFn>(
            GetProcAddress(static_cast<HMODULE>(handle_), "cuptiActivityGetNextRecord"));
        activity_get_num_dropped_records_ =
            reinterpret_cast<ActivityGetNumDroppedRecordsFn>(
                GetProcAddress(static_cast<HMODULE>(handle_),
                               "cuptiActivityGetNumDroppedRecords"));
        activity_push_external_correlation_id_ =
            reinterpret_cast<ActivityPushExternalCorrelationIdFn>(
                GetProcAddress(static_cast<HMODULE>(handle_),
                               "cuptiActivityPushExternalCorrelationId"));
        activity_pop_external_correlation_id_ =
            reinterpret_cast<ActivityPopExternalCorrelationIdFn>(
                GetProcAddress(static_cast<HMODULE>(handle_),
                               "cuptiActivityPopExternalCorrelationId"));
        get_timestamp_ = reinterpret_cast<GetTimestampFn>(
            GetProcAddress(static_cast<HMODULE>(handle_), "cuptiGetTimestamp"));
        get_result_string_ = reinterpret_cast<GetResultStringFn>(
            GetProcAddress(static_cast<HMODULE>(handle_), "cuptiGetResultString"));
#else
        handle_ = dlopen("libcupti.so.13", RTLD_LAZY);
        if (handle_ == nullptr) {
            handle_ = dlopen("libcupti.so", RTLD_LAZY);
        }
        if (handle_ == nullptr) {
            return;
        }
        activity_register_callbacks_ = reinterpret_cast<ActivityRegisterCallbacksFn>(
            dlsym(handle_, "cuptiActivityRegisterCallbacks"));
        activity_enable_ =
            reinterpret_cast<ActivityEnableFn>(dlsym(handle_, "cuptiActivityEnable"));
        activity_flush_all_ =
            reinterpret_cast<ActivityFlushAllFn>(dlsym(handle_, "cuptiActivityFlushAll"));
        activity_get_next_record_ = reinterpret_cast<ActivityGetNextRecordFn>(
            dlsym(handle_, "cuptiActivityGetNextRecord"));
        activity_get_num_dropped_records_ =
            reinterpret_cast<ActivityGetNumDroppedRecordsFn>(
                dlsym(handle_, "cuptiActivityGetNumDroppedRecords"));
        activity_push_external_correlation_id_ =
            reinterpret_cast<ActivityPushExternalCorrelationIdFn>(
                dlsym(handle_, "cuptiActivityPushExternalCorrelationId"));
        activity_pop_external_correlation_id_ =
            reinterpret_cast<ActivityPopExternalCorrelationIdFn>(
                dlsym(handle_, "cuptiActivityPopExternalCorrelationId"));
        get_timestamp_ =
            reinterpret_cast<GetTimestampFn>(dlsym(handle_, "cuptiGetTimestamp"));
        get_result_string_ =
            reinterpret_cast<GetResultStringFn>(dlsym(handle_, "cuptiGetResultString"));
#endif
        if (activity_register_callbacks_ == nullptr || activity_enable_ == nullptr ||
            activity_flush_all_ == nullptr || activity_get_next_record_ == nullptr ||
            activity_get_num_dropped_records_ == nullptr ||
            activity_push_external_correlation_id_ == nullptr ||
            activity_pop_external_correlation_id_ == nullptr || get_timestamp_ == nullptr ||
            get_result_string_ == nullptr) {
            return;
        }

        if (!EnableActivities()) {
            return;
        }
        available_ = true;
    }

    bool EnableActivities() {
        if (!Check(activity_register_callbacks_(BufferRequested, BufferCompleted),
                   "cuptiActivityRegisterCallbacks")) {
            return false;
        }
        return Check(activity_enable_(CUPTI_ACTIVITY_KIND_RUNTIME), "enable runtime") &&
               Check(activity_enable_(CUPTI_ACTIVITY_KIND_DRIVER), "enable driver") &&
               Check(activity_enable_(CUPTI_ACTIVITY_KIND_MEMCPY), "enable memcpy") &&
               Check(activity_enable_(CUPTI_ACTIVITY_KIND_MEMSET), "enable memset") &&
               Check(activity_enable_(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
                     "enable concurrent kernel") &&
               Check(activity_enable_(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION),
                     "enable external correlation");
    }

    bool PushCorrelation(CUpti_ExternalCorrelationKind kind, const std::string& text) {
        if (!available_ || text.empty()) {
            return false;
        }
        std::uint64_t external_id = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            external_id = next_external_id_++;
            external_id_to_text_[external_id] = text;
        }
        return Check(activity_push_external_correlation_id_(kind, external_id),
                     "push external correlation");
    }

    void PopCorrelation(CUpti_ExternalCorrelationKind kind) {
        if (!available_) {
            return;
        }
        std::uint64_t last_id = 0;
        activity_pop_external_correlation_id_(kind, &last_id);
    }

    bool Check(CUptiResult result, const char* action) {
        if (result == CUPTI_SUCCESS) {
            last_error_.clear();
            return true;
        }
        const char* text = nullptr;
        if (get_result_string_ != nullptr) {
            get_result_string_(result, &text);
        }
        last_error_ = std::string(action) + ": " + (text ? text : "unknown CUPTI error");
        return false;
    }

    std::optional<CorrelationContext> LookupCorrelation(std::uint32_t correlation_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = correlation_map_.find(correlation_id);
        if (it == correlation_map_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void UpdateCorrelation(std::uint32_t correlation_id, CUpti_ExternalCorrelationKind kind,
                           std::uint64_t external_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto text_it = external_id_to_text_.find(external_id);
        if (text_it == external_id_to_text_.end()) {
            return;
        }
        CorrelationContext& entry = correlation_map_[correlation_id];
        if (kind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
            entry.run_id = text_it->second;
        } else if (kind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1) {
            entry.span_id = text_it->second;
        }
    }

    std::int64_t ToRelativeNs(std::uint64_t timestamp) const {
        if (timestamp == 0 || cupti_origin_ns_ == 0 || timestamp < cupti_origin_ns_) {
            return 0;
        }
        return static_cast<std::int64_t>(timestamp - cupti_origin_ns_);
    }

    void EmitActivityEvent(EventSpec spec, std::uint32_t correlation_id,
                           std::uint32_t fallback_runtime_correlation_id,
                           std::uint64_t start, std::uint64_t end) {
        ProfileContext* ctx = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            ctx = bound_context_;
        }
        if (ctx == nullptr) {
            return;
        }

        // CUPTI 可能先上报 driver/kernel 记录，再上报对应 runtime id；
        // 主 correlation id 缺失时退回使用 runtime correlation id。
        std::optional<CorrelationContext> correlation =
            LookupCorrelation(correlation_id != 0 ? correlation_id
                                                  : fallback_runtime_correlation_id);
        const std::string run_id = correlation ? correlation->run_id : std::string();
        const std::string parent_span_id = correlation ? correlation->span_id : std::string();
        const std::string span_id = ctx->NextSpanId();
        const std::int64_t start_ns = ToRelativeNs(start);
        const std::int64_t end_ns =
            end > 0 ? ToRelativeNs(end) : start_ns;
        ctx->RecordCompletedSpan(spec, run_id, span_id, parent_span_id, start_ns, end_ns);
    }

    void ConsumeBuffer(CUcontext context, uint32_t stream_id, uint8_t* buffer,
                       size_t valid_size) {
        if (!available_ || buffer == nullptr || valid_size == 0) {
            return;
        }
        CUpti_Activity* record = nullptr;
        while (activity_get_next_record_(buffer, valid_size, &record) == CUPTI_SUCCESS) {
            switch (record->kind) {
                case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
                    auto* external =
                        reinterpret_cast<CUpti_ActivityExternalCorrelation*>(record);
                    UpdateCorrelation(external->correlationId, external->externalKind,
                                      external->externalId);
                    break;
                }
                case CUPTI_ACTIVITY_KIND_RUNTIME:
                case CUPTI_ACTIVITY_KIND_DRIVER: {
                    auto* api = reinterpret_cast<CUpti_ActivityAPI*>(record);
                    EventSpec spec;
                    spec.component = "backend.cuda";
                    spec.event_type = record->kind == CUPTI_ACTIVITY_KIND_DRIVER
                                          ? "cuda_driver_api"
                                          : "cuda_runtime_api";
                    spec.phase = "complete";
                    spec.device = "cuda";
                    spec.fields["backend.cuda.cbid"] = std::to_string(api->cbid);
                    spec.fields["backend.cuda.correlation_id"] =
                        std::to_string(api->correlationId);
                    spec.fields["backend.cuda.process_id"] =
                        std::to_string(api->processId);
                    spec.fields["backend.cuda.thread_id"] = std::to_string(api->threadId);
                    spec.fields["backend.cuda.return_value"] =
                        std::to_string(api->returnValue);
                    if (api->returnValue != 0) {
                        spec.status = "error";
                        spec.severity = LogSeverity::kWarn;
                    }
                    EmitActivityEvent(spec, api->correlationId, 0, api->start, api->end);
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMCPY: {
                    auto* memcpy = reinterpret_cast<CUpti_ActivityMemcpy6*>(record);
                    EventSpec spec;
                    spec.component = "backend.cuda";
                    spec.event_type = "cuda_memcpy";
                    spec.device = "cuda:" + std::to_string(memcpy->deviceId);
                    spec.fields["backend.cuda.copy_kind"] =
                        std::to_string(memcpy->copyKind);
                    spec.fields["backend.cuda.src_kind"] = std::to_string(memcpy->srcKind);
                    spec.fields["backend.cuda.dst_kind"] = std::to_string(memcpy->dstKind);
                    spec.fields["backend.cuda.stream_id"] =
                        std::to_string(memcpy->streamId);
                    spec.fields["backend.cuda.context_id"] =
                        std::to_string(memcpy->contextId);
                    spec.fields["backend.cuda.correlation_id"] =
                        std::to_string(memcpy->correlationId);
                    spec.fields["backend.cuda.runtime_correlation_id"] =
                        std::to_string(memcpy->runtimeCorrelationId);
                    spec.metrics["bytes"] = static_cast<double>(memcpy->bytes);
                    spec.metrics["copy_count"] = static_cast<double>(memcpy->copyCount);
                    EmitActivityEvent(spec, memcpy->runtimeCorrelationId, memcpy->correlationId,
                                      memcpy->start, memcpy->end);
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMSET: {
                    auto* memset = reinterpret_cast<CUpti_ActivityMemset4*>(record);
                    EventSpec spec;
                    spec.component = "backend.cuda";
                    spec.event_type = "cuda_memset";
                    spec.device = "cuda:" + std::to_string(memset->deviceId);
                    spec.fields["backend.cuda.stream_id"] =
                        std::to_string(memset->streamId);
                    spec.fields["backend.cuda.context_id"] =
                        std::to_string(memset->contextId);
                    spec.fields["backend.cuda.correlation_id"] =
                        std::to_string(memset->correlationId);
                    spec.fields["backend.cuda.memory_kind"] =
                        std::to_string(memset->memoryKind);
                    spec.metrics["bytes"] = static_cast<double>(memset->bytes);
                    spec.metrics["value"] = static_cast<double>(memset->value);
                    EmitActivityEvent(spec, memset->correlationId, 0, memset->start,
                                      memset->end);
                    break;
                }
                case CUPTI_ACTIVITY_KIND_KERNEL:
                case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                    auto* kernel = reinterpret_cast<CUpti_ActivityKernel10*>(record);
                    EventSpec spec;
                    spec.component = "backend.cuda";
                    spec.event_type = "cuda_kernel";
                    spec.device = "cuda:" + std::to_string(kernel->deviceId);
                    spec.kernel_symbol = kernel->name ? kernel->name : "";
                    spec.fields["backend.cuda.context_id"] =
                        std::to_string(kernel->contextId);
                    spec.fields["backend.cuda.stream_id"] =
                        std::to_string(kernel->streamId);
                    spec.fields["backend.cuda.correlation_id"] =
                        std::to_string(kernel->correlationId);
                    spec.fields["backend.cuda.grid_id"] = std::to_string(kernel->gridId);
                    spec.metrics["grid_x"] = static_cast<double>(kernel->gridX);
                    spec.metrics["grid_y"] = static_cast<double>(kernel->gridY);
                    spec.metrics["grid_z"] = static_cast<double>(kernel->gridZ);
                    spec.metrics["block_x"] = static_cast<double>(kernel->blockX);
                    spec.metrics["block_y"] = static_cast<double>(kernel->blockY);
                    spec.metrics["block_z"] = static_cast<double>(kernel->blockZ);
                    spec.metrics["dynamic_shared_memory"] =
                        static_cast<double>(kernel->dynamicSharedMemory);
                    spec.metrics["static_shared_memory"] =
                        static_cast<double>(kernel->staticSharedMemory);
                    EmitActivityEvent(spec, kernel->correlationId, 0, kernel->start,
                                      kernel->end);
                    break;
                }
                default:
                    break;
            }
        }

        std::size_t dropped = 0;
        if (activity_get_num_dropped_records_ != nullptr) {
            activity_get_num_dropped_records_(context, stream_id, &dropped);
        }
        if (dropped > 0) {
            ProfileContext* ctx = nullptr;
            {
                std::lock_guard<std::mutex> lock(mu_);
                ctx = bound_context_;
            }
            if (ctx != nullptr) {
                ctx->RecordLog(LogSeverity::kWarn, "backend.cuda",
                               "CUPTI dropped activity records",
                               MakeFields({{"backend.cuda.dropped_records",
                                            std::to_string(dropped)}}));
            }
        }
    }

    bool requested_{false};
    bool available_{false};
    std::string last_error_;
    void* handle_{nullptr};
    ActivityRegisterCallbacksFn activity_register_callbacks_{nullptr};
    ActivityEnableFn activity_enable_{nullptr};
    ActivityFlushAllFn activity_flush_all_{nullptr};
    ActivityGetNextRecordFn activity_get_next_record_{nullptr};
    ActivityGetNumDroppedRecordsFn activity_get_num_dropped_records_{nullptr};
    ActivityPushExternalCorrelationIdFn activity_push_external_correlation_id_{nullptr};
    ActivityPopExternalCorrelationIdFn activity_pop_external_correlation_id_{nullptr};
    GetTimestampFn get_timestamp_{nullptr};
    GetResultStringFn get_result_string_{nullptr};
    mutable std::mutex mu_;
    ProfileContext* bound_context_{nullptr};
    std::uint64_t cupti_origin_ns_{0};
    std::uint64_t next_external_id_{1};
    std::unordered_map<std::uint64_t, std::string> external_id_to_text_;
    std::unordered_map<std::uint32_t, CorrelationContext> correlation_map_;
};

#else

// 没有 CUDA/CUPTI 头文件时仍保留公共 profiling 路径可用。
class CuptiAdapter {
public:
    explicit CuptiAdapter(bool requested) : requested_(requested) {}

    bool requested() const { return requested_; }
    bool available() const { return false; }
    void BindContext(ProfileContext* ctx) { (void)ctx; }
    void UnbindContext(ProfileContext* ctx) { (void)ctx; }
    bool PushRunCorrelation(const std::string& run_id) {
        (void)run_id;
        return false;
    }
    void PopRunCorrelation() {}
    bool PushSpanCorrelation(const std::string& span_id) {
        (void)span_id;
        return false;
    }
    void PopSpanCorrelation() {}
    void Flush() {}

private:
    bool requested_{false};
};

#endif

CuptiAdapter* ResolveCuptiAdapter(bool requested) {
    static std::mutex mu;
    static std::unique_ptr<CuptiAdapter> adapter;
    std::lock_guard<std::mutex> lock(mu);
    if (requested && !adapter) {
        adapter = std::make_unique<CuptiAdapter>(true);
    }
    return adapter.get();
}

CuptiAdapter* AsCuptiAdapter(void* adapter) {
    return static_cast<CuptiAdapter*>(adapter);
}

// events.jsonl 是 agent 主要消费的数据流，因此这里显式维护稳定字段。
std::string SerializeEventLine(const std::string& trace_id, const std::string& session_id,
                               const std::string& run_id, const std::string& span_id,
                               const std::string& parent_span_id, const EventSpec& spec,
                               std::int64_t ts_ns, std::int64_t duration_ns) {
    StringMap fields = spec.fields;
    fields["thread_id"] = ThreadIdString();
    fields["schema_version"] = "1";

    std::ostringstream os;
    os << "{"
       << "\"trace_id\":\"" << EscapeJson(trace_id) << "\","
       << "\"session_id\":\"" << EscapeJson(session_id) << "\","
       << "\"run_id\":\"" << EscapeJson(run_id) << "\","
       << "\"span_id\":\"" << EscapeJson(span_id) << "\","
       << "\"parent_span_id\":\"" << EscapeJson(parent_span_id) << "\","
       << "\"component\":\"" << EscapeJson(spec.component) << "\","
       << "\"event_type\":\"" << EscapeJson(spec.event_type) << "\","
       << "\"phase\":\"" << EscapeJson(spec.phase) << "\","
       << "\"ts_ns\":" << ts_ns << ","
       << "\"duration_ns\":" << duration_ns << ","
       << "\"status\":\"" << EscapeJson(spec.status) << "\","
       << "\"severity\":\"" << EscapeJson(LogSeverityToString(spec.severity)) << "\","
       << "\"device\":\"" << EscapeJson(spec.device) << "\","
       << "\"worker_id\":" << spec.worker_id << ","
       << "\"op_name\":\"" << EscapeJson(spec.op_name) << "\","
       << "\"pass_name\":\"" << EscapeJson(spec.pass_name) << "\","
       << "\"kernel_symbol\":\"" << EscapeJson(spec.kernel_symbol) << "\","
       << "\"shape_signature\":\"" << EscapeJson(spec.shape_signature) << "\","
       << "\"message\":\"" << EscapeJson(spec.message) << "\","
       << "\"fields\":" << ToJsonObject(fields) << ","
       << "\"metrics\":" << ToJsonObject(spec.metrics) << "}";
    return os.str();
}

std::string SerializePerfettoEvent(const EventSpec& spec, std::int64_t ts_ns,
                                   std::int64_t duration_ns) {
    std::ostringstream os;
    const bool is_duration = duration_ns > 0;
    os << "{"
       << "\"name\":\"" << EscapeJson(spec.event_type.empty() ? spec.component : spec.event_type)
       << "\","
       << "\"cat\":\"" << EscapeJson(spec.component) << "\","
       << "\"ph\":\"" << (is_duration ? "X" : "i") << "\","
       << "\"ts\":" << (ts_ns / 1000) << ","
       << "\"dur\":" << (duration_ns / 1000) << ","
       << "\"pid\":" << ProcessIdNumeric() << ","
       << "\"tid\":" << ThreadIdNumeric() << ","
       << "\"s\":\"t\","
       << "\"args\":{"
       << "\"status\":\"" << EscapeJson(spec.status) << "\","
       << "\"severity\":\"" << EscapeJson(LogSeverityToString(spec.severity)) << "\","
       << "\"message\":\"" << EscapeJson(spec.message) << "\""
       << "}}";
    return os.str();
}

std::string DefaultBundleDir(const std::string& trace_id) {
    std::filesystem::path path =
        std::filesystem::current_path() / "profile_bundles" / trace_id;
    return path.string();
}

}  // namespace

ActivationScope::ActivationScope(std::shared_ptr<ProfileContext> ctx, std::string run_id)
    : previous_ctx_(tls_profile_ctx), previous_run_id_(tls_run_id),
      previous_span_stack_(tls_span_stack) {
    // 保存原有嵌套激活状态，让临时 profiling run 可以组合使用。
    tls_profile_ctx = std::move(ctx);
    tls_run_id = std::move(run_id);
    tls_span_stack.clear();
    if (tls_profile_ctx && tls_profile_ctx->cupti_adapter_ != nullptr) {
        pushed_cupti_run_ =
            AsCuptiAdapter(tls_profile_ctx->cupti_adapter_)->PushRunCorrelation(tls_run_id);
    }
    active_ = true;
}

ActivationScope::~ActivationScope() {
    if (!active_) {
        return;
    }
    if (tls_profile_ctx && pushed_cupti_run_ && tls_profile_ctx->cupti_adapter_ != nullptr) {
        AsCuptiAdapter(tls_profile_ctx->cupti_adapter_)->PopRunCorrelation();
    }
    tls_profile_ctx = std::move(previous_ctx_);
    tls_run_id = std::move(previous_run_id_);
    tls_span_stack = std::move(previous_span_stack_);
}

ScopedSpan::ScopedSpan(std::shared_ptr<ProfileContext> ctx, EventSpec spec, std::string run_id,
                       std::string parent_span_id)
    : ctx_(std::move(ctx)), spec_(std::move(spec)), run_id_(std::move(run_id)),
      parent_span_id_(std::move(parent_span_id)) {
    if (!ctx_) {
        return;
    }
    if (run_id_.empty()) {
        run_id_ = CurrentRunId();
    }
    // 默认父 span 是当前线程栈顶；跨线程串联任务时也可以显式传入父 span。
    if (parent_span_id_.empty() && !tls_span_stack.empty()) {
        parent_span_id_ = tls_span_stack.back();
    }
    span_id_ = ctx_->NextSpanId();
    start_ns_ = NowSteadyNs() - ctx_->start_monotonic_ns_;
    tls_span_stack.push_back(span_id_);
    pushed_to_stack_ = true;
    if (ctx_->options().enable_nvtx) {
        if (NvtxAdapter* adapter = ResolveNvtxAdapter(true)) {
            adapter->Push(spec_.event_type.empty() ? spec_.component : spec_.event_type);
        }
    }
    if (ctx_->cupti_adapter_ != nullptr) {
        pushed_cupti_span_ = AsCuptiAdapter(ctx_->cupti_adapter_)->PushSpanCorrelation(span_id_);
    }
    active_ = true;
}

ScopedSpan::~ScopedSpan() { Close(); }

ScopedSpan::ScopedSpan(ScopedSpan&& other) noexcept
    : ctx_(std::move(other.ctx_)), spec_(std::move(other.spec_)),
      run_id_(std::move(other.run_id_)), span_id_(std::move(other.span_id_)),
      parent_span_id_(std::move(other.parent_span_id_)), start_ns_(other.start_ns_),
      active_(other.active_), pushed_to_stack_(other.pushed_to_stack_) {
    other.active_ = false;
    other.pushed_to_stack_ = false;
}

ScopedSpan& ScopedSpan::operator=(ScopedSpan&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    Close();
    ctx_ = std::move(other.ctx_);
    spec_ = std::move(other.spec_);
    run_id_ = std::move(other.run_id_);
    span_id_ = std::move(other.span_id_);
    parent_span_id_ = std::move(other.parent_span_id_);
    start_ns_ = other.start_ns_;
    active_ = other.active_;
    pushed_to_stack_ = other.pushed_to_stack_;
    other.active_ = false;
    other.pushed_to_stack_ = false;
    return *this;
}

void ScopedSpan::SetStatus(const std::string& status) { spec_.status = status; }

void ScopedSpan::SetMessage(const std::string& message) { spec_.message = message; }

void ScopedSpan::AddField(const std::string& key, const std::string& value) {
    spec_.fields[key] = value;
}

void ScopedSpan::AddMetric(const std::string& key, double value) {
    spec_.metrics[key] = value;
}

void ScopedSpan::Close() {
    if (!active_) {
        return;
    }
    if (ctx_->options().enable_nvtx) {
        if (NvtxAdapter* adapter = ResolveNvtxAdapter(true)) {
            adapter->Pop();
        }
    }
    if (pushed_cupti_span_ && ctx_->cupti_adapter_ != nullptr) {
        AsCuptiAdapter(ctx_->cupti_adapter_)->PopSpanCorrelation();
    }
    if (pushed_to_stack_ && !tls_span_stack.empty() && tls_span_stack.back() == span_id_) {
        tls_span_stack.pop_back();
    }
    std::int64_t end_ns = NowSteadyNs() - ctx_->start_monotonic_ns_;
    ctx_->RecordCompletedSpan(spec_, run_id_, span_id_, parent_span_id_, start_ns_, end_ns);
    active_ = false;
}

ProfileContext::ProfileContext(ProfileOptions options)
    : options_(std::move(options)), trace_id_(MakeId("trace")),
      session_id_(MakeId("session")), start_wall_time_ns_(NowWallNs()),
      start_monotonic_ns_(NowSteadyNs()) {
    bundle_dir_ = options_.bundle_dir.empty() ? DefaultBundleDir(trace_id_) : options_.bundle_dir;
    // adapter 可能为空或不可用；此时其余 profiling 功能仍可继续工作。
    cupti_adapter_ = ResolveCuptiAdapter(options_.enable_cupti);
    if (options_.enable_cupti && cupti_adapter_ != nullptr &&
        AsCuptiAdapter(cupti_adapter_)->available()) {
        AsCuptiAdapter(cupti_adapter_)->BindContext(this);
        cupti_bound_ = true;
    } else if (options_.enable_cupti) {
        RecordLog(LogSeverity::kWarn, "backend.cuda",
                  "CUPTI requested but the activity collector is unavailable");
    }
}

ProfileContext::~ProfileContext() {
    Flush();
    if (cupti_adapter_ != nullptr && cupti_bound_) {
        AsCuptiAdapter(cupti_adapter_)->UnbindContext(this);
        cupti_bound_ = false;
    }
}

std::shared_ptr<ProfileContext> ProfileContext::Create(const ProfileOptions& options) {
    return std::shared_ptr<ProfileContext>(new ProfileContext(options));
}

std::string ProfileContext::NextRunId(const std::string& prefix) {
    std::lock_guard<std::mutex> lock(mu_);
    std::ostringstream os;
    os << prefix << "-" << ++run_counter_;
    return os.str();
}

std::string ProfileContext::NextSpanId() {
    std::lock_guard<std::mutex> lock(mu_);
    std::ostringstream os;
    os << "span-" << ++span_counter_;
    return os.str();
}

void ProfileContext::RecordCompletedSpan(const EventSpec& spec, const std::string& run_id,
                                         const std::string& span_id,
                                         const std::string& parent_span_id,
                                         std::int64_t start_ns, std::int64_t end_ns) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureBundleLayout();
    // 同一个事件同时追加到两种 bundle 格式，避免 JSONL 和 trace 脱节。
    serialized_events_.push_back(
        SerializeEventLine(trace_id_, session_id_, run_id, span_id, parent_span_id, spec,
                           start_ns, std::max<std::int64_t>(0, end_ns - start_ns)));
    perfetto_events_.push_back(
        SerializePerfettoEvent(spec, start_ns, std::max<std::int64_t>(0, end_ns - start_ns)));
    ++event_count_;
}

void ProfileContext::RecordInstant(EventSpec spec, const std::string& run_id,
                                   const std::string& parent_span_id) {
    const std::string effective_run_id = run_id.empty() ? CurrentRunId() : run_id;
    const std::string effective_parent =
        parent_span_id.empty() ? CurrentSpanId() : parent_span_id;
    std::int64_t ts_ns = NowSteadyNs() - start_monotonic_ns_;
    std::string span_id = NextSpanId();
    std::lock_guard<std::mutex> lock(mu_);
    EnsureBundleLayout();
    serialized_events_.push_back(
        SerializeEventLine(trace_id_, session_id_, effective_run_id, span_id, effective_parent,
                           spec, ts_ns, 0));
    perfetto_events_.push_back(SerializePerfettoEvent(spec, ts_ns, 0));
    ++event_count_;
}

void ProfileContext::RecordLog(LogSeverity severity, const std::string& component,
                               const std::string& message, StringMap fields,
                               MetricMap metrics, const std::string& run_id) {
    EventSpec spec;
    spec.component = component;
    spec.event_type = "log";
    spec.severity = severity;
    spec.message = message;
    spec.fields = std::move(fields);
    spec.metrics = std::move(metrics);
    spec.phase = "instant";
    RecordInstant(std::move(spec), run_id);
}

std::string ProfileContext::WriteArtifact(const std::string& relative_path,
                                          const std::string& content) {
    std::lock_guard<std::mutex> lock(mu_);
    EnsureBundleLayout();
    std::filesystem::path path = std::filesystem::path(bundle_dir_) / "artifacts" / relative_path;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    ofs << content;
    return path.string();
}

void ProfileContext::PublishDiagnostics(const std::vector<DiagnosticEntry>& diagnostics) {
    std::lock_guard<std::mutex> lock(mu_);
    diagnostics_ = diagnostics;
    WriteAllOutputsLocked();
}

void ProfileContext::Flush() {
    // 写文件前先强制刷出待处理的 CUDA activity 记录。
    if (cupti_adapter_ != nullptr && cupti_bound_) {
        AsCuptiAdapter(cupti_adapter_)->Flush();
    }
    std::lock_guard<std::mutex> lock(mu_);
    EnsureBundleLayout();
    WriteAllOutputsLocked();
}

void ProfileContext::EnsureBundleLayout() {
    if (bundle_initialized_) {
        return;
    }
    std::filesystem::create_directories(std::filesystem::path(bundle_dir_) / "artifacts");
    bundle_initialized_ = true;
}

void ProfileContext::WriteAllOutputsLocked() {
    // 调用方必须已持有 mu_；每次 flush 都重写一份自包含 bundle 快照。
    std::ofstream manifest(std::filesystem::path(bundle_dir_) / "manifest.json",
                           std::ios::out | std::ios::trunc);
    manifest << "{"
             << "\"schema_version\":" << options_.schema_version << ","
             << "\"trace_id\":\"" << EscapeJson(trace_id_) << "\","
             << "\"session_id\":\"" << EscapeJson(session_id_) << "\","
             << "\"bundle_dir\":\"" << EscapeJson(bundle_dir_) << "\","
             << "\"log_level\":\"" << EscapeJson(LogSeverityToString(options_.log_level))
             << "\","
             << "\"ir_capture_mode\":\"" << EscapeJson(IRCaptureModeToString(options_.ir_capture_mode))
             << "\","
             << "\"enable_nvtx\":" << (options_.enable_nvtx ? "true" : "false") << ","
             << "\"enable_cupti\":" << (options_.enable_cupti ? "true" : "false") << ","
             << "\"cupti_available\":"
             << ((cupti_adapter_ != nullptr && AsCuptiAdapter(cupti_adapter_)->available())
                     ? "true"
                     : "false")
             << ","
             << "\"record_execution_plan_details\":"
             << (options_.record_execution_plan_details ? "true" : "false") << ","
             << "\"event_count\":" << event_count_ << "}";

    std::ofstream events(std::filesystem::path(bundle_dir_) / "events.jsonl",
                         std::ios::out | std::ios::trunc);
    for (const auto& line : serialized_events_) {
        events << line << "\n";
    }

    std::ofstream trace(std::filesystem::path(bundle_dir_) / "trace.json",
                        std::ios::out | std::ios::trunc);
    trace << "{"
          << "\"traceEvents\":[";
    for (std::size_t i = 0; i < perfetto_events_.size(); ++i) {
        if (i) {
            trace << ",";
        }
        trace << perfetto_events_[i];
    }
    trace << "],"
          << "\"metadata\":{"
          << "\"trace_id\":\"" << EscapeJson(trace_id_) << "\","
          << "\"session_id\":\"" << EscapeJson(session_id_) << "\","
          << "\"start_time_ns\":" << start_wall_time_ns_ << "}}";

    StringMap component_counts;
    // summary 从 events.jsonl 派生，保证 bundle 只有一个事件事实源。
    for (const auto& line : serialized_events_) {
        std::size_t pos = line.find("\"component\":\"");
        if (pos == std::string::npos) {
            continue;
        }
        pos += std::string("\"component\":\"").size();
        std::size_t end = line.find("\"", pos);
        std::string key = line.substr(pos, end - pos);
        auto it = component_counts.find(key);
        std::uint64_t count = it == component_counts.end() ? 0 : std::stoull(it->second);
        component_counts[key] = std::to_string(count + 1);
    }

    std::ofstream summary(std::filesystem::path(bundle_dir_) / "summary.json",
                          std::ios::out | std::ios::trunc);
    summary << "{"
            << "\"trace_id\":\"" << EscapeJson(trace_id_) << "\","
            << "\"session_id\":\"" << EscapeJson(session_id_) << "\","
            << "\"event_count\":" << event_count_ << ","
            << "\"component_counts\":" << ToJsonObject(component_counts) << "}";

    if (diagnostics_.empty()) {
        diagnostics_.push_back({"not_analyzed", "info", "analysis",
                                "Bundle created. Run PYTHONPATH=python python -m kxc_agent.cli "
                                "analyze_bundle --bundle <path> to populate diagnostics."});
    }

    std::ofstream diagnosis_json(std::filesystem::path(bundle_dir_) / "diagnosis.json",
                                 std::ios::out | std::ios::trunc);
    diagnosis_json << "{"
                   << "\"trace_id\":\"" << EscapeJson(trace_id_) << "\","
                   << "\"diagnostics\":[";
    for (std::size_t i = 0; i < diagnostics_.size(); ++i) {
        if (i) {
            diagnosis_json << ",";
        }
        diagnosis_json << "{"
                       << "\"category\":\"" << EscapeJson(diagnostics_[i].category) << "\","
                       << "\"severity\":\"" << EscapeJson(diagnostics_[i].severity) << "\","
                       << "\"component\":\"" << EscapeJson(diagnostics_[i].component) << "\","
                       << "\"summary\":\"" << EscapeJson(diagnostics_[i].summary) << "\"}";
    }
    diagnosis_json << "]}";

    std::ofstream diagnosis_md(std::filesystem::path(bundle_dir_) / "diagnosis.md",
                               std::ios::out | std::ios::trunc);
    diagnosis_md << "# Diagnostics\n\n";
    for (const auto& item : diagnostics_) {
        diagnosis_md << "- [" << item.severity << "] " << item.category << " (" << item.component
                     << "): " << item.summary << "\n";
    }
}

std::shared_ptr<ProfileContext> CurrentContext() { return tls_profile_ctx; }

const std::string& CurrentRunId() { return tls_run_id; }

const std::string& CurrentSpanId() {
    static const std::string empty;
    return tls_span_stack.empty() ? empty : tls_span_stack.back();
}

}  // namespace profiling
}  // namespace kxc
