/*! \file include/kxc/profiling/profiling.h
 * \brief 定义 KXC profiling 的事件模型、RAII span、bundle 写出和诊断公共 API。
 */

// 职责简介：
// - 定义 KXC profiling 的公共事件模型、配置项和诊断结构。
// - 提供 ActivationScope/ScopedSpan 这类 RAII API，负责在线程内传播 run/span 上下文。
// - 暴露 ProfileContext，用于收集事件、写出 bundle、保存 IR artifact 和生成 trace。

#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace kxc {
namespace profiling {

/*! \brief profiling 日志和事件严重级别。 */
enum class LogSeverity : int {
    kDebug = 0,
    kInfo = 1,
    kWarn = 2,
    kError = 3,
};

/*! \brief IR artifact 捕获策略。 */
enum class IRCaptureMode : int {
    kDisabled = 0,
    kChangedOrFailed = 1,
    kVerbose = 2,
};

using StringMap = std::unordered_map<std::string, std::string>;
using MetricMap = std::unordered_map<std::string, double>;

/*! \brief 统一的事件载荷，同时供 JSONL bundle、Perfetto trace 和诊断逻辑使用。 */
struct EventSpec {
    std::string component;
    std::string event_type;
    std::string phase{"complete"};
    std::string status{"ok"};
    LogSeverity severity{LogSeverity::kInfo};
    std::string device;
    int worker_id{-1};
    std::string op_name;
    std::string pass_name;
    std::string kernel_symbol;
    std::string shape_signature;
    std::string message;
    StringMap fields;
    MetricMap metrics;
};

/*! \brief 运行时可配置的 profiling 开关。 */
struct ProfileOptions {
    bool enabled{false};
    std::string bundle_dir;
    LogSeverity log_level{LogSeverity::kInfo};
    IRCaptureMode ir_capture_mode{IRCaptureMode::kChangedOrFailed};
    bool enable_nvtx{false};
    bool enable_cupti{false};
    bool record_pass_ir{true};
    bool record_execution_plan_details{true};
    int schema_version{1};
};

/*! \brief 写入 profiling bundle 的诊断条目。 */
struct DiagnosticEntry {
    std::string category;
    std::string severity;
    std::string component;
    std::string summary;
};

class ProfileContext;

/*! \brief 在当前作用域内安装 ProfileContext/run id 到线程本地状态。 */
class ActivationScope {
public:
    /*! \brief 保存旧线程状态并激活新的 profiling 上下文。 */
    ActivationScope(std::shared_ptr<ProfileContext> ctx, std::string run_id,
                    std::string parent_span_id = {});
    ~ActivationScope();

    ActivationScope(const ActivationScope&) = delete;
    ActivationScope& operator=(const ActivationScope&) = delete;

private:
    std::shared_ptr<ProfileContext> previous_ctx_;
    std::string previous_run_id_;
    std::vector<std::string> previous_span_stack_;
    void* cupti_adapter_{nullptr};
    bool active_{false};
    bool pushed_cupti_run_{false};
    bool pushed_cupti_span_{false};
};

/*! \brief RAII 形式的耗时事件，析构时写入当前 ProfileContext。 */
class ScopedSpan {
public:
    ScopedSpan() = default;
    /*! \brief 创建并启动一个 span，可显式指定 run id 和 parent span。 */
    ScopedSpan(std::shared_ptr<ProfileContext> ctx, EventSpec spec, std::string run_id = "",
               std::string parent_span_id = "");
    ~ScopedSpan();

    ScopedSpan(const ScopedSpan&) = delete;
    ScopedSpan& operator=(const ScopedSpan&) = delete;
    ScopedSpan(ScopedSpan&& other) noexcept;
    ScopedSpan& operator=(ScopedSpan&& other) noexcept;

    /*! \brief 设置 span 完成状态。 */
    void SetStatus(const std::string& status);
    /*! \brief 设置 span 说明消息。 */
    void SetMessage(const std::string& message);
    /*! \brief 追加字符串字段。 */
    void AddField(const std::string& key, const std::string& value);
    /*! \brief 追加数值指标。 */
    void AddMetric(const std::string& key, double value);

    bool active() const { return active_; }
    const std::string& span_id() const { return span_id_; }

private:
    void Close() noexcept;

    std::shared_ptr<ProfileContext> ctx_;
    EventSpec spec_;
    std::string run_id_;
    std::string span_id_;
    std::string parent_span_id_;
    std::int64_t start_ns_{0};
    bool active_{false};
    bool pushed_to_stack_{false};
    bool pushed_cupti_span_{false};
};

/*! \brief 管理一次 profiling bundle，并以线程安全方式序列化所有事件。 */
class ProfileContext : public std::enable_shared_from_this<ProfileContext> {
public:
    /*! \brief 创建 profiling 上下文并按需初始化 bundle 目录。 */
    static std::shared_ptr<ProfileContext> Create(const ProfileOptions& options);
    ~ProfileContext();

    const ProfileOptions& options() const { return options_; }
    const std::string& trace_id() const { return trace_id_; }
    const std::string& session_id() const { return session_id_; }
    const std::string& bundle_dir() const { return bundle_dir_; }

    /*! \brief 生成带 prefix 的下一个 run id。 */
    std::string NextRunId(const std::string& prefix);
    /*! \brief 生成下一个 span id。 */
    std::string NextSpanId();
    /*! \brief 当前单调时钟相对 bundle 起点的纳秒值；供跨线程补记 span
     *  时与 RecordCompletedSpan 的相对时间轴保持一致。 */
    std::int64_t ElapsedMonotonicNs() const;

    /*! \brief 记录一个已完成的耗时 span。 */
    void RecordCompletedSpan(const EventSpec& spec, const std::string& run_id,
                             const std::string& span_id, const std::string& parent_span_id,
                             std::int64_t start_ns, std::int64_t end_ns);
    /*! \brief 记录一个瞬时事件。 */
    void RecordInstant(EventSpec spec, const std::string& run_id = "",
                       const std::string& parent_span_id = "");
    /*! \brief 记录一条日志事件。 */
    void RecordLog(LogSeverity severity, const std::string& component, const std::string& message,
                   StringMap fields = {}, MetricMap metrics = {},
                   const std::string& run_id = "");
    /*! \brief 写入一个相对 bundle 路径下的 artifact，并返回最终路径。 */
    std::string WriteArtifact(const std::string& relative_path, const std::string& content);
    /*! \brief 发布本次 profiling 的诊断结果。 */
    void PublishDiagnostics(const std::vector<DiagnosticEntry>& diagnostics);
    /*! \brief 将内存中的事件、trace 和诊断快照写入 bundle。 */
    void Flush();

private:
    friend class ScopedSpan;
    friend class ActivationScope;

    explicit ProfileContext(ProfileOptions options);

    void EnsureBundleLayout();
    void WriteAllOutputsLocked();

    ProfileOptions options_;
    std::string trace_id_;
    std::string session_id_;
    std::string bundle_dir_;
    // wall time 用于 trace 元数据；monotonic time 用于稳定计算事件耗时。
    std::int64_t start_wall_time_ns_{0};
    std::int64_t start_monotonic_ns_{0};
    // 事件序列化后保留在内存中，Flush 时重写完整 bundle 快照。
    std::vector<std::string> serialized_events_;
    std::vector<std::string> perfetto_events_;
    std::vector<DiagnosticEntry> diagnostics_;
    std::size_t event_count_{0};
    std::uint64_t run_counter_{0};
    std::uint64_t span_counter_{0};
    bool bundle_initialized_{false};
    void* cupti_adapter_{nullptr};
    bool cupti_bound_{false};
    mutable std::mutex mu_;
};

/*! \brief 将环境变量中的 profiling 配置覆盖应用到 options。 */
ProfileOptions ApplyEnvironmentOverrides(ProfileOptions options);

/*! \brief 返回当前线程激活的 ProfileContext。 */
std::shared_ptr<ProfileContext> CurrentContext();
/*! \brief 返回当前线程激活的 run id。 */
const std::string& CurrentRunId();
/*! \brief 返回当前线程栈顶 span id。 */
const std::string& CurrentSpanId();

/*! \brief 将 LogSeverity 转成稳定字符串。 */
std::string LogSeverityToString(LogSeverity severity);
/*! \brief 将 IRCaptureMode 转成稳定字符串。 */
std::string IRCaptureModeToString(IRCaptureMode mode);
/*! \brief 将输入 shape 列表格式化为诊断/事件字段。 */
std::string ShapeSignatureToString(const std::vector<std::vector<int64_t>>& shapes);
/*! \brief 从 initializer_list 构造 StringMap。 */
StringMap MakeFields(
    std::initializer_list<std::pair<std::string, std::string>> init);
/*! \brief 根据上下文配置和 pass 结果判断是否捕获 IR。 */
bool ShouldCaptureIR(const std::shared_ptr<ProfileContext>& ctx, bool changed, bool failed);

}  // namespace profiling
}  // namespace kxc
