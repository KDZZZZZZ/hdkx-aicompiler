/*! \file include/kxc/runtime/execution_observer.h
 * \brief 定义 runtime 层执行观测钩子、线程本地当前观测器和重入保护。
 */

// 职责简介：
// - 声明 ExecutionObserver 纯数据钩子接口，覆盖运行开始/结束、内核提交、
//   分配/复用/别名、拷贝和完成观测；runtime 只上报事实，不解释含义。
// - 提供线程本地"当前观测器"与 RAII 作用域，供 Storage::Alloc、
//   StorageCopySync/Async 这类没有 session 参数的底层路径查询。
// - 本头文件只依赖 runtime 自身类型；profiling 适配器由上层装配，
//   依赖方向始终是 profiling include runtime。

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include "kxc/runtime/device.h"

namespace kxc {
namespace runtime {

/*! \brief 调用方给一次运行附加的结构化关联字段。
 *
 * Runtime 只透传这些字段，既不解释其含义，也不把它们放进 kernel ABI
 * 或编译产物 identity。profiling 等观测适配器可以将它们复制到同一 run
 * 的事件上，用于模型阶段、导出回执和状态 extent 的关联。
 */
using ExecutionMetadata = std::unordered_map<std::string, std::string>;

/*! \brief 一次运行的观测关联；runtime 原样保存并返回，不解释内容。 */
struct ExecutionRunCorrelation {
    /*! \brief 观测方为本次运行生成的关联 id；空表示无关联。 */
    std::string run_id;
    /*! \brief 本次运行内部事件应指向的父 span id。 */
    std::string span_id;
};

/*! \brief 运行开始时上报的纯数据事实。 */
struct ExecutionRunStart {
    /*! \brief 执行设备。 */
    Device device;
    /*! \brief 本次运行的输入个数。 */
    std::size_t input_count{0};
    /*! \brief 执行计划声明的内核调用个数。 */
    std::size_t kernel_count{0};
    /*! \brief 本次运行的调用方关联字段；空表示未提供。 */
    ExecutionMetadata metadata;
};

/*! \brief 运行结束时上报的纯数据事实。 */
struct ExecutionRunEnd {
    /*! \brief 执行设备。 */
    Device device;
    /*! \brief 本次运行的输入个数。 */
    std::size_t input_count{0};
    /*! \brief 执行计划声明的内核调用个数。 */
    std::size_t kernel_count{0};
    /*! \brief 实际完成提交动作的内核个数。 */
    std::size_t submit_count{0};
    /*! \brief 运行是否未带异常走完提交路径。 */
    bool completed{false};
    /*! \brief 失败时原始异常文本；成功时为空。 */
    std::string error_message;
};

/*! \brief 一次内核提交的纯数据事实。 */
struct KernelSubmitInfo {
    /*! \brief 内核目标设备。 */
    Device device;
    /*! \brief 执行计划内的调用序号。 */
    std::size_t call_index{0};
    /*! \brief 内核符号。 */
    std::string kernel_symbol;
};

/*! \brief 完成观测回调；at_registration 表示注册时句柄已经完成。 */
using ExecutionCompletionCallback = std::function<void(bool at_registration)>;

// A synchronous observation activation returns same-thread cleanup. It never
// survives its lexical scope or becomes an asynchronous completion callback.
using ExecutionScopeExit = std::function<void()>;

/*! \brief 分配记账的形态：新分配、复用既有兼容存储、或就地别名绑定。 */
enum class AllocationKind {
    kFresh, /*!< 新分配的设备存储。 */
    kReuse, /*!< 复用命中的兼容存储，不重复计为新分配。 */
    kAlias, /*!< 输出在既有存储上的就地别名绑定。 */
};

/*! \brief 一次分配记账的纯数据事实。 */
struct AllocationInfo {
    /*! \brief 分配发生的设备。 */
    Device device;
    /*! \brief 分配/复用的字节量。 */
    std::uint64_t bytes{0};
    /*! \brief 地址对齐要求，单位为字节。 */
    std::size_t alignment{0};
    /*! \brief 分配形态。 */
    AllocationKind kind{AllocationKind::kFresh};
    /*! \brief 主机执行该分配（含复用判定）的耗时，单位为纳秒。 */
    std::int64_t duration_ns{0};
    /*! \brief 分配失败时的原始异常文本；成功时为空。 */
    std::string error_message;
};

/*! \brief 一次数据拷贝的纯数据事实。 */
struct CopyInfo {
    /*! \brief 源存储所属设备。 */
    Device from_device;
    /*! \brief 目标存储所属设备。 */
    Device to_device;
    /*! \brief 拷贝字节量。 */
    std::uint64_t bytes{0};
    /*! \brief 是否经异步拷贝接口提交；同步拷贝恒为 false。 */
    bool submitted_async{false};
    /*! \brief 主机实际执行拷贝的耗时（含同步完成的 CPU 异步接口）；
     *  设备异步提交路径为 0，单位为纳秒。 */
    std::int64_t duration_ns{0};
    /*! \brief 拷贝失败时的原始异常文本；成功时为空。 */
    std::string error_message;
};

/*! \brief runtime 执行观测钩子。实现方必须自行吞掉自身异常：
 *  观测永远不能改变执行结果，也不能覆盖原始错误。 */
class ExecutionObserver {
public:
    virtual ~ExecutionObserver() = default;

    // Optional backend/tool activation around the actual runtime work. The
    // kernel pointer is borrowed only during this call; implementations copy
    // any facts they retain. The returned cleanup must be safe during unwind.
    virtual ExecutionScopeExit OnScopeEnter(
        const ExecutionRunCorrelation&, const KernelSubmitInfo*) { return {}; }

    // A validated copy uses the same lexical activation contract. Existing
    // observers can keep the generic hook; adapters may use the copy facts.
    virtual ExecutionScopeExit OnCopyScopeEnter(
        const ExecutionRunCorrelation& correlation, const CopyInfo&) {
        return OnScopeEnter(correlation, nullptr);
    }

    /*! \brief 运行开始；返回本次运行的观测关联（可为空关联）。 */
    virtual ExecutionRunCorrelation OnRunStart(const ExecutionRunStart& run) = 0;
    /*! \brief 运行结束；成功与失败（status=error 语义）都恰好上报一次。 */
    virtual void OnRunEnd(const ExecutionRunEnd& run,
                          const ExecutionRunCorrelation& correlation) = 0;
    /*! \brief 内核提交动作开始前调用；观测方据此记录执行区间起点。 */
    virtual void OnKernelBegin(const KernelSubmitInfo& kernel,
                               const ExecutionRunCorrelation& correlation) = 0;
    /*! \brief 主机完成一次内核提交动作后调用；返回的回调（可为空）由
     *  runtime 安装到完成句柄上，在 Wait/IsReady/析构观测到完成时触发。 */
    virtual ExecutionCompletionCallback OnKernelSubmitted(
        const KernelSubmitInfo& kernel,
        const ExecutionRunCorrelation& correlation) = 0;
    /*! \brief 一次分配/复用/别名记账。 */
    virtual void OnAllocation(const AllocationInfo& allocation,
                              const ExecutionRunCorrelation& correlation) = 0;
    /*! \brief 一次同步拷贝（或异步拷贝的同步失败）记账。 */
    virtual void OnCopy(const CopyInfo& copy,
                        const ExecutionRunCorrelation& correlation) = 0;
    /*! \brief 异步拷贝提交成功后调用；返回的回调安装在完成句柄上。 */
    virtual ExecutionCompletionCallback OnCopySubmitted(
        const CopyInfo& copy,
        const ExecutionRunCorrelation& correlation) = 0;
};

/*! \brief 当前线程的观测器槽位；ExecutionObservationScope 需要写入。 */
inline ExecutionObserver*& CurrentExecutionObserverSlot() {
    thread_local ExecutionObserver* observer = nullptr;
    return observer;
}

/*! \brief 返回当前线程安装的观测器；未装配时为 nullptr。 */
inline ExecutionObserver* CurrentExecutionObserver() {
    return CurrentExecutionObserverSlot();
}

/*! \brief 当前线程的运行关联槽位；ExecutionObservationScope 需要写入。 */
inline ExecutionRunCorrelation& CurrentExecutionRunCorrelationSlot() {
    thread_local ExecutionRunCorrelation correlation;
    return correlation;
}

/*! \brief 返回当前线程的运行关联；未装配时 run_id/span_id 为空。 */
inline const ExecutionRunCorrelation& CurrentExecutionRunCorrelation() {
    return CurrentExecutionRunCorrelationSlot();
}

/*! \brief 当前线程活跃的观测 hold 深度。 */
inline std::size_t& ExecutionObservationHoldDepth() {
    thread_local std::size_t depth = 0;
    return depth;
}

/*! \brief hold 活跃期间，本线程的观测派发被抑制：记录路径自身分配或拷贝
 *  内存时不得递归触发钩子，也让外层记账可以独占一次分配事件。 */
class ExecutionObservationHold {
public:
    /*! \brief 深度加一；active=false 时不改变状态。 */
    explicit ExecutionObservationHold(bool active = true) : active_(active) {
        if (active_) ++ExecutionObservationHoldDepth();
    }
    /*! \brief 深度减一，恢复派发。 */
    ~ExecutionObservationHold() {
        if (active_) --ExecutionObservationHoldDepth();
    }

    ExecutionObservationHold(const ExecutionObservationHold&) = delete;
    ExecutionObservationHold& operator=(const ExecutionObservationHold&) = delete;

private:
    bool active_;
};

/*! \brief 派发一个观测钩子：未装配或被 hold 抑制时跳过；钩子异常被吞掉，
 *  保证观测不能改变执行结果。 */
template <typename Hook>
void DispatchExecutionObservation(ExecutionObserver* observer, Hook&& hook) {
    if (observer == nullptr || ExecutionObservationHoldDepth() != 0) return;
    const ExecutionObservationHold hold;
    try {
        hook(*observer);
    } catch (...) {
        // 观测永远不能改变执行结果；实现方也需自行吞掉异常，这里是双保险。
    }
}

/*! \brief RAII 作用域：在本线程安装当前观测器与运行关联，供底层分配和
 *  拷贝路径查询。离开作用域时恢复原值。 */
class ExecutionObservationScope {
public:
    /*! \brief 保存旧线程状态并安装新观测器与关联。 */
    ExecutionObservationScope(ExecutionObserver* observer,
                              ExecutionRunCorrelation correlation,
                              const KernelSubmitInfo* kernel = nullptr,
                              const CopyInfo* copy = nullptr)
        : previous_observer_(CurrentExecutionObserver()),
          previous_correlation_(CurrentExecutionRunCorrelation()) {
        CurrentExecutionObserverSlot() = observer;
        CurrentExecutionRunCorrelationSlot() = std::move(correlation);
        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
            exit_ = copy ? sink.OnCopyScopeEnter(CurrentExecutionRunCorrelation(), *copy)
                         : sink.OnScopeEnter(CurrentExecutionRunCorrelation(), kernel);
        });
    }
    /*! \brief 恢复旧的观测器与运行关联。 */
    ~ExecutionObservationScope() {
        if (exit_) {
            const ExecutionObservationHold hold;
            try { exit_(); } catch (...) {
                // An observer cannot replace an execution exception.
            }
        }
        CurrentExecutionObserverSlot() = previous_observer_;
        CurrentExecutionRunCorrelationSlot() = std::move(previous_correlation_);
    }

    ExecutionObservationScope(const ExecutionObservationScope&) = delete;
    ExecutionObservationScope& operator=(const ExecutionObservationScope&) = delete;

private:
    ExecutionObserver* previous_observer_;
    ExecutionRunCorrelation previous_correlation_;
    ExecutionScopeExit exit_;
};

}  // namespace runtime
}  // namespace kxc
