/*! \file src/distributed/execution_plan_json.cc
 * \brief Implements strict JSON persistence for distributed execution plans.
 */

#include "kxc/distributed/execution_plan.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kxc/runtime/device.h"

namespace kxc {

namespace {

// 表示执行计划 JSON 所需的最小值树，避免引入额外解析依赖。
struct J {
    // 区分 null、布尔、整数、字符串、数组和对象六类 JSON 值。
    enum K { N, B, I, S, A, O } k{N};
    bool b{false};
    int64_t i{0};
    std::string s;
    std::vector<J> a;
    std::unordered_map<std::string, J> o;
};

// 对可信契约仍执行严格语法和重复键检查的最小 JSON 解析器。
class P {
public:
    // 绑定待解析文本，解析器不取得字符串所有权。
    explicit P(const std::string& text) : t_(text) {}

    // 解析唯一根值并拒绝尾随字符。
    J Parse() {
        WS();
        J v = V();
        WS();
        if (p_ != t_.size()) Err("trailing characters");
        return v;
    }

private:
    const std::string& t_;
    size_t p_{0};

    // 抛出包含当前字节位置的统一解析错误。
    [[noreturn]] void Err(const std::string& m) const {
        std::stringstream ss;
        ss << "JSON parse error at " << p_ << ": " << m;
        throw std::runtime_error(ss.str());
    }

    // 查看当前字符但不推进游标，末尾返回哨兵零字符。
    char Peek() const { return p_ < t_.size() ? t_[p_] : '\0'; }
    // 读取当前字符并推进游标，禁止越过输入末尾。
    char Get() {
        if (p_ >= t_.size()) Err("unexpected EOF");
        return t_[p_++];
    }
    // 跳过 JSON 允许的空白字符。
    void WS() {
        while (p_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[p_])) != 0) ++p_;
    }
    // 消费指定结构字符，否则报告当前位置。
    void Expect(char c) {
        if (Get() != c) {
            std::stringstream ss;
            ss << "expected '" << c << "'";
            Err(ss.str());
        }
    }
    // 消费 true、false 或 null 等固定字面量。
    void Lit(const char* s) {
        while (*s) {
            if (Get() != *s) Err("invalid literal");
            ++s;
        }
    }

    // 根据首字符分派具体 JSON 值解析器。
    J V() {
        WS();
        char c = Peek();
        if (c == '{') return Obj();
        if (c == '[') return Arr();
        if (c == '"') return Str();
        if (c == 't') return True();
        if (c == 'f') return False();
        if (c == 'n') return Null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) != 0) return Num();
        Err("unexpected token");
        return J();
    }

    // 解析对象并拒绝重复键，避免后值静默覆盖计划字段。
    J Obj() {
        J v;
        v.k = J::O;
        Expect('{');
        WS();
        if (Peek() == '}') {
            Get();
            return v;
        }
        while (true) {
            J key = Str();
            WS();
            Expect(':');
            J val = V();
            auto ok = v.o.emplace(std::move(key.s), std::move(val));
            if (!ok.second) Err("duplicate key");
            WS();
            char c = Get();
            if (c == '}') break;
            if (c != ',') Err("expected ',' or '}'");
            WS();
        }
        return v;
    }

    // 解析有序 JSON 数组。
    J Arr() {
        J v;
        v.k = J::A;
        Expect('[');
        WS();
        if (Peek() == ']') {
            Get();
            return v;
        }
        while (true) {
            v.a.push_back(V());
            WS();
            char c = Get();
            if (c == ']') break;
            if (c != ',') Err("expected ',' or ']'");
            WS();
        }
        return v;
    }

    // 解析字符串转义；非 ASCII unicode 暂以问号保持单字节契约。
    J Str() {
        J v;
        v.k = J::S;
        Expect('"');
        while (true) {
            char c = Get();
            if (c == '"') break;
            if (c == '\\') {
                char e = Get();
                switch (e) {
                    case '"': v.s.push_back('"'); break;
                    case '\\': v.s.push_back('\\'); break;
                    case '/': v.s.push_back('/'); break;
                    case 'b': v.s.push_back('\b'); break;
                    case 'f': v.s.push_back('\f'); break;
                    case 'n': v.s.push_back('\n'); break;
                    case 'r': v.s.push_back('\r'); break;
                    case 't': v.s.push_back('\t'); break;
                    case 'u': {
                        int cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = Get();
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp += (h - '0');
                            else if (h >= 'a' && h <= 'f') cp += (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp += (h - 'A' + 10);
                            else Err("invalid unicode escape");
                        }
                        v.s.push_back(cp <= 0x7F ? static_cast<char>(cp) : '?');
                        break;
                    }
                    default:
                        Err("invalid escape");
                }
            } else {
                v.s.push_back(c);
            }
        }
        return v;
    }

    // 解析计划契约所需的有符号整数，并拒绝浮点或指数形式。
    J Num() {
        J v;
        v.k = J::I;
        size_t st = p_;
        if (Peek() == '-') ++p_;
        if (Peek() == '0') {
            ++p_;
        } else if (std::isdigit(static_cast<unsigned char>(Peek())) != 0) {
            while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) ++p_;
        } else {
            Err("invalid number");
        }
        if (Peek() == '.' || Peek() == 'e' || Peek() == 'E') Err("only integer supported");
        try {
            v.i = std::stoll(t_.substr(st, p_ - st));
        } catch (const std::exception&) {
            Err("invalid integer");
        }
        return v;
    }

    // 解析 true 字面量。
    J True() {
        Lit("true");
        J v;
        v.k = J::B;
        v.b = true;
        return v;
    }
    // 解析 false 字面量。
    J False() {
        Lit("false");
        J v;
        v.k = J::B;
        v.b = false;
        return v;
    }
    // 解析 null 字面量。
    J Null() {
        Lit("null");
        J v;
        v.k = J::N;
        return v;
    }
};

// 转义写入计划 JSON 的字符串字段。
std::string Esc(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: {
                unsigned char uc = static_cast<unsigned char>(c);
                if (uc < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(uc));
                    out += buf;
                } else {
                    out.push_back(c);
                }
            }
        }
    }
    return out;
}

// 校验 JSON 值类型并返回原值引用。
const J& RK(const J& v, J::K k, const std::string& ctx) {
    if (v.k != k) {
        std::stringstream ss;
        ss << "Expected kind " << static_cast<int>(k) << " in " << ctx << ", got "
           << static_cast<int>(v.k);
        throw std::runtime_error(ss.str());
    }
    return v;
}

// 读取必需对象字段，缺失时附带上下文失败。
const J& RF(const J& obj, const std::string& key, const std::string& ctx) {
    RK(obj, J::O, ctx);
    auto it = obj.o.find(key);
    if (it == obj.o.end()) {
        throw std::runtime_error("Missing required field '" + key + "' in " + ctx);
    }
    return it->second;
}

// 查找可选对象字段，缺失时返回空指针。
const J* FF(const J& obj, const std::string& key) {
    if (obj.k != J::O) return nullptr;
    auto it = obj.o.find(key);
    return it == obj.o.end() ? nullptr : &it->second;
}

// 读取 int 字段并检查从 int64_t 缩窄的范围。
int RI(const J& v, const std::string& ctx) {
    RK(v, J::I, ctx);
    if (v.i < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        v.i > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int out of range in " + ctx);
    }
    return static_cast<int>(v.i);
}
// 读取 int64_t 字段。
int64_t RI64(const J& v, const std::string& ctx) {
    RK(v, J::I, ctx);
    return v.i;
}
// 读取布尔字段。
bool RB(const J& v, const std::string& ctx) {
    RK(v, J::B, ctx);
    return v.b;
}
// 读取字符串字段。
std::string RS(const J& v, const std::string& ctx) {
    RK(v, J::S, ctx);
    return v.s;
}

// 读取带默认值的可选 int 字段。
int OI(const J& obj, const std::string& key, int d, const std::string& ctx) {
    const J* v = FF(obj, key);
    if (!v || v->k == J::N) return d;
    return RI(*v, ctx + "." + key);
}
// 读取带默认值的可选布尔字段。
bool OB(const J& obj, const std::string& key, bool d, const std::string& ctx) {
    const J* v = FF(obj, key);
    if (!v || v->k == J::N) return d;
    return RB(*v, ctx + "." + key);
}
// 读取带默认值的可选字符串字段。
std::string OS(const J& obj, const std::string& key, const std::string& d, const std::string& ctx) {
    const J* v = FF(obj, key);
    if (!v || v->k == J::N) return d;
    return RS(*v, ctx + "." + key);
}

// 将 JSON 整数数组转换为对象系统 Array<int>。
Array<int> AI(const J& arr, const std::string& ctx) {
    RK(arr, J::A, ctx);
    Array<int> out;
    for (size_t i = 0; i < arr.a.size(); ++i) {
        out.push_back(RI(arr.a[i], ctx + "[" + std::to_string(i) + "]"));
    }
    return out;
}
// 将 JSON 整数数组转换为对象系统 Array<int64_t>。
Array<int64_t> AI64(const J& arr, const std::string& ctx) {
    RK(arr, J::A, ctx);
    Array<int64_t> out;
    for (size_t i = 0; i < arr.a.size(); ++i) {
        out.push_back(RI64(arr.a[i], ctx + "[" + std::to_string(i) + "]"));
    }
    return out;
}

// 按稳定顺序写出 Array<int>。
void WI(std::ostream& os, const Array<int>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) os << ",";
        os << arr[i];
    }
    os << "]";
}
// 按稳定顺序写出 Array<int64_t>。
void WI64(std::ostream& os, const Array<int64_t>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) os << ",";
        os << arr[i];
    }
    os << "]";
}

// 从 JSON 设备对象恢复规范驻留的 Device。
Device ParseDeviceObj(const J& j, const std::string& ctx) {
    if (j.k == J::N) return Device();
    RK(j, J::O, ctx);
    int t = RI(RF(j, "device_type", ctx), ctx + ".device_type");
    int id = RI(RF(j, "device_id", ctx), ctx + ".device_id");
    // 反序列化统一回到 DeviceManager 的驻留对象，保持同一物理设备的对象身份稳定。
    return DeviceManager::Global()->Get(static_cast<DeviceTypeCode>(t), id);
}

// 从 JSON 恢复 Target 能力快照。
Target ParseTargetObj(const J& j, const std::string& ctx) {
    if (j.k == J::N) return Target();
    RK(j, J::O, ctx);
    auto* n = new TargetNode();
    n->kind = OS(j, "kind", "", ctx);
    n->device_type = static_cast<DeviceTypeCode>(OI(j, "device_type", static_cast<int>(kUnknown), ctx));
    n->device_id = OI(j, "device_id", -1, ctx);
    if (const J* a = FF(j, "attrs")) {
        if (a->k != J::N) {
            RK(*a, J::O, ctx + ".attrs");
            n->attrs.arch = OS(*a, "arch", "", ctx + ".attrs");
            n->attrs.compute_version = OS(*a, "compute_version", "0.0", ctx + ".attrs");
            n->attrs.max_threads_per_block = OI(*a, "max_threads_per_block", 1, ctx + ".attrs");
            n->attrs.warp_size = OI(*a, "warp_size", 1, ctx + ".attrs");
            n->attrs.max_shared_memory_per_block =
                OI(*a, "max_shared_memory_per_block", 0, ctx + ".attrs");
        }
    }
    return Target(ObjectRef(n));
}

// 从 JSON 恢复逻辑设备的物理、目标、内存域和逻辑编号约束。
VirtualDevice ParseVD(const J& j, const std::string& ctx) {
    if (j.k == J::N) return VirtualDevice();
    RK(j, J::O, ctx);
    std::string scope = OS(j, "memory_scope", "", ctx);
    int vid = OI(j, "virtual_device_id", kInvalidVirtualDeviceId, ctx);
    Device dev;
    if (const J* d = FF(j, "device")) dev = ParseDeviceObj(*d, ctx + ".device");
    Target target;
    if (const J* t = FF(j, "target")) target = ParseTargetObj(*t, ctx + ".target");
    if (dev.defined()) return VirtualDevice(dev, target, scope, vid);
    return VirtualDevice(target, scope, vid);
}

// 将物理设备身份写为 JSON 对象。
void WriteDeviceObj(std::ostream& os, const Device& dev) {
    if (!dev.defined()) {
        os << "null";
        return;
    }
    // 执行计划只持久化设备值；对象身份在读取时由 DeviceManager 重建。
    os << "{\"device_type\":" << static_cast<int>(dev.device_type()) << ",\"device_id\":"
       << dev.device_id() << "}";
}

// 将 Target 能力快照写为 JSON 对象。
void WriteTargetObj(std::ostream& os, const Target& t) {
    if (!t.defined()) {
        os << "null";
        return;
    }
    os << "{";
    os << "\"kind\":\"" << Esc(t->kind) << "\",";
    os << "\"device_type\":" << static_cast<int>(t->device_type) << ",";
    os << "\"device_id\":" << t->device_id << ",";
    os << "\"attrs\":{";
    os << "\"arch\":\"" << Esc(t->attrs.arch) << "\",";
    os << "\"compute_version\":\"" << Esc(t->attrs.compute_version) << "\",";
    os << "\"max_threads_per_block\":" << t->attrs.max_threads_per_block << ",";
    os << "\"warp_size\":" << t->attrs.warp_size << ",";
    os << "\"max_shared_memory_per_block\":" << t->attrs.max_shared_memory_per_block;
    os << "}}";
}

// 将可选 VirtualDevice 约束写为 JSON。
void WriteVD(std::ostream& os, const VirtualDevice& vd) {
    if (!vd.defined()) {
        os << "null";
        return;
    }
    os << "{";
    os << "\"virtual_device_id\":" << vd->virtual_device_id << ",";
    os << "\"memory_scope\":\"" << Esc(vd->memory_scope) << "\",";
    os << "\"device\":";
    WriteDeviceObj(os, vd->device);
    os << ",";
    os << "\"target\":";
    WriteTargetObj(os, vd->target);
    os << "}";
}

// 从 JSON 恢复 worker 放置表，并重建 VirtualDevice 映射。
DiscoPlacement ParsePlacement(const J& j, const std::string& ctx) {
    if (j.k == J::N) return DiscoPlacement();
    RK(j, J::O, ctx);
    int num_groups = OI(j, "num_groups", 1, ctx);
    const J& workers = RF(j, "workers", ctx);
    RK(workers, J::A, ctx + ".workers");
    Array<WorkerPlacement> ws;
    Map<VirtualDevice, int> map;
    for (size_t i = 0; i < workers.a.size(); ++i) {
        const J& item = workers.a[i];
        std::string wctx = ctx + ".workers[" + std::to_string(i) + "]";
        RK(item, J::O, wctx);
        int worker_id = RI(RF(item, "worker_id", wctx), wctx + ".worker_id");
        int group_id = OI(item, "group_id", 0, wctx);
        int local_rank = OI(item, "local_rank", worker_id, wctx);
        Device dev;
        if (const J* d = FF(item, "device")) dev = ParseDeviceObj(*d, wctx + ".device");
        Target target;
        if (const J* t = FF(item, "target")) target = ParseTargetObj(*t, wctx + ".target");
        VirtualDevice vd;
        if (const J* v = FF(item, "virtual_device")) {
            vd = ParseVD(*v, wctx + ".virtual_device");
        } else if (dev.defined()) {
            vd = VirtualDevice(dev, target);
        } else if (target.defined()) {
            vd = VirtualDevice(target);
        }
        ws.push_back(WorkerPlacement(worker_id, group_id, local_rank, dev, target, vd));
        if (vd.defined()) map.Set(vd, worker_id);
    }
    return DiscoPlacement(ws, map, num_groups);
}

// 将 PassContext 中可选的 DiscoPlacement 写入计划 JSON。
void WritePlacement(std::ostream& os, const DiscoPlacement& p) {
    if (!p.defined()) {
        os << "null";
        return;
    }
    os << "{";
    os << "\"num_groups\":" << p->num_groups << ",";
    os << "\"workers\":[";
    std::vector<WorkerPlacement> ws;
    ws.reserve(p->workers.size());
    for (const auto& w : p->workers) ws.push_back(w);
    std::sort(ws.begin(), ws.end(),
              [](const WorkerPlacement& a, const WorkerPlacement& b) {
                  if (!a.defined()) return false;
                  if (!b.defined()) return true;
                  return a->worker_id < b->worker_id;
              });
    for (size_t i = 0; i < ws.size(); ++i) {
        if (i) os << ",";
        const WorkerPlacement& w = ws[i];
        if (!w.defined()) {
            os << "null";
            continue;
        }
        os << "{";
        os << "\"worker_id\":" << w->worker_id << ",";
        os << "\"group_id\":" << w->group_id << ",";
        os << "\"local_rank\":" << w->local_rank << ",";
        os << "\"device\":";
        WriteDeviceObj(os, w->device);
        os << ",";
        os << "\"target\":";
        WriteTargetObj(os, w->target);
        os << ",";
        os << "\"virtual_device\":";
        WriteVD(os, w->virtual_device);
        os << "}";
    }
    os << "]}";
}

// 按已知通信 attrs 类型序列化结构化参数。
void WriteCommAttrs(std::ostream& os, const std::string& op_name,
                    const CommExecAttrs& attrs) {
    if (op_name == "device.copy") {
        os << "{";
        os << "\"src_virtual_device\":";
        WriteVD(os, attrs.src_virtual_device);
        os << ",";
        os << "\"dst_virtual_device\":";
        WriteVD(os, attrs.dst_virtual_device);
        os << ",";
        os << "\"async\":" << (attrs.async ? "true" : "false") << ",";
        os << "\"in_group\":" << (attrs.in_group ? "true" : "false");
        os << "}";
        return;
    }
    os << "{";
    os << "\"kind\":\"" << Esc(attrs.kind) << "\",";
    os << "\"reduce_kind\":\"" << Esc(attrs.reduce_kind) << "\",";
    os << "\"in_group\":" << (attrs.in_group ? "true" : "false") << ",";
    os << "\"group_id\":" << attrs.group_id << ",";
    os << "\"root_worker\":" << attrs.root_worker;
    os << "}";
}

// 根据通信算子名恢复对应 attrs 对象，未知字段不转为裸指针或弱类型数据。
CommExecAttrs ParseCommAttrs(const std::string& op_name, const J* attrs,
                             const std::string& ctx) {
    CommExecAttrs result;
    result.kind = op_name;
    if (op_name == "device.copy") {
        if (!attrs || attrs->k == J::N) {
            return result;
        }
        RK(*attrs, J::O, ctx);
        if (const J* src = FF(*attrs, "src_virtual_device")) {
            result.src_virtual_device =
                ParseVD(*src, ctx + ".src_virtual_device");
        }
        if (const J* dst = FF(*attrs, "dst_virtual_device")) {
            result.dst_virtual_device =
                ParseVD(*dst, ctx + ".dst_virtual_device");
        }
        result.async = OB(*attrs, "async", false, ctx);
        result.in_group = OB(*attrs, "in_group", true, ctx);
        return result;
    }
    if (!attrs || attrs->k == J::N) {
        return result;
    }
    RK(*attrs, J::O, ctx);
    result.kind = OS(*attrs, "kind", op_name, ctx);
    result.reduce_kind = OS(*attrs, "reduce_kind", "sum", ctx);
    result.in_group = OB(*attrs, "in_group", true, ctx);
    result.group_id = OI(*attrs, "group_id", 0, ctx);
    result.root_worker = OI(*attrs, "root_worker", 0, ctx);
    return result;
}

// 对 Map 键排序，保证 JSON 输出可复现。
std::vector<int> SortedIds(const Map<int, VirtualDevice>& m) {
    std::vector<int> ids;
    ids.reserve(m.size());
    for (const auto& kv : m) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// 合并 shape/dtype 的 value id 并排序，保证元数据输出稳定。
std::vector<int> SortedInfoIds(const Map<int, Array<int64_t>>& shapes,
                               const Map<int, std::string>& dtypes) {
    std::unordered_set<int> set;
    for (const auto& kv : shapes) set.insert(kv.first);
    for (const auto& kv : dtypes) set.insert(kv.first);
    std::vector<int> ids(set.begin(), set.end());
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

// 按 device. 命名空间识别执行计划通信算子。
bool IsCommunicationOpName(const std::string& op_name) {
    return op_name.rfind("device.", 0) == 0;
}

// 将完整执行计划序列化为可复现、可跨进程传输的 JSON。
std::string SerializeExecutionPlanToJson(const ExecutionPlan& plan) {
    if (!plan.defined()) {
        throw std::runtime_error("SerializeExecutionPlanToJson requires a defined ExecutionPlan");
    }

    std::stringstream os;
    os << "{";
    os << "\"schema_version\":1,";
    os << "\"num_values\":" << plan->num_values << ",";
    os << "\"output_value\":" << plan->output_value << ",";

    os << "\"nodes\":[";
    for (size_t i = 0; i < plan->nodes.size(); ++i) {
        if (i) os << ",";
        const ObjectRef& ref = plan->nodes[i];
        if (!ref.defined()) {
            os << "null";
            continue;
        }
        const Object* raw = ref.get();
        os << "{";
        if (raw->GetTypeId() == KernelExecNode::_type_index) {
            const auto* n = static_cast<const KernelExecNode*>(raw);
            os << "\"kind\":\"kernel\",";
            os << "\"input_values\":";
            WI(os, n->input_values);
            os << ",";
            os << "\"output_values\":";
            WI(os, n->output_values);
            os << ",";
            os << "\"worker_set\":";
            WI(os, n->worker_set);
            os << ",";
            os << "\"op_name\":\"" << Esc(n->op_name) << "\",";
            os << "\"kernel_symbol\":\"" << Esc(n->kernel_symbol) << "\"";
        } else if (raw->GetTypeId() == CommExecNode::_type_index) {
            const auto* n = static_cast<const CommExecNode*>(raw);
            os << "\"kind\":\"comm\",";
            os << "\"input_values\":";
            WI(os, n->input_values);
            os << ",";
            os << "\"output_values\":";
            WI(os, n->output_values);
            os << ",";
            os << "\"worker_set\":";
            WI(os, n->worker_set);
            os << ",";
            os << "\"op_name\":\"" << Esc(n->op_name) << "\",";
            os << "\"attrs\":";
            WriteCommAttrs(os, n->op_name, n->attrs);
        } else if (raw->GetTypeId() == BarrierExecNode::_type_index) {
            const auto* n = static_cast<const BarrierExecNode*>(raw);
            os << "\"kind\":\"barrier\",";
            os << "\"input_values\":[],\"output_values\":[],";
            os << "\"worker_set\":";
            WI(os, n->worker_set);
            os << ",";
            os << "\"tag\":\"" << Esc(n->tag) << "\"";
        } else {
            throw std::runtime_error("SerializeExecutionPlanToJson encountered unknown node type");
        }
        os << "}";
    }
    os << "],";

    os << "\"value_virtual_devices\":[";
    std::vector<int> vd_ids = SortedIds(plan->value_virtual_devices);
    for (size_t i = 0; i < vd_ids.size(); ++i) {
        if (i) os << ",";
        int id = vd_ids[i];
        os << "{\"value_id\":" << id << ",\"virtual_device\":";
        WriteVD(os, plan->value_virtual_devices.at(id));
        os << "}";
    }
    os << "],";

    os << "\"input_value_ids\":";
    WI(os, plan->input_value_ids);
    os << ",";
    os << "\"constant_value_ids\":";
    WI(os, plan->constant_value_ids);
    os << ",";

    os << "\"value_info\":[";
    std::vector<int> info_ids = SortedInfoIds(plan->value_shapes, plan->value_dtypes);
    for (size_t i = 0; i < info_ids.size(); ++i) {
        if (i) os << ",";
        int id = info_ids[i];
        os << "{\"value_id\":" << id << ",\"shape\":";
        if (plan->value_shapes.count(id)) WI64(os, plan->value_shapes.at(id));
        else os << "[]";
        os << ",\"dtype\":\"";
        std::string dt = plan->value_dtypes.count(id) ? plan->value_dtypes.at(id) : "";
        os << Esc(dt) << "\"}";
    }
    os << "],";

    os << "\"disco_placement\":";
    WritePlacement(os, plan->placement);
    os << ",";
    os << "\"reserved\":{}";
    os << "}";
    return os.str();
}

// 严格校验 JSON 契约并重建强类型执行计划对象。
ExecutionPlan DeserializeExecutionPlanFromJson(const std::string& json_text) {
    J root = P(json_text).Parse();
    RK(root, J::O, "root");
    int schema_version = RI(RF(root, "schema_version", "root"), "root.schema_version");
    if (schema_version != 1) {
        throw std::runtime_error("Unsupported ExecutionPlan JSON schema_version: " +
                                 std::to_string(schema_version));
    }

    int num_values = RI(RF(root, "num_values", "root"), "root.num_values");
    int output_value = RI(RF(root, "output_value", "root"), "root.output_value");

    const J& nodes_j = RF(root, "nodes", "root");
    RK(nodes_j, J::A, "root.nodes");
    Array<ObjectRef> nodes;
    for (size_t i = 0; i < nodes_j.a.size(); ++i) {
        const J& n = nodes_j.a[i];
        if (n.k == J::N) continue;
        std::string ctx = "root.nodes[" + std::to_string(i) + "]";
        RK(n, J::O, ctx);
        std::string kind = RS(RF(n, "kind", ctx), ctx + ".kind");
        Array<int> in = AI(RF(n, "input_values", ctx), ctx + ".input_values");
        Array<int> out = AI(RF(n, "output_values", ctx), ctx + ".output_values");
        Array<int> workers = AI(RF(n, "worker_set", ctx), ctx + ".worker_set");
        if (kind == "kernel") {
            std::string op_name = RS(RF(n, "op_name", ctx), ctx + ".op_name");
            std::string kernel_symbol = OS(n, "kernel_symbol", "", ctx);
            nodes.push_back(ObjectRef(KernelExec(op_name, tir::PrimFunc(), in, out, workers,
                                                 kernel_symbol)));
        } else if (kind == "comm") {
            std::string op_name = RS(RF(n, "op_name", ctx), ctx + ".op_name");
            const J* attrs = FF(n, "attrs");
            nodes.push_back(ObjectRef(CommExec(op_name, ParseCommAttrs(op_name, attrs, ctx + ".attrs"),
                                               in, out, workers)));
        } else if (kind == "barrier") {
            std::string tag = RS(RF(n, "tag", ctx), ctx + ".tag");
            nodes.push_back(ObjectRef(BarrierExec(tag, workers)));
        } else {
            throw std::runtime_error("Unsupported node kind in ExecutionPlan JSON: " + kind);
        }
    }

    const J& value_vd_j = RF(root, "value_virtual_devices", "root");
    RK(value_vd_j, J::A, "root.value_virtual_devices");
    Map<int, VirtualDevice> value_virtual_devices;
    for (size_t i = 0; i < value_vd_j.a.size(); ++i) {
        const J& item = value_vd_j.a[i];
        std::string ctx = "root.value_virtual_devices[" + std::to_string(i) + "]";
        RK(item, J::O, ctx);
        int id = RI(RF(item, "value_id", ctx), ctx + ".value_id");
        VirtualDevice vd = ParseVD(RF(item, "virtual_device", ctx), ctx + ".virtual_device");
        if (vd.defined()) value_virtual_devices.Set(id, vd);
    }

    Array<int> input_value_ids = AI(RF(root, "input_value_ids", "root"), "root.input_value_ids");
    Array<int> constant_value_ids =
        AI(RF(root, "constant_value_ids", "root"), "root.constant_value_ids");

    const J& info_j = RF(root, "value_info", "root");
    RK(info_j, J::A, "root.value_info");
    Map<int, Array<int64_t>> value_shapes;
    Map<int, std::string> value_dtypes;
    for (size_t i = 0; i < info_j.a.size(); ++i) {
        const J& item = info_j.a[i];
        std::string ctx = "root.value_info[" + std::to_string(i) + "]";
        RK(item, J::O, ctx);
        int id = RI(RF(item, "value_id", ctx), ctx + ".value_id");
        value_shapes.Set(id, AI64(RF(item, "shape", ctx), ctx + ".shape"));
        value_dtypes.Set(id, RS(RF(item, "dtype", ctx), ctx + ".dtype"));
    }

    DiscoPlacement placement =
        ParsePlacement(RF(root, "disco_placement", "root"), "root.disco_placement");
    Array<VirtualDevice> placement_virtual_devices;
    if (placement.defined()) {
        for (const auto& worker : placement->workers) {
            if (worker.defined() && worker->virtual_device.defined()) {
                placement_virtual_devices.push_back(worker->virtual_device);
            }
        }
    }
    PassContext pass_ctx =
        PassContext::FromVirtualDevices(placement_virtual_devices);

    RK(RF(root, "reserved", "root"), J::O, "root.reserved");

    return ExecutionPlan(nodes, value_virtual_devices, input_value_ids, constant_value_ids,
                          value_shapes, value_dtypes, num_values, pass_ctx,
                          placement, output_value);
}

// 从文件读取并反序列化执行计划。
ExecutionPlan LoadExecutionPlanFromJsonFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::in);
    if (!ifs) {
        throw std::runtime_error("Failed to open ExecutionPlan JSON file for reading: " + path);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return DeserializeExecutionPlanFromJson(ss.str());
}

// 将执行计划 JSON 完整写入文件并检查 I/O 失败。
void SaveExecutionPlanToJsonFile(const ExecutionPlan& plan, const std::string& path) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open ExecutionPlan JSON file for writing: " + path);
    }
    ofs << SerializeExecutionPlanToJson(plan);
}

}  // namespace kxc
