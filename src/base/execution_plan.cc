/*! \file src/base/execution_plan.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/execution_plan.h"

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

#include "base/device.h"

namespace kxc {

namespace {

struct J {
    enum K { N, B, I, S, A, O } k{N};
    bool b{false};
    int64_t i{0};
    std::string s;
    std::vector<J> a;
    std::unordered_map<std::string, J> o;
};

class P {
public:
    explicit P(const std::string& text) : t_(text) {}

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

    [[noreturn]] void Err(const std::string& m) const {
        std::stringstream ss;
        ss << "JSON parse error at " << p_ << ": " << m;
        throw std::runtime_error(ss.str());
    }

    char Peek() const { return p_ < t_.size() ? t_[p_] : '\0'; }
    char Get() {
        if (p_ >= t_.size()) Err("unexpected EOF");
        return t_[p_++];
    }
    void WS() {
        while (p_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[p_])) != 0) ++p_;
    }
    void Expect(char c) {
        if (Get() != c) {
            std::stringstream ss;
            ss << "expected '" << c << "'";
            Err(ss.str());
        }
    }
    void Lit(const char* s) {
        while (*s) {
            if (Get() != *s) Err("invalid literal");
            ++s;
        }
    }

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

    J True() {
        Lit("true");
        J v;
        v.k = J::B;
        v.b = true;
        return v;
    }
    J False() {
        Lit("false");
        J v;
        v.k = J::B;
        v.b = false;
        return v;
    }
    J Null() {
        Lit("null");
        J v;
        v.k = J::N;
        return v;
    }
};

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

const J& RK(const J& v, J::K k, const std::string& ctx) {
    if (v.k != k) {
        std::stringstream ss;
        ss << "Expected kind " << static_cast<int>(k) << " in " << ctx << ", got "
           << static_cast<int>(v.k);
        throw std::runtime_error(ss.str());
    }
    return v;
}

const J& RF(const J& obj, const std::string& key, const std::string& ctx) {
    RK(obj, J::O, ctx);
    auto it = obj.o.find(key);
    if (it == obj.o.end()) {
        throw std::runtime_error("Missing required field '" + key + "' in " + ctx);
    }
    return it->second;
}

const J* FF(const J& obj, const std::string& key) {
    if (obj.k != J::O) return nullptr;
    auto it = obj.o.find(key);
    return it == obj.o.end() ? nullptr : &it->second;
}

int RI(const J& v, const std::string& ctx) {
    RK(v, J::I, ctx);
    if (v.i < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        v.i > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int out of range in " + ctx);
    }
    return static_cast<int>(v.i);
}
int64_t RI64(const J& v, const std::string& ctx) {
    RK(v, J::I, ctx);
    return v.i;
}
bool RB(const J& v, const std::string& ctx) {
    RK(v, J::B, ctx);
    return v.b;
}
std::string RS(const J& v, const std::string& ctx) {
    RK(v, J::S, ctx);
    return v.s;
}

int OI(const J& obj, const std::string& key, int d, const std::string& ctx) {
    const J* v = FF(obj, key);
    if (!v || v->k == J::N) return d;
    return RI(*v, ctx + "." + key);
}
bool OB(const J& obj, const std::string& key, bool d, const std::string& ctx) {
    const J* v = FF(obj, key);
    if (!v || v->k == J::N) return d;
    return RB(*v, ctx + "." + key);
}
std::string OS(const J& obj, const std::string& key, const std::string& d, const std::string& ctx) {
    const J* v = FF(obj, key);
    if (!v || v->k == J::N) return d;
    return RS(*v, ctx + "." + key);
}

Array<int> AI(const J& arr, const std::string& ctx) {
    RK(arr, J::A, ctx);
    Array<int> out;
    for (size_t i = 0; i < arr.a.size(); ++i) {
        out.push_back(RI(arr.a[i], ctx + "[" + std::to_string(i) + "]"));
    }
    return out;
}
Array<int64_t> AI64(const J& arr, const std::string& ctx) {
    RK(arr, J::A, ctx);
    Array<int64_t> out;
    for (size_t i = 0; i < arr.a.size(); ++i) {
        out.push_back(RI64(arr.a[i], ctx + "[" + std::to_string(i) + "]"));
    }
    return out;
}

void WI(std::ostream& os, const Array<int>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) os << ",";
        os << arr[i];
    }
    os << "]";
}
void WI64(std::ostream& os, const Array<int64_t>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) os << ",";
        os << arr[i];
    }
    os << "]";
}

ObjectRef ParseDeviceObj(const J& j, const std::string& ctx) {
    if (j.k == J::N) return ObjectRef();
    RK(j, J::O, ctx);
    int t = RI(RF(j, "device_type", ctx), ctx + ".device_type");
    int id = RI(RF(j, "device_id", ctx), ctx + ".device_id");
    return DeviceManager::Global()->GetOrCreate(static_cast<DeviceTypeCode>(t), id);
}

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

VirtualDevice ParseVD(const J& j, const std::string& ctx) {
    if (j.k == J::N) return VirtualDevice();
    RK(j, J::O, ctx);
    std::string scope = OS(j, "memory_scope", "", ctx);
    int vid = OI(j, "virtual_device_id", kInvalidVirtualDeviceId, ctx);
    ObjectRef dev;
    if (const J* d = FF(j, "device")) dev = ParseDeviceObj(*d, ctx + ".device");
    Target target;
    if (const J* t = FF(j, "target")) target = ParseTargetObj(*t, ctx + ".target");
    if (dev.defined()) {
        const Object* raw = dev.get();
        if (!raw || raw->GetTypeId() != kKXC_DEVICE_TYPE) {
            throw std::runtime_error("Expected Device in " + ctx);
        }
        const auto* d = static_cast<const class Device*>(raw);
        return VirtualDevice(*d, target, scope, vid);
    }
    return VirtualDevice(target, scope, vid);
}

void WriteDeviceObj(std::ostream& os, const ObjectRef& dev) {
    if (!dev.defined()) {
        os << "null";
        return;
    }
    const Object* raw = dev.get();
    if (!raw || raw->GetTypeId() != kKXC_DEVICE_TYPE) {
        throw std::runtime_error("Expected Device object during ExecutionPlan serialization");
    }
    const auto* d = static_cast<const class Device*>(raw);
    os << "{\"device_type\":" << static_cast<int>(d->device_type()) << ",\"device_id\":"
       << d->device_id() << "}";
}

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

void WriteVD(std::ostream& os, const VirtualDevice& vd) {
    if (!vd.defined()) {
        os << "null";
        return;
    }
    os << "{";
    os << "\"virtual_device_id\":" << vd->virtual_device_id << ",";
    os << "\"memory_scope\":\"" << Esc(vd->memory_scope) << "\",";
    os << "\"device\":";
    WriteDeviceObj(os, vd->device_obj);
    os << ",";
    os << "\"target\":";
    WriteTargetObj(os, vd->target);
    os << "}";
}

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
        ObjectRef dev;
        if (const J* d = FF(item, "device")) dev = ParseDeviceObj(*d, wctx + ".device");
        Target target;
        if (const J* t = FF(item, "target")) target = ParseTargetObj(*t, wctx + ".target");
        VirtualDevice vd;
        if (const J* v = FF(item, "virtual_device")) {
            vd = ParseVD(*v, wctx + ".virtual_device");
        } else if (dev.defined()) {
            const Object* raw = dev.get();
            if (!raw || raw->GetTypeId() != kKXC_DEVICE_TYPE) {
                throw std::runtime_error("Expected Device in " + wctx);
            }
            vd = VirtualDevice(*static_cast<const class Device*>(raw), target);
        } else if (target.defined()) {
            vd = VirtualDevice(target);
        }
        ws.push_back(WorkerPlacement(worker_id, group_id, local_rank, dev, target, vd));
        if (vd.defined()) map.Set(vd, worker_id);
    }
    return DiscoPlacement(ws, map, num_groups);
}

void WritePlacement(std::ostream& os, const PassContext& pass_ctx) {
    if (!pass_ctx.has_disco_placement()) {
        os << "null";
        return;
    }
    DiscoPlacement p = pass_ctx.disco_placement();
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
        WriteDeviceObj(os, w->device_obj);
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

void WriteCommAttrs(std::ostream& os, const ObjectRef& attrs) {
    if (!attrs.defined()) {
        os << "null";
        return;
    }
    if (const auto* a = attrs.As<relay::DeviceCopyAttrsNode>()) {
        os << "{";
        os << "\"src_virtual_device\":";
        WriteVD(os, a->src_virtual_device);
        os << ",";
        os << "\"dst_virtual_device\":";
        WriteVD(os, a->dst_virtual_device);
        os << ",";
        os << "\"async\":" << (a->async ? "true" : "false") << ",";
        os << "\"in_group\":" << (a->in_group ? "true" : "false");
        os << "}";
        return;
    }
    if (const auto* a = attrs.As<relay::CollectiveAttrsNode>()) {
        os << "{";
        os << "\"kind\":\"" << Esc(a->kind) << "\",";
        os << "\"reduce_kind\":\"" << Esc(a->reduce_kind) << "\",";
        os << "\"in_group\":" << (a->in_group ? "true" : "false") << ",";
        os << "\"group_id\":" << a->group_id << ",";
        os << "\"root_worker\":" << a->root_worker;
        os << "}";
        return;
    }
    os << "null";
}

ObjectRef ParseCommAttrs(const std::string& op_name, const J* attrs, const std::string& ctx) {
    if (op_name == "device.copy") {
        if (!attrs || attrs->k == J::N) {
            return ObjectRef(
                relay::DeviceCopyAttrs::Create(VirtualDevice(), VirtualDevice(), false, true));
        }
        RK(*attrs, J::O, ctx);
        VirtualDevice src_vd;
        VirtualDevice dst_vd;
        if (const J* src = FF(*attrs, "src_virtual_device")) {
            src_vd = ParseVD(*src, ctx + ".src_virtual_device");
        }
        if (const J* dst = FF(*attrs, "dst_virtual_device")) {
            dst_vd = ParseVD(*dst, ctx + ".dst_virtual_device");
        }
        bool async = OB(*attrs, "async", false, ctx);
        bool in_group = OB(*attrs, "in_group", true, ctx);
        return ObjectRef(relay::DeviceCopyAttrs::Create(src_vd, dst_vd, async, in_group));
    }
    if (!attrs || attrs->k == J::N) {
        return ObjectRef(relay::CollectiveAttrs::Create(op_name, "sum", true, 0, 0));
    }
    RK(*attrs, J::O, ctx);
    std::string kind = OS(*attrs, "kind", op_name, ctx);
    std::string reduce_kind = OS(*attrs, "reduce_kind", "sum", ctx);
    bool in_group = OB(*attrs, "in_group", true, ctx);
    int group_id = OI(*attrs, "group_id", 0, ctx);
    int root_worker = OI(*attrs, "root_worker", 0, ctx);
    return ObjectRef(relay::CollectiveAttrs::Create(kind, reduce_kind, in_group, group_id, root_worker));
}

std::vector<int> SortedIds(const Map<int, VirtualDevice>& m) {
    std::vector<int> ids;
    ids.reserve(m.size());
    for (const auto& kv : m) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    return ids;
}

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

KernelExec::KernelExec(std::string op_name, tir::PrimFunc primfunc, Array<int> input_values,
                       Array<int> output_values, Array<int> worker_set,
                       std::string kernel_symbol) {
    KernelExecNode* node = new KernelExecNode();
    node->kind = ExecNodeKind::kKernel;
    node->op_name = std::move(op_name);
    node->kernel_symbol = std::move(kernel_symbol);
    node->primfunc = std::move(primfunc);
    node->input_values = std::move(input_values);
    node->output_values = std::move(output_values);
    node->worker_set = std::move(worker_set);
    SetData(node);
}

CommExec::CommExec(std::string op_name, ObjectRef attrs, Array<int> input_values,
                   Array<int> output_values, Array<int> worker_set) {
    CommExecNode* node = new CommExecNode();
    node->kind = ExecNodeKind::kComm;
    node->op_name = std::move(op_name);
    node->attrs = std::move(attrs);
    node->input_values = std::move(input_values);
    node->output_values = std::move(output_values);
    node->worker_set = std::move(worker_set);
    SetData(node);
}

BarrierExec::BarrierExec(std::string tag, Array<int> worker_set) {
    BarrierExecNode* node = new BarrierExecNode();
    node->kind = ExecNodeKind::kBarrier;
    node->tag = std::move(tag);
    node->worker_set = std::move(worker_set);
    SetData(node);
}

ExecutionPlan::ExecutionPlan(Array<ObjectRef> nodes, Map<int, VirtualDevice> value_virtual_devices,
                             Array<int> input_value_ids, Array<int> constant_value_ids,
                             Map<int, Array<int64_t>> value_shapes,
                             Map<int, std::string> value_dtypes, int num_values,
                             PassContext pass_ctx, int output_value) {
    ExecutionPlanNode* node = new ExecutionPlanNode();
    node->nodes = std::move(nodes);
    node->value_virtual_devices = std::move(value_virtual_devices);
    node->input_value_ids = std::move(input_value_ids);
    node->constant_value_ids = std::move(constant_value_ids);
    node->value_shapes = std::move(value_shapes);
    node->value_dtypes = std::move(value_dtypes);
    node->num_values = num_values;
    node->pass_ctx = std::move(pass_ctx);
    node->output_value = output_value;
    SetData(node);
}

std::string ExecutionPlan::ToString() const {
    if (!defined()) {
        return "ExecutionPlan(undefined)";
    }
    std::stringstream ss;
    ss << "ExecutionPlan(nodes=" << operator->()->nodes.size()
       << ", values=" << operator->()->num_values
       << ", inputs=" << operator->()->input_value_ids.size()
       << ", constants=" << operator->()->constant_value_ids.size()
       << ", output=" << operator->()->output_value << ")";
    return ss.str();
}

bool IsCommunicationOpName(const std::string& op_name) {
    return op_name.rfind("device.", 0) == 0;
}

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
            WriteCommAttrs(os, n->attrs);
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
    WritePlacement(os, plan->pass_ctx);
    os << ",";
    os << "\"reserved\":{}";
    os << "}";
    return os.str();
}

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
    PassContext pass_ctx = PassContext::WithDiscoPlacement(PassContext(), placement);

    RK(RF(root, "reserved", "root"), J::O, "root.reserved");

    return ExecutionPlan(nodes, value_virtual_devices, input_value_ids, constant_value_ids,
                         value_shapes, value_dtypes, num_values, pass_ctx, output_value);
}

ExecutionPlan LoadExecutionPlanFromJsonFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::in);
    if (!ifs) {
        throw std::runtime_error("Failed to open ExecutionPlan JSON file for reading: " + path);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return DeserializeExecutionPlanFromJson(ss.str());
}

void SaveExecutionPlanToJsonFile(const ExecutionPlan& plan, const std::string& path) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open ExecutionPlan JSON file for writing: " + path);
    }
    ofs << SerializeExecutionPlanToJson(plan);
}

}  // namespace kxc
