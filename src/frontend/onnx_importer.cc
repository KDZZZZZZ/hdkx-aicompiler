/*! \file src/frontend/onnx_importer.cc
 * \brief 实现 ONNX 图、属性和 Storage-backed 常量张量的导入。
 */

#include "kxc/frontend/onnx_importer.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"

namespace kxc {
namespace frontend {
namespace {

// 保存导入规范所需的最小 JSON 值树，避免解析阶段依赖外部 JSON 对象生命周期。
struct Json {
    // 标识 JSON 节点当前承载的值类别。
    enum Kind { Null, Bool, Int, Double, String, Array, Object } kind{Null};
    bool b{false};
    int64_t i{0};
    double d{0.0};
    std::string s;
    std::vector<Json> a;
    std::unordered_map<std::string, Json> o;
};

// 对受控 ONNX 导入规范执行严格递归下降解析，并报告精确字节位置。
class JsonParser {
public:
    // 绑定待解析文本；文本生命周期由调用方覆盖解析过程。
    explicit JsonParser(const std::string& text) : text_(text) {}

    // 解析单个根值并拒绝尾随字符。
    Json Parse() {
        SkipWhitespace();
        Json value = ParseValue();
        SkipWhitespace();
        if (pos_ != text_.size()) {
            Fail("trailing characters");
        }
        return value;
    }

private:
    const std::string& text_;
    size_t pos_{0};

    // 以当前位置构造统一解析异常。
    [[noreturn]] void Fail(const std::string& message) const {
        std::ostringstream os;
        os << "JSON parse error at " << pos_ << ": " << message;
        throw std::runtime_error(os.str());
    }

    // 查看当前字符而不推进游标。
    char Peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

    // 读取当前字符并推进游标，EOF 时失败。
    char Get() {
        if (pos_ >= text_.size()) {
            Fail("unexpected EOF");
        }
        return text_[pos_++];
    }

    // 跳过 JSON 允许的空白字符。
    void SkipWhitespace() {
        while (pos_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
            ++pos_;
        }
    }

    // 消费一个必需的分隔字符。
    void Expect(char expected) {
        if (Get() != expected) {
            std::ostringstream os;
            os << "expected '" << expected << "'";
            Fail(os.str());
        }
    }

    // 消费 true、false 或 null 等固定字面量。
    void Literal(const char* text) {
        while (*text) {
            if (Get() != *text) {
                Fail("invalid literal");
            }
            ++text;
        }
    }

    // 根据首字符分派到具体 JSON 值解析器。
    Json ParseValue() {
        SkipWhitespace();
        char c = Peek();
        if (c == '{') return ParseObject();
        if (c == '[') return ParseArray();
        if (c == '"') return ParseString();
        if (c == 't') return ParseTrue();
        if (c == 'f') return ParseFalse();
        if (c == 'n') return ParseNull();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) != 0) return ParseNumber();
        Fail("unexpected token");
    }

    // 解析对象并拒绝重复字段。
    Json ParseObject() {
        Json out;
        out.kind = Json::Object;
        Expect('{');
        SkipWhitespace();
        if (Peek() == '}') {
            Get();
            return out;
        }
        while (true) {
            Json key = ParseString();
            SkipWhitespace();
            Expect(':');
            Json value = ParseValue();
            if (!out.o.emplace(std::move(key.s), std::move(value)).second) {
                Fail("duplicate key");
            }
            SkipWhitespace();
            char c = Get();
            if (c == '}') break;
            if (c != ',') Fail("expected ',' or '}'");
            SkipWhitespace();
        }
        return out;
    }

    // 解析保持原始顺序的 JSON 数组。
    Json ParseArray() {
        Json out;
        out.kind = Json::Array;
        Expect('[');
        SkipWhitespace();
        if (Peek() == ']') {
            Get();
            return out;
        }
        while (true) {
            out.a.push_back(ParseValue());
            SkipWhitespace();
            char c = Get();
            if (c == ']') break;
            if (c != ',') Fail("expected ',' or ']'");
            SkipWhitespace();
        }
        return out;
    }

    // 解析字符串转义；最小导入格式仅原样保留 ASCII Unicode 转义。
    Json ParseString() {
        Json out;
        out.kind = Json::String;
        Expect('"');
        while (true) {
            char c = Get();
            if (c == '"') break;
            if (c == '\\') {
                char e = Get();
                switch (e) {
                    case '"': out.s.push_back('"'); break;
                    case '\\': out.s.push_back('\\'); break;
                    case '/': out.s.push_back('/'); break;
                    case 'b': out.s.push_back('\b'); break;
                    case 'f': out.s.push_back('\f'); break;
                    case 'n': out.s.push_back('\n'); break;
                    case 'r': out.s.push_back('\r'); break;
                    case 't': out.s.push_back('\t'); break;
                    case 'u': {
                        int codepoint = 0;
                        for (int n = 0; n < 4; ++n) {
                            char h = Get();
                            codepoint <<= 4;
                            if (h >= '0' && h <= '9') codepoint += h - '0';
                            else if (h >= 'a' && h <= 'f') codepoint += h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') codepoint += h - 'A' + 10;
                            else Fail("invalid unicode escape");
                        }
                        out.s.push_back(codepoint <= 0x7F ? static_cast<char>(codepoint) : '?');
                        break;
                    }
                    default:
                        Fail("invalid escape");
                }
            } else {
                out.s.push_back(c);
            }
        }
        return out;
    }

    // 区分整数与浮点语法并执行范围检查。
    Json ParseNumber() {
        Json out;
        size_t start = pos_;
        if (Peek() == '-') ++pos_;
        if (Peek() == '0') {
            ++pos_;
        } else if (std::isdigit(static_cast<unsigned char>(Peek())) != 0) {
            while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) ++pos_;
        } else {
            Fail("invalid number");
        }

        bool is_double = false;
        if (Peek() == '.') {
            is_double = true;
            ++pos_;
            if (std::isdigit(static_cast<unsigned char>(Peek())) == 0) Fail("invalid fraction");
            while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) ++pos_;
        }
        if (Peek() == 'e' || Peek() == 'E') {
            is_double = true;
            ++pos_;
            if (Peek() == '+' || Peek() == '-') ++pos_;
            if (std::isdigit(static_cast<unsigned char>(Peek())) == 0) Fail("invalid exponent");
            while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) ++pos_;
        }

        std::string text = text_.substr(start, pos_ - start);
        try {
            if (is_double) {
                out.kind = Json::Double;
                out.d = std::stod(text);
            } else {
                out.kind = Json::Int;
                out.i = std::stoll(text);
            }
        } catch (const std::exception&) {
            Fail("invalid number");
        }
        return out;
    }

    // 解析布尔真值。
    Json ParseTrue() {
        Literal("true");
        Json out;
        out.kind = Json::Bool;
        out.b = true;
        return out;
    }

    // 解析布尔假值。
    Json ParseFalse() {
        Literal("false");
        Json out;
        out.kind = Json::Bool;
        out.b = false;
        return out;
    }

    // 解析空值。
    Json ParseNull() {
        Literal("null");
        return Json();
    }
};

// 校验 JSON 节点类别，并在错误中携带字段上下文。
const Json& RequireKind(const Json& value, Json::Kind kind, const std::string& ctx) {
    if (value.kind != kind) {
        std::ostringstream os;
        os << "Expected JSON kind " << static_cast<int>(kind) << " in " << ctx << ", got "
           << static_cast<int>(value.kind);
        throw std::runtime_error(os.str());
    }
    return value;
}

// 读取必需对象字段。
const Json& Field(const Json& object, const std::string& key, const std::string& ctx) {
    RequireKind(object, Json::Object, ctx);
    auto it = object.o.find(key);
    if (it == object.o.end()) {
        throw std::runtime_error("Missing required field '" + key + "' in " + ctx);
    }
    return it->second;
}

// 查询可选对象字段，缺失时返回空指针。
const Json* OptionalField(const Json& object, const std::string& key) {
    if (object.kind != Json::Object) return nullptr;
    auto it = object.o.find(key);
    return it == object.o.end() ? nullptr : &it->second;
}

// 将 JSON 字符串读取为标准字符串。
std::string ReadString(const Json& value, const std::string& ctx) {
    return RequireKind(value, Json::String, ctx).s;
}

// 读取有符号 64 位整数。
int64_t ReadInt64(const Json& value, const std::string& ctx) {
    RequireKind(value, Json::Int, ctx);
    return value.i;
}

// 读取并校验平台 int 范围。
int ReadInt(const Json& value, const std::string& ctx) {
    int64_t value64 = ReadInt64(value, ctx);
    if (value64 < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        value64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int out of range in " + ctx);
    }
    return static_cast<int>(value64);
}

// 接受整数或浮点 JSON 数字并转换为 float。
float ReadFloat(const Json& value, const std::string& ctx) {
    if (value.kind == Json::Int) return static_cast<float>(value.i);
    if (value.kind == Json::Double) return static_cast<float>(value.d);
    throw std::runtime_error("Expected numeric value in " + ctx);
}

// 读取布尔字段。
bool ReadBool(const Json& value, const std::string& ctx) {
    return RequireKind(value, Json::Bool, ctx).b;
}

// 读取有序整数数组，用于算子维度属性。
std::vector<int64_t> ReadInt64Vector(const Json& value, const std::string& ctx) {
    RequireKind(value, Json::Array, ctx);
    std::vector<int64_t> out;
    out.reserve(value.a.size());
    for (size_t i = 0; i < value.a.size(); ++i) {
        out.push_back(ReadInt64(value.a[i], ctx + "[" + std::to_string(i) + "]"));
    }
    return out;
}

// 读取 JSON 静态 shape；零维合法，负数和非整数维度一律拒绝。
std::vector<int64_t> ReadStaticShape(const Json& value, const std::string& ctx) {
    std::vector<int64_t> shape = ReadInt64Vector(value, ctx);
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] < 0) {
            throw std::runtime_error(
                "Expected non-negative static dimension in " + ctx + "[" +
                std::to_string(axis) + "]");
        }
    }
    return shape;
}

// 读取有序字符串数组，用于值名称列表。
std::vector<std::string> ReadStringVector(const Json& value, const std::string& ctx) {
    RequireKind(value, Json::Array, ctx);
    std::vector<std::string> out;
    out.reserve(value.a.size());
    for (size_t i = 0; i < value.a.size(); ++i) {
        out.push_back(ReadString(value.a[i], ctx + "[" + std::to_string(i) + "]"));
    }
    return out;
}

// 将解析期 vector 转入对象系统 Array，供 Relay/NDArray 节点长期持有。
Array<int64_t> ToArray(const std::vector<int64_t>& values) {
    Array<int64_t> out;
    for (int64_t value : values) {
        out.push_back(value);
    }
    return out;
}

// 完整读取参数二进制文件并校验读取结果。
std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open ONNX params file for reading: " + path);
    }
    input.seekg(0, std::ios::end);
    std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to determine ONNX params file size: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!data.empty()) {
        input.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    if (!input) {
        throw std::runtime_error("Failed to read ONNX params file: " + path);
    }
    return data;
}

// 完整读取导入规范文本。
std::string ReadTextFile(const std::string& path) {
    std::ifstream input(path, std::ios::in);
    if (!input) {
        throw std::runtime_error("Failed to open ONNX import JSON for reading: " + path);
    }
    std::ostringstream os;
    os << input.rdbuf();
    return os.str();
}

// 按算子名称把 JSON 属性转换为对应的强类型 Relay Attrs 对象。
// node_name 仅用于诊断：手写非法 spec 的报错必须能定位到具体节点。
ObjectRef MakeAttrs(const std::string& op_name, const Json& attrs,
                    const std::string& node_name) {
    RequireKind(attrs, Json::Object, "attrs for " + op_name);
    if (op_name == "nn_conv2d") {
        return ObjectRef(relay::Conv2DAttrs::Create(
            ReadInt64Vector(Field(attrs, "strides", "conv attrs"), "conv attrs.strides"),
            ReadInt64Vector(Field(attrs, "pads", "conv attrs"), "conv attrs.pads"),
            ReadInt64Vector(Field(attrs, "dilations", "conv attrs"), "conv attrs.dilations"),
            ReadInt(Field(attrs, "group", "conv attrs"), "conv attrs.group"),
            ReadInt(Field(attrs, "channels", "conv attrs"), "conv attrs.channels"),
            ReadInt64Vector(Field(attrs, "kernel_size", "conv attrs"), "conv attrs.kernel_size"),
            ReadString(Field(attrs, "data_layout", "conv attrs"), "conv attrs.data_layout"),
            ReadString(Field(attrs, "kernel_layout", "conv attrs"), "conv attrs.kernel_layout"),
            ReadString(Field(attrs, "out_layout", "conv attrs"), "conv attrs.out_layout"),
            ReadString(Field(attrs, "out_dtype", "conv attrs"), "conv attrs.out_dtype")));
    }
    if (op_name == "nn_relu") {
        return ObjectRef(relay::ReluAttrs::Create());
    }
    if (op_name == "nn_max_pool2d") {
        return ObjectRef(relay::MaxPool2DAttrs::Create(
            ReadInt64Vector(Field(attrs, "strides", "pool attrs"), "pool attrs.strides"),
            ReadInt64Vector(Field(attrs, "pads", "pool attrs"), "pool attrs.pads"),
            ReadInt64Vector(Field(attrs, "dilations", "pool attrs"), "pool attrs.dilations"),
            ReadInt64Vector(Field(attrs, "pool_size", "pool attrs"), "pool attrs.pool_size"),
            ReadString(Field(attrs, "layout", "pool attrs"), "pool attrs.layout"),
            ReadBool(Field(attrs, "ceil_mode", "pool attrs"), "pool attrs.ceil_mode")));
    }
    if (op_name == "where") {
        if (!attrs.o.empty()) {
            throw std::runtime_error("Where import attrs must be empty");
        }
        return ObjectRef();
    }
    if (op_name == "equal") {
        // ONNX Equal(opset 17) 是 fieldless 算子：两输入一输出、无属性。
        if (!attrs.o.empty()) {
            throw std::runtime_error("Node '" + node_name +
                                     "' (equal) import attrs must be empty");
        }
        return ObjectRef();
    }
    if (op_name == "add" || op_name == "matmul") {
        return ObjectRef();
    }
    if (op_name == "mul" || op_name == "subtract" || op_name == "divide" ||
        op_name == "sqrt") {
        if (!attrs.o.empty()) {
            throw std::runtime_error("Node '" + node_name + "' (" + op_name +
                                     ") import attrs must be empty");
        }
        return ObjectRef();
    }
    if (op_name == "neg" || op_name == "sigmoid" || op_name == "pow") {
        // ONNX Neg/Sigmoid/Pow(opset 13+) 是 fieldless 算子：无属性。
        if (!attrs.o.empty()) {
            throw std::runtime_error("Node '" + node_name + "' (" + op_name +
                                     ") import attrs must be empty");
        }
        return ObjectRef();
    }
    if (op_name == "cast") {
        const std::string ctx = "cast attrs";
        if (attrs.o.size() != 1 || !OptionalField(attrs, "to")) {
            throw std::runtime_error("Cast import attrs must contain exactly 'to': " +
                                     node_name);
        }
        const int to = ReadInt(Field(attrs, "to", ctx), ctx + ".to");
        if (to < 0 || to > 6) {
            throw std::runtime_error("Cast import 'to' dtype code " + std::to_string(to) +
                                     " is outside the supported Relay codes 0..6: " +
                                     node_name);
        }
        return ObjectRef(relay::CastAttrs::Create(to));
    }
    if (op_name == "reduce_mean") {
        const std::string ctx = "reduce_mean attrs";
        if (attrs.o.size() != 2 || !OptionalField(attrs, "axes") ||
            !OptionalField(attrs, "keepdims")) {
            throw std::runtime_error(
                "ReduceMean import attrs must contain exactly axes and keepdims: " +
                node_name);
        }
        const int64_t keepdims =
            ReadInt64(Field(attrs, "keepdims", ctx), ctx + ".keepdims");
        if (keepdims != 0 && keepdims != 1) {
            throw std::runtime_error(
                "ReduceMean import keepdims must be 0 or 1: " + node_name);
        }
        return ObjectRef(relay::ReduceMeanAttrs::Create(
            ToArray(ReadInt64Vector(Field(attrs, "axes", ctx), ctx + ".axes")),
            keepdims));
    }
    if (op_name == "reshape") {
        const std::string ctx = "reshape attrs";
        if (attrs.o.size() != 2 || !OptionalField(attrs, "newshape") ||
            !OptionalField(attrs, "allowzero")) {
            throw std::runtime_error(
                "Reshape import attrs must contain exactly newshape and allowzero: " +
                node_name);
        }
        const std::vector<int64_t> newshape =
            ReadInt64Vector(Field(attrs, "newshape", ctx), ctx + ".newshape");
        for (size_t axis = 0; axis < newshape.size(); ++axis) {
            if (newshape[axis] < 0) {
                throw std::runtime_error(
                    "Reshape import newshape must be the fully resolved target "
                    "shape; negative dimensions are rejected: " +
                    node_name);
            }
        }
        const int64_t allowzero =
            ReadInt64(Field(attrs, "allowzero", ctx), ctx + ".allowzero");
        if (allowzero != 0) {
            throw std::runtime_error(
                "Reshape import requires allowzero=0 in the static S1 subset: " +
                node_name);
        }
        return ObjectRef(relay::ReshapeAttrs::Create(ToArray(newshape), 0));
    }
    if (op_name == "expand") {
        // Expand 的目标 shape 是导入期已解析的常量控制输入，canonical attrs
        // 携带完整的非负目标形状；手写 spec 不允许省略或注入负维度。
        const std::string ctx = "expand attrs";
        if (attrs.o.size() != 1 || !OptionalField(attrs, "target_shape")) {
            throw std::runtime_error(
                "Expand import attrs must contain exactly target_shape: " +
                node_name);
        }
        const std::vector<int64_t> target_shape =
            ReadInt64Vector(Field(attrs, "target_shape", ctx), ctx + ".target_shape");
        for (size_t axis = 0; axis < target_shape.size(); ++axis) {
            if (target_shape[axis] < 0) {
                throw std::runtime_error(
                    "Expand import target_shape must be non-negative static "
                    "dimensions: " + node_name);
            }
        }
        return ObjectRef(relay::ExpandAttrs::Create(ToArray(target_shape)));
    }
    if (op_name == "softmax") {
        return ObjectRef(relay::SoftmaxAttrs::Create(
            ReadInt(Field(attrs, "axis", "softmax attrs"), "softmax attrs.axis")));
    }
    if (op_name == "nn_layer_norm") {
        const std::string ctx = "layer_norm attrs";
        if (attrs.o.size() != 3 || !OptionalField(attrs, "axis") ||
            !OptionalField(attrs, "epsilon") || !OptionalField(attrs, "accumulation_dtype")) {
            throw std::runtime_error("LayerNorm import attrs must contain axis, epsilon, and accumulation_dtype");
        }
        return ObjectRef(relay::LayerNormAttrs::Create(
            ReadInt(Field(attrs, "axis", ctx), ctx + ".axis"),
            ReadFloat(Field(attrs, "epsilon", ctx), ctx + ".epsilon"),
            ReadString(Field(attrs, "accumulation_dtype", ctx),
                       ctx + ".accumulation_dtype")));
    }
    if (op_name == "transpose") {
        return ObjectRef(relay::TransposeAttrs::Create(
            ToArray(ReadInt64Vector(Field(attrs, "perm", "transpose attrs"),
                                    "transpose attrs.perm"))));
    }
    if (op_name == "gather") {
        const std::string ctx = "gather attrs";
        if (attrs.o.size() != 1 || !OptionalField(attrs, "axis")) {
            throw std::runtime_error("Gather import attrs must contain exactly axis");
        }
        return ObjectRef(relay::GatherAttrs::Create(
            ReadInt(Field(attrs, "axis", ctx), ctx + ".axis")));
    }
    if (op_name == "slice") {
        const std::string ctx = "slice attrs";
        if (attrs.o.size() != 4 || !OptionalField(attrs, "starts") ||
            !OptionalField(attrs, "ends") || !OptionalField(attrs, "axes") ||
            !OptionalField(attrs, "steps")) {
            throw std::runtime_error("Slice import attrs must contain exactly starts, ends, axes, and steps");
        }
        return ObjectRef(relay::SliceAttrs::Create(
            ToArray(ReadInt64Vector(Field(attrs, "starts", ctx), ctx + ".starts")),
            ToArray(ReadInt64Vector(Field(attrs, "ends", ctx), ctx + ".ends")),
            ToArray(ReadInt64Vector(Field(attrs, "axes", ctx), ctx + ".axes")),
            ToArray(ReadInt64Vector(Field(attrs, "steps", ctx), ctx + ".steps"))));
    }
    if (op_name == "concatenate") {
        const std::string ctx = "concatenate attrs";
        if (attrs.o.size() != 1 || !OptionalField(attrs, "axis")) {
            throw std::runtime_error("Concatenate import attrs must contain exactly axis");
        }
        return ObjectRef(relay::ConcatenateAttrs::Create(
            ReadInt(Field(attrs, "axis", ctx), ctx + ".axis")));
    }
    if (op_name == "nn_global_avg_pool2d") {
        return ObjectRef(relay::GlobalAvgPool2DAttrs::Create());
    }
    if (op_name == "nn_flatten") {
        return ObjectRef(relay::FlattenAttrs::Create(
            ReadInt(Field(attrs, "axis", "flatten attrs"), "flatten attrs.axis")));
    }
    if (op_name == "nn_gemm") {
        return ObjectRef(relay::GemmAttrs::Create(
            ReadFloat(Field(attrs, "alpha", "gemm attrs"), "gemm attrs.alpha"),
            ReadFloat(Field(attrs, "beta", "gemm attrs"), "gemm attrs.beta"),
            ReadInt(Field(attrs, "transA", "gemm attrs"), "gemm attrs.transA"),
            ReadInt(Field(attrs, "transB", "gemm attrs"), "gemm attrs.transB")));
    }
    throw std::runtime_error("Unsupported Relay op in ONNX import spec: " + op_name);
}

// 推导节点实参的静态类型；导入期先于最终 InferType 消费。
void InferArgTypes(const Array<Expr>& args, const Array<Var>& function_params) {
    relay::InferTypePass(Function(function_params, Tuple(args)));
}

// S1 算术/开方节点只接收 float32：手写 spec 不能依赖 Python 已校验的假设。
void ValidateFloat32Inputs(const std::string& op_name, const Array<Expr>& args,
                           const Array<Var>& function_params, const std::string& node_name) {
    InferArgTypes(args, function_params);
    for (size_t i = 0; i < args.size(); ++i) {
        const auto* type = args[i].checked_type().As<TensorTypeNode>();
        if (!type || type->dtype != "float32") {
            throw std::runtime_error(
                op_name + " import requires float32 inputs in the static S1 subset: " +
                node_name);
        }
    }
}

// M4/M5 Neg/Sigmoid/Pow 只开放 float32：手写 spec 不能依赖 Python 已校验的假设，
// dtype 越界必须携带节点名失败。
void ValidateFloat32MathInputs(const std::string& op_name, const Array<Expr>& args,
                               const Array<Var>& function_params,
                               const std::string& node_name) {
    InferArgTypes(args, function_params);
    for (size_t i = 0; i < args.size(); ++i) {
        const auto* type = args[i].checked_type().As<TensorTypeNode>();
        if (!type || type->dtype != "float32") {
            throw std::runtime_error(
                op_name + " import requires float32 input(s) in the M4/M5 static subset: " +
                node_name);
        }
    }
}

// M5 S2 Pow 只开放同 dtype float32，且广播必须在 reifier 处以节点名失败：
// 函数级 InferType 的广播错误不携带节点名，手写 spec 的诊断要求能定位节点。
void ValidatePowSubset(const Array<Expr>& args, const Array<Var>& function_params,
                       const std::string& node_name) {
    if (args.size() != 2) {
        throw std::runtime_error("pow import expects exactly two inputs: " + node_name);
    }
    InferArgTypes(args, function_params);
    const auto* lhs = args[0].checked_type().As<TensorTypeNode>();
    const auto* rhs = args[1].checked_type().As<TensorTypeNode>();
    if (!lhs || !rhs || lhs->dtype != "float32" || rhs->dtype != "float32") {
        throw std::runtime_error(
            "pow import requires same-dtype float32 input(s) in the M4/M5 static "
            "subset: " + node_name);
    }
    const auto format_shape = [](const Array<int64_t>& shape) {
        std::string text = "[";
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            if (axis != 0) text += ", ";
            text += std::to_string(shape[axis]);
        }
        return text + "]";
    };
    size_t lhs_axis = lhs->shape.size();
    size_t rhs_axis = rhs->shape.size();
    while (lhs_axis > 0 && rhs_axis > 0) {
        --lhs_axis;
        --rhs_axis;
        const int64_t left = lhs->shape[lhs_axis];
        const int64_t right = rhs->shape[rhs_axis];
        if (left != right && left != 1 && right != 1) {
            throw std::runtime_error(
                "pow import cannot broadcast shapes " + format_shape(lhs->shape) +
                " and " + format_shape(rhs->shape) + ": " + node_name);
        }
    }
}

// Expand 的 reifier 复校验：不信任 Python。data 维度必须静态非负，且每个
// 对齐后的 data 维度是 1 或等于目标维度（numpy broadcast_to 规则）。
void ValidateExpandSubset(const Array<Expr>& args, const ObjectRef& attrs,
                          const Array<Var>& function_params,
                          const std::string& node_name) {
    if (args.size() != 1) {
        throw std::runtime_error("expand import expects exactly one data input: " +
                                 node_name);
    }
    const auto* expand_attrs = attrs.As<relay::ExpandAttrsNode>();
    if (!expand_attrs) {
        throw std::runtime_error("Expand import requires ExpandAttrs: " + node_name);
    }
    InferArgTypes(args, function_params);
    const auto* data = args[0].checked_type().As<TensorTypeNode>();
    if (!data) {
        throw std::runtime_error("Expand import expects a TensorType data input: " +
                                 node_name);
    }
    const size_t target_rank = expand_attrs->target_shape.size();
    if (data->shape.size() > target_rank) {
        throw std::runtime_error(
            "Expand import data rank must not exceed the target rank: " + node_name);
    }
    const size_t offset = target_rank - data->shape.size();
    for (size_t axis = 0; axis < data->shape.size(); ++axis) {
        const int64_t dim = data->shape[axis];
        if (dim < 0) {
            throw std::runtime_error(
                "Expand import requires non-negative static data dimensions: " +
                node_name);
        }
        const int64_t target = expand_attrs->target_shape[axis + offset];
        if (dim != target && dim != 1) {
            throw std::runtime_error(
                "Expand import data dimension " + std::to_string(dim) + " at axis " +
                std::to_string(axis) + " must be 1 or equal to the target dimension " +
                std::to_string(target) + ": " + node_name);
        }
    }
}

// S1 Equal 接线只开放同 dtype 的 int32/int64/float32（与 M5 EqualInferType 合同
// 一致）：手写 spec 不能依赖 Python 已校验的假设。dtype 子集、dtype 一致性和
// trailing 广播都在此用携带节点名的诊断复校验；输出 bool 由 InferType 与图
// 输出声明合同共同钉住（Y 声明为非 bool 会在输出契约检查中失败）。
void ValidateEqualSubset(const Array<Expr>& args, const Array<Var>& function_params,
                         const std::string& node_name) {
    if (args.size() != 2) {
        throw std::runtime_error("equal import expects exactly two inputs: " +
                                 node_name);
    }
    InferArgTypes(args, function_params);
    const auto* lhs = args[0].checked_type().As<TensorTypeNode>();
    const auto* rhs = args[1].checked_type().As<TensorTypeNode>();
    const auto is_supported_dtype = [](const std::string& dtype) {
        return dtype == "int32" || dtype == "int64" || dtype == "float32";
    };
    if (!lhs || !rhs || !is_supported_dtype(lhs->dtype) || !is_supported_dtype(rhs->dtype)) {
        throw std::runtime_error(
            "equal import supports same-dtype int32, int64, or float32 inputs: " +
            node_name);
    }
    if (lhs->dtype != rhs->dtype) {
        throw std::runtime_error(
            "equal import requires matching input dtypes: " + node_name);
    }
    const auto format_shape = [](const Array<int64_t>& shape) {
        std::string text = "[";
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            if (axis != 0) text += ", ";
            text += std::to_string(shape[axis]);
        }
        return text + "]";
    };
    size_t lhs_axis = lhs->shape.size();
    size_t rhs_axis = rhs->shape.size();
    while (lhs_axis > 0 && rhs_axis > 0) {
        --lhs_axis;
        --rhs_axis;
        const int64_t left = lhs->shape[lhs_axis];
        const int64_t right = rhs->shape[rhs_axis];
        if (left != right && left != 1 && right != 1) {
            throw std::runtime_error(
                "equal import cannot broadcast shapes " + format_shape(lhs->shape) +
                " and " + format_shape(rhs->shape) + ": " + node_name);
        }
    }
}

// S1 Cast 只开放 int32/int64 到 float32；不信任 Python 已校验的手写 spec。
void ValidateCastSubset(const Array<Expr>& args, const ObjectRef& attrs,
                        const Array<Var>& function_params, const std::string& node_name) {
    const auto* cast_attrs = attrs.As<relay::CastAttrsNode>();
    if (!cast_attrs || cast_attrs->to != 0) {
        throw std::runtime_error(
            "Cast import supports only to=float32 (Relay code 0) in the static S1 "
            "subset: " +
            node_name);
    }
    InferArgTypes(args, function_params);
    const auto* type = args[0].checked_type().As<TensorTypeNode>();
    // float32 源是恒等转换：MiniMind 的 RMSNorm 用 `.float()` 提升精度，在已是
    // float32 的图上导出成恒等 Cast。Relay cast 对 dtype 不设限，同 dtype 经
    // topi 落成一次拷贝。
    if (!type || (type->dtype != "int32" && type->dtype != "int64" &&
                  type->dtype != "float32")) {
        throw std::runtime_error(
            "Cast import supports only int32/int64/float32 sources in the static S1 "
            "subset: " +
            node_name);
    }
}

// S1 ReduceMean 只接收 float32，且不得在零尺寸轴上归约（结果未定义）。
void ValidateReduceMeanSubset(const Array<Expr>& args, const ObjectRef& attrs,
                              const Array<Var>& function_params,
                              const std::string& node_name) {
    InferArgTypes(args, function_params);
    const auto* type = args[0].checked_type().As<TensorTypeNode>();
    if (!type || type->dtype != "float32") {
        throw std::runtime_error(
            "ReduceMean import requires a float32 input in the static S1 subset: " +
            node_name);
    }
    const auto* reduce_attrs = attrs.As<relay::ReduceMeanAttrsNode>();
    if (!reduce_attrs) {
        throw std::runtime_error("ReduceMean import requires ReduceMeanAttrs: " +
                                 node_name);
    }
    const int rank = static_cast<int>(type->shape.size());
    for (int64_t axis : reduce_attrs->axes) {
        int64_t normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank) {
            throw std::runtime_error("ReduceMean import axis " + std::to_string(axis) +
                                     " is out of range for rank " +
                                     std::to_string(rank) + ": " + node_name);
        }
        const int64_t extent = type->shape[static_cast<size_t>(normalized)];
        if (extent == 0) {
            throw std::runtime_error(
                "ReduceMean import reduces over a zero-extent axis, which is "
                "undefined and rejected in the static S1 subset: " +
                node_name);
        }
    }
}

// 索引可以是常量，也可以是运行时张量（embedding 查表就是后者）。两条路径都
// 固定 data rank / axis / extent / 索引 dtype 合同；差别只在值域：常量索引在
// 导入期逐值证明落在 ONNX 域内，运行时索引的值到 launch 才存在，由 lowering
// 后的 GatherCompute 守卫承担——负索引按 ONNX 语义折回，越界经 Select 取零
// 且不形成越界 Load。
void ValidateGatherIndices(const Array<Expr>& args,
                           const ObjectRef& attrs,
                           const Array<Var>& function_params,
                           const std::string& node_name) {
    if (args.size() != 2) {
        throw std::runtime_error(
            "Gather import node must contain exactly data and indices: " + node_name);
    }
    const auto* gather_attrs = attrs.As<relay::GatherAttrsNode>();
    if (!gather_attrs) {
        throw std::runtime_error("Gather import requires GatherAttrs: " + node_name);
    }

    relay::InferTypePass(Function(function_params, args[0]));
    const auto* data_type = args[0].checked_type().As<TensorTypeNode>();
    if (!data_type || data_type->shape.empty()) {
        throw std::runtime_error("Gather import data must have rank >= 1: " +
                                 node_name);
    }
    int axis = gather_attrs->axis;
    const int rank = static_cast<int>(data_type->shape.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("Gather import axis is out of range: " + node_name);
    }
    const int64_t extent = data_type->shape[static_cast<size_t>(axis)];
    if (extent < 0) {
        throw std::runtime_error(
            "Gather import requires a non-negative static axis extent: " +
            node_name);
    }

    const auto* constant = args[1].As<ConstantNode>();
    if (!constant) {
        relay::InferTypePass(Function(function_params, args[1]));
        const auto* index_type = args[1].checked_type().As<TensorTypeNode>();
        if (!index_type ||
            (index_type->dtype != "int32" && index_type->dtype != "int64")) {
            throw std::runtime_error(
                "Gather import runtime indices must be an int32 or int64 tensor: " +
                node_name);
        }
        return;
    }
    if (!constant->data.defined()) {
        throw std::runtime_error(
            "Gather import constant indices have no payload: " + node_name);
    }
    const DLDataType dtype = constant->data.dtype();
    if (dtype.code != kDLInt || dtype.lanes != 1 ||
        (dtype.bits != 32 && dtype.bits != 64)) {
        throw std::runtime_error(
            "Gather import constant indices must be int32 or int64: " + node_name);
    }
    const size_t count = constant->data.NBytes() / (dtype.bits / 8);
    const auto validate = [&](int64_t index) {
        if (index < -extent || index >= extent) {
            throw std::runtime_error(
                "Gather import constant index " + std::to_string(index) +
                " is outside the ONNX domain [" + std::to_string(-extent) +
                ", " + std::to_string(extent - 1) + "]: " + node_name);
        }
    };
    if (dtype.bits == 32) {
        std::vector<int32_t> values(count);
        constant->data.CopyToBytes(values.data(), constant->data.NBytes());
        for (int32_t value : values) validate(value);
    } else {
        std::vector<int64_t> values(count);
        constant->data.CopyToBytes(values.data(), constant->data.NBytes());
        for (int64_t value : values) validate(value);
    }
}

}  // namespace

// 装载 ONNX 中间规范、参数 Storage 和 Relay 数据流，返回可编译函数及参数表。
ImportedONNXModel LoadONNXImportSpec(const std::string& json_path,
                                     const std::string& params_path) {
    Json root = JsonParser(ReadTextFile(json_path)).Parse();
    RequireKind(root, Json::Object, "root");
    std::string format = ReadString(Field(root, "format", "root"), "root.format");
    if (format != "kxc.onnx_import.v1") {
        throw std::runtime_error("Unsupported ONNX import spec format: " + format);
    }

    const Json& function_json = Field(root, "function", "root");
    const Json& inputs_json = Field(function_json, "inputs", "function");
    const Json& outputs_json = Field(function_json, "outputs", "function");
    const Json& params_json = Field(root, "params", "root");
    const Json& nodes_json = Field(function_json, "nodes", "function");
    RequireKind(inputs_json, Json::Array, "function.inputs");
    RequireKind(outputs_json, Json::Array, "function.outputs");
    RequireKind(params_json, Json::Array, "root.params");
    RequireKind(nodes_json, Json::Array, "function.nodes");

    ImportedONNXModel result;
    std::unordered_map<std::string, Expr> values;
    // 记录每个中间值由哪个节点产出，供图输出契约失配诊断定位到节点。
    std::unordered_map<std::string, std::string> producer_node_by_value;
    Array<Var> function_params;

    for (size_t i = 0; i < inputs_json.a.size(); ++i) {
        const Json& input = inputs_json.a[i];
        std::string ctx = "function.inputs[" + std::to_string(i) + "]";
        std::string name = ReadString(Field(input, "name", ctx), ctx + ".name");
        std::vector<int64_t> shape = ReadStaticShape(Field(input, "shape", ctx), ctx + ".shape");
        std::string dtype = ReadString(Field(input, "dtype", ctx), ctx + ".dtype");
        Var var(name, TensorType(ToArray(shape), dtype));
        function_params.push_back(var);
        values[name] = var;
        result.input_names.push_back(name);
    }

    std::vector<char> param_bytes = ReadBinaryFile(params_path);
    for (size_t i = 0; i < params_json.a.size(); ++i) {
        const Json& param = params_json.a[i];
        std::string ctx = "root.params[" + std::to_string(i) + "]";
        std::string name = ReadString(Field(param, "name", ctx), ctx + ".name");
        std::vector<int64_t> shape = ReadStaticShape(Field(param, "shape", ctx), ctx + ".shape");
        std::string dtype = ReadString(Field(param, "dtype", ctx), ctx + ".dtype");
        int64_t offset = ReadInt64(Field(param, "offset", ctx), ctx + ".offset");
        int64_t nbytes = ReadInt64(Field(param, "nbytes", ctx), ctx + ".nbytes");
        if (offset < 0 || nbytes < 0 ||
            static_cast<uint64_t>(offset) + static_cast<uint64_t>(nbytes) > param_bytes.size()) {
            throw std::runtime_error("Param bytes range out of bounds for: " + name);
        }

        // ONNX 参数先落到 CPU Storage；后续放置阶段再显式复制到目标设备。
        runtime::NDArray array = runtime::NDArray::Empty(
            ToArray(shape), runtime::DataTypeFromString(dtype), Device::CPU());
        array.CopyFromBytes(param_bytes.data() + offset, static_cast<size_t>(nbytes));
        result.params.emplace(name, array);
        result.param_order.push_back(name);
        if (!values.emplace(name, Constant(array)).second) {
            throw std::runtime_error(
                "ONNX import param name conflicts with an existing value: " + name);
        }
    }

    const Json* param_order_json = OptionalField(root, "param_order");
    if (param_order_json) {
        std::vector<std::string> declared_order =
            ReadStringVector(*param_order_json, "root.param_order");
        if (declared_order != result.param_order) {
            throw std::runtime_error("ONNX import spec param_order does not match params metadata");
        }
    }

    for (size_t i = 0; i < nodes_json.a.size(); ++i) {
        const Json& node = nodes_json.a[i];
        std::string ctx = "function.nodes[" + std::to_string(i) + "]";
        std::string node_name = ReadString(Field(node, "name", ctx), ctx + ".name");
        std::string op_name = ReadString(Field(node, "op_name", ctx), ctx + ".op_name");
        std::vector<std::string> input_names =
            ReadStringVector(Field(node, "inputs", ctx), ctx + ".inputs");
        std::vector<std::string> output_names =
            ReadStringVector(Field(node, "outputs", ctx), ctx + ".outputs");
        if (output_names.size() != 1 || output_names[0].empty()) {
            throw std::runtime_error(
                "ONNX import node must have one non-empty output: " + node_name);
        }

        Array<Expr> args;
        for (const std::string& input_name : input_names) {
            auto it = values.find(input_name);
            if (it == values.end()) {
                throw std::runtime_error("Missing input value '" + input_name + "' for node '" +
                                         node_name + "'");
            }
            args.push_back(it->second);
        }
        ObjectRef attrs = MakeAttrs(op_name, Field(node, "attrs", ctx), node_name);
        if (op_name == "gather") {
            ValidateGatherIndices(args, attrs, function_params, node_name);
        }
        if (op_name == "mul" || op_name == "subtract" || op_name == "divide" ||
            op_name == "sqrt") {
            ValidateFloat32Inputs(op_name, args, function_params, node_name);
        }
        if (op_name == "neg" || op_name == "sigmoid") {
            ValidateFloat32MathInputs(op_name, args, function_params, node_name);
        }
        if (op_name == "pow") {
            ValidatePowSubset(args, function_params, node_name);
        }
        if (op_name == "expand") {
            ValidateExpandSubset(args, attrs, function_params, node_name);
        }
        if (op_name == "equal") {
            ValidateEqualSubset(args, function_params, node_name);
        }
        if (op_name == "cast") {
            ValidateCastSubset(args, attrs, function_params, node_name);
        }
        if (op_name == "reduce_mean") {
            ValidateReduceMeanSubset(args, attrs, function_params, node_name);
        }
        Call call(relay::Op::Get(op_name), args, attrs);
        if (!values.emplace(output_names[0], call).second) {
            throw std::runtime_error(
                "ONNX import node output name conflicts with an existing value: " +
                node_name);
        }
        producer_node_by_value.emplace(output_names[0], node_name);
    }

    Array<Expr> output_exprs;
    std::vector<std::vector<int64_t>> declared_output_shapes;
    std::vector<std::string> declared_output_dtypes;
    const auto output_contract_error = [&](const std::string& name) {
        std::string message = "ONNX import output contract mismatch for '" + name + "'";
        const auto producer = producer_node_by_value.find(name);
        if (producer != producer_node_by_value.end()) {
            message += " (produced by node '" + producer->second + "')";
        }
        return message;
    };
    for (size_t i = 0; i < outputs_json.a.size(); ++i) {
        const Json& output = outputs_json.a[i];
        std::string ctx = "function.outputs[" + std::to_string(i) + "]";
        std::string name = ReadString(Field(output, "name", ctx), ctx + ".name");
        declared_output_shapes.push_back(
            ReadStaticShape(Field(output, "shape", ctx), ctx + ".shape"));
        declared_output_dtypes.push_back(
            ReadString(Field(output, "dtype", ctx), ctx + ".dtype"));
        auto it = values.find(name);
        if (it == values.end()) {
            throw std::runtime_error("Missing graph output value in ONNX import spec: " + name);
        }
        output_exprs.push_back(it->second);
        result.output_names.push_back(name);
    }
    if (output_exprs.empty()) {
        throw std::runtime_error("ONNX import spec has no graph outputs");
    }
    Expr body = output_exprs.size() == 1 ? output_exprs[0] : Expr(Tuple(output_exprs));
    result.function = relay::InferTypePass(Function(function_params, body));
    for (size_t i = 0; i < output_exprs.size(); ++i) {
        const auto* inferred = output_exprs[i].checked_type().As<TensorTypeNode>();
        if (!inferred || inferred->dtype != declared_output_dtypes[i] ||
            inferred->shape.size() != declared_output_shapes[i].size()) {
            throw std::runtime_error(output_contract_error(result.output_names[i]));
        }
        for (size_t axis = 0; axis < declared_output_shapes[i].size(); ++axis) {
            if (inferred->shape[axis] != declared_output_shapes[i][axis]) {
                throw std::runtime_error(output_contract_error(result.output_names[i]));
            }
        }
    }
    return result;
}

}  // namespace frontend
}  // namespace kxc
