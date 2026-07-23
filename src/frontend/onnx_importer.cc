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
ObjectRef MakeAttrs(const std::string& op_name, const Json& attrs) {
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
    if (op_name == "add" || op_name == "matmul" || op_name == "where") {
        return ObjectRef();
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
        return ObjectRef(relay::GatherAttrs::Create(
            ReadInt(Field(attrs, "axis", "gather attrs"), "gather attrs.axis")));
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
        values[name] = Constant(array);
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
        if (output_names.size() != 1) {
            throw std::runtime_error("ONNX import node must have one output: " + node_name);
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
        ObjectRef attrs = MakeAttrs(op_name, Field(node, "attrs", ctx));
        Call call(relay::Op::Get(op_name), args, attrs);
        values[output_names[0]] = call;
    }

    Array<Expr> output_exprs;
    std::vector<std::vector<int64_t>> declared_output_shapes;
    std::vector<std::string> declared_output_dtypes;
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
            throw std::runtime_error("ONNX import output contract mismatch for '" +
                                     result.output_names[i] + "'");
        }
        for (size_t axis = 0; axis < declared_output_shapes[i].size(); ++axis) {
            if (inferred->shape[axis] != declared_output_shapes[i][axis]) {
                throw std::runtime_error("ONNX import output contract mismatch for '" +
                                         result.output_names[i] + "'");
            }
        }
    }
    return result;
}

}  // namespace frontend
}  // namespace kxc
