/*! \file test/lower_multi_output_test.cpp
 * \brief 测试 Relay 多输出 lowering 和 output buffer ABI。
 */

#include "relay/op.h"
#include "relay/relay.h"
#include "relay/transforms/lower.h"
#include "tir/expr.h"
#include "tir/pass/print_ir.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool Check(bool cond, const std::string& message) {
    if (!cond) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

bool HasAttrInt(const kxc::tir::PrimFunc& func, const std::string& key, int64_t expected) {
    if (!func->attrs.count(kxc::String(key))) {
        std::cerr << "FAIL: missing PrimFunc attr " << key << "\n";
        return false;
    }
    const auto* value = func->attrs.at(kxc::String(key)).As<kxc::tir::IntImmNode>();
    if (!value) {
        std::cerr << "FAIL: PrimFunc attr " << key << " is not IntImm\n";
        return false;
    }
    if (value->value != expected) {
        std::cerr << "FAIL: PrimFunc attr " << key << " = " << value->value
                  << ", expected " << expected << "\n";
        return false;
    }
    return true;
}

bool CheckBufferShape(const kxc::tir::PrimFunc& func,
                      size_t param_index,
                      const std::vector<int64_t>& expected) {
    if (param_index >= func->params.size()) {
        std::cerr << "FAIL: param index out of range: " << param_index << "\n";
        return false;
    }
    const kxc::tir::Var& var = func->params[param_index];
    if (!func->buffer_map.count(var)) {
        std::cerr << "FAIL: missing buffer for param " << param_index << "\n";
        return false;
    }
    const kxc::tir::Buffer& buffer = func->buffer_map.at(var);
    if (buffer->shape.size() != expected.size()) {
        std::cerr << "FAIL: buffer rank mismatch at param " << param_index << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto* dim = buffer->shape[i].As<kxc::tir::IntImmNode>();
        if (!dim || dim->value != expected[i]) {
            std::cerr << "FAIL: buffer shape mismatch at param " << param_index
                      << ", dim " << i << "\n";
            return false;
        }
    }
    return true;
}

std::string TIRText(const kxc::tir::PrimFunc& func) {
    std::ostringstream os;
    kxc::tir::pass::DumpPrimFunc(func, os);
    return os.str();
}

bool TestExplicitTupleOutput() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Var y("y", kxc::TensorType({4}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    kxc::Call relu(kxc::relay::Op::Get("nn_relu"), {y});
    kxc::Function func({x, y}, kxc::Tuple({add, relu}));

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    return Check(lowered->params.size() == 4, "tuple output should add 2 output params") &&
           HasAttrInt(lowered, "kxc.input_count", 2) &&
           HasAttrInt(lowered, "kxc.constant_count", 0) &&
           HasAttrInt(lowered, "kxc.output_count", 2) &&
           HasAttrInt(lowered, "kxc.output_param_start", 2) &&
           CheckBufferShape(lowered, 2, {4}) &&
           CheckBufferShape(lowered, 3, {4});
}

bool TestTupleGetItemOutput() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Var y("y", kxc::TensorType({4}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    kxc::Call relu(kxc::relay::Op::Get("nn_relu"), {y});
    kxc::Function func({x, y}, kxc::TupleGetItem(kxc::Tuple({add, relu}), 1));

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    const std::string tir = TIRText(lowered);
    return Check(lowered->params.size() == 3, "tuple_get_item should expose 1 output param") &&
           HasAttrInt(lowered, "kxc.output_count", 1) &&
           HasAttrInt(lowered, "kxc.output_param_start", 2) &&
           CheckBufferShape(lowered, 2, {4}) &&
           Check(tir.find("T_add_out") == std::string::npos,
                 "unselected tuple field should not become an output buffer");
}

bool TestSplitTupleGetItemLowering() {
    kxc::Var x("x", kxc::TensorType({2, 6}, "float32"));
    kxc::Call split(kxc::relay::Op::Get("split"), {x},
                    kxc::relay::SplitAttrs::Create({3}, 1));
    kxc::Function func({x}, kxc::TupleGetItem(split, 1));

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    return Check(lowered->params.size() == 2, "split projection should expose 1 output") &&
           HasAttrInt(lowered, "kxc.input_count", 1) &&
           HasAttrInt(lowered, "kxc.output_count", 1) &&
           HasAttrInt(lowered, "kxc.output_param_start", 1) &&
           CheckBufferShape(lowered, 1, {2, 2});
}

bool TestSplitTupleOutputLowering() {
    kxc::Var x("x", kxc::TensorType({2, 6}, "float32"));
    kxc::Call split(kxc::relay::Op::Get("split"), {x},
                    kxc::relay::SplitAttrs::Create({3}, 1));
    kxc::Function func({x}, split);

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    return Check(lowered->params.size() == 4, "split tuple output should expose 3 outputs") &&
           HasAttrInt(lowered, "kxc.input_count", 1) &&
           HasAttrInt(lowered, "kxc.output_count", 3) &&
           HasAttrInt(lowered, "kxc.output_param_start", 1) &&
           CheckBufferShape(lowered, 1, {2, 2}) &&
           CheckBufferShape(lowered, 2, {2, 2}) &&
           CheckBufferShape(lowered, 3, {2, 2});
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, bool (*)()>> tests = {
        {"explicit_tuple_output", TestExplicitTupleOutput},
        {"tuple_get_item_output", TestTupleGetItemOutput},
        {"split_tuple_get_item_lowering", TestSplitTupleGetItemLowering},
        {"split_tuple_output_lowering", TestSplitTupleOutputLowering},
    };

    for (const auto& test : tests) {
        bool ok = false;
        try {
            ok = test.second();
        } catch (const std::exception& e) {
            std::cerr << "FAIL: " << test.first << " threw: " << e.what() << "\n";
            return 1;
        }
        if (!ok) {
            std::cerr << "Test failed: " << test.first << "\n";
            return 1;
        }
        std::cout << "PASS: " << test.first << "\n";
    }
    std::cout << "All lower multi-output tests passed.\n";
    return 0;
}
