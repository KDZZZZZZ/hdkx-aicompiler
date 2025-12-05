#include "../include/base/op.h"
#include "../include/base/relay.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;

void test_new_ops_registration() {
    std::cout << "Testing New Operators Registration..." << std::endl;

    // List of ops to check
    std::vector<std::string> ops_to_check = {
        "matmul", "mul", "pow", "sqrt", "sub", // math
        "reduce_mean", // reduce
        "softmax", // nn
        "reshape", "shape", "slice", "split", "squeeze", "transpose", "unsqueeze", "where" // transform
    };

    for (const auto& name : ops_to_check) {
        try {
            Op op = Op::Get(name);
            std::cout << "Found Op: " << op->name << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "FAILED to find Op: " << name << " - " << e.what() << std::endl;
            exit(1);
        }
    }

    std::cout << "PASS: All new ops found." << std::endl;
}

void test_new_attrs() {
    std::cout << "Testing New Attributes..." << std::endl;

    // ReduceMeanAttrs
    {
        std::vector<int64_t> axes = {1};
        ReduceMeanAttrs attrs = ReduceMeanAttrs::Create(axes, 1);
        assert(attrs->axes[0] == 1);
        assert(attrs->keepdims == 1);
        std::cout << "ReduceMeanAttrs created." << std::endl;
    }

    // ReshapeAttrs
    {
        ReshapeAttrs attrs = ReshapeAttrs::Create(1);
        assert(attrs->allowzero == 1);
        std::cout << "ReshapeAttrs created." << std::endl;
    }

    // SplitAttrs
    {
        SplitAttrs attrs = SplitAttrs::Create(2);
        assert(attrs->axis == 2);
        std::cout << "SplitAttrs created." << std::endl;
    }

    // TransposeAttrs
    {
        std::vector<int64_t> perm = {0, 2, 1};
        TransposeAttrs attrs = TransposeAttrs::Create(perm);
        assert(attrs->perm[1] == 2);
        std::cout << "TransposeAttrs created." << std::endl;
    }

    std::cout << "PASS: All new attrs verified." << std::endl;
}

int main() {
    test_new_ops_registration();
    test_new_attrs();
    return 0;
}
