/*! \file include/base/ndarray.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "base/container.h"
#include "base/object.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    kDLInt = 0U,
    kDLUint = 1U,
    kDLFloat = 2U,
    kDLBfloat = 4U,
    kDLComplex = 5U,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef enum {
    kDLCPU = 1,
    kDLGPU = 2,
} DLDeviceType;

typedef struct {
    int device_type;
    int device_id;
} DLContext;

typedef struct {
    void* data;
    DLContext ctx;
    int ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;

#ifdef __cplusplus
}
#endif

namespace kxc {
namespace runtime {

class NDArrayNode : public Object {
public:
    DLTensor dl_tensor;
    std::vector<int64_t> shape;

    ~NDArrayNode() override;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(NDArrayNode)

class NDArray : public ObjectRef {
public:
    using ObjectRef::ObjectRef;

    NDArray(Array<int64_t> shape, std::string dtype_str);

    const NDArrayNode* operator->() const;
    const DLTensor* operator*() const;
};

}  // namespace runtime
}  // namespace kxc