#pragma once
#include "base/object.h"
#include "base/container.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstring> // for malloc/free/memcpy

// --- DLPack Compatible Definitions ---
// These are standard DLPack definitions used for tensor data exchange.
// We define them here to avoid external dependencies for now.

#ifdef __cplusplus
extern "C" {
#endif

// DLDataTypeCode
typedef enum {
    kDLInt = 0U,
    kDLUint = 1U,
    kDLFloat = 2U,
    kDLBfloat = 4U,
    kDLComplex = 5U,
} DLDataTypeCode;

// DLDataType: Data type code, bits, and lanes
typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

// DLDeviceType: CPU, GPU, etc.
typedef enum {
    kDLCPU = 1,
    kDLGPU = 2,
} DLDeviceType;

// DLContext: Device type and ID
typedef struct {
    int device_type; 
    int device_id;
} DLContext;

// DLTensor: The core tensor structure
typedef struct {
    void* data;           // Data pointer
    DLContext ctx;        // Device context
    int ndim;             // Number of dimensions
    DLDataType dtype;     // Data type
    int64_t* shape;       // Shape array
    int64_t* strides;     // Strides array (can be NULL for contiguous)
    uint64_t byte_offset; // Byte offset within the data
} DLTensor;

#ifdef __cplusplus
}
#endif
// -------------------------------------

namespace kxc {
namespace runtime {

class NDArrayNode : public Object {
public:
    // Core DLTensor structure that interoperates with other frameworks
    DLTensor dl_tensor;

    // Backing storage for shape (since DLTensor uses int64_t*)
    std::vector<int64_t> shape;
    
    // We don't store dtype string anymore, we rely on dl_tensor.dtype

    // Destructor to free data if we own it
    // For this simple implementation, we assume NDArray owns the data allocated via new/malloc
    ~NDArrayNode() {
        if (dl_tensor.data) {
            // std::free(dl_tensor.data); // Using delete[] for C++ new[]
            delete[] static_cast<char*>(dl_tensor.data);
        }
    }

    // Use macro for type registration
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(NDArrayNode)

class NDArray : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    
    NDArray(Array<int64_t> shape, std::string dtype_str) {
        NDArrayNode* node = new NDArrayNode();
        // Convert Array to std::vector for internal storage
        for(auto s : shape) node->shape.push_back(s);
        
        // Setup DLTensor
        node->dl_tensor.shape = node->shape.data();
        node->dl_tensor.ndim = static_cast<int>(node->shape.size());
        node->dl_tensor.strides = nullptr; // Compact
        node->dl_tensor.byte_offset = 0;
        node->dl_tensor.ctx = {kDLCPU, 0}; // Default to CPU

        // Parse dtype
        if (dtype_str == "float32") {
            node->dl_tensor.dtype = {kDLFloat, 32, 1};
        } else if (dtype_str == "int32") {
            node->dl_tensor.dtype = {kDLInt, 32, 1};
        } else if (dtype_str == "float64") {
            node->dl_tensor.dtype = {kDLFloat, 64, 1};
        } else if (dtype_str == "int64") {
            node->dl_tensor.dtype = {kDLInt, 64, 1};
        } else if (dtype_str == "int8") {
            node->dl_tensor.dtype = {kDLInt, 8, 1};
        } else {
            // Default or error?
             node->dl_tensor.dtype = {kDLFloat, 32, 1};
        }

        // Allocate Data
        int64_t size = 1;
        for (auto s : node->shape) size *= s;
        size_t bytes = size * (node->dl_tensor.dtype.bits / 8);
        
        node->dl_tensor.data = new char[bytes];
        // Initialize to 0
        std::memset(node->dl_tensor.data, 0, bytes);

        SetData(node);
    }

    const NDArrayNode* operator->() const {
        return static_cast<const NDArrayNode*>(object_);
    }
    
    // Helper to get DLTensor*
    const DLTensor* operator*() const {
        return &operator->()->dl_tensor;
    }
};

} // namespace runtime
} // namespace kxc
