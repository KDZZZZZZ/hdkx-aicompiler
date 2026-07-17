/*! \file include/base/py_runtime.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "object.h"
#include "packedfunc.h"
#include "registry.h"
#include "device.h"
#include "typemanager.h"
#include "ndarray.h"

namespace py = pybind11;

namespace kxc {

// 辅助函数：构造 Value
inline Value MakeValue(int64_t v) { Value val; val.v_int = v; return val; }
inline Value MakeValue(double v) { Value val; val.v_float = v; return val; }
inline Value MakeValue(const char* v) { Value val; val.v_str = v; return val; }
inline Value MakeValue(const Object* v) { Value val; val.v_object = v; return val; }
inline Value MakeNullValue() { Value val; val.v_object = nullptr; return val; }

inline py::object WrapObjectRef(const ObjectRef& object) {
    if (!object.defined()) return py::none();
    if (object.As<runtime::NDArrayNode>()) {
        return py::cast(runtime::NDArray(object.get()));
    }
    return py::cast(object);
}

// 模块初始化函数
inline void InitKXCRuntime(py::module_& m) {
    py::class_<TypeInfo>(m, "TypeInfo")
        .def_property_readonly("runtime_index", &TypeInfo::runtime_index)
        .def_property_readonly("type_key", [](const TypeInfo& info) {
            return std::string(info.type_key());
        });

    // 绑定 ObjectRef
    py::class_<ObjectRef>(m, "ObjectRef")
        .def(py::init<>())
        .def_property_readonly("defined", [](const ObjectRef& self) { return (bool)self; })
        .def_property_readonly(
            "type_info",
            [](const ObjectRef& self) -> const TypeInfo* {
                return self.get() ? &self.get()->GetTypeInfo() : nullptr;
            },
            py::return_value_policy::reference)
        .def_property_readonly("type_id", [](const ObjectRef& self) -> py::object {
            return self.get() ? py::cast(self.get()->GetTypeId()) : py::none();
        })
        .def_property_readonly("type_key", [](const ObjectRef& self) -> py::object {
            return self.get() ? py::cast(std::string(self.get()->GetTypeKey())) : py::none();
        });

    // 绑定 Object
    // 使用 nodelete，因为 Object 是引用计数的，我们不希望 pybind11 直接 delete 它
    // 而是通过 ObjectRef 或手动 DecRef 管理
    py::class_<Object, std::unique_ptr<Object, py::nodelete>> object_base(m, "Object");
    object_base
        .def_property_readonly(
            "type_info", &Object::GetTypeInfo,
            py::return_value_policy::reference_internal)
        .def_property_readonly("type_id", &Object::GetTypeId)
        .def_property_readonly("type_key", [](const Object& object) {
            return std::string(object.GetTypeKey());
        })
        .def("get_type_id", &Object::GetTypeId, "Get the runtime type ID of the object.");
    
    // 绑定 PackedFunc
    py::class_<PackedFunc, ObjectRef>(m, "PackedFunc")
        .def("__call__", [](const PackedFunc& func, py::args args) -> py::object {
            std::vector<Value> values;
            std::vector<TypeCode> type_codes;
            std::vector<ObjectRef> object_holders;
            values.reserve(args.size());
            type_codes.reserve(args.size());
            object_holders.reserve(args.size());

            // 保持 string 的生命周期
            std::vector<std::string> str_holders;
            str_holders.reserve(args.size());

            for (const auto& arg : args) {
                if (py::isinstance<ObjectRef>(arg)) {
                    object_holders.push_back(arg.cast<ObjectRef>());
                    values.push_back(MakeValue(object_holders.back().get()));
                    type_codes.push_back(kObjectRef);
                } else if (py::isinstance<Object>(arg)) {
                    const Object* object = arg.cast<const Object*>();
                    object_holders.emplace_back(object);
                    values.push_back(MakeValue(object_holders.back().get()));
                    type_codes.push_back(kObjectRef);
                } else if (py::isinstance<py::int_>(arg)) {
                    values.push_back(MakeValue(arg.cast<int64_t>()));
                    type_codes.push_back(kInt);
                } else if (py::isinstance<py::float_>(arg)) {
                    values.push_back(MakeValue(arg.cast<double>()));
                    type_codes.push_back(kFloat);
                } else if (py::isinstance<py::str>(arg)) {
                    str_holders.push_back(arg.cast<std::string>());
                    values.push_back(MakeValue(str_holders.back().c_str()));
                    type_codes.push_back(kString);
                } else if (arg.is_none()) {
                    values.push_back(MakeNullValue());
                    type_codes.push_back(kNull);
                } else {
                    throw py::type_error("Unsupported argument type");
                }
            }
            
            Args kxc_args(values.data(), type_codes.data(), (int)args.size());
            RetValue rv;
            
            func(kxc_args, &rv);
            
            // 处理返回值
            switch (rv.type_code()) {
                case kInt: return py::cast(rv.As<int64_t>());
                case kFloat: return py::cast(rv.As<double>());
                case kString: return py::cast(rv.As<const char*>());
                case kObjectRef: return WrapObjectRef(rv.As<ObjectRef>());
                case kNull: return py::none();
                default: throw py::type_error("Unsupported return type");
            }
        });
        
    // Bind NDArrayNode
    py::class_<runtime::NDArrayNode, Object, std::unique_ptr<runtime::NDArrayNode, py::nodelete>>(m, "NDArrayNode")
        .def_readonly("shape", &runtime::NDArrayNode::shape);

    // Bind NDArray
    py::class_<runtime::NDArray, ObjectRef>(m, "NDArray", py::buffer_protocol())
        .def(py::init<std::vector<int64_t>, std::string>())
        .def_buffer([](runtime::NDArray& m) -> py::buffer_info {
            const runtime::NDArrayNode* node = m.operator->();
            std::string format;
            if (node->dl_tensor.dtype.code == kDLFloat) format = "f";
            else if (node->dl_tensor.dtype.code == kDLInt) format = "i"; 
            else format = "B"; // fallback

            std::vector<py::ssize_t> strides;
            std::vector<py::ssize_t> shape;
            for(auto s : node->shape) shape.push_back(s);
            
            // Calc strides (row-major)
            py::ssize_t stride = node->dl_tensor.dtype.bits / 8;
            for (int i = node->dl_tensor.ndim - 1; i >= 0; --i) {
                strides.insert(strides.begin(), stride);
                stride *= node->dl_tensor.shape[i];
            }
            
            return py::buffer_info(
                node->dl_tensor.data,
                node->dl_tensor.dtype.bits / 8,
                format,
                node->dl_tensor.ndim,
                shape,
                strides
            );
        });

    // 绑定 get_global_func
    m.def("get_global_func", [](const std::string& name) -> py::object {
        // Registry::Global() 返回 Registry&，使用 .Get()
        PackedFunc func = Registry::Global().Get(name);
        // PackedFunc 现在是 Object，有 operator bool
        if (!func) {
            return py::none();
        }
        return py::cast(func);
    }, py::arg("name"));

    // 绑定 DeviceTypeCode 枚举
    py::enum_<DeviceTypeCode>(m, "DeviceTypeCode")
        .value("CPU", kCPU)
        .value("GPU", kGPU)
        .value("OpenCL", kOpenCL)
        .value("Metal", kMetal)
        .value("Unknown", kUnknown)
        .export_values();

    // 绑定 Device 类
    // 使用别名消除歧义
    using DeviceCls = class ::kxc::Device;
    py::class_<DeviceCls, Object>(m, "Device")
        .def_property_readonly("device_type", &DeviceCls::device_type)
        .def_property_readonly("device_id", &DeviceCls::device_id)
        .def("__repr__", [](const DeviceCls& self) { return self.ToString(); })
        .def("__str__", [](const DeviceCls& self) { return self.ToString(); });

    // 绑定 device 工厂函数
    // DeviceManager 的缓存持有 ObjectRef，Python 只借用对应节点。
    m.def("device", [](DeviceTypeCode type, int id) -> DeviceCls* {
         ObjectRef ref = kxc::DeviceManager::Global()->GetOrCreate(type, id);
         return const_cast<DeviceCls*>(ref.As<DeviceCls>());
    }, py::arg("type_code"), py::arg("device_id"), py::return_value_policy::reference);

    // 绑定 create_object 工厂函数
    // 使用 TypeManager 通过字符串键创建对象
    m.def("create_object", [](const std::string& type_key) -> py::object {
        ObjectRef obj = TypeManager::Get()->CreateObject(type_key);
        if (!obj.defined()) {
            // 如果创建失败（未注册的类型），返回 None
            return py::none();
        }
        // 返回持有侵入式引用的 ObjectRef Python 句柄。
        return WrapObjectRef(obj);
    }, py::arg("type_key"));
}

} // namespace kxc
