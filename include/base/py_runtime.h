#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "object.h"
#include "packedfunc.h"
#include "registry.h"
#include "device.h"
#include "typemanager.h"

namespace py = pybind11;

// --- 1. pybind11 类型转换特化 (必须在 ::pybind11::detail 命名空间下) ---
namespace pybind11 {
namespace detail {

template<>
struct type_caster<kxc::ObjectRef> {
public:
    // 定义 Python 中的类型名称
    PYBIND11_TYPE_CASTER(kxc::ObjectRef, _("kxc_runtime.ObjectRef"));

    // Python -> C++ (load)
    bool load(py::handle src, bool convert) {
        if (!src) return false;
        if (src.is_none()) {
            value = kxc::ObjectRef();
            return true;
        }
        try {
            // 尝试将 Python 对象转换为 C++ Object 指针
            // 注意：这要求 Python 对象必须是由 pybind11 包装的 kxc::Object 或其子类
            const kxc::Object* cpp_obj_ptr = py::cast<const kxc::Object*>(src);
            value = kxc::ObjectRef(cpp_obj_ptr);
            return true;
        } catch (const py::cast_error&) {
            return false;
        }
    }

    // C++ -> Python (cast)
    static py::handle cast(const kxc::ObjectRef& src, py::return_value_policy policy, py::handle parent) {
        if (!src.get()) {
            return py::none().release();
        }
        // 将内部的 Object* 转换为 Python 对象
        // pybind11 会查找该 Object 实际类型对应的 Python 类
        return py::cast(src.get(), policy, parent);
    }
};

} // namespace detail
} // namespace pybind11

namespace kxc {

// 辅助函数：构造 Value
inline Value MakeValue(int64_t v) { Value val; val.v_int = v; return val; }
inline Value MakeValue(double v) { Value val; val.v_float = v; return val; }
inline Value MakeValue(const char* v) { Value val; val.v_str = v; return val; }
inline Value MakeValue(const Object* v) { Value val; val.v_object = v; return val; }
inline Value MakeNullValue() { Value val; val.v_object = nullptr; return val; }

// --- 2. 模块初始化函数 ---
inline void InitKXCRuntime(py::module_& m) {
    // 绑定 ObjectRef
    py::class_<ObjectRef>(m, "ObjectRef")
        .def(py::init<>())
        .def_property_readonly("defined", [](const ObjectRef& self) { return (bool)self; });

    // 绑定 Object
    // 使用 nodelete，因为 Object 是引用计数的，我们不希望 pybind11 直接 delete 它
    // 而是通过 ObjectRef 或手动 DecRef 管理
    py::class_<Object, std::unique_ptr<Object, py::nodelete>> object_base(m, "Object");
    object_base.def("get_type_id", &Object::GetTypeId, "Get the type ID of the object.");
    
    // 绑定 PackedFunc
    py::class_<PackedFunc, Object, std::unique_ptr<PackedFunc, py::nodelete>>(m, "PackedFunc")
        .def("__call__", [](const PackedFunc& func, py::args args) -> py::object {
            std::vector<Value> values;
            std::vector<TypeCode> type_codes;
            values.reserve(args.size());
            type_codes.reserve(args.size());

            // 保持 string 的生命周期
            std::vector<std::string> str_holders;
            str_holders.reserve(args.size());

            for (const auto& arg : args) {
                if (py::isinstance<Object>(arg)) {
                    ObjectRef obj_ref = arg.cast<ObjectRef>(); 
                    // 增加引用计数，因为 Value 只持有裸指针
                    values.push_back(MakeValue(obj_ref.get()));
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
                case kObjectRef: return py::cast(rv.As<ObjectRef>());
                case kNull: return py::none();
                default: throw py::type_error("Unsupported return type");
            }
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
    // Removed explicit holder to match MyObj pattern which works
    py::class_<DeviceCls, Object>(m, "Device") 
        .def(py::init<DeviceTypeCode, int>(),
             py::arg("type_code"), py::arg("device_id"))
        .def_property_readonly("device_type", &DeviceCls::device_type)
        .def_property_readonly("device_id", &DeviceCls::device_id)
        .def("__repr__", [](const DeviceCls& self) { return self.ToString(); })
        .def("__str__", [](const DeviceCls& self) { return self.ToString(); });

    // 绑定 device 工厂函数
    // 使用 DeviceManager 显式创建，确保返回 ObjectRef 且使用缓存
    m.def("device", [](DeviceTypeCode type, int id) -> DeviceCls* {
         // Return raw pointer to bypass potential ObjectRef type_caster issues with m.def return values.
         // Use DeviceManager to get ObjectRef (which manages the device lifecycle).
         // We IncRef because Python will hold a reference to the Device object.
         // Since Device is bound with nodelete, Python won't delete it, but it won't DecRef it either via unique_ptr deleter.
         // This effectively relies on DeviceManager keeping it alive or intentional leak of the python reference increment.
         ObjectRef ref = kxc::DeviceManager::Global()->GetOrCreate(type, id);
         ref.get()->IncRef(); // Add a ref for Python
         return (DeviceCls*)ref.get();
    }, py::arg("type_code"), py::arg("device_id"));

    // 绑定 create_object 工厂函数
    // 使用 TypeManager 通过字符串键创建对象
    m.def("create_object", [](const std::string& type_key) -> py::object {
        ObjectRef obj = TypeManager::Get()->CreateObject(type_key);
        if (!obj.defined()) {
            // 如果创建失败（未注册的类型），返回 None
            return py::none();
        }
        // 返回 ObjectRef，pybind11 会自动转换为对应的 Python 对象
        // 注意：这里利用了 type_caster<ObjectRef> 的特化
        return py::cast(obj);
    }, py::arg("type_key"));
}

} // namespace kxc
