# KXC Runtime 抽象层架构指南

本文档详细介绍了 `kxc_runtime` 的核心抽象层设计、C++ 实现细节以及 Python 端的使用方法。该架构深受 TVM 和 PyTorch 的设计启发，旨在提供高性能、可扩展且跨语言的深度学习编译器运行时支持。

## 1. 核心对象系统 (Object System)

整个运行时系统的基石是侵入式引用计数对象系统，它允许 C++ 对象在 Python 中被安全地管理和访问，实现了零拷贝的对象传递。

### 1.1 C++ 层：`Object` 与 `ObjectRef`

- **`kxc::Object`**: 所有运行时对象的基类。
  - **功能**: 维护一个原子引用计数器 (`_refCount`)。
  - **内存管理**: 通过 `IncRef()` 和 `DecRef()` 手动管理生命周期。当引用计数降为 0 时，自动调用 `delete this`。
  - **RTTI**: 通过 `GetTypeId()` 提供运行时类型识别。

- **`kxc::ObjectRef`**: `Object` 的智能指针封装。
  - **功能**: 类似于 `std::shared_ptr`，但在对象内部存储引用计数（侵入式）。
  - **设计目的**: 作为函数参数和返回值，确保跨语言边界时的内存安全。
  - **关键特性**: 是所有句柄类（Handle Class）的基类。

**代码示例 (C++):**
```cpp
// 定义一个继承自 Object 的类
class MyObj : public kxc::Object {
public:
    // 唯一的类型 ID
    const kxc::TypeIndex GetTypeId() const override { return 7; }
    std::string name = "MyObject";
};

// 使用 ObjectRef 进行管理
ObjectRef CreateMyObj() {
    // 此时引用计数为 0
    MyObj* obj = new MyObj(); 
    // 包装进 ObjectRef，引用计数 +1
    return ObjectRef(obj); 
}
```

### 1.2 Python 层：映射与绑定

通过 `pybind11` 的自定义 `type_caster`，C++ 的 `ObjectRef` 会自动转换为对应的 Python 类。

- **自动向下转型**: 如果 C++ 返回 `ObjectRef`，但底层是 `MyObj`，Python 端会自动获得 `MyObj` 类型的实例。
- **引用保持**: Python 对象持有 C++ 对象的引用，防止其在 Python 使用期间被释放。

---

## 2. 类型擦除与函数调用 (PackedFunc)

为了在 C++ 和 Python 之间动态注册和调用函数，我们使用了 `PackedFunc` 机制。这是一种类型擦除的函数包装器。

### 2.1 C++ 层：`PackedFunc` 与 `Registry`

- **`PackedFunc`**: 通用函数签名 `void(Args, RetValue*)`。
  - **Args**: 包含参数值 (`Value` union) 和类型码 (`TypeCode`) 的数组。
  - **RetValue**: 用于存储返回值的容器。
  - **自动转换**: `ArgConverter` 模板自动将 `Args` 中的原始数据转换为 C++ 类型（如 `int`, `std::string`, `ObjectRef`）。

- **`Registry`**: 全局函数注册表。
  - 允许在 C++ 中注册函数，并在 Python 中按名称查找。

**注册示例 (C++):**
```cpp
// 定义普通 C++ 函数
int Add(int a, int b) { return a + b; }

// 注册到全局表
KXC_REGISTER_GLOBAL("my_module.add")
.set_body(ToPackedFunc(Add));
```

### 2.2 Python 层：调用注册函数

Python 端可以通过 `get_global_func` 获取并调用这些函数。

```python
import kxc_runtime

# 获取函数
add_func = kxc_runtime.get_global_func("my_module.add")

# 调用 (自动处理类型转换)
result = add_func(10, 20)
print(result) # 输出 30
```

---

## 3. 设备抽象 (Device Abstraction)

运行时需要统一管理异构硬件（CPU, GPU, etc.）。

### 3.1 `Device` 类

表示一个逻辑计算设备。

- **属性**:
  - `device_type` (`DeviceTypeCode`): 设备类型枚举 (CPU=0, GPU=1, etc.)。
  - `device_id` (int): 设备索引。
- **管理**: 通过 `DeviceManager` 单例进行缓存管理，确保相同的 `(type, id)` 对应同一个 `Device` 对象实例。

### 3.2 `DeviceAPI` 接口

定义了底层硬件操作的统一接口（纯虚基类）。

- **核心方法**:
  - `AllocDataSpace`: 分配显存/内存。
  - `FreeDataSpace`: 释放内存。
  - `CopyDataFromTo`: 设备间数据拷贝 (Host->Device, Device->Host, Device->Device)。

### 3.3 使用方法

**Python 端创建设备:**
```python
# 获取 CPU 设备 (id=0)
cpu_dev = kxc_runtime.device(kxc_runtime.DeviceTypeCode.CPU, 0)

# 获取 GPU 设备 (id=0)
gpu_dev = kxc_runtime.device(kxc_runtime.DeviceTypeCode.GPU, 0)
```

**C++ 端获取 API:**
```cpp
// 获取对应设备的 API 实现
DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(kGPU);
void* ptr = api->AllocDataSpace(gpu_dev, 1024, 256);
```

---

## 4. 内存管理与数据传输

内存操作通过全局注册的 PackedFunc 暴露给 Python，底层转发给具体的 `DeviceAPI` 实现。

### 4.1 核心 API (Python)

以下函数注册在 `device_api` 命名空间下：

1.  **`device_api.AllocDataSpace(device, nbytes, alignment)`**
    - 返回: 内存指针的整数地址 (`int64`)。

2.  **`device_api.FreeDataSpace(device, ptr_val)`**
    - 参数: `ptr_val` 是之前分配的指针地址。

3.  **`device_api.CopyDataFromTo(src_dev, src_ptr, dst_dev, dst_ptr, nbytes)`**
    - 功能: 执行同步数据拷贝。自动处理不同设备间的流向。

### 4.2 完整示例 (Python)

```python
import kxc_runtime
import ctypes

# 1. 准备设备
cpu = kxc_runtime.device(kxc_runtime.DeviceTypeCode.CPU, 0)
gpu = kxc_runtime.device(kxc_runtime.DeviceTypeCode.GPU, 0)

# 2. 获取全局函数
alloc = kxc_runtime.get_global_func("device_api.AllocDataSpace")
free = kxc_runtime.get_global_func("device_api.FreeDataSpace")
copy = kxc_runtime.get_global_func("device_api.CopyDataFromTo")

# 3. 分配内存
size = 1024
cpu_ptr = alloc(cpu, size, 64)
gpu_ptr = alloc(gpu, size, 64)

# 4. 初始化数据 (使用 ctypes 访问 CPU 内存)
# 假设我们写入一些 float 数据
FLOAT_SIZE = 4
num_floats = size // FLOAT_SIZE
# 将 int 地址转换为 ctypes 指针
c_float_p = ctypes.POINTER(ctypes.c_float)
cpu_buffer = ctypes.cast(cpu_ptr, c_float_p)

for i in range(num_floats):
    cpu_buffer[i] = float(i)

# 5. 拷贝 Host -> Device
copy(cpu, cpu_ptr, gpu, gpu_ptr, size)

# 6. 拷贝 Device -> Host (验证)
# 先清空 CPU buffer 验证是否真的拷贝回来了
for i in range(num_floats):
    cpu_buffer[i] = 0.0

copy(gpu, gpu_ptr, cpu, cpu_ptr, size)

# 验证
print(f"Value at index 5: {cpu_buffer[5]}") # 应该输出 5.0

# 7. 释放内存
free(gpu, gpu_ptr)
free(cpu, cpu_ptr)
```

## 5. 扩展指南

### 如何添加新的 C++ 类并暴露给 Python?

1.  **定义类**: 继承自 `kxc::Object`，实现 `GetTypeId`。
2.  **编写绑定**: 在 `py_bindings.cpp` 中使用 `py::class_` 绑定。
3.  **注册工厂函数**: 编写一个返回 `ObjectRef` 的 C++ 函数，并使用 `KXC_REGISTER_GLOBAL` 注册。
4.  **Python 使用**: 通过 `get_global_func` 获取工厂函数创建实例。

### 如何添加新的设备后端?

1.  **继承 `DeviceAPI`**: 实现 `AllocDataSpace`, `FreeDataSpace`, `CopyDataFromTo` 等纯虚函数。
2.  **注册 API**: 在 `DeviceAPIManager::GetAPI` 中添加新的 `DeviceTypeCode` 分支，返回新的 API 单例。
3.  **编译**: 将新的 `.cc` 文件加入构建系统（如 `build_pybind.bat`）。
