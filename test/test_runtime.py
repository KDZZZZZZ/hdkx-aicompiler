import kxc_runtime
import sys

print("Successfully imported kxc_runtime")

# Test 1: ObjectRef
obj_ref = kxc_runtime.ObjectRef()
print("ObjectRef created")

# Test 2: MyObj
my_obj = kxc_runtime.MyObj()
print(f"MyObj created. Greet: {my_obj.greet()}")
my_obj.name = "PythonUser"
print(f"MyObj updated. Greet: {my_obj.greet()}")

# Test 3: Global Functions
try:
    add_func = kxc_runtime.get_global_func("add")
    result = add_func(10, 20)
    print(f"Call add(10, 20) = {result}")
    assert result == 30
except Exception as e:
    print(f"Test add failed: {e}")

try:
    greet_func = kxc_runtime.get_global_func("greet")
    msg = greet_func("World")
    print(f"Call greet('World') = {msg}")
    assert msg == "Hello, World"
except Exception as e:
    print(f"Test greet failed: {e}")

# Test 4: Object Passing
try:
    process_func = kxc_runtime.get_global_func("process_obj")
    print("Testing process_obj...")
    # Pass my_obj directly. pybind11 should handle the conversion to ObjectRef via our custom type_caster
    returned_obj = process_func(my_obj)
    
    print(f"Call process_obj success. Returned type: {type(returned_obj)}")
    
    # Check if we got back the correct object
    # Since process_obj returns the input object, and our type_caster::cast converts it back to Python object
    if hasattr(returned_obj, "name"):
         print(f"Returned object name: {returned_obj.name}")
         assert returned_obj.name == "PythonUser"
    else:
         print("Returned object does not have 'name' attribute (might be raw ObjectRef wrapper?)")

except Exception as e:
    print(f"Test process_obj failed: {e}")

def log_debug(msg):
    with open("debug_log.txt", "a") as f:
        f.write(msg + "\n")

# Test 5: Device
try:
    print("Testing Device...")
    log_debug("Testing Device...")
    # Test Enum
    cpu_type = kxc_runtime.DeviceTypeCode.CPU
    gpu_type = kxc_runtime.DeviceTypeCode.GPU
    print(f"Device Types: CPU={int(cpu_type)}, GPU={int(gpu_type)}")
    
    # Test Device Creation via Constructor
    dev1 = kxc_runtime.Device(cpu_type, 0)
    print(f"Created Device via constructor: {dev1}")
    log_debug(f"Created Device via constructor: {dev1}")
    sys.stdout.flush()
    assert dev1.device_type == cpu_type
    assert dev1.device_id == 0
    print("Device constructor check passed")
    sys.stdout.flush()
    
    # Test Device Creation via Factory Function (should return ObjectRef wrapping Device)
    print("Calling kxc_runtime.device(gpu_type, 1)...")
    log_debug("Calling kxc_runtime.device(gpu_type, 1)...")
    sys.stdout.flush()
    dev2 = kxc_runtime.device(gpu_type, 1)
    print("Returned from kxc_runtime.device")
    log_debug("Returned from kxc_runtime.device")
    sys.stdout.flush()
    
    # Check type first
    print(f"Type of dev2: {type(dev2)}")
    log_debug(f"Type of dev2: {type(dev2)}")
    sys.stdout.flush()
    
    try:
        print(f"Representation of dev2: {dev2}")
        log_debug(f"Representation of dev2: {dev2}")
    except Exception as e:
        print(f"Could not print dev2: {e}")
        log_debug(f"Could not print dev2: {e}")
    sys.stdout.flush()

    print("Checking dev2.device_type...")
    sys.stdout.flush()
    try:
        # Ensure it is a Device object before accessing attributes
        if not isinstance(dev2, kxc_runtime.Device):
             print(f"Warning: dev2 is {type(dev2)}, not kxc_runtime.Device")
        
        dt = dev2.device_type
        print(f"dev2.device_type = {dt}")
        log_debug(f"dev2.device_type = {dt}")
    except AttributeError as e:
        print(f"AttributeError accessing device_type: {e}")
        log_debug(f"AttributeError accessing device_type: {e}")
        # If it's bytes, let's see what it contains
        if isinstance(dev2, bytes):
             print(f"dev2 is bytes: {dev2}")
             log_debug(f"dev2 is bytes: {dev2}")
        raise e
    sys.stdout.flush()

    # print(f"Created Device via factory: {dev2}")
    # sys.stdout.flush()
    # Note: If dev2 is strictly ObjectRef in C++, pybind cast should convert it to Python Device object 
    # if the underlying pointer is a Device instance and RTTI works / type matches.
    
    assert dev2.device_type == gpu_type
    print("dev2.device_type matches gpu_type")
    sys.stdout.flush()
    assert dev2.device_id == 1
    print("dev2.device_id matches 1")
    sys.stdout.flush()
    
    print("Device tests passed!")

    # Test 6: DeviceAPI
    print("Testing DeviceAPI...")
    alloc_func = kxc_runtime.get_global_func("device_api.AllocDataSpace")
    free_func = kxc_runtime.get_global_func("device_api.FreeDataSpace")
    
    # Alloc on CPU
    nbytes = 1024
    alignment = 64
    ptr_val = alloc_func(dev1, nbytes, alignment)
    print(f"Allocated {nbytes} bytes on {dev1}, ptr={ptr_val}")
    assert ptr_val != 0
    
    # Test CUDA and CopyDataFromTo
    try:
        gpu_ptr_val = alloc_func(dev2, nbytes, alignment)
        print(f"Allocated {nbytes} bytes on {dev2}, ptr={gpu_ptr_val}")
        
        # Verify CopyDataFromTo
        import ctypes
        # Access CPU memory via ctypes
        # array of ints
        num_ints = nbytes // 4
        CpuArrayType = ctypes.c_int * num_ints
        cpu_array = CpuArrayType.from_address(ptr_val)
        
        # Initialize data
        print("Initializing CPU data...")
        for i in range(num_ints):
            cpu_array[i] = i
            
        copy_func = kxc_runtime.get_global_func("device_api.CopyDataFromTo")
        
        # Copy CPU -> GPU
        print("Copying CPU -> GPU...")
        copy_func(dev1, ptr_val, dev2, gpu_ptr_val, nbytes)
        
        # Clear CPU memory to verify copy back works
        for i in range(num_ints):
            cpu_array[i] = 0
            
        # Copy GPU -> CPU
        print("Copying GPU -> CPU...")
        copy_func(dev2, gpu_ptr_val, dev1, ptr_val, nbytes)
        
        # Verify
        print("Verifying data...")
        errors = 0
        for i in range(num_ints):
            if cpu_array[i] != i:
                errors += 1
                if errors < 5:
                    print(f"Mismatch at index {i}: expected {i}, got {cpu_array[i]}")
        
        if errors == 0:
            print("CopyDataFromTo test passed: Data verified successfully!")
        else:
            print(f"CopyDataFromTo test failed with {errors} errors.")
            
        free_func(dev2, gpu_ptr_val)
        print("Freed memory on GPU")
        
    except Exception as e:
        print(f"CUDA Alloc/Copy failed (expected if no GPU or CUDA not configured): {e}")

    # Free on CPU
    free_func(dev1, ptr_val)
    print("Freed memory on CPU")
        
    print("DeviceAPI tests passed!")

    # Test 7: Device Detection
    print("Testing Device Detection...")
    try:
        detect_func = kxc_runtime.get_global_func("device.DetectAndRegister")
        detect_func()
        
        # Try to retrieve info for device 0 (if exists)
        try:
            get_name = kxc_runtime.get_global_func("device_info.cuda.0.name")
            if get_name:
                name = get_name()
                print(f"Detected CUDA Device 0 Name: {name}")
                
                get_mem = kxc_runtime.get_global_func("device_info.cuda.0.memory")
                mem = get_mem()
                print(f"Detected CUDA Device 0 Memory: {mem / (1024*1024):.2f} MB")
            else:
                print("No CUDA device info found (maybe no GPU detected)")
        except Exception as e:
             print(f"Could not get device info: {e}")

    except Exception as e:
        print(f"Device Detection failed: {e}")

except Exception as e:
    print(f"Test Device failed: {e}")
