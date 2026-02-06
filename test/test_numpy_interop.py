import sys
import os
import numpy as np

# Add current directory to sys.path to find kxc_runtime.pyd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import kxc_runtime
except ImportError:
    # Try adding the directory where build_python_ext.bat ran (which is test dir)
    # If we are running from root, it might be in test/
    sys.path.append(os.path.join(os.getcwd(), 'test'))
    try:
        import kxc_runtime
    except ImportError as e:
        print(f"Failed to import kxc_runtime: {e}")
        print(f"sys.path: {sys.path}")
        sys.exit(1)

def test_ndarray_interop():
    print("Testing NDArray interop...")
    
    # Create NDArray
    shape = [2, 3]
    dtype = "float32"
    arr = kxc_runtime.NDArray(shape, dtype)
    
    print(f"Created NDArray with shape {shape} and dtype {dtype}")
    
    # Convert to numpy (zero copy)
    # Note: np.array(obj, copy=False) might still copy if interface doesn't support writeable buffer or other reasons.
    # But with buffer protocol it should work.
    np_arr = np.array(arr, copy=False)
    
    print(f"Converted to numpy array:\n{np_arr}")
    print(f"Numpy shape: {np_arr.shape}")
    print(f"Numpy dtype: {np_arr.dtype}")
    
    # Verify values are initialized to 0
    assert np.all(np_arr == 0), "Initial values should be 0"
    
    # Modify numpy array
    print("Modifying numpy array element [0, 0] to 123.0")
    np_arr[0, 0] = 123.0
    
    # Check if we can read it back via another numpy view (since we don't have C++ accessor exposed yet)
    # But np.array(arr, copy=False) creates a view on the same memory.
    # So if I create another view, it should see the change.
    np_arr2 = np.array(arr, copy=False)
    print(f"Read back via new numpy view:\n{np_arr2}")
    
    assert np_arr2[0, 0] == 123.0, "Modification not reflected in new view"
    
    # Verify memory sharing
    # If copy=False worked, they should share memory interface.
    if np_arr.base is not None:
         print("Numpy array has a base, likely sharing memory.")
    else:
         print("Warning: Numpy array base is None, might be a copy (or it owns the memory?).")
         # If it's a view of C++ object, base should be the C++ object wrapper.
    
    print("Test Passed!")

if __name__ == "__main__":
    try:
        test_ndarray_interop()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
