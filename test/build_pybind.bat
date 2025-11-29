@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
echo Building kxc_runtime.pyd using MSVC...
cl /O2 /LD /MD /std:c++17 /EHsc /utf-8 ^
   /I..\include ^
   /ID:\Anaconda\Include ^
   /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" ^
   /ID:\Anaconda\Lib\site-packages\pybind11\include ^
   py_bindings.cpp ^
   ..\src\base\device_api.cc ^
   ..\src\base\device\cpu_device_api.cc ^
   ..\src\base\device\cuda_device_api.cc ^
   /link /LIBPATH:D:\Anaconda\libs python312.lib ^
   /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" cudart.lib ^
   /OUT:kxc_runtime.pyd
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build success! kxc_runtime.pyd created.
