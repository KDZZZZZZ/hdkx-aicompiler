@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 /MD /std:c++17 /EHsc /utf-8 ^
   /I..\include ^
   /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include" ^
   test_relay_op_ffi.cpp ^
   ..\src\base\device_api.cc ^
   ..\src\base\device\cpu_device_api.cc ^
   ..\src\base\device\cuda_device_api.cc ^
   ..\src\base\op.cc ^
   ..\src\relay\op\op_ffi.cc ^
   ..\src\relay\op\tensor\math.cc ^
   ..\src\relay\op\tensor\reduce.cc ^
   ..\src\relay\op\tensor\transform.cc ^
   ..\src\relay\op\nn\convolution.cc ^
   ..\src\relay\op\nn\pooling.cc ^
   ..\src\relay\op\nn\softmax.cc ^
   ..\src\relay\op\nn\activation.cc ^
   ..\src\relay\op\nn\dense.cc ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64" cudart.lib ^
   /OUT:test_relay_ffi.exe
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build success!
test_relay_ffi.exe
