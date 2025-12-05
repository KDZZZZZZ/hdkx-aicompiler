@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 /MD /std:c++17 /EHsc /utf-8 ^
   /I..\include ^
   /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" ^
   generated_network.cpp ^
   ..\src\base\op.cc ^
   ..\src\base\device_api.cc ^
   ..\src\base\device\cpu_device_api.cc ^
   ..\src\base\device\cuda_device_api.cc ^
   ..\src\relay\op\tensor\math.cc ^
   ..\src\relay\op\tensor\reduce.cc ^
   ..\src\relay\op\tensor\transform.cc ^
   ..\src\relay\op\nn\softmax.cc ^
   ..\src\relay\common_ops.cc ^
   /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" cudart.lib ^
   /OUT:generated_network.exe

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build success!
generated_network.exe
