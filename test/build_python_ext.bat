@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

set PYTHON_INCLUDE=D:\Anaconda\Include
set PYTHON_LIBS=D:\Anaconda\libs
set PYBIND11_INCLUDE=D:\Anaconda\Lib\site-packages\pybind11\include
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

cl /O2 /MD /std:c++17 /EHsc /utf-8 ^
   /I..\include ^
   /I"%PYTHON_INCLUDE%" ^
   /I"%PYBIND11_INCLUDE%" ^
   /I"%CUDA_PATH%\include" ^
   /LD ^
   ..\src\base\py_module.cc ^
   ..\src\base\op.cc ^
   ..\src\base\device_api.cc ^
   ..\src\base\device\cpu_device_api.cc ^
   ..\src\base\device\cuda_device_api.cc ^
   /link /LIBPATH:"%PYTHON_LIBS%" ^
   /LIBPATH:"%CUDA_PATH%\lib\x64" ^
   cudart_static.lib cuda.lib ^
   /OUT:kxc_runtime.pyd

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build success!
