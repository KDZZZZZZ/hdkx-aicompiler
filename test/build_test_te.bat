@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 /MD /std:c++17 /EHsc /utf-8 ^
   /I..\include ^
   test_te.cpp ^
   /Fe:test_te.exe

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)
echo Build success!
test_te.exe
