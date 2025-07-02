@echo off
echo 编译文件操作和类型注册系统测试...

REM 检查是否有cl编译器（Visual Studio）
where cl >nul 2>nul
if %errorlevel% equ 0 (
    echo 使用 Visual Studio 编译器...
    echo 编译主测试...
    cl /std:c++17 /EHsc /I..\include ..\src\base\capi.cpp test_file_op.cpp example_types.cpp /Fe:test_file_op.exe
    if %errorlevel% equ 0 (
        echo 编译结构验证...
        cl /std:c++17 /EHsc /I..\include ..\src\base\capi.cpp verify_structure.cpp /Fe:verify_structure.exe
        if %errorlevel% equ 0 (
            echo 编译成功！
            echo.
            echo 运行主测试...
            test_file_op.exe
            echo.
            echo 运行结构验证...
            verify_structure.exe
        ) else (
            echo 结构验证编译失败！
        )
    ) else (
        echo 主测试编译失败！
    )
    goto :end
)

REM 检查是否有g++编译器（MinGW）
where g++ >nul 2>nul
if %errorlevel% equ 0 (
    echo 使用 MinGW 编译器...
    echo 编译主测试...
    g++ -std=c++17 -Wall -Wextra -I../include ../src/base/capi.cpp test_file_op.cpp example_types.cpp -o test_file_op.exe
    if %errorlevel% equ 0 (
        echo 编译结构验证...
        g++ -std=c++17 -Wall -Wextra -I../include ../src/base/capi.cpp verify_structure.cpp -o verify_structure.exe
        if %errorlevel% equ 0 (
            echo 编译成功！
            echo.
            echo 运行主测试...
            test_file_op.exe
            echo.
            echo 运行结构验证...
            verify_structure.exe
        ) else (
            echo 结构验证编译失败！
        )
    ) else (
        echo 主测试编译失败！
    )
    goto :end
)

REM 检查是否有clang++编译器
where clang++ >nul 2>nul
if %errorlevel% equ 0 (
    echo 使用 Clang 编译器...
    echo 编译主测试...
    clang++ -std=c++17 -Wall -Wextra -I../include ../src/base/capi.cpp test_file_op.cpp example_types.cpp -o test_file_op.exe
    if %errorlevel% equ 0 (
        echo 编译结构验证...
        clang++ -std=c++17 -Wall -Wextra -I../include ../src/base/capi.cpp verify_structure.cpp -o verify_structure.exe
        if %errorlevel% equ 0 (
            echo 编译成功！
            echo.
            echo 运行主测试...
            test_file_op.exe
            echo.
            echo 运行结构验证...
            verify_structure.exe
        ) else (
            echo 结构验证编译失败！
        )
    ) else (
        echo 主测试编译失败！
    )
    goto :end
)

echo 错误：未找到可用的C++编译器！
echo 请安装以下编译器之一：
echo - Visual Studio (包含cl编译器)
echo - MinGW-w64 (包含g++编译器)
echo - Clang (包含clang++编译器)

:end
pause 