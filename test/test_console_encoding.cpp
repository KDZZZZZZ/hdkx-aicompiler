#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <iostream>
#include <string>

int main() {
    UINT cp_before = GetConsoleOutputCP();
    std::cout << "CP(before): " << cp_before << std::endl;
    std::cout << "ASCII: ok" << std::endl;
    std::cout << u8"UTF8 narrow: 中文 测试" << std::endl;

    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    _setmode(_fileno(stdout), _O_U8TEXT);

    UINT cp_after = GetConsoleOutputCP();
    std::wcout << L"CP(after): " << cp_after << std::endl;
    std::wcout << L"UTF16 wide: 中文 测试" << std::endl;

    _setmode(_fileno(stdout), _O_TEXT);
    return 0;
}

