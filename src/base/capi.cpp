#include "capi.hpp"
#include <cstdio>

namespace kxcomp {


// file_op类的实现
std::string file_op::read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";  // 文件不存在或无法打开时返回空字符串
    }
    
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    return content;
}

void file_op::write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << content;
    }
}

void file_op::delete_file(const std::string& path) {
    std::remove(path.c_str());
}

} // namespace kxcomp
