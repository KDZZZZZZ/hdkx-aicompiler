#include "capi.hpp"
namespace kxcomp{
namespace base{

class file_op{
std::string file_op::read_file(const std::string& path){
    std::ifstream file(path);
    std::string content;
    std::string line;
    while(std::getline(file, line)){
        content += line + "\n";
    }
    return content;
};
void file_op::write_file(const std::string& path, const std::string& content){
    std::ofstream file(path);
    file << content;
};
void file_op::delete_file(const std::string& path){
    std::remove(path.c_str());
}
};

}}
