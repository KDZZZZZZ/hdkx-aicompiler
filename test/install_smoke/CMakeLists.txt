cmake_minimum_required(VERSION 3.23)
project(kxc_install_smoke LANGUAGES CXX)

find_package(kxc CONFIG REQUIRED)

add_executable(kxc_install_smoke main.cpp)
target_compile_features(kxc_install_smoke PRIVATE cxx_std_17)
target_link_libraries(kxc_install_smoke PRIVATE kxc::runtime)
