cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(cpp_tic_tac_toe CXX)

set(CMAKE_PREFIX_PATH "~/libtorch")

# Find Package
find_package(Torch REQUIRED)


# Create Executable File
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)