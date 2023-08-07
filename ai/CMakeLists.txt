cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(cpp_tic_tac_toe CXX)

set(CMAKE_PREFIX_PATH "~/libtorch")
set(THREADS_PREFER_PTHREAD_FLAG ON)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# Find Package
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native")

# Create Executable File
add_executable(${PROJECT_NAME} main.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} "-pthread")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
