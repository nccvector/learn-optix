cmake_minimum_required(VERSION 3.0.0)

# Specify project and languages (notice: CUDA)
project(cwc LANGUAGES CXX CUDA)


# Set CUDA architecture
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "50")

include_directories(
    "${CUDAToolkit_INCLUDE_DIRS}"
)


# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/../CMake
    ${CMAKE_MODULE_PATH}
)

# IMPORTANT: Must set(OptiX_INSTALL_DIR "<optix-install-dir>")
find_package(OptiX REQUIRED)
include_directories(
    "${OptiX_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../ext"
)

add_library(programs OBJECT solid_color.cu)
set_property(TARGET programs PROPERTY CUDA_OPTIX_COMPILATION ON)

configure_file(
    "${CMAKE_SOURCE_DIR}/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
    @ONLY
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Add cpp files to target
add_executable(cwc
    main.cpp
)

target_link_libraries(cwc
    ${GLFW_LIB_NAME}
    CUDA::cudart
)

