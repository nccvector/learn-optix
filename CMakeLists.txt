cmake_minimum_required(VERSION 3.0.0)

# Specify project and languages (notice: CUDA)
project(cwc LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/CMake
        ${CMAKE_MODULE_PATH}
)

set(OptiX_INSTALL_DIR "/home/vector/optix77")
find_package(OptiX REQUIRED)

include_directories(
        "${OptiX_INCLUDE}"
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(PROGRAMS
        solid_color.cu
)

# Compile all cuda files to ptx
cuda_compile_ptx(cuda_ptx_files
        ${PROGRAMS}
)

add_custom_target(ptx ALL
        DEPENDS ${cuda_ptx_files} ${PROGRAMS}
        SOURCES ${PROGRAMS}
)

# Add cpp files to target
add_executable(cwc
        main.cpp
)

target_link_libraries(cwc
        ${GLFW_LIB_NAME}
        ${CUDA_LIBRARIES}
)

