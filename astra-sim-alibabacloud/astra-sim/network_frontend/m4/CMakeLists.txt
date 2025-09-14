# CMake requirement
cmake_minimum_required(VERSION 3.15)

# Project name and settings
project(SimAI_m4)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for maximum performance (matching inference CMakeLists.txt)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

# Prefer a standalone LibTorch (CUDA + cuDNN) if provided, otherwise fall back to conda Torch
if(DEFINED ENV{LIBTORCH_DIR})
    message(STATUS "Using LibTorch from LIBTORCH_DIR: $ENV{LIBTORCH_DIR}")
    set(CMAKE_PREFIX_PATH "$ENV{LIBTORCH_DIR}/share/cmake/Torch")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "${CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake/Torch")
    message(STATUS "Using Torch from conda: ${CMAKE_PREFIX_PATH}")
else()
    message(FATAL_ERROR "Neither LIBTORCH_DIR nor CONDA_PREFIX is set. Please set LIBTORCH_DIR to a CUDA+cuDNN LibTorch, or activate conda.")
endif()

# Find the Torch package
find_package(Torch REQUIRED)

# Find source files
file(GLOB SOURCES "*.cc" "*.cpp")
file(GLOB HEADERS "*.h")

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/../../../")
include_directories("${PROJECT_SOURCE_DIR}/../../system/")
include_directories("${PROJECT_SOURCE_DIR}/../../")
include_directories(${TORCH_INCLUDE_DIRS})

# Add RapidYAML subdirectory
add_subdirectory(rapidyaml)

# Create executable
add_executable(SimAI_m4 ${SOURCES} ${HEADERS})

# Let PyTorch's ABI settings take precedence (PyTorch uses old ABI)
# Remove ABI override to avoid mismatch

# Ensure linker doesn't drop static libs and place AstraSim last for symbol resolution
target_link_options(SimAI_m4 PRIVATE -Wl,--no-as-needed)
# Link libraries: Torch, ryml, then AstraSim (AstraSim last!)
target_link_libraries(SimAI_m4 "${TORCH_LIBRARIES}" ryml AstraSim)

# Set C++ standard
set_property(TARGET SimAI_m4 PROPERTY CXX_STANDARD 17)