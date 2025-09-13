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

# Use CONDA_PREFIX to dynamically locate the LibTorch path (like m4/inference)
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
    message(FATAL_ERROR "CONDA_PREFIX is not set. Please activate your conda environment.")
endif()

# Adjust the Python version in the path if necessary
set(CMAKE_PREFIX_PATH "${CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake/Torch")

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