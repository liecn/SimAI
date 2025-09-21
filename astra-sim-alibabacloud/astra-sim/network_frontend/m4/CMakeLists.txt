cmake_minimum_required(VERSION 3.15)
project(SimAI_m4)

# C++ standard and build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG")

# ------------------------------------------------------------------
# Torch discovery: LIBTORCH_DIR (explicit) -> Python torch (uv/venv/conda/pip)
# No guessed paths, no placeholders.
# ------------------------------------------------------------------
set(_torch_cmake_dir "")

if(DEFINED ENV{LIBTORCH_DIR})
  message(STATUS "Using LibTorch from LIBTORCH_DIR: $ENV{LIBTORCH_DIR}")
  set(_torch_cmake_dir "$ENV{LIBTORCH_DIR}/share/cmake/Torch")
endif()

if(NOT _torch_cmake_dir)
  # Require Python since we're not using LIBTORCH_DIR
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import os, importlib.util as u; s=u.find_spec('torch'); \
p=os.path.dirname(s.origin) if s else ''; cm=os.path.join(p,'share','cmake','Torch'); \
print(cm if os.path.isdir(cm) else '')"
    OUTPUT_VARIABLE _torch_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _torch_cmake_dir)
    message(FATAL_ERROR
      "Unable to locate the PyTorch CMake package.\n"
      "- LIBTORCH_DIR is not set, and\n"
      "- the active Python interpreter does not have a usable 'torch' installation.\n"
      "Activate the Python environment that has PyTorch installed (e.g., uv/venv/conda) or set LIBTORCH_DIR.")
  endif()
  message(STATUS "Using Torch discovered via Python at: ${_torch_cmake_dir}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_dir}")

# Find Torch
find_package(Torch REQUIRED)
message(STATUS "Torch found. Torch_DIR='${Torch_DIR}'")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ------------------------------------------------------------------
# Sources / Headers
# ------------------------------------------------------------------
file(GLOB SOURCES "*.cc" "*.cpp")
file(GLOB HEADERS "*.h")

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/../../../")
include_directories("${PROJECT_SOURCE_DIR}/../../system/")
include_directories("${PROJECT_SOURCE_DIR}/../../")
include_directories(${TORCH_INCLUDE_DIRS})

# RapidYAML
add_subdirectory(rapidyaml)

# Target
add_executable(SimAI_m4 ${SOURCES} ${HEADERS})

# Link options & libraries (AstraSim last)
target_link_options(SimAI_m4 PRIVATE -Wl,--no-as-needed)
target_link_libraries(SimAI_m4 "${TORCH_LIBRARIES}" ryml AstraSim)

# Ensure C++17 on the target
set_property(TARGET SimAI_m4 PROPERTY CXX_STANDARD 17)
