# CMake requirement
cmake_minimum_required(VERSION 3.15)

# Project name and settings
project(SimAI_m4)

# Find source files
file(GLOB SOURCES "*.cc")
file(GLOB HEADERS "*.h")

# Include directories (restrict to AstraSim system; avoid flowsim-specific headers)
include_directories("${PROJECT_SOURCE_DIR}/../../../")
include_directories("${PROJECT_SOURCE_DIR}/../../system/")
include_directories("${PROJECT_SOURCE_DIR}/../../")

# Create executable
add_executable(SimAI_m4 ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(SimAI_m4 AstraSim)

# Set C++ standard
set_property(TARGET SimAI_m4 PROPERTY CXX_STANDARD 17)


