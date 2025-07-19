# CMake requirement
cmake_minimum_required(VERSION 3.15)

# Project name and settings
project(SimAI_flowsim)

# Find source files
file(GLOB SOURCES "*.cc")
file(GLOB HEADERS "*.h")

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/../../../")
include_directories("${PROJECT_SOURCE_DIR}/../../system/")
include_directories("${PROJECT_SOURCE_DIR}/../../")

# Create executable
add_executable(SimAI_flowsim ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(SimAI_flowsim AstraSim)

# Set C++ standard
set_property(TARGET SimAI_flowsim PROPERTY CXX_STANDARD 17) 