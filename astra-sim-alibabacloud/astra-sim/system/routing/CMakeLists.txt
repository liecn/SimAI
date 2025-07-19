# Routing Framework CMakeLists.txt

# Set minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set project name
project(RoutingFramework)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(ROUTING_SOURCES
    src/TopologyParser.cc
    src/RoutingFramework.cc
)

# Create routing library
add_library(routing STATIC ${ROUTING_SOURCES})

# Set library properties
set_target_properties(routing PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Topology routing test executable
add_executable(test_topology_routing test/test_spectrum_x.cpp)
target_link_libraries(test_topology_routing routing)

# Enable testing
enable_testing()
add_test(NAME TopologyRoutingTest COMMAND test_topology_routing)

# Installation (optional)
install(TARGETS routing
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/routing
    FILES_MATCHING PATTERN "*.h"
) 