/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "../include/RoutingFramework.h"
#include <iostream>
#include <fstream>
#include <cassert>

using namespace AstraSim;

void TestNS3Compatibility(const std::string& topology_file) {
    std::cout << "Testing NS3-compatible routing with: " << topology_file << std::endl;
    
    RoutingFramework routing;
    bool success = routing.ParseTopology(topology_file);
    
    if (!success) {
        std::cerr << "Failed to parse topology file" << std::endl;
        return;
    }
    
    assert(routing.IsTopologyLoaded() && "Topology not loaded");
    std::cout << "✓ Topology loaded: " << routing.GetTopology().GetNodeCount() << " nodes" << std::endl;
    
    // Pre-calculate routing tables (NS3-style)
    routing.PrecalculateRoutingTables();
    std::cout << "✓ Routing tables pre-calculated" << std::endl;
    
    const auto& topology = routing.GetTopology();
    
    // Reasonable host pairs based on Spectrum-X topology
    std::vector<std::pair<int, int>> test_pairs = {
        {0, 7},    // Same NVSwitch group
        {0, 8},    // Different NVSwitch group
        {0, 64},   // Different switch group
        {0, 127},  // Far apart
        {8, 15},   // Same NVSwitch group
        {16, 24},  // Same NVSwitch group
        {120, 127} // Same NVSwitch group
    };
    
    std::cout << "\nTesting routing (host pairs):" << std::endl;
    for (const auto& pair : test_pairs) {
        int src = pair.first;
        int dst = pair.second;
        
        uint32_t src_ip = topology.NodeIdToIp(src);
        uint32_t dst_ip = topology.NodeIdToIp(dst);
        
        // Get next-hop interfaces
        std::vector<int> next_hops = routing.GetPrecalculatedNextHops(src, dst);
        std::cout << "  Host " << src << " -> Host " << dst << ": ";
        if (next_hops.empty()) {
            std::cout << "NO PATH" << std::endl;
        } else {
            std::cout << "Next-hop interfaces: ";
            for (int iface : next_hops) {
                std::cout << iface << " ";
            }
            std::cout << "| Selected: ";
            int interface = routing.GetOutInterface(src, dst, src_ip, dst_ip, 0x11, 10006, 100);
            std::cout << interface << std::endl;
        }
    }
    
    // Test routing consistency (same flow should get same interface)
    int src = 0, dst = 8;
    uint32_t src_ip = topology.NodeIdToIp(src);
    uint32_t dst_ip = topology.NodeIdToIp(dst);
    
    int interface1 = routing.GetOutInterface(src, dst, src_ip, dst_ip, 0x11, 10006, 100);
    int interface2 = routing.GetOutInterface(src, dst, src_ip, dst_ip, 0x11, 10006, 100);
    int interface3 = routing.GetOutInterface(src, dst, src_ip, dst_ip, 0x11, 10006, 100);
    
    assert(interface1 == interface2 && interface2 == interface3 && "Same flow should get same interface");
    std::cout << "✓ Routing consistency verified" << std::endl;
    
    std::cout << "✓ NS3 compatibility test passed!" << std::endl;
}

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <topology_file>" << std::endl;
    std::cout << "Example: " << program_name << " Spectrum-X_128g_8gps_100Gbps_A100" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Error: Missing topology file argument" << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
    
    std::string topology_file = argv[1];
    
    // Check if file exists
    std::ifstream file_check(topology_file);
    if (!file_check.good()) {
        std::cerr << "Error: Cannot open topology file: " << topology_file << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
    file_check.close();
    
    try {
        TestNS3Compatibility(topology_file);
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 