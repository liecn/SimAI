/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "TopologyParser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace AstraSim {

bool TopologyParser::ParseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open topology file: " << filename << std::endl;
        // Reset state when parsing fails
        node_count_ = 0;
        node_types_.clear();
        graph_.clear();
        link_info_.clear();
        interface_map_.clear();
        server_addresses_.clear();
        return false;
    }
    
    // Parse header: node_num gpus_per_server nvswitch_num switch_num link_num gpu_type_str
    int node_num, gpus_per_server, nvswitch_num, switch_num, link_num;
    std::string gpu_type_str;
    file >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str;
    
    if (node_num <= 0 || nvswitch_num < 0 || switch_num < 0 || link_num < 0) {
        std::cerr << "Error: Invalid topology file format" << std::endl;
        // Reset state when parsing fails
        node_count_ = 0;
        node_types_.clear();
        graph_.clear();
        link_info_.clear();
        interface_map_.clear();
        server_addresses_.clear();
        return false;
    }
    
    node_count_ = node_num;
    
    // Initialize node types (all hosts by default)
    node_types_.resize(node_num, 0);
    
    // Parse NV switch nodes (type 2)
    for (int i = 0; i < nvswitch_num; i++) {
        int sid;
        file >> sid;
        if (sid >= 0 && sid < node_num) {
            node_types_[sid] = 2; // Mark as NV switch
        }
    }
    
    // Parse regular switch nodes (type 1)
    for (int i = 0; i < switch_num; i++) {
        int sid;
        file >> sid;
        if (sid >= 0 && sid < node_num) {
            node_types_[sid] = 1; // Mark as regular switch
        }
    }
    
    // Initialize graph
    graph_.resize(node_num);
    
    // Parse links
    for (int i = 0; i < link_num; i++) {
        int src, dst;
        std::string data_rate, link_delay;
        double error_rate;
        
        file >> src >> dst >> data_rate >> link_delay >> error_rate;
        
        if (src >= 0 && src < node_num && dst >= 0 && dst < node_num) {
            // Add to adjacency list
            graph_[src].push_back(dst);
            graph_[dst].push_back(src);
            
            // Parse bandwidth (convert from string like "100Gbps" to bits per second)
            uint64_t bandwidth = ParseBandwidth(data_rate);
            
            // Parse delay (convert from string like "1us" to nanoseconds)
            uint64_t delay = ParseDelay(link_delay);
            
            // Store link information
            LinkInfo link_info;
            link_info.bandwidth = bandwidth;
            link_info.delay = delay;
            link_info.error_rate = error_rate;
            
            link_info_[src][dst] = link_info;
            link_info_[dst][src] = link_info;
            
            // Assign interface numbers (simple sequential assignment)
            interface_map_[src][dst] = graph_[src].size();
            interface_map_[dst][src] = graph_[dst].size();
        }
    }
    
    // Generate server IP addresses (same as NS3 node_id_to_ip)
    server_addresses_.resize(node_num);
    for (int i = 0; i < node_num; i++) {
        if (IsHost(i)) {
            server_addresses_[i] = NodeIdToIp(i);
        }
    }
    
    std::cout << "Topology parsed successfully: " << node_num << " nodes, " 
              << nvswitch_num << " NV switches, " << switch_num << " switches, " 
              << link_num << " links, GPU type: " << gpu_type_str << std::endl;
    
    return true;
}

uint32_t TopologyParser::NodeIdToIp(int node_id) const {
    // Same logic as NS3 node_id_to_ip function
    // Convert node ID to IP address in 10.x.y.z format
    if (node_id < 0 || node_id >= 65536) {
        return 0;
    }
    
    uint32_t x = (node_id >> 8) & 0xFF;
    uint32_t y = node_id & 0xFF;
    
    return (10 << 24) | (x << 16) | (y << 8) | 1;
}

uint64_t TopologyParser::ParseBandwidth(const std::string& data_rate) {
    // Parse bandwidth strings like "100Gbps", "10Mbps", etc.
    std::string rate = data_rate;
    uint64_t multiplier = 1;
    
    // Convert to lowercase for easier parsing
    std::transform(rate.begin(), rate.end(), rate.begin(), ::tolower);
    
    if (rate.find("gbps") != std::string::npos) {
        multiplier = 1000000000ULL; // 1 Gbps = 10^9 bps
        rate = rate.substr(0, rate.find("gbps"));
    } else if (rate.find("mbps") != std::string::npos) {
        multiplier = 1000000ULL; // 1 Mbps = 10^6 bps
        rate = rate.substr(0, rate.find("mbps"));
    } else if (rate.find("kbps") != std::string::npos) {
        multiplier = 1000ULL; // 1 Kbps = 10^3 bps
        rate = rate.substr(0, rate.find("kbps"));
    } else if (rate.find("bps") != std::string::npos) {
        multiplier = 1ULL;
        rate = rate.substr(0, rate.find("bps"));
    }
    
    double value = std::stod(rate);
    return static_cast<uint64_t>(value * multiplier);
}

uint64_t TopologyParser::ParseDelay(const std::string& link_delay) {
    // Parse delay strings like "1us", "100ns", "1ms", etc.
    std::string delay = link_delay;
    uint64_t multiplier = 1;
    
    // Convert to lowercase for easier parsing
    std::transform(delay.begin(), delay.end(), delay.begin(), ::tolower);
    
    if (delay.find("ms") != std::string::npos) {
        multiplier = 1000000ULL; // 1 ms = 10^6 ns
        delay = delay.substr(0, delay.find("ms"));
    } else if (delay.find("us") != std::string::npos) {
        multiplier = 1000ULL; // 1 us = 10^3 ns
        delay = delay.substr(0, delay.find("us"));
    } else if (delay.find("ns") != std::string::npos) {
        multiplier = 1ULL;
        delay = delay.substr(0, delay.find("ns"));
    }
    
    double value = std::stod(delay);
    return static_cast<uint64_t>(value * multiplier);
}

} // namespace AstraSim 