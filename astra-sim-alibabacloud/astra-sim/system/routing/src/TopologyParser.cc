/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "../include/TopologyParser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <map>

namespace AstraSim {

bool TopologyParser::ParseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open topology file: " << filename << std::endl;
        node_count_ = 0;
        node_types_.clear();
        graph_.clear();
        link_info_.clear();
        interface_map_.clear();
        server_addresses_.clear();
        return false;
    }
    
    int node_num, gpus_per_server, nvswitch_num, switch_num, link_num;
    std::string gpu_type_str;
    file >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str;
    
    if (node_num <= 0 || nvswitch_num < 0 || switch_num < 0 || link_num < 0) {
        std::cerr << "Error: Invalid topology file format" << std::endl;
        node_count_ = 0;
        node_types_.clear();
        graph_.clear();
        link_info_.clear();
        interface_map_.clear();
        server_addresses_.clear();
        return false;
    }
    
    node_count_ = node_num;
    
    node_types_.resize(node_num, 0);
    
    for (int i = 0; i < nvswitch_num; i++) {
        int sid;
        file >> sid;
        if (sid >= 0 && sid < node_num) {
            node_types_[sid] = 2;
        }
    }
    
    for (int i = 0; i < switch_num; i++) {
        int sid;
        file >> sid;
        if (sid >= 0 && sid < node_num) {
            node_types_[sid] = 1;
        }
    }
    
    graph_.resize(node_num);
    
    std::map<int, int> node_interface_count;
    for (int i = 0; i < node_num; i++) {
        node_interface_count[i] = 0;
    }
    
    for (int i = 0; i < link_num; i++) {
        int src, dst;
        std::string data_rate, link_delay;
        double error_rate;
        
        file >> src >> dst >> data_rate >> link_delay >> error_rate;
        
        if (src >= 0 && src < node_num && dst >= 0 && dst < node_num) {
            graph_[src].push_back(dst);
            graph_[dst].push_back(src);
            
            uint64_t bandwidth = ParseBandwidth(data_rate);
            uint64_t delay = ParseDelay(link_delay);
            
            LinkInfo link_info;
            link_info.bandwidth = bandwidth;
            link_info.delay = delay;
            link_info.error_rate = error_rate;
            
            link_info_[src][dst] = link_info;
            link_info_[dst][src] = link_info;
            
            interface_map_[src][dst] = node_interface_count[src] + 1;
            interface_map_[dst][src] = node_interface_count[dst] + 1;
            
            node_interface_count[src]++;
            node_interface_count[dst]++;
        }
    }
    
    server_addresses_.resize(node_num);
    for (int i = 0; i < node_num; i++) {
        if (IsHost(i)) {
            server_addresses_[i] = NodeIdToIp(i);
        }
    }
    
    return true;
}

uint32_t TopologyParser::NodeIdToIp(int node_id) const {
    if (node_id < 0 || node_id >= 65536) {
        return 0;
    }
    
    uint32_t x = (node_id >> 8) & 0xFF;
    uint32_t y = node_id & 0xFF;
    
    return (10 << 24) | (x << 16) | (y << 8) | 1;
}

uint64_t TopologyParser::ParseBandwidth(const std::string& data_rate) {
    std::string rate = data_rate;
    uint64_t multiplier = 1;
    
    std::transform(rate.begin(), rate.end(), rate.begin(), ::tolower);
    
    if (rate.find("gbps") != std::string::npos) {
        multiplier = 1000000000ULL;
        rate = rate.substr(0, rate.find("gbps"));
    } else if (rate.find("mbps") != std::string::npos) {
        multiplier = 1000000ULL;
        rate = rate.substr(0, rate.find("mbps"));
    } else if (rate.find("kbps") != std::string::npos) {
        multiplier = 1000ULL;
        rate = rate.substr(0, rate.find("kbps"));
    } else if (rate.find("bps") != std::string::npos) {
        multiplier = 1ULL;
        rate = rate.substr(0, rate.find("bps"));
    }
    
    double value = std::stod(rate);
    return static_cast<uint64_t>(value * multiplier);
}

uint64_t TopologyParser::ParseDelay(const std::string& link_delay) {
    std::string delay = link_delay;
    uint64_t multiplier = 1;
    
    std::transform(delay.begin(), delay.end(), delay.begin(), ::tolower);
    
    if (delay.find("ms") != std::string::npos) {
        multiplier = 1000000ULL;
        delay = delay.substr(0, delay.find("ms"));
    } else if (delay.find("us") != std::string::npos) {
        multiplier = 1000ULL;
        delay = delay.substr(0, delay.find("us"));
    } else if (delay.find("ns") != std::string::npos) {
        multiplier = 1ULL;
        delay = delay.substr(0, delay.find("ns"));
    }
    
    double value = std::stod(delay);
    return static_cast<uint64_t>(value * multiplier);
}

} // namespace AstraSim 