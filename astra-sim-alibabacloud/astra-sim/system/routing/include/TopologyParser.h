/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __TOPOLOGYPARSER_H__
#define __TOPOLOGYPARSER_H__

#include <string>
#include <vector>
#include <unordered_map>

namespace AstraSim {

class TopologyParser {
public:
    // Parse NS3 topology file format
    bool ParseFile(const std::string& filename);
    
    // Get topology graph (adjacency list)
    const std::vector<std::vector<int>>& GetGraph() const { return graph_; }
    
    // Get node types (0=host, 1=switch)
    const std::vector<int>& GetNodeTypes() const { return node_types_; }
    
    // Get number of nodes
    int GetNodeCount() const { return node_count_; }
    
    // Get interface mapping (same as NS3 nbr2if)
    const std::unordered_map<int, std::unordered_map<int, int>>& GetInterfaceMap() const { 
        return interface_map_; 
    }
    
    // Get link information (bandwidth, delay)
    struct LinkInfo {
        uint64_t bandwidth;  // bits per second
        uint64_t delay;      // nanoseconds
        double error_rate;
    };
    
    const std::unordered_map<int, std::unordered_map<int, LinkInfo>>& GetLinkInfo() const {
        return link_info_;
    }
    
    // Get server IP addresses (same as NS3 serverAddress)
    const std::vector<uint32_t>& GetServerAddresses() const { return server_addresses_; }
    
    // Convert node ID to IP address (same as NS3 node_id_to_ip)
    uint32_t NodeIdToIp(int node_id) const;
    
    // Check if node is a host
    bool IsHost(int node_id) const { 
        return node_id < node_types_.size() && node_types_[node_id] == 0; 
    }
    
    // Check if node is a switch
    bool IsSwitch(int node_id) const { 
        return node_id < node_types_.size() && node_types_[node_id] == 1; 
    }
    
    // Check if node is an NV switch
    bool IsNVSwitch(int node_id) const { 
        return node_id < node_types_.size() && node_types_[node_id] == 2; 
    }
    
    // Get node type
    int GetNodeType(int node_id) const {
        return (node_id < node_types_.size()) ? node_types_[node_id] : -1;
    }

private:
    std::vector<std::vector<int>> graph_;  // Adjacency list
    std::vector<int> node_types_;          // 0=host, 1=switch
    int node_count_;
    std::unordered_map<int, std::unordered_map<int, int>> interface_map_; // src->dst->interface
    std::unordered_map<int, std::unordered_map<int, LinkInfo>> link_info_; // src->dst->link_info
    std::vector<uint32_t> server_addresses_; // IP addresses for hosts
    
    // Helper functions for parsing
    uint64_t ParseBandwidth(const std::string& data_rate);
    uint64_t ParseDelay(const std::string& link_delay);
};

} // namespace AstraSim

#endif // __TOPOLOGYPARSER_H__ 