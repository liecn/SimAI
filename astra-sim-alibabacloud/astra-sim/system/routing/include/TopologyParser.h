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
    bool ParseFile(const std::string& filename);
    
    const std::vector<std::vector<int>>& GetGraph() const { return graph_; }
    
    const std::vector<int>& GetNodeTypes() const { return node_types_; }
    
    int GetNodeCount() const { return node_count_; }
    
    const std::unordered_map<int, std::unordered_map<int, int>>& GetInterfaceMap() const { 
        return interface_map_; 
    }
    
    struct LinkInfo {
        uint64_t bandwidth;
        uint64_t delay;
        double error_rate;
    };
    
    const std::unordered_map<int, std::unordered_map<int, LinkInfo>>& GetLinkInfo() const {
        return link_info_;
    }
    
    const std::vector<uint32_t>& GetServerAddresses() const { return server_addresses_; }
    
    uint32_t NodeIdToIp(int node_id) const;
    
    bool IsHost(int node_id) const { 
        return node_id < node_types_.size() && node_types_[node_id] == 0; 
    }
    
    bool IsSwitch(int node_id) const { 
        return node_id < node_types_.size() && node_types_[node_id] == 1; 
    }
    
    bool IsNVSwitch(int node_id) const { 
        return node_id < node_types_.size() && node_types_[node_id] == 2; 
    }
    
    int GetNodeType(int node_id) const {
        return (node_id < node_types_.size()) ? node_types_[node_id] : -1;
    }

private:
    std::vector<std::vector<int>> graph_;
    std::vector<int> node_types_;
    int node_count_;
    std::unordered_map<int, std::unordered_map<int, int>> interface_map_;
    std::unordered_map<int, std::unordered_map<int, LinkInfo>> link_info_;
    std::vector<uint32_t> server_addresses_;
    
    uint64_t ParseBandwidth(const std::string& data_rate);
    uint64_t ParseDelay(const std::string& link_delay);
};

} // namespace AstraSim

#endif // __TOPOLOGYPARSER_H__ 