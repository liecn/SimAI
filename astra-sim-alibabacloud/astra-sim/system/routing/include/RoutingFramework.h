/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ROUTINGFRAMEWORK_H__
#define __ROUTINGFRAMEWORK_H__

#include "TopologyParser.h"
#include "FlowKey.h"
#include <vector>
#include <unordered_map>
#include <map>

namespace AstraSim {

class RoutingFramework {
public:
    RoutingFramework();
    
    bool ParseTopology(const std::string& topology_file);
    
    int GetOutInterface(int src_node, int dst_node, uint32_t src_ip, uint32_t dst_ip, 
                       uint8_t protocol, uint16_t src_port, uint16_t dst_port, uint32_t seed = 0);
    
    int GetOutInterface(const FlowKey& flow_key, uint32_t seed = 0);
    
    std::vector<int> GetNextHopInterfaces(int src_node, int dst_node);
    
    void PrecalculateRoutingTables();
    
    std::vector<int> GetPrecalculatedNextHops(int src_node, int dst_node);
    
    const TopologyParser& GetTopology() const { return topology_; }
    
    bool IsTopologyLoaded() const { return topology_.GetNodeCount() > 0; }
    
    const std::unordered_map<uint32_t, std::vector<int>>& GetRoutingTable(int node_id) const;

private:
    TopologyParser topology_;
    std::map<int, std::map<int, std::vector<int>>> next_hop_tables_;
    std::map<int, std::unordered_map<uint32_t, std::vector<int>>> routing_tables_;
    uint32_t ecmp_seed_;
    
    void CalculateRouteNS3Style(int host_node);
    void BuildRoutingTablesFromNextHop();
    int GetInterfaceIndex(int from_node, int to_node) const;
    std::vector<int> FindPath(int src, int dst);
    
    uint32_t EcmpHash(const uint8_t* key, size_t len, uint32_t seed);
    FlowKey CreateFlowKey(int src_node, int dst_node, uint8_t protocol = 0x11, 
                         uint16_t src_port = 10006, uint16_t dst_port = 100) const;
};

} // namespace AstraSim

#endif // __ROUTINGFRAMEWORK_H__ 