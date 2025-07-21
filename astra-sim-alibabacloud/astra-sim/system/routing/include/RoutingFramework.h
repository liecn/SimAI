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
    
    std::vector<int> GetPrecalculatedNextHops(int src_node, int dst_node) const;
    
    const TopologyParser& GetTopology() const { return topology_; }
    
    /**
     * Get the number of pre-calculated flow paths
     * @return Number of flow paths in the map
     */
    size_t GetFlowPathCount() const;

    /**
     * Check if topology is loaded
     * @return True if topology is loaded
     */
    bool IsTopologyLoaded() const;
    
    // Get routing table for a specific node
    const std::unordered_map<uint32_t, std::vector<int>>& GetRoutingTable(int node_id) const;
    
    // Pre-calculate flow paths with path tracing and ECMP selection (used by NS3)
    std::vector<std::pair<FlowKey, int>> PrecalculateFlowPathsWithTracing(int node_count,
                                                                          const std::function<uint32_t(int)>& node_id_to_ip,
                                                                          const std::function<int(int,int)>& get_node_type,
                                                                          uint16_t src_port = 10006,
                                                                          uint16_t dst_port = 100,
                                                                          uint8_t protocol = 0x11);
    
    /**
     * Get pre-calculated path for FlowSim (returns list of node IDs)
     * @param flow_key The flow key identifying the flow
     * @return List of node IDs representing the path from source to destination
     */
    std::vector<int> GetFlowSimPath(const FlowKey& flow_key) const;
    
    /**
     * Pre-calculate flow paths for FlowSim backend
     * @param topology_file Path to topology file
     * @param network_config_file Path to network configuration file
     * @return True if successful, false otherwise
     */
    bool PrecalculateFlowPathsForFlowSim(const std::string& topology_file, 
                                        const std::string& network_config_file);

private:
    TopologyParser topology_;
    std::unordered_map<FlowKey, uint32_t, FlowKeyHash> flow_to_path_map_;
    std::map<int, std::map<int, std::vector<int>>> next_hop_tables_;
    std::map<int, std::unordered_map<uint32_t, std::vector<int>>> routing_tables_;
    uint32_t ecmp_seed_;
    
    void CalculateRouteNS3Style(int host_node);
    void BuildRoutingTablesFromNextHop();
    int GetInterfaceIndex(int from_node, int to_node) const;
    std::vector<int> FindPath(int src, int dst);
    
    // Get next node from interface index (used by NS3)
    int GetNextNodeFromInterface(int from_node, int interface) const;
    
    uint32_t EcmpHash(const uint8_t* key, size_t len, uint32_t seed) const;
};

} // namespace AstraSim

#endif // __ROUTINGFRAMEWORK_H__ 