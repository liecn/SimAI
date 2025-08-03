/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "../include/RoutingFramework.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <queue>
#include <set>

namespace AstraSim {

RoutingFramework::RoutingFramework() : topology_() {
    ecmp_seed_ = 0;
}

bool RoutingFramework::ParseTopology(const std::string& topology_file) {
    return topology_.ParseFile(topology_file);
}

int RoutingFramework::GetOutInterface(int src_node, int dst_node, uint32_t src_ip, uint32_t dst_ip, 
                                     uint8_t protocol, uint16_t src_port, uint16_t dst_port, uint32_t seed) {
    if (!IsTopologyLoaded()) {
        return -1;
    }
    
    int node_count = topology_.GetNodeCount();
    if (src_node < 0 || src_node >= node_count || dst_node < 0 || dst_node >= node_count) {
        return -1;
    }
    
    if (src_node == dst_node) {
        return -1;
    }
    
    auto next_hops = GetPrecalculatedNextHops(src_node, dst_node);
    if (next_hops.empty()) {
        return -1;
    }
    
    if (next_hops.size() == 1) {
        return next_hops[0];
    }
    
    union {
        uint8_t u8[4+4+2+2];
        uint32_t u32[3];
    } buf;
    
    buf.u32[0] = src_ip;
    buf.u32[1] = dst_ip;
    buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
    
    uint32_t hash = EcmpHash(buf.u8, 12, seed);
    uint32_t interface_idx = hash % next_hops.size();
    
    return next_hops[interface_idx];
}

int RoutingFramework::GetOutInterface(const FlowKey& flow_key, uint32_t seed) {
    int src_node = -1, dst_node = -1;
    int node_count = topology_.GetNodeCount();
    
    for (int i = 0; i < node_count; i++) {
        if (topology_.NodeIdToIp(i) == flow_key.src_ip) {
            src_node = i;
        }
        if (topology_.NodeIdToIp(i) == flow_key.dst_ip) {
            dst_node = i;
        }
    }
    
    if (src_node == -1 || dst_node == -1) {
        return -1;
    }
    
    return GetOutInterface(src_node, dst_node, flow_key.src_ip, flow_key.dst_ip, 
                          flow_key.protocol, flow_key.src_port, flow_key.dst_port, seed);
}

std::vector<int> RoutingFramework::GetNextHopInterfaces(int src_node, int dst_node) {
    return GetPrecalculatedNextHops(src_node, dst_node);
}

void RoutingFramework::PrecalculateRoutingTables() {
    if (!IsTopologyLoaded()) {
        routing_tables_.clear();
        return;
    }
    
    int node_count = topology_.GetNodeCount();
    
    for (int src_id = 0; src_id < node_count; src_id++) {
        if (topology_.GetNodeType(src_id) == 0) {
            CalculateRouteNS3Style(src_id);
        }
    }
    
    BuildRoutingTablesFromNextHop();
}

std::vector<int> RoutingFramework::GetPrecalculatedNextHops(int src_node, int dst_node) const {
    auto it = routing_tables_.find(src_node);
    if (it == routing_tables_.end()) {
        return {};
    }
    
    uint32_t dst_ip = topology_.NodeIdToIp(dst_node);
    auto dst_it = it->second.find(dst_ip);
    if (dst_it == it->second.end()) {
        return {};
    }
    
    return dst_it->second;
}

const std::unordered_map<uint32_t, std::vector<int>>& RoutingFramework::GetRoutingTable(int node_id) const {
    static std::unordered_map<uint32_t, std::vector<int>> empty_table;
    auto it = routing_tables_.find(node_id);
    if (it == routing_tables_.end()) {
        return empty_table;
    }
    return it->second;
}

void RoutingFramework::CalculateRouteNS3Style(int host_node) {
    std::map<int, int> dis;
    std::map<int, std::vector<int>> nextHop;
    std::vector<int> q;
    q.push_back(host_node);
    dis[host_node] = 0;

    for (int i = 0; i < (int)q.size(); i++) {
        int now = q[i];
        int d = dis[now];
        const auto& neighbors = topology_.GetGraph()[now];
        
        for (int next : neighbors) {
            if (dis.find(next) == dis.end()) {
                dis[next] = d + 1;
                if (topology_.GetNodeType(next) == 1 || topology_.GetNodeType(next) == 2) {
                    q.push_back(next);
                }
            }
            
            if (d + 1 == dis[next]) {
                bool via_nvswitch = false;
                
                if (nextHop.find(next) != nextHop.end()) {
                    for (int x : nextHop[next]) {
                        if (topology_.GetNodeType(x) == 2) {
                            via_nvswitch = true;
                            break;
                        }
                    }
                }
                
                if (via_nvswitch == false) {
                    if (topology_.GetNodeType(now) == 2) {
                        nextHop[next].clear();
                    }
                    nextHop[next].push_back(now);
                } else if (via_nvswitch == true && topology_.GetNodeType(now) == 2) {
                    nextHop[next].push_back(now);
                }
                
                if (topology_.GetNodeType(next) == 0 && nextHop.find(next) == nextHop.end()) {
                    nextHop[next].push_back(now);
                }
            }
        }
    }
    
    next_hop_tables_[host_node] = nextHop;
}

void RoutingFramework::BuildRoutingTablesFromNextHop() {
    int node_count = topology_.GetNodeCount();
    
    for (const auto& host_pair : next_hop_tables_) {
        int host_node = host_pair.first;
        const auto& nextHop = host_pair.second;
        
        for (const auto& dst_pair : nextHop) {
            int dst_node = dst_pair.first;
            const auto& nexts = dst_pair.second;
            
            uint32_t dst_ip = topology_.NodeIdToIp(dst_node);
            
            for (int next : nexts) {
                int interface = GetInterfaceIndex(host_node, next);
                if (interface >= 0) {
                    routing_tables_[host_node][dst_ip].push_back(interface);
                } else {
                    std::vector<int> path = FindPath(host_node, next);
                    if (!path.empty() && path.size() > 1) {
                        int first_hop = path[1];
                        int interface = GetInterfaceIndex(host_node, first_hop);
                        if (interface >= 0) {
                            routing_tables_[host_node][dst_ip].push_back(interface);
                        }
                    }
                }
            }
        }
    }
    
    for (int src_host = 0; src_host < node_count; src_host++) {
        if (topology_.GetNodeType(src_host) != 0) continue;
        
        for (int dst_host = 0; dst_host < node_count; dst_host++) {
            if (topology_.GetNodeType(dst_host) != 0) continue;
            if (src_host == dst_host) continue;
            
            uint32_t dst_ip = topology_.NodeIdToIp(dst_host);
            auto it = routing_tables_.find(src_host);
            if (it != routing_tables_.end() && it->second.find(dst_ip) != it->second.end()) {
                continue;
            }
            
            const auto& neighbors = topology_.GetGraph()[src_host];
            bool direct_connection = false;
            for (int neighbor : neighbors) {
                if (neighbor == dst_host) {
                    direct_connection = true;
                    break;
                }
            }
            
            if (direct_connection) {
                int interface = GetInterfaceIndex(src_host, dst_host);
                if (interface >= 0) {
                    routing_tables_[src_host][dst_ip].push_back(interface);
                }
            } else {
                auto host_it = next_hop_tables_.find(src_host);
                if (host_it != next_hop_tables_.end()) {
                    auto dst_it = host_it->second.find(dst_host);
                    if (dst_it != host_it->second.end()) {
                        const auto& next_hops = dst_it->second;
                        for (int next_hop : next_hops) {
                            int interface = GetInterfaceIndex(src_host, next_hop);
                            if (interface >= 0) {
                                routing_tables_[src_host][dst_ip].push_back(interface);
                            } else {
                                std::vector<int> path = FindPath(src_host, next_hop);
                                if (!path.empty() && path.size() > 1) {
                                    int first_hop = path[1];
                                    int interface = GetInterfaceIndex(src_host, first_hop);
                                    if (interface >= 0) {
                                        routing_tables_[src_host][dst_ip].push_back(interface);
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (routing_tables_[src_host][dst_ip].empty()) {
                    std::vector<int> path = FindPath(src_host, dst_host);
                    if (!path.empty() && path.size() > 1) {
                        int first_hop = path[1];
                        int interface = GetInterfaceIndex(src_host, first_hop);
                        if (interface >= 0) {
                            routing_tables_[src_host][dst_ip].push_back(interface);
                        }
                    }
                }
            }
        }
    }
}

std::vector<int> RoutingFramework::FindPath(int src, int dst) {
    const auto& graph = topology_.GetGraph();
    int node_count = graph.size();
    
    std::vector<bool> visited(node_count, false);
    std::vector<int> parent(node_count, -1);
    std::queue<int> q;
    
    q.push(src);
    visited[src] = true;
    
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
        if (current == dst) {
            std::vector<int> path;
            int node = dst;
            while (node != -1) {
                path.push_back(node);
                node = parent[node];
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                if (topology_.GetNodeType(neighbor) == 0 && neighbor != src && neighbor != dst) {
                    continue;
                }
                
                visited[neighbor] = true;
                parent[neighbor] = current;
                q.push(neighbor);
            }
        }
    }
    
    return {};
}

int RoutingFramework::GetInterfaceIndex(int from_node, int to_node) const {
    const auto& graph = topology_.GetGraph();
    if (from_node < 0 || from_node >= graph.size()) {
        return -1;
    }
    
    const auto& neighbors = graph[from_node];
    for (int neighbor : neighbors) {
        if (neighbor == to_node) {
            const auto& interface_map = topology_.GetInterfaceMap();
            auto it = interface_map.find(from_node);
            if (it != interface_map.end()) {
                auto dst_it = it->second.find(to_node);
                if (dst_it != it->second.end()) {
                    return dst_it->second;
                }
            }
            
            return to_node + 1;
        }
    }
    
    return -1;
}

int RoutingFramework::GetNextNodeFromInterface(int from_node, int interface) const {
    const auto& graph = topology_.GetGraph();
    if (from_node < 0 || from_node >= graph.size()) {
        return -1;
    }
    
    const auto& neighbors = graph[from_node];
    for (int neighbor : neighbors) {
        int neighbor_interface = GetInterfaceIndex(from_node, neighbor);
        if (neighbor_interface == interface) {
            return neighbor;
        }
    }
    
    return -1;
}

uint32_t RoutingFramework::EcmpHash(const uint8_t* key, size_t len, uint32_t seed) const {
    uint32_t h = seed;
    if (len > 3) {
        const uint32_t* key_x4 = (const uint32_t*) key;
        size_t i = len >> 2;
        do {
            uint32_t k = *key_x4++;
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h += (h << 2) + 0xe6546b64;
        } while (--i);
        key = (const uint8_t*) key_x4;
    }
    if (len & 3) {
        size_t i = len & 3;
        uint32_t k = 0;
        key = &key[i - 1];
        do {
            k <<= 8;
            k |= *key--;
        } while (--i);
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

std::vector<std::pair<FlowKey, int>> RoutingFramework::PrecalculateFlowPathsWithTracing(
    int node_count, const std::function<uint32_t(int)>& node_id_to_ip,
    const std::function<int(int,int)>& get_node_type,
    uint16_t src_port, uint16_t dst_port, uint8_t protocol) {
    
    std::vector<std::pair<FlowKey, int>> flow_paths;
    
    if (!IsTopologyLoaded()) {
        std::cout << "[ROUTING] Topology not loaded, cannot pre-calculate flows" << std::endl;
        return flow_paths;
    }
    
    int topology_node_count = topology_.GetNodeCount();
    std::cout << "[ROUTING] Pre-calculating flows for " << node_count << " nodes (topology has " << topology_node_count << " nodes)" << std::endl;
    
    // Count host nodes
    int host_count = 0;
    for (int i = 0; i < topology_node_count; i++) {
        if (topology_.GetNodeType(i) == 0) {
            host_count++;
        }
    }
    std::cout << "[ROUTING] Found " << host_count << " host nodes in topology" << std::endl;
    
    for (int src_id = 0; src_id < node_count; src_id++) {
        if (get_node_type(src_id, 0) != 0) {
            // std::cout << "[ROUTING] Skipping node " << src_id << " (type " << get_node_type(src_id, 0) << ")" << std::endl;
            continue; // Only hosts
        }
        
        for (int dst_id = 0; dst_id < node_count; dst_id++) {
            if (get_node_type(dst_id, 0) != 0) continue; // Only hosts
            if (src_id == dst_id) continue;
            
            uint32_t src_ip = node_id_to_ip(src_id);
            uint32_t dst_ip = node_id_to_ip(dst_id);
            
            // Create a simple flow key for this host-to-host communication
            FlowKey key;
            key.src_ip = src_ip;
            key.dst_ip = dst_ip;
            key.protocol = protocol;
            key.src_port = src_port;
            key.dst_port = dst_port;
            
            // Get the next hop interface for this flow
            std::vector<int> interfaces = GetNextHopInterfaces(src_id, dst_id);
            if (!interfaces.empty()) {
                // Use ECMP hash to select next hop (same as NS-3 runtime)
                union {
                    uint8_t u8[4+4+2+2];
                    uint32_t u32[3];
                } buf;
                buf.u32[0] = src_ip;
                buf.u32[1] = dst_ip;
                buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
                
                uint32_t hash = EcmpHash(buf.u8, 12, src_id);  // Use source node ID as seed (same as NS3)
                uint32_t path_idx = hash % interfaces.size();
                int selected_interface = interfaces[path_idx];
                
                // Store the pre-calculated ECMP decision
                flow_paths.push_back(std::make_pair(key, selected_interface));
            }
        }
    }
    
    std::cout << "[ROUTING] Pre-calculated " << flow_paths.size() << " flow paths" << std::endl;
    return flow_paths;
}

// FlowSim Integration Methods
std::vector<int> RoutingFramework::GetFlowSimPath(const FlowKey& flow_key) const {
    // Find source and destination nodes
    int src_node = -1, dst_node = -1;
    int node_count = topology_.GetNodeCount();
    
    for (int i = 0; i < node_count; i++) {
        if (topology_.NodeIdToIp(i) == flow_key.src_ip) {
            src_node = i;
        }
        if (topology_.NodeIdToIp(i) == flow_key.dst_ip) {
            dst_node = i;
        }
    }
    
    if (src_node == -1 || dst_node == -1) {
        return {};
    }
    
    // If source and destination are the same, return just the source
    if (src_node == dst_node) {
        return {src_node};
    }
    
    // Use the routing framework to find the actual path
    // First, get the next hop interface for this flow
    auto it = flow_to_path_map_.find(flow_key);
    if (it == flow_to_path_map_.end()) {
        // No pre-calculated path, use routing tables
        auto next_hops = GetPrecalculatedNextHops(src_node, dst_node);
        if (next_hops.empty()) {
            // Fallback: direct path
            return {src_node, dst_node};
        }
        
        // Use ECMP hash to select next hop (same as NS3)
        union {
            uint8_t u8[4+4+2+2];
            uint32_t u32[3];
        } buf;
        buf.u32[0] = flow_key.src_ip;
        buf.u32[1] = flow_key.dst_ip;
        buf.u32[2] = flow_key.src_port | ((uint32_t)flow_key.dst_port << 16);
        
        uint32_t hash = EcmpHash(buf.u8, 12, src_node);  // Use source node ID as seed
        uint32_t path_idx = hash % next_hops.size();
        int selected_interface = next_hops[path_idx];
        
        // Get the next node from the selected interface
        int next_node = GetNextNodeFromInterface(src_node, selected_interface);
        if (next_node == -1) {
            return {src_node, dst_node};
        }
        
        // Build the complete path
        std::vector<int> path = {src_node};
        
        // Trace the path through the network
        int current_node = next_node;
        std::set<int> visited = {src_node};
        
        while (current_node != dst_node && visited.find(current_node) == visited.end()) {
            path.push_back(current_node);
            visited.insert(current_node);
            
            // Get next hop from current node to destination
            auto current_next_hops = GetPrecalculatedNextHops(current_node, dst_node);
            if (current_next_hops.empty()) {
                // No path found, add destination and break
                path.push_back(dst_node);
                break;
            }
            
            // Use ECMP hash for consistency
            uint32_t current_hash = EcmpHash(buf.u8, 12, current_node);
            uint32_t current_path_idx = current_hash % current_next_hops.size();
            int current_interface = current_next_hops[current_path_idx];
            
            int next_current_node = GetNextNodeFromInterface(current_node, current_interface);
            if (next_current_node == -1) {
                path.push_back(dst_node);
                break;
            }
            
            current_node = next_current_node;
        }
        
        if (current_node == dst_node) {
            path.push_back(dst_node);
        }
        
        return path;
    } else {
        // Use pre-calculated path
        int selected_interface = it->second;
        int next_node = GetNextNodeFromInterface(src_node, selected_interface);
        
        if (next_node == -1) {
            return {src_node, dst_node};
        }
        
        // Build path similar to above but using the pre-calculated interface
        std::vector<int> path = {src_node};
        int current_node = next_node;
        std::set<int> visited = {src_node};
        
        while (current_node != dst_node && visited.find(current_node) == visited.end()) {
            path.push_back(current_node);
            visited.insert(current_node);
            
            auto current_next_hops = GetPrecalculatedNextHops(current_node, dst_node);
            if (current_next_hops.empty()) {
                path.push_back(dst_node);
                break;
            }
            
            // Use ECMP hash for consistency
            union {
                uint8_t u8[4+4+2+2];
                uint32_t u32[3];
            } buf;
            buf.u32[0] = flow_key.src_ip;
            buf.u32[1] = flow_key.dst_ip;
            buf.u32[2] = flow_key.src_port | ((uint32_t)flow_key.dst_port << 16);
            
            uint32_t current_hash = EcmpHash(buf.u8, 12, current_node);
            uint32_t current_path_idx = current_hash % current_next_hops.size();
            int current_interface = current_next_hops[current_path_idx];
            
            int next_current_node = GetNextNodeFromInterface(current_node, current_interface);
            if (next_current_node == -1) {
                path.push_back(dst_node);
                break;
            }
            
            current_node = next_current_node;
        }
        
        if (current_node == dst_node) {
            path.push_back(dst_node);
        }
        
        return path;
    }
}

std::vector<int> RoutingFramework::GetFlowSimPathByNodeIds(int src_node, int dst_node) const {
    // Create FlowKey using the same IP format as pre-calculation (TopologyParser::NodeIdToIp)
    FlowKey flow_key;
    
    // Use the same IP format as TopologyParser::NodeIdToIp
    uint32_t src_x = (src_node >> 8) & 0xFF;
    uint32_t src_y = src_node & 0xFF;
    flow_key.src_ip = (10 << 24) | (src_x << 16) | (src_y << 8) | 1;
    
    uint32_t dst_x = (dst_node >> 8) & 0xFF;
    uint32_t dst_y = dst_node & 0xFF;
    flow_key.dst_ip = (10 << 24) | (dst_x << 16) | (dst_y << 8) | 1;
    
    flow_key.protocol = 17;  // UDP default
    flow_key.src_port = 10006;  // Default source port
    flow_key.dst_port = 100;    // Default destination port
    
    // Use the existing GetFlowSimPath function
    return GetFlowSimPath(flow_key);
}

size_t RoutingFramework::GetFlowPathCount() const {
    return flow_to_path_map_.size();
}

bool RoutingFramework::IsTopologyLoaded() const {
    return topology_.GetNodeCount() > 0;
}

bool RoutingFramework::PrecalculateFlowPathsForFlowSim(const std::string& topology_file, 
                                                      const std::string& network_config_file) {
    // Parse topology first
    if (!ParseTopology(topology_file)) {
        std::cerr << "[ROUTING] Failed to parse topology file: " << topology_file << std::endl;
        return false;
    }
    
    // Pre-calculate routing tables
    PrecalculateRoutingTables();
    
    // Generate flow paths using the existing method with proper parameters
    int node_count = topology_.GetNodeCount();
    
    // Create node_id_to_ip function
    auto node_id_to_ip = [this](int node_id) -> uint32_t {
        return topology_.NodeIdToIp(node_id);
    };
    
    // Create get_node_type function
    auto get_node_type = [this](int node_id, int unused) -> int {
        return topology_.GetNodeType(node_id);
    };
    
    // Pre-calculate flow paths (same as NS3)
    auto flow_paths = PrecalculateFlowPathsWithTracing(node_count, node_id_to_ip, get_node_type);
    
    // Store the results in flow_to_path_map_ (same as NS3)
    flow_to_path_map_.clear();
    for (const auto& pair : flow_paths) {
        flow_to_path_map_[pair.first] = pair.second;
    }
    
    std::cout << "[ROUTING] FlowSim paths pre-calculated successfully" << std::endl;
    std::cout << "[ROUTING] Total flows: " << flow_to_path_map_.size() << std::endl;
    
    return true;
}

} // namespace AstraSim 