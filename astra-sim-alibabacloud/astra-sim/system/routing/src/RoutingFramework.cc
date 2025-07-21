/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "../include/RoutingFramework.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <queue>

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

std::vector<int> RoutingFramework::GetPrecalculatedNextHops(int src_node, int dst_node) {
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

uint32_t RoutingFramework::EcmpHash(const uint8_t* key, size_t len, uint32_t seed) {
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
        return flow_paths;
    }
    
    for (int src_id = 0; src_id < node_count; src_id++) {
        if (get_node_type(src_id, 0) != 0) continue; // Only hosts
        
        for (int dst_id = 0; dst_id < node_count; dst_id++) {
            if (get_node_type(dst_id, 0) != 0) continue; // Only hosts
            if (src_id == dst_id) continue;
            
            uint32_t src_ip = node_id_to_ip(src_id);
            uint32_t dst_ip = node_id_to_ip(dst_id);
            
            // Trace the path from source to destination using routing framework
            // but store these decisions as fixed input for deterministic routing
            
            int current = src_id;
            
            while (current != dst_id) {
                // Use routing framework to get next hop interfaces for current node
                std::vector<int> interfaces = GetNextHopInterfaces(current, dst_id);
                
                if (interfaces.empty()) break;
                
                // Use ECMP hash to select next hop (same as NS-3 runtime)
                union {
                    uint8_t u8[4+4+2+2];
                    uint32_t u32[3];
                } buf;
                buf.u32[0] = src_ip;
                buf.u32[1] = dst_ip;
                buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
                
                uint32_t hash = EcmpHash(buf.u8, 12, current);  // Use current node ID as seed (same as NS3)
                uint32_t path_idx = hash % interfaces.size();
                int selected_interface = interfaces[path_idx];
                
                // Store the ECMP decision for this switch
                int current_node_type = get_node_type(current, 0);
                if (current_node_type == 1 || current_node_type == 2) { // Switch or NVSwitch
                    FlowKey key;
                    key.src_ip = src_ip;
                    key.dst_ip = dst_ip;
                    key.protocol = protocol;
                    key.src_port = src_port;
                    key.dst_port = dst_port;
                    
                    // Store the pre-calculated ECMP decision
                    flow_paths.push_back(std::make_pair(key, selected_interface));
                }
                
                // Find the next hop node using this interface
                // This requires the routing framework to provide interface-to-node mapping
                int next_node = GetNextNodeFromInterface(current, selected_interface);
                
                if (next_node >= 0) {
                    current = next_node;
                } else {
                    break; // Can't find next hop
                }
                
                // Safety check to prevent infinite loops
                if (get_node_type(current, 0) == 0) break; // Reached a host
            }
        }
    }
    
    return flow_paths;
}

} // namespace AstraSim 