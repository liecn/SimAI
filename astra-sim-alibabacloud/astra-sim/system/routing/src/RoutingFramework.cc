/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "RoutingFramework.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <queue> // Added for BFS queue

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
    
    // Check bounds
    int node_count = topology_.GetNodeCount();
    if (src_node < 0 || src_node >= node_count || dst_node < 0 || dst_node >= node_count) {
        return -1;
    }
    
    if (src_node == dst_node) {
        return -1; // Same node, no forwarding needed
    }
    
    // Use pre-calculated routing table (NS3-style)
    auto next_hops = GetPrecalculatedNextHops(src_node, dst_node);
    if (next_hops.empty()) {
        return -1;
    }
    
    // If only one next-hop, return it directly
    if (next_hops.size() == 1) {
        return next_hops[0];
    }
    
    // Use ECMP hash to select interface (exactly like NS3)
    union {
        uint8_t u8[4+4+2+2];
        uint32_t u32[3];
    } buf;
    
    buf.u32[0] = src_ip;  // Source IP
    buf.u32[1] = dst_ip;  // Destination IP
    
    if (protocol == 0x6) {  // TCP
        buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
    } else if (protocol == 0x11) {  // UDP
        buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
    } else if (protocol == 0xFC || protocol == 0xFD) {  // ACK
        buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
    } else {
        buf.u32[2] = src_port | ((uint32_t)dst_port << 16);
    }
    
    uint32_t hash = EcmpHash(buf.u8, 12, seed);
    uint32_t interface_idx = hash % next_hops.size();
    
    return next_hops[interface_idx];
}

int RoutingFramework::GetOutInterface(const FlowKey& flow_key, uint32_t seed) {
    // Extract node IDs from flow key IPs
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
    
    // NS3-style: only calculate routes from HOST nodes (node type 0)
    for (int src_id = 0; src_id < node_count; src_id++) {
        if (topology_.GetNodeType(src_id) == 0) {  // HOST nodes only
            CalculateRouteNS3Style(src_id);
        }
    }
    
    // Build routing tables from nextHop structure (like NS3's SetRoutingEntries)
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

// NS3-style routing calculation (exact copy of CalculateRoute function)
void RoutingFramework::CalculateRouteNS3Style(int host_node) {
    int node_count = topology_.GetNodeCount();
    // NS3 data structures
    std::map<int, int> dis;  // distance from host
    std::map<int, std::vector<int>> nextHop;  // nextHop[node] = vector of next hop nodes
    std::queue<int> q;
    q.push(host_node);
    dis[host_node] = 0;

    while (!q.empty()) {
        int now = q.front();
        q.pop();
        int d = dis[now];
        const auto& neighbors = topology_.GetGraph()[now];
        for (int next : neighbors) {
            if (dis.find(next) == dis.end()) {
                dis[next] = d + 1;
                // Only add switches and NV switches to queue (like NS3)
                if (topology_.GetNodeType(next) == 1 || topology_.GetNodeType(next) == 2) {
                    q.push(next);
                }
            }
            // NS3's key logic: if 'now' is on the shortest path from 'next' to 'host'
            if (d + 1 == dis[next]) {
                nextHop[next].push_back(now);
            }
        }
    }
    next_hop_tables_[host_node] = nextHop;
}

void RoutingFramework::BuildRoutingTablesFromNextHop() {
    int node_count = topology_.GetNodeCount();
    // For each host that has nextHop information
    for (const auto& host_pair : next_hop_tables_) {
        int host_node = host_pair.first;
        const auto& nextHop = host_pair.second;
        
        // For each destination in this host's nextHop table
        for (const auto& dst_pair : nextHop) {
            int dst_node = dst_pair.first;
            const auto& nexts = dst_pair.second;
            
            // Get destination IP
            uint32_t dst_ip = topology_.NodeIdToIp(dst_node);
            
            // For each next hop, get the interface from host to that next hop
            for (int next : nexts) {
                int interface = GetInterfaceIndex(host_node, next);
                if (interface >= 0) {
                    routing_tables_[host_node][dst_ip].push_back(interface);
                }
            }
        }
    }
    
    // Remove duplicates and sort (like NS3)
    for (auto& node_table : routing_tables_) {
        for (auto& dst_entry : node_table.second) {
            std::vector<int>& interfaces = dst_entry.second;
            std::sort(interfaces.begin(), interfaces.end());
            interfaces.erase(std::unique(interfaces.begin(), interfaces.end()), interfaces.end());
        }
    }
}

int RoutingFramework::GetInterfaceIndex(int from_node, int to_node) const {
    const auto& graph = topology_.GetGraph();
    if (from_node < 0 || from_node >= graph.size()) {
        return -1;
    }
    
    // Check if there's a direct link
    const auto& neighbors = graph[from_node];
    for (int neighbor : neighbors) {
        if (neighbor == to_node) {
            // Return interface index (simulating NS3's GetIfIndex())
            return to_node + 1;  // Simple mapping to avoid interface 0
        }
    }
    
    return -1;  // No direct link
}

uint32_t RoutingFramework::EcmpHash(const uint8_t* key, size_t len, uint32_t seed) {
    // Exact copy of NS3's EcmpHash function from switch-node.cc
    uint32_t h = seed;
    uint32_t k;
    
    const uint32_t* key_x4 = (const uint32_t*)key;
    size_t i = len >> 2;
    
    for (; i > 0; i--) {
        k = *key_x4++;
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h += (h << 2) + 0xe6546b64;
    }
    
    key = (const uint8_t*)key_x4;
    
    if (len & 3) {
        size_t i = len & 3;
        k = 0;
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

FlowKey RoutingFramework::CreateFlowKey(int src_node, int dst_node, uint8_t protocol, 
                                       uint16_t src_port, uint16_t dst_port) const {
    uint32_t src_ip = topology_.NodeIdToIp(src_node);
    uint32_t dst_ip = topology_.NodeIdToIp(dst_node);
    return FlowKey(src_ip, dst_ip, protocol, src_port, dst_port);
}

} // namespace AstraSim 