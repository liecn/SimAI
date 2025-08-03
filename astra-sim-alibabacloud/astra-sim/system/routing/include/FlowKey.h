/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef FLOW_KEY_H
#define FLOW_KEY_H

#include <cstdint>
#include <functional>

namespace AstraSim {

struct FlowKey {
    uint16_t cur_node;   // ID of the node performing lookup (current hop)
    uint32_t src_ip;
    uint32_t dst_ip;
    uint8_t  protocol;
    uint16_t src_port;
    uint16_t dst_port;

    FlowKey() : src_ip(0), dst_ip(0), protocol(0), src_port(0), dst_port(0) {}
    
    FlowKey(uint32_t src, uint32_t dst, uint8_t proto, uint16_t sport, uint16_t dport)
        : src_ip(src), dst_ip(dst), protocol(proto), src_port(sport), dst_port(dport) {}
    
    bool operator==(const FlowKey& other) const {
        return cur_node == other.cur_node &&
               src_ip == other.src_ip && dst_ip == other.dst_ip &&
               protocol == other.protocol && src_port == other.src_port &&
               dst_port == other.dst_port;
    }
};

struct FlowKeyHash {
    std::size_t operator()(const FlowKey& key) const {
        std::size_t h0 = std::hash<uint16_t>{}(key.cur_node);
        std::size_t h1 = std::hash<uint32_t>{}(key.src_ip);
        std::size_t h2 = std::hash<uint32_t>{}(key.dst_ip);
        std::size_t h3 = std::hash<uint8_t>{}(key.protocol);
        std::size_t h4 = std::hash<uint16_t>{}(key.src_port);
        std::size_t h5 = std::hash<uint16_t>{}(key.dst_port);
        return h0 ^ (h1 << 1) ^ (h2 << 2) ^ (h3 << 3) ^ (h4 << 4) ^ (h5 << 5);
    }
};

} // namespace AstraSim

#endif // FLOW_KEY_H 