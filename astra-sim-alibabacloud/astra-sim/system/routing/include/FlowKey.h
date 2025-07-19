/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __FLOWKEY_H__
#define __FLOWKEY_H__

#include <cstdint>
#include <functional>

namespace AstraSim {

struct FlowKey {
    uint32_t src_ip;      // Source IP address
    uint32_t dst_ip;      // Destination IP address  
    uint8_t protocol;     // Protocol (0x11 for UDP, 0x06 for TCP)
    uint16_t src_port;    // Source port (typically 10006)
    uint16_t dst_port;    // Destination port (typically 100)
    
    FlowKey() : src_ip(0), dst_ip(0), protocol(0), src_port(0), dst_port(0) {}
    
    FlowKey(uint32_t src, uint32_t dst, uint8_t proto, uint16_t sport, uint16_t dport)
        : src_ip(src), dst_ip(dst), protocol(proto), src_port(sport), dst_port(dport) {}
    
    bool operator==(const FlowKey& other) const {
        return src_ip == other.src_ip && 
               dst_ip == other.dst_ip && 
               protocol == other.protocol && 
               src_port == other.src_port && 
               dst_port == other.dst_port;
    }
};

struct FlowKeyHash {
    size_t operator()(const FlowKey& key) const {
        size_t hash = 0;
        hash ^= std::hash<uint32_t>{}(key.src_ip) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.dst_ip) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint8_t>{}(key.protocol) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint16_t>{}(key.src_port) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint16_t>{}(key.dst_port) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

} // namespace AstraSim

#endif // __FLOWKEY_H__ 