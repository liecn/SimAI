#include "TopologyBuilder.h"
#include "Topology.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

std::tuple<int, int, std::vector<int>, std::vector<std::tuple<int, int, double, double, double>>> parse_fat_tree_topology_file(const std::string& topology_file) {
    std::ifstream file(topology_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open topology file");
    }

    int npus_count;
    int switch_node_count;
    int link_count;
    std::vector<int> switch_node_ids;
    std::vector<std::tuple<int, int, double, double, double>> links;

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> npus_count >> switch_node_count >> link_count;
    npus_count-=switch_node_count;

    std::getline(file, line);
    std::istringstream iss_switches(line);
    int switch_node_id;
    while (iss_switches >> switch_node_id) {
        switch_node_ids.push_back(switch_node_id);
    }

    while (std::getline(file, line)) {
        int src, dst;
        double rate, delay, error_rate;
        std::string rate_str, delay_str, error_rate_str;
        std::istringstream iss_link(line);
        iss_link >> src >> dst >> rate_str >> delay_str >> error_rate_str;
        // Parse bandwidth: input is in Gbps, need to convert to Bytes/ns
        double rate_gbps = std::stod(rate_str.substr(0, rate_str.size() - 3)); // Removing "bps", get Gbps
        // Convert Gbps to Bytes/ns:
        // Gbps -> GB/s: divide by 8 (bits to bytes)
        // GB/s -> Bytes/ns: multiply by 1e9/1e9 = 1 (since 1 GB/s = 1 Byte/ns numerically)
        rate = rate_gbps / 8.0;  // Convert Gbps to GB/s, which equals Bytes/ns numerically
        
        // Parse delay: input is in milliseconds (e.g., "0.000025ms"), need to convert to nanoseconds
        double delay_ms = std::stod(delay_str.substr(0, delay_str.size() - 2)); // Removing "ms"
        delay = delay_ms * 1e6; // Convert milliseconds to nanoseconds (1 ms = 1e6 ns)
        error_rate = std::stod(error_rate_str);
        links.emplace_back(src, dst, rate, delay, error_rate);
    }

    return std::make_tuple(npus_count, switch_node_count, switch_node_ids, links);
}

std::shared_ptr<Topology> construct_fat_tree_topology(const std::string& topology_file) noexcept {
    //std::cerr << "Constructing Fat-Tree topology from file: " << topology_file << std::endl;

    // Parse the topology file
    auto [npus_count, switch_node_count, switch_node_ids, links] = parse_fat_tree_topology_file(topology_file);

    // Create an instance of FatTreeTopology
    auto fat_tree_topology = std::make_shared<Topology>(npus_count + switch_node_count, npus_count);
    for (const auto& link : links) {
        int src = std::get<0>(link);
        int dest = std::get<1>(link);
        double bandwidth = std::get<2>(link);
        double latency = std::get<3>(link);
        bool bidirectional = true; // Assuming bidirectional links
        //std::cerr << "Connecting " << src << " to " << dest << " with bandwidth " << bandwidth << " GBps and latency " << latency << " ns" << std::endl;
        fat_tree_topology->connect(src, dest, bandwidth, latency, bidirectional);
    }

    return fat_tree_topology;
}

// Convert bandwidth expressed in Gbps (gigabits per second) to Bytes per nanosecond.
//   1 Gbit  = 1e9 bits
//   1 Byte  = 8 bits
//   1 second= 1e9 nanoseconds
Bandwidth bw_GBps_to_Bpns(const Bandwidth bw_Gbps) noexcept {
    assert(bw_Gbps > 0);
    return bw_Gbps; // Bytes per ns
}
