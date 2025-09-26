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

    // Sweep topo header: node_num gpus_per_server nvswitch_num switch_num link_num gpu_type
    int node_num = 0;
    int gpus_per_server = 0;
    int nvswitch_num = 0;
    int switch_num = 0;
    int link_num = 0;
    std::vector<int> switch_node_ids;
    std::vector<std::tuple<int, int, double, double, double>> links;

    std::string line;
    std::getline(file, line);
    {
        std::istringstream iss(line);
        std::string gpu_type_str;
        iss >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str;
        if (!iss) {
            throw std::runtime_error("Invalid topology header: " + line);
        }
    }

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
        if (rate_str.size() >= 4 && rate_str.substr(rate_str.size() - 4) == "Gbps") {
            double gbps = std::stod(rate_str.substr(0, rate_str.size() - 4));
            rate = gbps / 8.0;
        } else {
            throw std::runtime_error("Unsupported bandwidth unit in: " + rate_str);
        }
        
        if (delay_str.size() >= 2 && delay_str.substr(delay_str.size() - 2) == "ms") {
            double ms = std::stod(delay_str.substr(0, delay_str.size() - 2));
            delay = ms * 1e6;
        } else if (delay_str.size() >= 2 && delay_str.substr(delay_str.size() - 2) == "ns") {
            delay = std::stod(delay_str.substr(0, delay_str.size() - 2));
        } else {
            throw std::runtime_error("Unsupported latency unit in: " + delay_str);
        }
        error_rate = std::stod(error_rate_str);
        links.emplace_back(src, dst, rate, delay, error_rate);
    }
    
    int npus_count = node_num - nvswitch_num - switch_num;
    if (npus_count <= 0) {
        throw std::runtime_error("Invalid computed NPU count from topology header");
    }
    int non_npu_nodes = nvswitch_num + switch_num;
    return std::make_tuple(npus_count, non_npu_nodes, switch_node_ids, links);
}

std::shared_ptr<Topology> construct_fat_tree_topology(const std::string& topology_file) noexcept {
    //std::cerr << "Constructing Fat-Tree topology from file: " << topology_file << std::endl;

    // Parse the topology file
    auto [npus_count, non_npu_nodes, switch_node_ids, links] = parse_fat_tree_topology_file(topology_file);

    // Create an instance of FatTreeTopology
    auto fat_tree_topology = std::make_shared<Topology>(npus_count + non_npu_nodes, npus_count);
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
