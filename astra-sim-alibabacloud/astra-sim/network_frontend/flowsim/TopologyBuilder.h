#include <memory>
#include "Topology.h"
#include <tuple>
// Forward declaration so other translation units (e.g. FlowsimAstra.cc) can reuse the parser
[[nodiscard]] std::tuple<int, int, std::vector<int>, std::vector<std::tuple<int, int, double, double, double>>>
parse_fat_tree_topology_file(const std::string &topology_file);

/**
 * Construct a Fat-Tree topology from a topology file.
 *
 * @param topology_file path to the topology file
 * @return pointer to the constructed Fat-Tree topology
 */
[[nodiscard]] std::shared_ptr<Topology> construct_fat_tree_topology(
    const std::string& topology_file) noexcept;