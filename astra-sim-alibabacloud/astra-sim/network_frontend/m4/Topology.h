/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef _TOPOLOGY_
#define _TOPOLOGY_

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <set>
#include <queue>
#include <stdexcept>
#include "EventQueue.h"
#include "Chunk.h"
#include "Device.h"
#include "Link.h"

// Hash function for std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

class Topology {
 public:
  static void set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept;
  Topology(int device_count, int npus_count) noexcept;
  //[[nodiscard]] virtual Route route(uint32_t flow_id, DeviceId src, DeviceId dest) const noexcept = 0;
  void send(std::unique_ptr<Chunk> chunk) noexcept;
  [[nodiscard]] int get_npus_count() const noexcept;
  [[nodiscard]] int get_devices_count() const noexcept;
  [[nodiscard]] int get_dims_count() const noexcept;
  [[nodiscard]] std::vector<int> get_npus_count_per_dim() const noexcept;
  [[nodiscard]] std::vector<Bandwidth> get_bandwidth_per_dim() const noexcept;
  void connect(DeviceId src, DeviceId dest, Bandwidth bandwidth, Latency latency, bool bidirectional = true) noexcept;
  std::shared_ptr<Node> get_device(int index);
  bool contains_chunk(int id);
  double chunk_time(int id);

  bool has_completion_time();
  int get_next_completion();
  EventTime get_next_completion_time();

  void chunk_completion(int chunk_id);

  void set_time(EventTime time);
  EventTime get_current_time();

  float get_latency();
  float get_bandwidth();
  float bandwidth;
  float latency;

  // Per-link accessors (bandwidth in bytes/ns, latency in ns)
  // Throws if the link does not exist
  double get_link_bandwidth(DeviceId src, DeviceId dst);
  float get_link_latency(DeviceId src, DeviceId dst);

  // Record of node roles to emulate NS3 BFS expansion
  void set_switch_node_ids(const std::vector<int>& ids) { switch_node_ids.clear(); switch_node_ids.insert(ids.begin(), ids.end()); }
  void set_nvswitch_count(int gpu_count, int nvswitch_count) {
    nvswitch_node_ids.clear();
    for (int i = 0; i < nvswitch_count; ++i) {
      nvswitch_node_ids.insert(gpu_count + i);
    }
  }

  // Find NS3-like route: expand only through switches/NVSwitches
  std::vector<int> find_ns3_route(int src, int dst);



 protected:
  int devices_count;
  int npus_count;
  int dims_count;
  std::vector<int> npus_count_per_dim;
  std::vector<std::shared_ptr<Node>> devices;
  std::vector<Bandwidth> bandwidth_per_dim;

  static std::shared_ptr<EventQueue> event_queue;

  std::unordered_map<std::pair<DeviceId, DeviceId>, std::shared_ptr<Link>, pair_hash> link_map;
  std::unordered_set<std::pair<DeviceId, DeviceId>, pair_hash> active_links;
  std::vector<Chunk*> active_chunks;
  std::vector<std::unique_ptr<Chunk>> active_chunks_ptrs; // Store unique pointers to chunks
  std::unordered_map<int, double> completion_time_map;

  std::unordered_set<int> switch_node_ids;
  std::unordered_set<int> nvswitch_node_ids;

  int next_completion_id;
  EventTime next_completion_time;

  EventTime current_time;

  //void instantiate_devices() noexcept;
  void update_link_states();
  double calculate_bottleneck_rate(const std::pair<DeviceId, DeviceId>& link, const std::unordered_set<Chunk*>& fixed_chunks);
  void reschedule_active_chunks();
  void add_chunk_to_links(Chunk* chunk);
  void remove_chunk_from_links(Chunk* chunk);
  //static void chunk_completion_callback(void* arg) noexcept;
  void cancel_all_events() noexcept;
};

#endif // _TOPOLOGY_
