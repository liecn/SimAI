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
  
  // Original individual chunk send
  void send(std::unique_ptr<Chunk> chunk) noexcept;
  
  // New batched send - groups chunks from collective operations
  void send_with_batching(std::unique_ptr<Chunk> chunk) noexcept;
  
  [[nodiscard]] int get_npus_count() const noexcept;
  [[nodiscard]] int get_devices_count() const noexcept;
  [[nodiscard]] int get_dims_count() const noexcept;
  [[nodiscard]] std::vector<int> get_npus_count_per_dim() const noexcept;
  [[nodiscard]] std::vector<Bandwidth> get_bandwidth_per_dim() const noexcept;
  void connect(DeviceId src, DeviceId dest, Bandwidth bandwidth, Latency latency, bool bidirectional = true) noexcept;
  std::shared_ptr<Device> get_device(int index);

 protected:
  /// Topology-related member variables
  int npus_count;
  int devices_count;
  int dims_count;
  std::vector<int> npus_count_per_dim;
  std::vector<Bandwidth> bandwidth_per_dim;
  std::vector<std::shared_ptr<Device>> devices;
  std::map<std::pair<DeviceId, DeviceId>, std::shared_ptr<Link>> link_map;
  std::set<std::pair<DeviceId, DeviceId>> active_links;
  
  // Chunk-based simulation (proven approach)
  std::list<std::unique_ptr<Chunk>> active_chunks_ptrs;
  std::list<Chunk*> active_chunks;
  
  // Batching support for collective operations
  std::vector<Chunk*> pending_chunks_;      // Chunks waiting to be processed as a batch
  uint64_t last_batch_time_;                // When the current batch started
  int batch_timeout_event_id_;              // Event ID for batch timeout
  bool recalc_event_scheduled_ = false;     // Ensure single post-batch processing event
  static constexpr uint64_t BATCH_TIMEOUT_NS = 1000; // 10 microseconds batching window

  static std::shared_ptr<EventQueue> event_queue;

  //void instantiate_devices() noexcept;
  
  // Proven chunk-based simulation methods
  void add_chunk_to_links(Chunk* chunk);
  void associate_chunk_with_links(Chunk* chunk);
  void update_link_states();
  double calculate_bottleneck_rate(const std::pair<DeviceId, DeviceId>& link, const std::set<Chunk*>& fixed_chunks);
  double calculate_path_latency(Chunk* chunk);
  void remove_chunk_from_links(Chunk* chunk);
  static void chunk_completion_callback(void* arg) noexcept;
  
  // Batched completion processing
  static void post_batch_completion_callback(void* arg) noexcept;

  // Batching methods (performance optimization)
  void process_batch_of_chunks();
  static void batch_timeout_callback(void* arg) noexcept;
};

#endif // _TOPOLOGY_
