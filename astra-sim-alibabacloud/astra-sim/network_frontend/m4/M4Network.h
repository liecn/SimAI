/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#ifndef __SIMAI_M4_NETWORK_HH__
#define __SIMAI_M4_NETWORK_HH__

#include"astra-sim/system/AstraNetworkAPI.hh"
#include <map>
#include <utility>
#include <memory>
#include <vector>
#include <unordered_map>
// Re-enable PyTorch headers
#include <torch/torch.h>
#include <torch/script.h>
#include "Topology.h"
#include "EventQueue.h"
#include "Type.h"
#include "astra-sim/system/routing/include/RoutingFramework.h"

// Forward declarations
struct M4CallbackData;

// Copy FlowSim's task1 struct for callback management
struct M4Task {
  int src;
  int dest;
  uint64_t count;
  int type;
  void* fun_arg;
  void (*msg_handler)(void* fun_arg);
  uint64_t schTime;
};

/**
 * M4 Network Interface
 * Implements AstraSim::AstraNetworkAPI for M4 backend
 */
class M4Network: public AstraSim::AstraNetworkAPI {
private:
  int npu_offset;
  
  // M4 Core Components (like FlowSim)
  std::shared_ptr<EventQueue> event_queue;
  std::shared_ptr<Topology> topology;
  std::vector<Route> routing;
  
  // M4 Inference Components - Re-enabled
  torch::Device device;
  torch::jit::script::Module lstmcell_time, lstmcell_rate, lstmcell_time_link, lstmcell_rate_link;
  torch::jit::script::Module output_layer, gnn_layer_0, gnn_layer_1, gnn_layer_2;
  
  // M4 State Management - Re-enabled
  torch::Tensor h_vec, flowid_active_mask;
  torch::Tensor fat_tensor, i_fct_tensor, size_tensor;
  torch::Tensor params_tensor;
  torch::Tensor release_time_tensor;
  torch::Tensor flowid_to_nlinks_tensor;
  torch::Tensor flowid_to_linkid_flat_tensor;
  torch::Tensor flowid_to_linkid_offsets_tensor;
  torch::Tensor edge_index;
  torch::Tensor z_t_link;
  torch::Tensor link_to_graph_id;
  torch::Tensor link_to_nflows;
  torch::Tensor flow_to_graph_id;
  torch::Tensor time_last;
  static torch::Tensor ones_cache;
  float flow_arrival_time, flow_completion_time;
  int32_t n_flows, flow_id_in_prop, n_flows_active;
  
  // M4 Model Parameters
  std::vector<double> params;
  bool models_loaded;
  int32_t hidden_size_;
  int32_t n_links_max_;

  // Flow/link bookkeeping (parity with inference no_flowsim)
  std::unordered_map<int, int> flowIdToIndex;
  int next_flow_index = 0;
  std::vector<int32_t> flowid_to_linkid_offsets_v;
  std::vector<int32_t> flowid_to_linkid_flat_v;
  std::unordered_map<long long, int> edgeKeyToLinkId; // key=(min(u,v)<<32)|max(u,v)
  int graph_id_counter = 0;
  std::vector<int32_t> edges_flow_ids_v;
  std::vector<int32_t> edges_link_ids_v;
  int graph_id_cur = 0;

  // Cached topology defaults (from RoutingFramework TopologyParser)
  double default_link_bandwidth_bytes_per_ns = 0.0;
  double default_link_latency_ns = 0.0;
  bool link_params_initialized = false;

  // Online flow features (no_flowsim inputs built on the fly)
  std::vector<int64_t> fsize_vec;
  std::vector<int32_t> nlinks_vec;
  std::vector<float> release_time_vec;
  int completed_flow_id = -1;

  // Routing framework shared with FlowSim
  static std::unique_ptr<AstraSim::RoutingFramework> s_routing;

public:
    M4Network(int _local_rank);
    ~M4Network();
    
    // Override backend type
    AstraSim::AstraNetworkAPI::BackendType get_backend_type() override {
        return AstraSim::AstraNetworkAPI::BackendType::M4;
    }
    
    // AstraNetworkAPI interface implementations
    int sim_comm_size(AstraSim::sim_comm comm, int * size) {
        return 0;
    }
    
    int sim_finish();
    
    double sim_time_resolution() {
        return 0;
    }
    
    int sim_init(AstraSim::AstraMemoryAPI* MEM) {
        return 0;
    }
    
    AstraSim::timespec_t sim_get_time();
    
    virtual void sim_schedule(
        AstraSim::timespec_t delta,
        void (*fun_ptr)(void* fun_arg),
        void* fun_arg);
        
    virtual int sim_send(
        void* buffer,
        uint64_t count,
        int type,
        int dst,
        int tag,
        AstraSim::sim_request* request,
        void (*msg_handler)(void* fun_arg),
        void* fun_arg);
        
    virtual int sim_recv(
        void* buffer,
        uint64_t count,
        int type,
        int src,
        int tag,
        AstraSim::sim_request* request,
        void (*msg_handler)(void* fun_arg),
        void* fun_arg);
        
    // Sender completion notification callback
    void notify_sender_sending_finished(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag);
    
    // Receiver packet arrival notification callback  
    void notify_receiver_packet_arrived(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag);
    
    // M4 Inference Functions (core ML pipeline)
    void setup_m4();           // Load PyTorch models
    void process_m4_send(M4CallbackData* data);  // Process delayed M4 send
    
    // M4 network simulation implementation complete
    
    // Routing framework wiring (same style as FlowSim)
    static void SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing) {
      s_routing = std::move(routing);
    }
    static const AstraSim::RoutingFramework* GetRoutingFramework() { return s_routing.get(); }

    // M4 processes sends/receives immediately like FlowSim
    // No event queue needed
};

#endif // __SIMAI_M4_NETWORK_HH__