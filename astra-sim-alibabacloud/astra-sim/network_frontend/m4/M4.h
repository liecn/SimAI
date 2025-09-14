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

#ifndef __M4_H__
#define __M4_H__

#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "astra-sim/system/routing/include/RoutingFramework.h"
#include <torch/torch.h>
#include <torch/script.h>
#include "Type.h"  // For Callback and CallbackArg types

// Forward declarations
class EventQueue;
class Topology;
class Device;
class Node;

// Use the Callback type from Type.h (function pointer, not std::function)
// using Callback = std::function<void(void*)>;
// using CallbackArg = void*;

// M4Flow structure for temporal batching (like FlowSim's Chunk)
struct M4Flow {
    int src, dst;
    uint64_t size;
    std::vector<int> node_path;
    void (*callback)(void*);
    void* callbackArg;
    uint64_t start_time;
    int flow_id;
    
    M4Flow(int s, int d, uint64_t sz, const std::vector<int>& path, void (*cb)(void*), void* arg)
        : src(s), dst(d), size(sz), node_path(path), callback(cb), callbackArg(arg), start_time(0), flow_id(-1) {}
};

/**
 * M4 Backend - ML-based network simulation
 * Similar structure to FlowSim but uses ML inference for completion time prediction
 */
class M4 {
private:
    // Core components (same as FlowSim)
    static std::shared_ptr<EventQueue> event_queue;
    static std::shared_ptr<Topology> topology;
    static std::unique_ptr<AstraSim::RoutingFramework> routing_framework_;
    
    // M4-specific ML components
    static torch::Device device;
    static torch::jit::script::Module lstmcell_time, lstmcell_rate, lstmcell_time_link, lstmcell_rate_link;
    static torch::jit::script::Module output_layer, gnn_layer_0, gnn_layer_1, gnn_layer_2;
    static torch::Tensor params_tensor;
    static bool models_loaded;
    static int32_t hidden_size_;
    static int32_t n_links_max_;
    
    // Multi-flow state management (from @inference/ ground truth)
    static torch::Tensor h_vec;
    static torch::Tensor flowid_active_mask;
    static torch::Tensor edge_index;
    static torch::Tensor z_t_link;
    static torch::Tensor link_to_graph_id;
    static torch::Tensor link_to_nflows;
    static torch::Tensor flow_to_graph_id;
    static torch::Tensor time_last;
    static torch::Tensor release_time_tensor;
    static torch::Tensor flowid_to_nlinks_tensor;
    static torch::Tensor i_fct_tensor;
    
    // Additional tensors from @inference/ for complete ML pipeline
    static torch::Tensor flowid_to_linkid_flat_tensor;
    static torch::Tensor flowid_to_linkid_offsets_tensor;
    static torch::Tensor edges_flow_ids_tensor;
    static torch::Tensor edges_link_ids_tensor;
    static torch::Tensor ones_cache;
    
    // Flow and graph management
    static int32_t n_flows_max;
    static int32_t n_flows_active;
    static int32_t n_flows_completed;
    static int32_t graph_id_counter;
    static int32_t graph_id_cur;
    static float time_clock;
    static int32_t next_flow_id;

    // Link indexing for graph construction (derived from RoutingFramework paths)
    static std::unordered_map<long long, int32_t> link_key_to_index; // key = ((int64_t)u<<32)|v
    static int32_t next_link_index;

    // Store per-flow link indices (built from RoutingFramework paths)
    static std::vector<std::vector<int32_t>> flowid_to_link_indices;
    
    // FlowSim-style temporal batching
    static std::vector<M4Flow*> pending_flows_;
    static std::list<std::unique_ptr<M4Flow>> active_flows_ptrs;
    static uint64_t last_batch_time_;
    static int batch_timeout_event_id_;
    static constexpr uint64_t BATCH_TIMEOUT_NS = 0;
    
    // Inference-style flow completion tracking
    static float flow_completion_time;
    static int32_t completed_flow_id;

public:
    // Type definitions (same as FlowSim)
    using Callback = void (*)(void*);
    using CallbackArg = void*;
    using Route = std::vector<std::shared_ptr<Node>>;
    using ChunkSize = uint64_t;
    
    
    // Core M4 functions (mirror FlowSim interface)
    static void Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo);
    static void Run();
    static void Stop();
    static void Destroy();
    
    // Scheduling and sending (same interface as FlowSim)
    static void Schedule(uint64_t delay, void (*fun_ptr)(void* fun_arg), void* fun_arg);
    static void Send(int src, int dst, uint64_t size, int tag, Callback callback, CallbackArg callbackArg);
    static double Now();
    
    // Flow management functions
    static int AddActiveFlow(int src, int dst, uint64_t size, const std::vector<int>& node_path, Callback callback, CallbackArg callbackArg);
    
    // ML batch processing function (uses @inference/ approach: MLP for prediction, LSTM+GNN for state updates)
    // FlowSim-style batch processing
    static void process_batch_of_flows();
    static void batch_timeout_callback(void* arg);
    
    // Inference-style event-driven ML processing
    static void update_times_m4();
    static void step_m4();
    // Post-batch state evolution only (no scheduling, no completion selection)
    static void step_m4_state_only(float new_time_clock);
    
    // Routing framework management (same as FlowSim)
    static void SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework);
    static const AstraSim::RoutingFramework* GetRoutingFramework();
    static bool IsRoutingFrameworkLoaded();
    
    // Topology access for FCT calculation
    static float GetTopologyLatency();
    static float GetTopologyBandwidth();
    
    // M4-specific ML setup
    static void SetupML();
    static void OnFlowCompleted(int flow_id);
    static void ScheduleNextForGraph(int32_t graph_id);
    
private:
};

#endif // __M4_H__
