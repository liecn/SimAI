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

#include "M4.h"
#include "EventQueue.h"
#include "Topology.h"
#include "Chunk.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <ATen/Context.h>
#include "M4Network.h"

// Static members initialization (same pattern as FlowSim)
std::shared_ptr<EventQueue> M4::event_queue = nullptr;
std::shared_ptr<Topology> M4::topology = nullptr;
std::unique_ptr<AstraSim::RoutingFramework> M4::routing_framework_ = nullptr;


// M4-specific ML components
torch::Device M4::device(torch::kCUDA, 0);
torch::jit::script::Module M4::lstmcell_time;
torch::jit::script::Module M4::lstmcell_rate;
torch::jit::script::Module M4::lstmcell_time_link;
torch::jit::script::Module M4::lstmcell_rate_link;
torch::jit::script::Module M4::output_layer;
torch::jit::script::Module M4::gnn_layer_0;
torch::jit::script::Module M4::gnn_layer_1;
torch::jit::script::Module M4::gnn_layer_2;
torch::Tensor M4::params_tensor;
bool M4::models_loaded = false;

// Multi-flow state management (from @inference/ ground truth)
// NOTE: These tensors are initialized in SetupML() to avoid static initialization issues
torch::Tensor M4::h_vec;
torch::Tensor M4::flowid_active_mask;
torch::Tensor M4::z_t_link;
torch::Tensor M4::link_to_graph_id;
torch::Tensor M4::link_to_nflows;
torch::Tensor M4::flow_to_graph_id;
torch::Tensor M4::time_last;
torch::Tensor M4::release_time_tensor;
torch::Tensor M4::flowid_to_nlinks_tensor;
torch::Tensor M4::i_fct_tensor;

// Flow and graph management
int32_t M4::hidden_size_ = 200; // Model expects 214 total: 1+13+200=214 (matches main_m4_noflowsim.cpp)
int32_t M4::n_links_max_ = 768;
int32_t M4::n_flows_max = 50000;  // Large enough for simulation
float M4::time_clock = 0.0f;

// M4 configuration parameters (hardcoded, no YAML config needed for SimAI integration)
uint64_t M4::batch_time_ns_ = 10000; // Temporal batching interval (ns) - smaller = smoother slowdowns
int32_t M4::reschedule_flow_count_ = 8; // Reschedule all active flows every N new arrivals
std::unordered_map<long long, int32_t> M4::link_key_to_index;
int32_t M4::next_link_index = 0;
std::vector<std::vector<int32_t>> M4::flowid_to_link_indices;
std::unordered_set<int32_t> M4::current_batch_link_set;

// Flow lifecycle tracking counters
static int32_t n_flows_arrived = 0;
static int32_t n_flows_completed = 0;
static int32_t n_flows_since_last_reschedule = 0; // Track arrivals since last reschedule



// FlowSim-style temporal batching
std::vector<M4Flow*> M4::pending_flows_;
std::list<std::unique_ptr<M4Flow>> M4::active_flows_ptrs;
int M4::batch_timeout_event_id_ = 0;
bool M4::is_processing_batch_ = false;

void M4::Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo) {

    M4::event_queue = event_queue;
    M4::topology = topo;
    M4::topology->set_event_queue(event_queue);
    
    // Setup ML models
    SetupML();

    // Prepare per-flow link storage
    flowid_to_link_indices.assign(n_flows_max, {});
    
    
    std::cout << "[M4] Init() completed successfully! models_loaded=" << models_loaded << ", n_flows_max=" << n_flows_max << ", hidden_size_=" << hidden_size_ << std::endl;
}

void M4::SetupML() {
    if (models_loaded) return;
    
    auto setup_start = std::chrono::high_resolution_clock::now();
    
    // Hardcoded network parameters (from test_config.yaml)
    const float buffer_size_cfg = 10.0f;   // Buffer size (bfsz parameter)
    const float fwin_cfg = 10.0f;            // Flow window parameter
    const float dctcp_k_cfg = 10.0f;        // DCTCP threshold parameter
    
    if (!torch::cuda::is_available()) {
        std::cerr << "[M4] ERROR: CUDA is not available!" << std::endl;
        return;
    }
    
    std::cout << "[M4] CUDA is available, proceeding with setup..." << std::endl;
    // Enable cuDNN benchmarking for optimal algorithm selection
    try {
        at::globalContext().setBenchmarkCuDNN(true);
        std::cout << "[M4] cuDNN benchmark enabled: " << at::globalContext().benchmarkCuDNN() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[M4] WARNING: Failed to enable cuDNN benchmark: " << e.what() << std::endl;
    }
    
    torch::NoGradGuard no_grad;
    
    // Model directory: use local models directory
    const std::string model_dir = "./astra-sim-alibabacloud/astra-sim/network_frontend/m4/models/";

    
    // Load ALL models as required by M4 inference (same as inference main_m4_noflowsim.cpp)
    try {
        lstmcell_time = torch::jit::load(model_dir + "lstmcell_time.pt", device);
        lstmcell_rate = torch::jit::load(model_dir + "lstmcell_rate.pt", device);
        lstmcell_rate_link = torch::jit::load(model_dir + "lstmcell_rate_link.pt", device);
        lstmcell_time_link = torch::jit::load(model_dir + "lstmcell_time_link.pt", device);
        output_layer = torch::jit::load(model_dir + "output_layer.pt", device);
        gnn_layer_0 = torch::jit::load(model_dir + "gnn_layer_0.pt", device);
        gnn_layer_1 = torch::jit::load(model_dir + "gnn_layer_1.pt", device);
        gnn_layer_2 = torch::jit::load(model_dir + "gnn_layer_2.pt", device);
        std::cout << "[M4] All models loaded successfully" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "[M4] ERROR: Failed to load models: " << e.what() << std::endl;
        std::cerr << "[M4] ERROR: Model directory: " << model_dir << std::endl;
        models_loaded = false;
        return;
    }

    // Set models to evaluation mode
    lstmcell_time.eval();
    lstmcell_rate.eval();
    lstmcell_rate_link.eval();
    lstmcell_time_link.eval();
    output_layer.eval();
    gnn_layer_0.eval();
    gnn_layer_1.eval();
    gnn_layer_2.eval();

    // Optimize models for inference
    lstmcell_time = torch::jit::optimize_for_inference(lstmcell_time);
    lstmcell_rate = torch::jit::optimize_for_inference(lstmcell_rate);
    lstmcell_time_link = torch::jit::optimize_for_inference(lstmcell_time_link);
    lstmcell_rate_link = torch::jit::optimize_for_inference(lstmcell_rate_link);
    output_layer = torch::jit::optimize_for_inference(output_layer);
    gnn_layer_0 = torch::jit::optimize_for_inference(gnn_layer_0);
    gnn_layer_1 = torch::jit::optimize_for_inference(gnn_layer_1);
    gnn_layer_2 = torch::jit::optimize_for_inference(gnn_layer_2);
    std::cout << "[M4] Model optimization completed" << std::endl;

    models_loaded = true;
    
    std::cout << "[M4] Using network parameters from config: n_links_max=" << n_links_max_ 
              << ", hidden_size=" << hidden_size_ 
              << ", buffer_size=" << buffer_size_cfg 
              << ", fwin=" << fwin_cfg 
              << ", dctcp_k=" << dctcp_k_cfg << std::endl;
    
    // Structure from consts.py: [bfsz(0), fwin(1), dctcp_flag(2), dcqcn_flag(3), hp_flag(4), timely_flag(5), 
    //                           dctcp_k(6), dcqcn_k_min(7), dcqcn_k_max(8), u_tgt(9), hpai(10), timely_t_low(11), timely_t_high(12)]
    std::vector<float> param_values(13, 0.0f);
    
    // Create parameter vector to match inference expectation (loaded from .npy file in inference)
    param_values[0] = buffer_size_cfg;   // bfsz (buffer size from config)
    param_values[1] = fwin_cfg;          // fwin (flow window from config)
    
    // Set CC type: CC_MODE=8 corresponds to DCTCP (index 0 in CC_LIST = ["dctcp", "dcqcn_paper_vwin", "hp", "timely_vwin"])
    // From consts.py: CC_DICT = {"dctcp": 8, ...} - so CC_MODE=8 is indeed DCTCP
    param_values[2] = 1.0f;
    
    // Set DCTCP-specific parameters from config file
    param_values[6] = dctcp_k_cfg;
    
    // Additional parameters to match @inference/ exactly
    param_values[3] = 0.0f;    // dcqcn_flag (not used for DCTCP)
    param_values[4] = 0.0f;    // hp_flag (not used for DCTCP)
    param_values[5] = 0.0f;    // timely_flag (not used for DCTCP)
    param_values[7] = 0.0f;    // dcqcn_k_min (not used for DCTCP)
    param_values[8] = 0.0f;    // dcqcn_k_max (not used for DCTCP)
    param_values[9] = 0.0f;    // u_tgt
    param_values[10] = 0.0f;   // hpai (not used for DCTCP)
    param_values[11] = 0.0f;   // timely_t_low (not used for DCTCP)
    param_values[12] = 0.0f;   // timely_t_high (not used for DCTCP)
    
    params_tensor = torch::tensor(param_values, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Read topology parameters for logging
    float topo_bandwidth = topology->get_bandwidth(); // in bps
    float topo_latency = topology->get_latency(); // in ns
    
    std::cout << "[M4] Loaded network parameters from config: bfsz=" << param_values[0] << ", fwin=" << param_values[1] 
              << ", cc=dctcp, u_tgt=" << param_values[9] << ", dctcp_k=" << param_values[6] << ", topology_bw=" << (topo_bandwidth * 8.0) << "Gbps, topology_lat=" << topo_latency << "ns, batch_time_ns=" << batch_time_ns_ 
              << ", reschedule_flow_count=" << reschedule_flow_count_ << std::endl;
    
    // Initialize multi-flow state tensors (from @inference/ ground truth)
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);
    
    // Initialize flow and link state tensors (NOT all zeros - need flow-specific features!)
    h_vec = torch::zeros({n_flows_max, hidden_size_}, options_float);
    // Set first column to 1.0 for all flows (matching inference code)
    h_vec.index_put_({torch::arange(n_flows_max, device=device), 0}, 1.0f);
    // Note: Flow sizes and hop counts will be set dynamically in Send() when flows are created
    flowid_active_mask = torch::zeros({n_flows_max}, options_bool);
    time_last = torch::zeros({n_flows_max}, options_float);
    release_time_tensor = torch::zeros({n_flows_max}, options_float);
    flowid_to_nlinks_tensor = torch::zeros({n_flows_max}, options_int32);
    i_fct_tensor = torch::zeros({n_flows_max}, options_float);
    
    // Initialize per-link state
    z_t_link = torch::zeros({n_links_max_, hidden_size_}, options_float);
    z_t_link.index_put_({torch::arange(n_links_max_, device=device), 1}, 1.0f);
    z_t_link.index_put_({torch::arange(n_links_max_, device=device), 2}, 1.0f);
    link_to_graph_id = -torch::ones({n_links_max_}, options_int32);
    link_to_nflows = torch::zeros({n_links_max_}, options_int32);
    
    // Initialize graph management tensors
    flow_to_graph_id = -torch::ones({n_flows_max}, options_int32);
    
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start).count();
    std::cout << "[M4] SetupML() completed in " << setup_duration << "ms!" << std::endl;
}

void M4::OnFlowCompleted(const int flow_id) {
    // Simple: flow_id is already the internal monotonic ID
    if (flow_id < 0 || flow_id >= n_flows_max) return;
    const auto &links = flowid_to_link_indices[flow_id];
    if (links.empty()) {
        flowid_active_mask[flow_id] = false;
        flow_to_graph_id[flow_id] = -1;
        return;
    }
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::vector<int32_t> links_vec(links.begin(), links.end());
    auto idx = torch::from_blob(links_vec.data(), {(int)links_vec.size()}, torch::TensorOptions().dtype(torch::kInt32)).to(device);
    auto ones_i32 = torch::ones({(int)links.size()}, options_int32);
    auto cur = link_to_nflows.index_select(0, idx);
    auto new_counts = torch::clamp(cur - ones_i32, 0);
    link_to_nflows.index_put_({idx}, new_counts);

    // Links that became idle now
    auto idle_mask = (new_counts == 0);
    if (idle_mask.any().item<bool>()) {
        auto idle_links = torch::nonzero(idle_mask).flatten().to(torch::kInt64);
        if (idle_links.numel() > 0) {
        // Clear graph id and reset z_t_link rows
            link_to_graph_id.index_put_({idle_links}, torch::full({idle_links.size(0)}, -1, options_int32));
        auto reset_values = torch::zeros({idle_links.size(0), z_t_link.size(1)}, options_float);
        z_t_link.index_put_({idle_links, torch::indexing::Slice()}, reset_values);
            auto ones_vec = torch::ones({idle_links.size(0)}, options_float);
            z_t_link.index_put_({idle_links, 1}, ones_vec);
            z_t_link.index_put_({idle_links, 2}, ones_vec);
        }
    }

    // Clear flow state
    flowid_active_mask[flow_id] = false;
    flow_to_graph_id[flow_id] = -1;
    
    // Remove any stored completion event id (should be gone already after firing)
    auto it_e = flow_id_to_completion_event_id.find(flow_id);
    if (it_e != flow_id_to_completion_event_id.end()) {
        flow_id_to_completion_event_id.erase(it_e);
    }
    
    // Update counter and check equation: active = arrived - completed
    n_flows_completed++;
    int32_t n_flows_active_current = torch::nonzero(flowid_active_mask).flatten().numel();
    assert(n_flows_active_current == (n_flows_arrived - n_flows_completed));
    
    // Clean up completed flow from active tracking
    CleanupCompletedFlow(flow_id);
}

void M4::CleanupCompletedFlow(const int flow_id) {
    // OPTIMIZATION: Remove from fast lookup map
    flow_id_to_ptr_.erase(flow_id);
    
    // Remove completed flow from active_flows_ptrs to prevent memory leak
    auto it = std::remove_if(active_flows_ptrs.begin(), active_flows_ptrs.end(),
        [flow_id](const std::unique_ptr<M4Flow>& flow_ptr) {
            return flow_ptr && flow_ptr->flow_id == flow_id;
        });
    
    if (it != active_flows_ptrs.end()) {
        active_flows_ptrs.erase(it, active_flows_ptrs.end());
    }
}

void M4::SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework) {
    routing_framework_ = std::move(routing_framework);
}

void M4::Run() {
    // New design: completions are scheduled per temporal batch in process_batch_of_flows.
    while (!event_queue->finished()) {
        event_queue->proceed();
    }
}

void M4::Schedule(uint64_t delay, void (*fun_ptr)(void* fun_arg), void* fun_arg) {
    // Use M4's event queue (same as FlowSim)
    uint64_t time = event_queue->get_current_time() + delay;
    event_queue->schedule_event(time, fun_ptr, fun_arg);
}

double M4::Now() {
    // Use M4's event queue time (same as FlowSim)
    return event_queue->get_current_time();
}

// CRITICAL FIX: Single ideal FCT calculation function
// This ensures ML processing and FCT logging use identical values
uint64_t M4::CalculateIdealFCT(int src, int dst, uint64_t size) {
    if (!routing_framework_) {
        throw std::runtime_error("[M4 ERROR] RoutingFramework is null when computing ideal FCT");
    }
    
    uint64_t base_rtt = routing_framework_->GetPairRtt(src, dst);
    uint64_t b_bps = routing_framework_->GetPairBandwidth(src, dst);
    
    const uint32_t packet_payload_size = 1000u;
    const uint32_t header_overhead = 48u;
    uint64_t num_pkts = (size + packet_payload_size - 1) / packet_payload_size;
    uint64_t total_bytes = size + num_pkts * header_overhead;
    
    return base_rtt + total_bytes * 8000000000lu / b_bps;
}

// Unified scheduling helper used everywhere we schedule/reschedule a flow
void M4::ScheduleWithRemainingTime(int32_t flow_id, uint64_t now_ns, uint64_t remaining_ns) {
    if (remaining_ns == 0) remaining_ns = 1ULL;
    uint64_t completion_time = now_ns + remaining_ns;

    // OPTIMIZATION: O(1) lookup instead of O(N) linear search
    auto it_ptr = flow_id_to_ptr_.find(flow_id);
    if (it_ptr == flow_id_to_ptr_.end()) return; // flow completed or not found
    M4Flow* fptr = it_ptr->second;

    auto it_e = flow_id_to_completion_event_id.find(flow_id);
    if (it_e != flow_id_to_completion_event_id.end()) {
        // Bidirectional: allow pull-in or push-out, but never schedule in the past
        uint64_t prev = flow_id_to_scheduled_time_ns[flow_id];
        if (completion_time != prev && completion_time > now_ns) {
            event_queue->cancel_event(it_e->second);
            EventId new_eid = event_queue->schedule_event(completion_time, fptr->callback, fptr->callbackArg);
            it_e->second = new_eid;
            flow_id_to_scheduled_time_ns[flow_id] = completion_time;
        }
    } else {
        EventId eid = event_queue->schedule_event(completion_time, fptr->callback, fptr->callbackArg);
        flow_id_to_completion_event_id[flow_id] = eid;
        flow_id_to_scheduled_time_ns[flow_id] = completion_time;
    }
}

// Remove old completion callback - now handled by event-driven processing

void M4::Send(int src, int dst, uint64_t size, int tag, Callback callback, CallbackArg callbackArg) {
    // M4 integration with ASTRA-Sim following FlowSim's pattern
    
    if (!models_loaded) {
        std::cerr << "[M4 ERROR] ML models not loaded! Cannot process flow." << std::endl;
        throw std::runtime_error("M4 ML models not loaded");
    }
    
    // Check AS_NVLS_ENABLE for hardware acceleration simulation (same as FlowSim)
    const char* nvls_env = std::getenv("AS_NVLS_ENABLE");
    if (nvls_env && std::stoi(nvls_env) == 1) {
        if (size < 4096 && size > 0) {
            size = 4096; // Minimum chunk size with NVLS
        }
    }
    
    // Get pre-calculated path from routing framework (same as FlowSim)
    std::vector<int> node_path = routing_framework_->GetFlowSimPathByNodeIds(src, dst);
    
    // CRITICAL FIX: Use shared ideal FCT calculation
    uint64_t correct_ideal_fct = CalculateIdealFCT(src, dst, size);

        // Create M4Flow and add to pending batch (following FlowSim's temporal batching)
        auto m4_flow = std::make_unique<M4Flow>(src, dst, size, node_path, callback, callbackArg);
    
    // Use ASTRA-Sim flow id and actual send start time if available
        if (callbackArg) {
            auto* cd = reinterpret_cast<M4CallbackData*>(callbackArg);
            m4_flow->flow_id = cd->flowTag.current_flow_id;
        m4_flow->start_time = cd->start_time; // actual network start after AS_SEND_LAT
        
        // Store start time
        flow_id_to_start_time_ns[cd->flowTag.current_flow_id] = cd->start_time;
        // Baseline scheduling at ideal FCT; ML will only push later
        uint64_t base_completion = cd->start_time + correct_ideal_fct;
        EventId base_eid = event_queue->schedule_event(base_completion, callback, callbackArg);
        flow_id_to_completion_event_id[cd->flowTag.current_flow_id] = base_eid;
        flow_id_to_scheduled_time_ns[cd->flowTag.current_flow_id] = base_completion;
        
        // Mark flow as active for ML immediately when scheduled
        // This ensures ML graph only includes truly in-flight flows
        flowid_active_mask[cd->flowTag.current_flow_id] = true;
        flow_to_graph_id[cd->flowTag.current_flow_id] = 0;
        n_flows_arrived++;
    } else {
        m4_flow->start_time = static_cast<uint64_t>(event_queue->get_current_time());
    }

    // Add to pending batch for flow-count processing
        int32_t flow_id = m4_flow->flow_id; // Save flow_id before moving m4_flow
        pending_flows_.push_back(m4_flow.get());
        // Keep ownership in active_flows_ptrs until batch processing
        active_flows_ptrs.push_back(std::move(m4_flow));
        // OPTIMIZATION: Register flow pointer for O(1) lookup during rescheduling
        flow_id_to_ptr_[flow_id] = active_flows_ptrs.back().get();
        
    // Temporal batching: arm one update at now + batch_time_ns_
        const auto current_time = event_queue->get_current_time();
    if (batch_timeout_event_id_ == 0) {
        batch_timeout_event_id_ = event_queue->schedule_event(current_time + batch_time_ns_, batch_timeout_callback, nullptr);
    }
}

// Batch processing callback (following FlowSim's pattern)
void M4::batch_timeout_callback(void* arg) {
    // Drain all pending flows in a single update, then re-arm timer for temporal batching
    // Clear the current timer id to allow re-arming
    batch_timeout_event_id_ = 0;
    process_batch_of_flows_count((int32_t)pending_flows_.size());
    const auto now = event_queue->get_current_time();
    // Re-arm only if there is work (active or pending flows)
    bool has_work = !pending_flows_.empty() || (torch::nonzero(flowid_active_mask).flatten().numel() > 0);
    if (has_work) {
        batch_timeout_event_id_ = event_queue->schedule_event(now + batch_time_ns_, batch_timeout_callback, nullptr);
    }
}

// Process final batch at simulation end (handles remaining flows in final time window)
void M4::process_final_batch() {
    // Drain-and-process loop until no pending flows and no scheduled events remain
    for (;;) {
        // Fully drain all scheduled events (safe: EventQueue pops before invoke)
        while (!event_queue->finished()) {
            event_queue->proceed();
        }

        // If no pending flows remain after draining, we're done
        if (pending_flows_.empty()) {
            break;
        }

        // Drain all remaining flows regardless of size
        if (!pending_flows_.empty()) {
            process_batch_of_flows_count((int32_t)pending_flows_.size());
        }

        // Loop again to drain the completions we just scheduled and catch any
        // new pending flows triggered by callbacks.
    }
}

// FlowSim-style batch processing with inference ML logic
void M4::process_batch_of_flows() {
    process_batch_of_flows_count((int32_t)pending_flows_.size());
}

void M4::process_batch_of_flows_count(int32_t max_flows) {
    if (pending_flows_.empty() || max_flows <= 0) {
        return;
    }
    // Reset batch event ID to allow next scheduling
    batch_timeout_event_id_ = 0;
    // Take exactly max_flows from the front
    int32_t take = std::min((int32_t)pending_flows_.size(), max_flows);
    std::vector<M4Flow*> flows_to_process;
    flows_to_process.reserve(take);
    for (int32_t i = 0; i < take; ++i) flows_to_process.push_back(pending_flows_[i]);
    pending_flows_.erase(pending_flows_.begin(), pending_flows_.begin() + take);
    
    const auto current_time = event_queue->get_current_time();
    time_clock = static_cast<float>(current_time);
    // Use global graph so all flows can interact (no batch fragmentation)
    
    // Initialize per-flow state for this batch
    current_batch_link_set.clear();
    std::unordered_map<int32_t, int32_t> batch_link_counts;
    std::vector<int64_t> flow_ids_batch;
    flow_ids_batch.reserve(flows_to_process.size());
    int flows_arriving_this_batch = 0;
    for (M4Flow* flow : flows_to_process) {
        if (!flow) continue;
        
        // Use ASTRA-Sim's flow ID directly (already set in Send())
        int flow_id = flow->flow_id;
        
        // Ensure ASTRA-Sim's flow ID is within our tensor capacity
        if (flow_id < 0 || flow_id >= n_flows_max) {
            throw std::runtime_error("[M4 ERROR] ASTRA-Sim flow ID out of range: " + std::to_string(flow_id) + " (valid: 0-" + std::to_string(n_flows_max-1) + ")");
        }
        
        // Do not overwrite actual start_time set at send; just count arrivals
        flows_arriving_this_batch++;

        uint64_t size = flow->size;
        double size_bytes = static_cast<double>(size);
        
        // CRITICAL FIX: Use shared ideal FCT calculation
        double ideal_fct = (double)CalculateIdealFCT(flow->src, flow->dst, size);
        // Get route for hop count (still needed for ML features) via RoutingFramework
        std::vector<int> ns3_route = routing_framework_->GetFlowSimPathByNodeIds(flow->src, flow->dst);
        if (ns3_route.size() < 2) {
            throw std::runtime_error("[M4 ERROR] Empty/invalid NS3 route for hop count");
        }
        // {
        //     static int route_debug_count = 0;
        //     if (route_debug_count < 20) {
        //         std::cout << "[ROUTE CHECK] src=" << flow->src
        //                   << " dst=" << flow->dst
        //                   << " b_bps=" << b_bps
        //                   << " rtt_ns=" << base_rtt
        //                   << " path=";
        //         for (size_t k = 0; k < ns3_route.size(); ++k) {
        //             if (k) std::cout << "->";
        //             std::cout << ns3_route[k];
        //         }
        //         std::cout << std::endl;
        //         route_debug_count++;
        //     }
        // }
        int ns3_num_links = static_cast<int>(ns3_route.size()) - 1;
        flowid_to_nlinks_tensor[flow_id] = ns3_num_links;
        i_fct_tensor[flow_id] = static_cast<float>(ideal_fct);
        release_time_tensor[flow_id] = time_clock;
        // Preserve true start time set at send; only set if missing
        if (!flow_id_to_start_time_ns.count(flow_id)) {
            flow_id_to_start_time_ns[flow_id] = flow->start_time;
        }
        time_last[flow_id] = time_clock;  // Initialize time_last to avoid massive time deltas
        // Note: flowid_active_mask already set to true in M4::Send when scheduled
        // Initialize h_vec features
        h_vec[flow_id].zero_();
        h_vec[flow_id][0] = 1.0f;
        h_vec[flow_id][2] = std::log2(size_bytes / 1000.0 + 1.0);
        h_vec[flow_id][3] = static_cast<float>(ns3_num_links);
        if (flow_id >= (int)flowid_to_link_indices.size()) flowid_to_link_indices.resize(flow_id + 1);
        if (flowid_to_link_indices[flow_id].empty()) {
            std::vector<int32_t> flow_link_indices;
            for (int i = 0; i < ns3_num_links; i++) {
                int src_node = ns3_route[i];
                int dst_node = ns3_route[i + 1];
                long long link_key = ((long long)src_node << 32) | dst_node;
                if (link_key_to_index.find(link_key) == link_key_to_index.end()) {
                    link_key_to_index[link_key] = next_link_index++;
                }
                int32_t lid = link_key_to_index[link_key];
                flow_link_indices.push_back(lid);
                current_batch_link_set.insert(lid);
                batch_link_counts[lid] += 1;
            }
            flowid_to_link_indices[flow_id] = std::move(flow_link_indices);
        } else {
            // Add existing links to batch set
            for (int32_t lid : flowid_to_link_indices[flow_id]) {
                current_batch_link_set.insert(lid);
                batch_link_counts[lid] += 1;
            }
        }

        flow_ids_batch.push_back(flow_id);
    }
    
    // Update counters and check equation: active = arrived - completed
    // Note: n_flows_arrived incremented in M4::Send when flows are scheduled
    int32_t n_flows_active_current = torch::nonzero(flowid_active_mask).flatten().numel();
    assert(n_flows_active_current == (n_flows_arrived - n_flows_completed));

    // Increment link_to_nflows and tag graph id for links touched by this batch (parity with @inference)
    if (!batch_link_counts.empty()) {
        std::vector<int32_t> link_ids_vec;
        std::vector<int32_t> link_incrs_vec;
        link_ids_vec.reserve(batch_link_counts.size());
        link_incrs_vec.reserve(batch_link_counts.size());
        for (const auto &kv : batch_link_counts) {
            link_ids_vec.push_back(kv.first);
            link_incrs_vec.push_back(kv.second);
        }
        auto link_idx64 = torch::from_blob(link_ids_vec.data(), {(int)link_ids_vec.size()}, torch::TensorOptions().dtype(torch::kInt32))
                               .to(torch::kInt64).to(device);
        auto incr_i32 = torch::from_blob(link_incrs_vec.data(), {(int)link_incrs_vec.size()}, torch::TensorOptions().dtype(torch::kInt32))
                               .to(device);
        auto cur_counts = link_to_nflows.index_select(0, link_idx64);
        link_to_nflows.index_put_({link_idx64}, cur_counts + incr_i32);
        auto graph_id_fill = torch::full({(int)link_ids_vec.size()}, 0, torch::TensorOptions().dtype(torch::kInt32).device(device));
        link_to_graph_id.index_put_({link_idx64}, graph_id_fill);
    }

    // Evolve states for interacting flows (LSTM -> GNN) before prediction
    // Declare interacting set outside the block to reuse below for rescheduling
    std::set<int32_t> interacting_flows;
    {
        torch::NoGradGuard no_grad;
        time_clock = static_cast<float>(current_time);
        
        // Batch-local graph: Only reschedule flows that interact with the current batch
        // Add current batch flows
        for (int64_t fid : flow_ids_batch) {
            interacting_flows.insert((int32_t)fid);
        }
        // Add other active flows that share at least one link with current batch
        auto flowid_active_list_all = torch::nonzero(flowid_active_mask).flatten();
        if (flowid_active_list_all.numel() > 0) {
            // Identify interacting active flows by link intersection
            for (int i = 0; i < flowid_active_list_all.size(0); i++) {
                int32_t fid = flowid_active_list_all[i].item<int32_t>();
                if (fid < (int32_t)flowid_to_link_indices.size()) {
                    const auto &links = flowid_to_link_indices[fid];
                    for (int32_t lid : links) {
                        if (current_batch_link_set.find(lid) != current_batch_link_set.end()) {
                            interacting_flows.insert(fid);
                            break;
                        }
                    }
                }
            }
            // Build edges over ALL active flows and all their links (match inference main_m4_noflowsim)
            std::vector<int32_t> flow_edges;
            std::vector<int32_t> link_edges;
            for (int i = 0; i < flowid_active_list_all.size(0); i++) {
                int32_t fid = flowid_active_list_all[i].item<int32_t>();
                if (fid < (int32_t)flowid_to_link_indices.size()) {
                    const auto &links = flowid_to_link_indices[fid];
                    for (int32_t lid : links) {
                        flow_edges.push_back(fid);
                        link_edges.push_back(lid);
                    }
                }
            }
            if (!flow_edges.empty()) {
                auto edge_flow_tensor = torch::from_blob(flow_edges.data(), {(int)flow_edges.size()}, torch::TensorOptions().dtype(torch::kInt32))
                                           .to(torch::kInt64).to(device);
                auto edge_link_tensor = torch::from_blob(link_edges.data(), {(int)link_edges.size()}, torch::TensorOptions().dtype(torch::kInt32))
                                           .to(torch::kInt64).to(device);

                // Map to compact indices for active flows/links
                auto flowid_active_tensor = flowid_active_list_all.to(torch::kInt64);
                auto new_flow_indices = torch::searchsorted(flowid_active_tensor, edge_flow_tensor);
                int n_flows_active_cur = (int)flowid_active_tensor.size(0);

                auto unique_links_tuple = torch::_unique(edge_link_tensor, true, true);
                auto active_link_idx = std::get<0>(unique_links_tuple);
                auto new_link_indices = std::get<1>(unique_links_tuple);
                new_link_indices += n_flows_active_cur;

                auto edges_list_active = torch::cat({
                    torch::stack({new_flow_indices, new_link_indices}, 0),
                    torch::stack({new_link_indices, new_flow_indices}, 0)
                }, 1);

                auto subset_indices = flowid_active_tensor;
                auto time_deltas = (time_clock - time_last.index_select(0, subset_indices).squeeze()).view({-1, 1});
                auto h_vec_time_updated = h_vec.index_select(0, subset_indices);
                auto h_vec_time_link_updated = z_t_link.index_select(0, active_link_idx);
                auto max_time_delta = torch::max(time_deltas).template item<float>();
                if (max_time_delta > 0.0f) {
                    time_deltas.fill_(max_time_delta / 1000.0f);
                    h_vec_time_updated = lstmcell_time.forward(std::vector<c10::IValue>{time_deltas, h_vec_time_updated}).toTensor();
                    auto time_deltas_link = torch::zeros({active_link_idx.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
                    time_deltas_link.fill_(max_time_delta / 1000.0f);
                    h_vec_time_link_updated = lstmcell_time_link.forward(std::vector<c10::IValue>{time_deltas_link, h_vec_time_link_updated}).toTensor();
                }

                // Optional debug
                // std::cout << "[GNN Update] Flow nodes: " << n_flows_active_cur
                //           << ", Link nodes: " << active_link_idx.size(0)
                //           << ", Edges: " << edges_list_active.size(1) << std::endl;

                auto x_combined = torch::cat({h_vec_time_updated, h_vec_time_link_updated}, 0);
                auto gnn_output_0 = gnn_layer_0.forward(std::vector<c10::IValue>{x_combined, edges_list_active}).toTensor();
                auto gnn_output_1 = gnn_layer_1.forward(std::vector<c10::IValue>{gnn_output_0, edges_list_active}).toTensor();
                auto gnn_output_2 = gnn_layer_2.forward(std::vector<c10::IValue>{gnn_output_1, edges_list_active}).toTensor();
                auto h_vec_rate_updated = gnn_output_2.slice(0, 0, n_flows_active_cur);
                auto h_vec_rate_link = gnn_output_2.slice(0, n_flows_active_cur, gnn_output_2.size(0));
                auto params_data = params_tensor.repeat({n_flows_active_cur, 1});
                h_vec_rate_updated = torch::cat({h_vec_rate_updated, params_data}, 1);
                h_vec_rate_updated = lstmcell_rate.forward(std::vector<c10::IValue>{h_vec_rate_updated, h_vec_time_updated}).toTensor();
                h_vec_rate_link = lstmcell_rate_link.forward(std::vector<c10::IValue>{h_vec_rate_link, h_vec_time_link_updated}).toTensor();
                h_vec.index_copy_(0, subset_indices, h_vec_rate_updated);
                z_t_link.index_copy_(0, active_link_idx.to(torch::kInt64), h_vec_rate_link);
                time_last.index_put_({torch::indexing::TensorIndex(subset_indices)}, time_clock);

                // MLP prediction for ALL active flows
                auto nlinks_batch_all = flowid_to_nlinks_tensor.index_select(0, subset_indices).unsqueeze(1);
                auto params_batch_all = params_tensor.unsqueeze(0).repeat({n_flows_active_cur, 1});
                auto input_batch_all = torch::cat({nlinks_batch_all, params_batch_all, h_vec_rate_updated}, 1);
                auto sldn_all = output_layer.forward(std::vector<c10::IValue>{input_batch_all}).toTensor().view(-1);
                sldn_all = torch::clamp(sldn_all, 1.0f, std::numeric_limits<float>::infinity());
                
                // Transfer slowdown predictions from GPU to CPU for processing
                auto sldn_cpu = sldn_all.to(torch::kCPU);
                auto sldn_data = sldn_cpu.data_ptr<float>();
                auto flowid_active_cpu = flowid_active_tensor.to(torch::kCPU);
                auto flowid_active_data = flowid_active_cpu.data_ptr<int64_t>();
                
                // SMART RESCHEDULING: Balance correctness vs performance
                // - Always reschedule flows in current batch (new arrivals)
                // - Periodically reschedule ALL active flows (every reschedule_flow_count arrivals)
                // This avoids flows getting stuck at baseline while keeping most batches fast
                uint64_t now_ns = current_time;
                
                // Build set of flow IDs in current batch for fast lookup
                std::unordered_set<int32_t> batch_flow_set;
                for (int64_t fid : flow_ids_batch) {
                    batch_flow_set.insert((int32_t)fid);
                }
                
                // Decide whether to reschedule all flows or just new arrivals
                bool reschedule_all = false;
                if (reschedule_flow_count_ == 0) {
                    // Always reschedule all (slow but most accurate)
                    reschedule_all = true;
                } else if (n_flows_since_last_reschedule >= reschedule_flow_count_) {
                    // Periodic full reschedule to prevent stale predictions
                    reschedule_all = true;
                    n_flows_since_last_reschedule = 0; // Reset counter
                }
                
                // Update arrival counter for next decision
                n_flows_since_last_reschedule += flows_arriving_this_batch;
                
                // Reschedule flows with their updated predictions
                for (int i = 0; i < n_flows_active_cur; i++) {
                    int32_t flow_id = (int32_t)flowid_active_data[i];
                    
                    // Skip if not in batch AND we're not doing a full reschedule
                    bool is_in_batch = batch_flow_set.count(flow_id) > 0;
                    if (!is_in_batch && !reschedule_all) {
                        continue; // Skip to save time
                    }
                    
                    float raw_slowdown = sldn_data[i];
                    float scaled_slowdown = (raw_slowdown < 1.0f) ? 1.0f : raw_slowdown;
                    
                    float ideal_fct = i_fct_tensor[flow_id].item<float>();
                    float predicted_total_fct = scaled_slowdown * ideal_fct;
                    
                    // Get flow start time
                    auto start_it = flow_id_to_start_time_ns.find(flow_id);
                    if (start_it == flow_id_to_start_time_ns.end()) {
                        continue;
                    }
                    uint64_t start_ns = start_it->second;
                    uint64_t elapsed = now_ns > start_ns ? (now_ns - start_ns) : 0ULL;
                    uint64_t remaining = (predicted_total_fct > (float)elapsed) 
                                        ? (uint64_t)(predicted_total_fct - (float)elapsed) 
                                        : 1ULL;
                    
                    // ScheduleWithRemainingTime is efficient: only reschedules if time changed
                    ScheduleWithRemainingTime(flow_id, now_ns, remaining);
                }
                
                // Schedule new arrivals (flows in flows_to_process that have short/empty routes)
                for (size_t i = 0; i < flows_to_process.size(); i++) {
                    M4Flow* flow = flows_to_process[(int)i];
                    int flow_id = flow->flow_id;
                    
                    // Handle short/empty routes (immediate completion with minimal FCT)
                    if (flow->node_path.empty() || flow->node_path.size() < 2) {
                        float minimal_fct = 1.0f; // 1ns minimal FCT for local/short flows
                        uint64_t completion_time = current_time + (uint64_t)minimal_fct;
                        event_queue->schedule_event(completion_time, flow->callback, flow->callbackArg);
                        continue; // Skip ML inference for short routes
                    }
                    // Normal flows already rescheduled above via ALL active flows loop
                }
            }
        }
    }
    
    // All flows (both new arrivals and existing active flows) have been rescheduled above
}

const AstraSim::RoutingFramework* M4::GetRoutingFramework() {
    return routing_framework_.get();
}

void M4::Stop() {
    // Stop processing events (same as FlowSim)
    // EventQueue doesn't have a clear method, so just let it finish naturally
    if (event_queue) {
        // Event queue will be cleared when destroyed
    }
}

void M4::Destroy() {
    // Clear static resources in proper order (same as FlowSim)
    routing_framework_.reset();
    topology.reset();
    event_queue.reset();
}

std::unordered_map<int32_t, EventId> M4::flow_id_to_completion_event_id;
std::unordered_map<int32_t, M4Flow*> M4::flow_id_to_ptr_;

static inline float clamp_ge1(float v) { return v < 1.0f ? 1.0f : v; }

// New: track per-flow times for MLP push-out
std::unordered_map<int32_t, uint64_t> M4::flow_id_to_start_time_ns;
std::unordered_map<int32_t, uint64_t> M4::flow_id_to_scheduled_time_ns;

