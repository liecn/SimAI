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
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <fstream>
#include <sstream>
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
torch::Tensor M4::edge_index;
torch::Tensor M4::z_t_link;
torch::Tensor M4::link_to_graph_id;
torch::Tensor M4::link_to_nflows;
torch::Tensor M4::flow_to_graph_id;
torch::Tensor M4::time_last;
torch::Tensor M4::release_time_tensor;
torch::Tensor M4::flowid_to_nlinks_tensor;
torch::Tensor M4::i_fct_tensor;

// Additional tensors from @inference/ for complete ML pipeline
torch::Tensor M4::flowid_to_linkid_flat_tensor;
torch::Tensor M4::flowid_to_linkid_offsets_tensor;
torch::Tensor M4::edges_flow_ids_tensor;
torch::Tensor M4::edges_link_ids_tensor;
torch::Tensor M4::ones_cache;

// Flow and graph management
int32_t M4::hidden_size_ = 200; // Model expects 214 total: 1+13+200=214 (matches main_m4_noflowsim.cpp)
int32_t M4::n_links_max_ = 128;
int32_t M4::batch_size_flows_ = 128; // Flow-count batching size
int32_t M4::n_flows_max = 50000;  // Large enough for simulation
int32_t M4::graph_id_cur = 0;
float M4::time_clock = 0.0f;
std::unordered_map<long long, int32_t> M4::link_key_to_index;
int32_t M4::next_link_index = 0;
std::vector<std::vector<int32_t>> M4::flowid_to_link_indices;
std::unordered_set<int32_t> M4::current_batch_link_set;

// Flow lifecycle tracking counters
static int32_t n_flows_arrived = 0;
static int32_t n_flows_completed = 0;



// FlowSim-style temporal batching
std::vector<M4Flow*> M4::pending_flows_;
std::list<std::unique_ptr<M4Flow>> M4::active_flows_ptrs;
int M4::batch_timeout_event_id_ = 0;
bool M4::is_processing_batch_ = false;

// (removed) legacy inference-style flow completion tracking

// Remove old scheduling logic - now handled by event-driven processing

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

    // Parse config for model and network parameters
    float buffer_size_cfg = 1000.0f;  // Default fallback
    float fwin_cfg = 15.0f;           // Default fallback  
    float dctcp_k_cfg = 30.0f;        // Default fallback
    
    // Check for environment variable overrides first
    const char* fwin_env = std::getenv("AS_FWIN");
    if (fwin_env) {
        fwin_cfg = std::stof(fwin_env);
    }
    
    try {
        const std::string cfg_path = "./astra-sim-alibabacloud/astra-sim/network_frontend/m4/config/test_config.yaml";
        std::ifstream infile(cfg_path);
        if (infile.good()) {
            std::ostringstream contents;
            contents << infile.rdbuf();
            std::string config_contents = contents.str();
            ryml::Tree config = ryml::parse_in_place(ryml::to_substr(config_contents));
            
            // Parse existing model parameters
            ryml::NodeRef hidden_size_node = config["model"]["hidden_size"];
            ryml::NodeRef n_links_node = config["dataset"]["n_links_max"];
            hidden_size_node >> hidden_size_;
            n_links_node >> n_links_max_;
            
            // Parse network parameters from config (environment variables take precedence)
            try {
                auto network_node = config["network"];
                if (!network_node.invalid()) {
                    try { network_node["buffer_size"] >> buffer_size_cfg; } catch(...) {}
                    // Only use config fwin if environment variable wasn't set
                    if (!fwin_env) {
                        try { network_node["fwin"] >> fwin_cfg; } catch(...) {}
                    }
                    try { network_node["dctcp_k"] >> dctcp_k_cfg; } catch(...) {}
                }
            } catch(...) {
                // network section doesn't exist, use defaults
            }
            
            // Parse M4-specific parameters from config
            try {
                auto m4_node = config["m4"];
                if (!m4_node.invalid()) {
                    try { m4_node["batch_size_flows"] >> batch_size_flows_; } catch(...) {}
                }
            } catch(...) {
                // m4 section doesn't exist, use defaults
            }
            
        } else {
            throw std::runtime_error("M4 config missing");
        }
    } catch (const std::exception& e) {
        std::cerr << "[M4] Config parse error: " << e.what() << std::endl;
        hidden_size_ = 64;
        n_links_max_ = 128;
    }
    
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
              << ", cc=dctcp, u_tgt=" << param_values[9] << ", dctcp_k=" << param_values[6] << ", topology_bw=" << (topo_bandwidth * 8.0) << "Gbps, topology_lat=" << topo_latency << "ns, batch_size_flows=" << batch_size_flows_ << std::endl;
    
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
    
    // Initialize link state tensors
    z_t_link = torch::zeros({n_links_max_, hidden_size_}, options_float);
    z_t_link.index_put_({torch::arange(n_links_max_, device=device), 1}, 1.0f);
    z_t_link.index_put_({torch::arange(n_links_max_, device=device), 2}, 1.0f);
    
    // Initialize graph management tensors
    link_to_graph_id = -torch::ones({n_links_max_}, options_int32);
    link_to_nflows = torch::zeros({n_links_max_}, options_int32);
    flow_to_graph_id = -torch::ones({n_flows_max}, options_int32);
    
    // Initialize empty edge_index - will be built dynamically as flows are added
    edge_index = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    
    // Initialize additional tensors for complete ML pipeline
    flowid_to_linkid_flat_tensor = torch::empty({0}, options_int32);
    flowid_to_linkid_offsets_tensor = torch::empty({0}, options_int32);
    edges_flow_ids_tensor = torch::empty({0}, options_int32);
    edges_link_ids_tensor = torch::empty({0}, options_int32);
    ones_cache = torch::ones({1000}, options_float); // Pre-allocate for efficiency
    
    
    
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
        auto idle_links = idx.masked_select(idle_mask);
        // Clear graph id and reset z_t_link rows
        link_to_graph_id.index_put_({idle_links}, -1);
        auto reset_values = torch::zeros({idle_links.size(0), z_t_link.size(1)}, options_float);
        z_t_link.index_put_({idle_links, torch::indexing::Slice()}, reset_values);
        z_t_link.index_put_({idle_links, 1}, torch::ones({idle_links.size(0)}, options_float));
        z_t_link.index_put_({idle_links, 2}, torch::ones({idle_links.size(0)}, options_float));
    }

    // Clear flow state
    flowid_active_mask[flow_id] = false;
    flow_to_graph_id[flow_id] = -1;
    
    // Update counter and check equation: active = arrived - completed
    n_flows_completed++;
    int32_t n_flows_active_current = torch::nonzero(flowid_active_mask).flatten().numel();
    assert(n_flows_active_current == (n_flows_arrived - n_flows_completed));
    
    // Clean up completed flow from active tracking
    CleanupCompletedFlow(flow_id);
}

void M4::CleanupCompletedFlow(const int flow_id) {
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
    
    // Create M4Flow and add to pending batch (following FlowSim's temporal batching)
    auto m4_flow = std::make_unique<M4Flow>(src, dst, size, node_path, callback, callbackArg);
    // Try to use flowTag.current_flow_id as global flow_id if available
    if (callbackArg) {
        auto* cd = reinterpret_cast<M4CallbackData*>(callbackArg);
        m4_flow->flow_id = cd->flowTag.current_flow_id;
    }

    // Add to pending batch for flow-count processing
    pending_flows_.push_back(m4_flow.get());
    // Keep ownership in active_flows_ptrs until batch processing
    active_flows_ptrs.push_back(std::move(m4_flow));
    
    // Flow-count batching: trigger when we have enough flows
    const auto current_time = event_queue->get_current_time();
    if (batch_timeout_event_id_ == 0 && (int)pending_flows_.size() >= batch_size_flows_) {
        batch_timeout_event_id_ = event_queue->schedule_event(current_time, batch_timeout_callback, nullptr);
    }
}

// Batch processing callback (following FlowSim's pattern)
void M4::batch_timeout_callback(void* arg) {
    process_batch_of_flows_count(batch_size_flows_);
}

// Process final batch at simulation end (handles remaining flows in final time window)
void M4::process_final_batch() {
    // Stable finalize loop with safety limits to prevent infinite loops
    int max_iterations = 1000; // Safety limit to prevent infinite loops
    
    while (max_iterations-- > 0) {
        // Drain any outstanding events to enqueue delayed sends (with limit)
        int drain_limit = 10000; // Prevent infinite event processing
        while (!event_queue->finished() && drain_limit-- > 0) {
            // Safety check: only proceed if event queue is in valid state
            if (event_queue->finished()) break;
            try {
                event_queue->proceed();
            } catch (...) {
                // If proceed fails, break out of drain loop
                break;
            }
        }
        
        // If no pending flows, we are done
        if (pending_flows_.empty()) {
            break;
        }
        
        // Process all full N-chunks
        while ((int)pending_flows_.size() >= batch_size_flows_) {
            process_batch_of_flows_count(batch_size_flows_);
        }
        
        // Process remaining partial once
        if (!pending_flows_.empty()) {
            process_batch_of_flows_count((int32_t)pending_flows_.size());
        }
        
        // Drain scheduled completions before checking again (with limit)
        drain_limit = 10000;
        while (!event_queue->finished() && drain_limit-- > 0) {
            // Safety check: only proceed if event queue is in valid state
            if (event_queue->finished()) break;
            try {
                event_queue->proceed();
            } catch (...) {
                // If proceed fails, break out of drain loop
                break;
            }
        }
        
        // If nothing new arrived during completion processing, exit
        if (pending_flows_.empty()) {
            break;
        }
    }
    
    if (max_iterations <= 0) {
        std::cerr << "[M4] Warning: process_final_batch() hit iteration limit, may have incomplete flows" << std::endl;
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
        
        flow->start_time = current_time;
        flows_arriving_this_batch++;

        uint64_t size = flow->size;
        double size_bytes = static_cast<double>(size);
        
        // Use routing framework's pre-computed RTT and bandwidth (same as NS3)
        if (!routing_framework_) {
            throw std::runtime_error("[M4 ERROR] RoutingFramework is null when computing ideal FCT");
        }
        
        uint64_t base_rtt = routing_framework_->GetPairRtt(flow->src, flow->dst);
        uint64_t b_bps = routing_framework_->GetPairBandwidth(flow->src, flow->dst);
        
        const uint32_t packet_payload_size = 1000u;
        const uint32_t header_overhead = 52u; //
        uint64_t num_pkts = (size + packet_payload_size - 1) / packet_payload_size;
        uint64_t total_bytes = size + num_pkts * header_overhead;
        
        // Use NS3's exact calculation: base_rtt + total_bytes * 8000000000lu / b
        double ideal_fct = (double)base_rtt + (double)(total_bytes * 8000000000ULL) / (double)b_bps;
        // std::cout << "[M4 DBG] ideal_fct=" << ideal_fct << " base_rtt=" << base_rtt << " total_bytes=" << total_bytes << " b_bps=" << b_bps << std::endl;
        // Get route for hop count (still needed for ML features)
        std::vector<int> ns3_route = topology->find_ns3_route(flow->src, flow->dst);
        if (ns3_route.size() < 2) {
            throw std::runtime_error("[M4 ERROR] Empty/invalid NS3 route for hop count");
        }
        int ns3_num_links = static_cast<int>(ns3_route.size()) - 1;
        flowid_to_nlinks_tensor[flow_id] = ns3_num_links;
        i_fct_tensor[flow_id] = static_cast<float>(ideal_fct);
        release_time_tensor[flow_id] = time_clock;
        time_last[flow_id] = time_clock;  // Initialize time_last to avoid massive time deltas
        flowid_active_mask[flow_id] = true;
        flow_to_graph_id[flow_id] = 0;  // All flows use global graph
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
    n_flows_arrived += flows_arriving_this_batch;
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
    {
        torch::NoGradGuard no_grad;
        time_clock = static_cast<float>(current_time);
        
        // Batch-local graph: Only process flows that interact with the current batch
        // Step 1: Find flows that share links with the current batch
        std::set<int32_t> interacting_flows;
        // Add current batch flows
        for (int64_t fid : flow_ids_batch) {
            interacting_flows.insert((int32_t)fid);
        }
        // Add other active flows that share links with current batch
        auto flowid_active_list_all = torch::nonzero(flowid_active_mask).flatten();
        for (int i = 0; i < flowid_active_list_all.size(0); i++) {
            int32_t fid = flowid_active_list_all[i].item<int32_t>();
            if (fid < flowid_to_link_indices.size()) {
                // Check if this flow shares any links with current batch
                for (int32_t lid : flowid_to_link_indices[fid]) {
                    if (current_batch_link_set.find(lid) != current_batch_link_set.end()) {
                        interacting_flows.insert(fid);
                        break; // Found overlap, add flow and move on
                    }
                }
            }
        }
        
        if (!interacting_flows.empty()) {
            // Step 2: Build compact flow list and edges for interacting flows only
            std::vector<int32_t> flowid_interacting_vec(interacting_flows.begin(), interacting_flows.end());
            std::sort(flowid_interacting_vec.begin(), flowid_interacting_vec.end()); // For consistent indexing
            
            // Step 3: Build edges using actual flow IDs and link IDs (like inference code)
            std::vector<int32_t> flow_edges, link_edges;
            for (int32_t fid : flowid_interacting_vec) {
                if (fid < flowid_to_link_indices.size()) {
                    for (int32_t lid : flowid_to_link_indices[fid]) {
                        // Only include links that are active in current batch
                        if (current_batch_link_set.find(lid) != current_batch_link_set.end()) {
                            flow_edges.push_back(fid);  // Use actual flow ID, not relative index
                            link_edges.push_back(lid);  // Use actual link ID
                        }
                    }
                }
            }
            if (!flow_edges.empty()) {
                auto edge_flow_tensor = torch::from_blob(flow_edges.data(), {(int)flow_edges.size()}, torch::TensorOptions().dtype(torch::kInt32))
                                           .to(torch::kInt64).to(device);
                auto edge_link_tensor = torch::from_blob(link_edges.data(), {(int)link_edges.size()}, torch::TensorOptions().dtype(torch::kInt32))
                                           .to(torch::kInt64).to(device);
                
                // Step 4: Convert interacting flows to compact indices (like inference searchsorted)
                auto flowid_interacting_tensor = torch::tensor(flowid_interacting_vec, torch::TensorOptions().dtype(torch::kInt32).device(device)).to(torch::kInt64);
                auto new_flow_indices = torch::searchsorted(flowid_interacting_tensor, edge_flow_tensor);
                int n_flows_active_cur = flowid_interacting_vec.size();
                
                // Step 5: Get unique links and remap them (exactly like inference code)
                auto unique_links_tuple = torch::_unique(edge_link_tensor, true, true);
                auto active_link_idx = std::get<0>(unique_links_tuple);  // Unique link IDs
                auto new_link_indices = std::get<1>(unique_links_tuple); // Inverse indices for remapping
                new_link_indices += n_flows_active_cur;

                auto edges_list_active = torch::cat({
                    torch::stack({new_flow_indices, new_link_indices}, 0),
                    torch::stack({new_link_indices, new_flow_indices}, 0)
                }, 1);

                // Use actual flow IDs for tensor indexing (like inference code)
                auto subset_indices = flowid_interacting_tensor; // These are the actual flow IDs
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
                
                // Log GNN update details
                // int n_link_nodes = active_link_idx.size(0);
                // int n_edges = edges_list_active.size(1);
                // int total_active_flows = flowid_active_list_all.size(0);
                // std::cout << "[GNN Update] Flow nodes: " << n_flows_active_cur << "/" << total_active_flows << " (interacting/total_active)"
                //           << ", Link nodes: " << n_link_nodes 
                //           << ", Edges: " << n_edges 
                //           << ", Time: " << (max_time_delta) << "μs" << std::endl;
                
                std::vector<torch::Tensor> tensors_to_cat = {h_vec_time_updated, h_vec_time_link_updated};
                auto x_combined = torch::cat(tensors_to_cat, 0);
                auto gnn_output_0 = gnn_layer_0.forward(std::vector<c10::IValue>{x_combined, edges_list_active}).toTensor();
                auto gnn_output_1 = gnn_layer_1.forward(std::vector<c10::IValue>{gnn_output_0, edges_list_active}).toTensor();
                auto gnn_output_2 = gnn_layer_2.forward(std::vector<c10::IValue>{gnn_output_1, edges_list_active}).toTensor();
                auto h_vec_rate_updated = gnn_output_2.slice(0, 0, n_flows_active_cur);
                auto h_vec_rate_link = gnn_output_2.slice(0, n_flows_active_cur, gnn_output_2.size(0));
                auto params_data = params_tensor.repeat({n_flows_active_cur, 1});
                std::vector<torch::Tensor> rate_tensors = {h_vec_rate_updated, params_data};
                h_vec_rate_updated = torch::cat(rate_tensors, 1);
                h_vec_rate_updated = lstmcell_rate.forward(std::vector<c10::IValue>{h_vec_rate_updated, h_vec_time_updated}).toTensor();
                h_vec_rate_link = lstmcell_rate_link.forward(std::vector<c10::IValue>{h_vec_rate_link, h_vec_time_link_updated}).toTensor();
                h_vec.index_copy_(0, subset_indices, h_vec_rate_updated);
                z_t_link.index_copy_(0, active_link_idx.to(torch::kInt64), h_vec_rate_link);
                // Update time_last only for flows that participated in this GNN update
                time_last.index_put_({torch::indexing::TensorIndex(subset_indices)}, time_clock);
            }
        }
    }

    // Create flow ID batch tensor for ML inference
    int batch_size = static_cast<int>(flow_ids_batch.size());
    auto flowid_batch = torch::tensor(flow_ids_batch, torch::TensorOptions().dtype(torch::kInt64).device(device));
    // Batch tensor operations for efficiency
    auto h_vec_batch = h_vec.index_select(0, flowid_batch);
    auto nlinks_batch = flowid_to_nlinks_tensor.index_select(0, flowid_batch).unsqueeze(1);
    auto params_batch = params_tensor.unsqueeze(0).repeat({batch_size, 1});
    std::vector<torch::Tensor> input_tensors = {nlinks_batch, params_batch, h_vec_batch};
    auto input_batch = torch::cat(input_tensors, 1);

    // Simple logging: ML update number and active flows
    // std::cout << "[ML Update " << ml_update_counter << "] " << flow_ids_batch.size() << " active flows" << std::endl;
    
    // Single MLP forward for the entire temporal batch (predict after step)
    auto sldn_raw = output_layer.forward(std::vector<c10::IValue>{input_batch}).toTensor().view(-1);
    // Debug: print slowdown stats for first few batches only
    // static int debug_batches = 0;
    // if (debug_batches < 3) {
    //     auto to_cpu = [](const torch::Tensor& t){ return t.detach().to(torch::kCPU); };
    //     auto s = to_cpu(sldn_raw);
    //     double s_min = s.min().item<double>();
    //     double s_max = s.max().item<double>();
    //     double s_mean = s.mean().item<double>();
    //     auto below_one = (s < 1.0).to(torch::kFloat32);
    //     double frac_below_one = below_one.mean().item<double>();
    //     auto nlinks_cpu = to_cpu(nlinks_batch.squeeze(1).to(torch::kFloat32));
    //     auto size_feat_cpu = to_cpu(h_vec_batch.index({torch::indexing::Slice(), 2}));
    //     double nlinks_min = nlinks_cpu.min().item<double>();
    //     double nlinks_max = nlinks_cpu.max().item<double>();
    //     double size_min = size_feat_cpu.min().item<double>();
    //     double size_max = size_feat_cpu.max().item<double>();
    //     std::cout << "[M4 DBG] slowdown_raw min/mean/max=" << s_min << "/" << s_mean << "/" << s_max
    //               << " frac<1=" << frac_below_one
    //               << " nlinks[min,max]=" << nlinks_min << "," << nlinks_max
    //               << " size_feat[min,max]=" << size_min << "," << size_max
    //               << std::endl;
    //     // print a small slice of the actual feature vectors fed to MLP for the first row
    //     double nlinks0 = nlinks_cpu[0].item<double>();
    //     double size0 = size_feat_cpu[0].item<double>();
    //     std::cout << "[M4 DBG] feat nlinks0=" << nlinks0 << " size0=" << size0 << " h0[0..5]=";
    //     int h_cols = (int)h_vec_batch.size(1);
    //     for (int k = 0; k < std::min(6, h_cols); ++k) {
    //         double v = to_cpu(h_vec_batch.index({0, k})).item<double>();
    //         if (k) std::cout << ",";
    //         std::cout << v;
    //     }
    //     std::cout << " in0[0..9]=";
    //     int in_cols = (int)input_batch.size(1);
    //     for (int k = 0; k < std::min(10, in_cols); ++k) {
    //         double v = to_cpu(input_batch.index({0, k})).item<double>();
    //         if (k) std::cout << ",";
    //         std::cout << v;
    //     }
    //     std::cout << std::endl;
    //     debug_batches++;
    // }
    auto sldn = torch::clamp(sldn_raw, 1.0f, std::numeric_limits<float>::infinity());

    // Transfer slowdown predictions from GPU to CPU for processing
    auto sldn_cpu = sldn.to(torch::kCPU);
    auto sldn_data = sldn_cpu.data_ptr<float>();
    
    // Legitimate batch processing: Use runtime batch statistics for normalization
    // Step 1: Calculate batch statistics (available at runtime)
    float sum = 0.0f, sum_sq = 0.0f;
    int n = flow_ids_batch.size();
    for (int i = 0; i < n; i++) {
        sum += sldn_data[i];
        sum_sq += sldn_data[i] * sldn_data[i];
    }
    float batch_mean = sum / n;
    float batch_var = (sum_sq / n) - (batch_mean * batch_mean);
    float batch_std = std::sqrt(std::max(batch_var, 1e-8f));
    
    // Step 2: Schedule flow completions (M4Network.cc handles collective completion)
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
        
        // Find the ML prediction for this flow in the batch
        int batch_idx = -1;
        for (size_t j = 0; j < flow_ids_batch.size(); j++) {
            if ((int)flow_ids_batch[j] == flow_id) {
                batch_idx = j;
                break;
            }
        }
        
        if (batch_idx >= 0) {
            // Normal ML inference for flows with valid routes
            float raw_slowdown = sldn_data[batch_idx];
            
            float z_score = (raw_slowdown - batch_mean) / batch_std;
            z_score = std::max(-1.0f, std::min(1.0f, z_score));
            float scaled_slowdown = raw_slowdown;
            
            // Only enforce physical constraint: slowdown >= 1.0
            scaled_slowdown = std::max(scaled_slowdown, 1.0f);
            
            float predicted_fct = scaled_slowdown * i_fct_tensor[flow_id].item<float>();
            uint64_t completion_time = current_time + (uint64_t)predicted_fct;
            
            // Schedule original callback - M4Network.cc handles collective completion correctly
            event_queue->schedule_event(completion_time, flow->callback, flow->callbackArg);
        }
    }
    
    // No need to schedule next batch - all pending flows processed
}

// LSTM+GNN state evolution only (no completion selection/scheduling)


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

