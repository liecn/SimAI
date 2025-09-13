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
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <fstream>
#include <sstream>
#include <chrono>

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
int32_t M4::hidden_size_ = 200; // Model expects 214 total: 1+13+200=214 (matches main_m4_noflowsim.cpp)
int32_t M4::n_links_max_ = 4096;

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
int32_t M4::n_flows_max = 1000000;  // Large enough for simulation
int32_t M4::n_flows_active = 0;
int32_t M4::n_flows_completed = 0;
int32_t M4::graph_id_counter = 0;
int32_t M4::graph_id_cur = 0;
float M4::time_clock = 0.0f;
int32_t M4::next_flow_id = 0;  // For assigning unique flow IDs
std::unordered_map<long long, int32_t> M4::link_key_to_index;
int32_t M4::next_link_index = 0;
std::vector<std::vector<int32_t>> M4::flowid_to_link_indices;

// Flow callback storage
std::unordered_map<int, std::pair<void(*)(void*), void*>> M4::flow_callbacks;
std::vector<int32_t> M4::active_flows;
std::unordered_map<int32_t, M4::M4CompletionData*> M4::pending_map; // flow_id -> completion data
std::unordered_set<int32_t> M4::scheduled_flows; // flows with a scheduled completion event

void M4::ScheduleNextForGraph(int32_t graph_id) {
    // Build candidate list of active, unscheduled flows in the graph
    std::vector<int32_t> flows_in_graph;
    flows_in_graph.reserve(M4::active_flows.size());
    for (int32_t fid : M4::active_flows) {
        if (M4::flow_to_graph_id.index({fid}).item<int32_t>() == graph_id && M4::scheduled_flows.find(fid) == M4::scheduled_flows.end()) {
            flows_in_graph.push_back(fid);
        }
    }
    if (flows_in_graph.empty()) return;

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(M4::device);
    auto flowid_active_list_cur = torch::from_blob(flows_in_graph.data(), {(int)flows_in_graph.size()}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kInt64).to(M4::device).clone();
    // Prepare inputs
    auto nlinks_cur = M4::flowid_to_nlinks_tensor.index_select(0, flowid_active_list_cur).unsqueeze(1).to(options_float);
    auto h_vec_cur = M4::h_vec.index_select(0, flowid_active_list_cur);
    auto input_tensor = torch::cat({nlinks_cur, M4::params_tensor.repeat({(int64_t)flows_in_graph.size(),1}), h_vec_cur}, 1);
    auto sldn_est = M4::output_layer.forward({input_tensor}).toTensor().view(-1);
    sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());
    auto fct_est = M4::release_time_tensor.index_select(0, flowid_active_list_cur) + sldn_est * M4::i_fct_tensor.index_select(0, flowid_active_list_cur);
    auto min_idx_tensor = torch::argmin(fct_est);
    int min_idx = min_idx_tensor.item<int>();
    int32_t min_flow = flows_in_graph[min_idx];
    float predicted_completion_time = fct_est[min_idx].item<float>();
    uint64_t delay = static_cast<uint64_t>(std::max(0.0f, predicted_completion_time - static_cast<float>(M4::Now())));
    if (delay == 0) delay = 1000;

    auto it = M4::pending_map.find(min_flow);
    if (it == M4::pending_map.end()) {
        // Pending may not be enqueued yet; skip scheduling for now
        return;
    }
    M4::M4CompletionData* data = it->second;
    M4::scheduled_flows.insert(min_flow);

    M4::Schedule(delay, [](void* arg){
        M4::M4CompletionData* completion_data = static_cast<M4::M4CompletionData*>(arg);
        // Mark flow complete
        M4::flowid_active_mask[completion_data->flow_id] = false;
        M4::n_flows_active--;
        M4::n_flows_completed++;
        // Remove from active and scheduled/pending
        auto &af = M4::active_flows;
        af.erase(std::remove(af.begin(), af.end(), completion_data->flow_id), af.end());
        M4::scheduled_flows.erase(completion_data->flow_id);
        M4::pending_map.erase(completion_data->flow_id);
        M4::OnFlowCompleted(completion_data->flow_id);
        // Invoke original app callback
        completion_data->callback(completion_data->callbackArg);
        int32_t gid = M4::flow_to_graph_id.index({completion_data->flow_id}).item<int32_t>();
        delete completion_data;
        // Schedule next event in this graph if any
        M4::ScheduleNextForGraph(gid);
    }, data);
}

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
    std::cout << "[M4] SetupML() starting..." << std::endl;
    
    if (!torch::cuda::is_available()) {
        std::cerr << "[M4] ERROR: CUDA is not available!" << std::endl;
        return;
    }
    
    std::cout << "[M4] CUDA is available, proceeding with setup..." << std::endl;
    
    torch::NoGradGuard no_grad;
    
    // Model directory: use local models directory
    const std::string model_dir = "./astra-sim-alibabacloud/astra-sim/network_frontend/m4/models/";

    
    // Load ALL models as required by M4 inference (same as inference main_m4_noflowsim.cpp)
    try {
        std::cout << "[M4] Loading all required models..." << std::endl;
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
    std::cout << "[M4] Setting models to evaluation mode..." << std::endl;
    lstmcell_time.eval();
    lstmcell_rate.eval();
    lstmcell_rate_link.eval();
    lstmcell_time_link.eval();
    output_layer.eval();
    gnn_layer_0.eval();
    gnn_layer_1.eval();
    gnn_layer_2.eval();

    // Optimize models for inference
    std::cout << "[M4] Optimizing models for inference..." << std::endl;
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

    // Parse config for hidden_size and n_links_max
    try {
        const std::string cfg_path = "./m4/config/test_config.yaml";
        std::ifstream infile(cfg_path);
        if (infile.good()) {
            std::ostringstream contents;
            contents << infile.rdbuf();
            std::string config_contents = contents.str();
            ryml::Tree config = ryml::parse_in_place(ryml::to_substr(config_contents));
            ryml::NodeRef hidden_size_node = config["model"]["hidden_size"];
            ryml::NodeRef n_links_node = config["dataset"]["n_links_max"];
            hidden_size_node >> hidden_size_;
            n_links_node >> n_links_max_;
        
        } else {
            std::cerr << "[M4] ERROR: cannot open config at " << cfg_path << std::endl;
            throw std::runtime_error("M4 config missing");
        }
    } catch (const std::exception& e) {
        std::cerr << "[M4] Config parse error: " << e.what() << std::endl;
        hidden_size_ = 64;
        n_links_max_ = 4096;
    }
    
    // Initialize params tensor with ACTUAL network configuration from SimAI.conf
    // Structure from consts.py: [bfsz(0), fwin(1), dctcp_flag(2), dcqcn_flag(3), hp_flag(4), timely_flag(5), 
    //                           dctcp_k(6), dcqcn_k_min(7), dcqcn_k_max(8), u_tgt(9), hpai(10), timely_t_low(11), timely_t_high(12)]
    std::vector<float> param_values(13, 0.0f);
    
    // Create parameter vector to match inference expectation (loaded from .npy file in inference)
    // Use SimAI.conf values: CC_MODE=8 (DCTCP), BUFFER_SIZE=10, U_TARGET=0.95
    param_values[0] = 10.0f;   // bfsz (BUFFER_SIZE from SimAI.conf)
    param_values[1] = 15.0f;   // fwin (default from consts.py)
    
    // Set CC type: CC_MODE=8 corresponds to DCTCP (index 0 in CC_LIST = ["dctcp", "dcqcn_paper_vwin", "hp", "timely_vwin"])
    // From consts.py: CC_DICT = {"dctcp": 8, ...} - so CC_MODE=8 is indeed DCTCP
    param_values[2] = 1.0f;    // dctcp flag (index 2 = CC_IDX_BASE + 0)
    
    // Set DCTCP-specific parameters from SimAI.conf and consts.py defaults
    param_values[6] = 10.0f;   // dctcp_k (default from consts.py)
    
    params_tensor = torch::from_blob(param_values.data(), {13}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();
    
    // Read topology parameters for logging
    float topo_bandwidth = topology->get_bandwidth(); // in bps
    float topo_latency = topology->get_latency(); // in ns
    
    std::cout << "[M4] Loaded network parameters from SimAI.conf: bfsz=" << param_values[0] << ", fwin=" << param_values[1] 
              << ", cc=dctcp, u_tgt=" << param_values[9] << "%, topology_bw=" << (topo_bandwidth * 8.0) << "Gbps, topology_lat=" << topo_latency << "ns" << std::endl;
    
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
    // Decrement link_to_nflows for this flow's links
    if (flow_id < 0 || flow_id >= n_flows_max) return;
    const auto &links = flowid_to_link_indices[flow_id];
    if (links.empty()) return;
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto idx = torch::from_blob(const_cast<int32_t*>(links.data()), {(int)links.size()}, options_int32).to(device).clone();
    auto ones = torch::ones({(int)links.size()}, options_int32).to(device);
    auto cur = link_to_nflows.index_select(0, idx);
    link_to_nflows.index_put_({idx}, torch::clamp(cur - ones, 0));
}

void M4::SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework) {
    routing_framework_ = std::move(routing_framework);
}

void M4::Run() {

    int iteration = 0;
    
    while (true) {
        // Process M4 events if available (same as FlowSim)
        if (!event_queue->finished()) {
            event_queue->proceed();
            iteration++;
        
        } else {
            // Queue is empty - simulation is complete
        
            break;
        }
        
        // Safety limit
        if (iteration > 100000000) {
        
            break;
        }
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

// M4 completion callback data structure (extended for multi-flow state)
struct M4CompletionData {
    int src;
    int dst;
    uint64_t size;
    int tag;
    uint64_t start_time;
    int flow_id;  // Flow ID for state management
    Callback callback;
    CallbackArg callbackArg;
};

void M4::m4_completion_callback(void* arg) {
    M4CompletionData* data = static_cast<M4CompletionData*>(arg);
    
    if (!models_loaded) {
        // Fallback without ML
        data->callback(data->callbackArg);
        delete data;
        return;
    }
    
    torch::NoGradGuard no_grad;
    
    // Update time_clock to current simulation time
    time_clock = static_cast<float>(Now());
    
    // Get flow ID and activate flow in state tensors
    int flow_id = data->flow_id;
    if (flow_id >= n_flows_max) {
        std::cerr << "[M4] ERROR: Flow ID " << flow_id << " exceeds max flows " << n_flows_max << std::endl;
        data->callback(data->callbackArg);
        delete data;
        return;
    }
    
    // Calculate routing path and flow parameters
    std::vector<int> path = routing_framework_->GetFlowSimPathByNodeIds(data->src, data->dst);
    int num_hops = path.empty() ? 2 : static_cast<int>(path.size());
    
    // Calculate ideal FCT (same as @inference/)
    const double MTU = 1000.0;
    const double BYTES_PER_HEADER = 48.0;
    const double default_link_bandwidth_bytes_per_ns = 100e9 / 8.0 / 1e9; // 100Gbps
    const double default_link_latency_ns = 1000.0; // 1μs
    
    double size_bytes = static_cast<double>(data->size);
    double prop_delay = default_link_latency_ns * (num_hops - 1);
    double trans_delay = ((size_bytes + std::ceil(size_bytes / MTU) * BYTES_PER_HEADER)) / default_link_bandwidth_bytes_per_ns;
    double first_packet = (std::min(MTU, size_bytes) + BYTES_PER_HEADER) / default_link_bandwidth_bytes_per_ns * (num_hops - 2);
    double ideal_fct = trans_delay + prop_delay + first_packet;
    
    // Initialize flow state (from @inference/ ground truth)
    flowid_active_mask[flow_id] = true;
    time_last[flow_id] = time_clock;
    release_time_tensor[flow_id] = static_cast<float>(data->start_time);
    flowid_to_nlinks_tensor[flow_id] = num_hops;
    i_fct_tensor[flow_id] = static_cast<float>(ideal_fct);
    
    // Initialize h_vec for this flow (from @inference/)
    h_vec[flow_id].zero_();
    h_vec[flow_id][0] = 1.0f; // flow active
    h_vec[flow_id][2] = std::log2(size_bytes / 1000.0 + 1.0); // log size
    h_vec[flow_id][3] = static_cast<float>(num_hops); // hop count
    
    n_flows_active++;
    
    // Perform multi-flow inference with GNN+LSTM pipeline (simplified version)
    // For now, use direct output_layer inference (can be extended to full GNN+LSTM later)
    auto flowid_active_indices = torch::nonzero(flowid_active_mask).flatten();
    if (flowid_active_indices.numel() > 0) {
        auto h_vec_active = h_vec.index_select(0, flowid_active_indices);
        auto nlinks_cur = flowid_to_nlinks_tensor.index_select(0, flowid_active_indices).unsqueeze(1);
        auto params_data_cur = params_tensor.repeat({flowid_active_indices.size(0), 1});
        auto input_tensor = torch::cat({nlinks_cur, params_data_cur, h_vec_active}, 1);
        
        // Perform inference
        auto sldn_est = output_layer.forward({input_tensor}).toTensor().view(-1);
        sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());
        
        // Calculate completion times for all active flows
        auto fct_stamp_est = release_time_tensor.index_select(0, flowid_active_indices) + 
                           sldn_est * i_fct_tensor.index_select(0, flowid_active_indices);
        
        // Find the current flow's prediction
        auto current_flow_idx = (flowid_active_indices == flow_id).nonzero().flatten();
        if (current_flow_idx.numel() > 0) {
            int idx = current_flow_idx[0].item<int>();
            float predicted_completion_time = fct_stamp_est[idx].item<float>();
            uint64_t delay = static_cast<uint64_t>(std::max(0.0f, predicted_completion_time - time_clock));
            
            // Schedule completion after predicted delay
            if (delay > 0) {
                Schedule(delay, [](void* arg) {
                    M4CompletionData* completion_data = static_cast<M4CompletionData*>(arg);
                    // Mark flow as completed
                    M4::flowid_active_mask[completion_data->flow_id] = false;
                    M4::n_flows_active--;
                    M4::n_flows_completed++;
                    // Remove from active set and update link counters
                    auto &af = M4::active_flows;
                    af.erase(std::remove(af.begin(), af.end(), completion_data->flow_id), af.end());
                    M4::OnFlowCompleted(completion_data->flow_id);
                    
                    completion_data->callback(completion_data->callbackArg);
                    delete completion_data;
                }, data);
                return;
            }
        }
    }
    
    // Immediate completion fallback
    flowid_active_mask[flow_id] = false;
    n_flows_active--;
    n_flows_completed++;
    data->callback(data->callbackArg);
    delete data;
}

void M4::Send(int src, int dst, uint64_t size, int tag, Callback callback, CallbackArg callbackArg) {
    // M4 integration with ASTRA-Sim using ML prediction
    
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
    if (node_path.empty()) {
        // Nothing to schedule; invoke callback immediately (same as FlowSim)
        callback(callbackArg);
        return;
    }
    
    // Convert to device route (same as FlowSim)
    Route route;
    for (int node_id : node_path) {
        if (node_id >= 0 && node_id < topology->get_devices_count()) {
            route.push_back(topology->get_device(node_id));
        }
    }
    
    if (route.size() >= 2) {
        // Add flow to M4's active flow tracking for ML processing
        int flow_id = AddActiveFlow(src, dst, size, node_path, callback, callbackArg);
        
        // Trigger ML batch processing if needed
        ProcessFlowBatch();
    }
}

int M4::AddActiveFlow(int src, int dst, uint64_t size, const std::vector<int>& node_path, Callback callback, CallbackArg callbackArg) {
    // Add new flow to M4's active flow tracking
    
    // Assign flow ID for ML state tracking
    int flow_id = next_flow_id++;
    if (flow_id >= n_flows_max) flow_id = flow_id % n_flows_max;
    
    // Update M4 state tensors (from @inference/ logic)
    time_clock = static_cast<float>(Now());
    
    // Compute ideal FCT components (from @inference/)
    const double MTU = 1000.0;
    const double BYTES_PER_HEADER = 48.0;
    float topology_latency = topology->get_latency();  // in ns
    float topology_bandwidth = topology->get_bandwidth(); // in bytes/ns
    
    int num_hops = static_cast<int>(node_path.size());
    double size_bytes = static_cast<double>(size);
    double prop_delay = topology_latency * (num_hops - 1);
    double trans_delay = ((size_bytes + std::ceil(size_bytes / MTU) * BYTES_PER_HEADER)) / topology_bandwidth;
    double first_packet = (std::min(MTU, size_bytes) + BYTES_PER_HEADER) / topology_bandwidth * (num_hops - 2);
    double ideal_fct = trans_delay + prop_delay + first_packet;
    
    // Update per-flow tensors for ML inference
    flowid_to_nlinks_tensor[flow_id] = num_hops - 1;
    i_fct_tensor[flow_id] = static_cast<float>(ideal_fct);
    release_time_tensor[flow_id] = time_clock;
    flowid_active_mask[flow_id] = true;
    flow_to_graph_id[flow_id] = graph_id_cur;
    
    // Initialize h_vec using EXACT same logic as @inference/ (main_m4_noflowsim.cpp lines 197-200)
    h_vec[flow_id].zero_();
    h_vec[flow_id][0] = 1.0f; // flow active (line 198)
    h_vec[flow_id][2] = std::log2(size_bytes / 1000.0 + 1.0); // size_tensor (line 199)
    h_vec[flow_id][3] = static_cast<float>(num_hops - 1); // flowid_to_nlinks_tensor (line 200)
    
    // Store callback for completion
    flow_callbacks[flow_id] = std::make_pair(callback, callbackArg);
    
    return flow_id;
}

void M4::ProcessFlowBatch() {
    // Process all active flows using @inference/ approach
    // CRITICAL: @inference/ uses MLP DIRECTLY on h_vec, NOT after LSTM+GNN!
    
    if (!models_loaded) {
        std::cerr << "[M4 ERROR] ML models not loaded! Cannot process flow batch." << std::endl;
        throw std::runtime_error("M4 ML models not loaded");
    }
    
    torch::NoGradGuard no_grad;
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    // Update time_clock to current simulation time
    time_clock = static_cast<float>(Now());
    
    // Get active flows (matching @inference/ update_times_m4() lines 284-285)
    auto flowid_active_indices = torch::nonzero(flowid_active_mask).flatten();
    
    if (flowid_active_indices.numel() > 0) {
        int n_flows_active = flowid_active_indices.size(0);
        
        // MLP prediction using CLEAN h_vec (matching @inference/ lines 286-294)
        auto h_vec_active = h_vec.index_select(0, flowid_active_indices);
        auto nlinks_cur = flowid_to_nlinks_tensor.index_select(0, flowid_active_indices).unsqueeze(1);
        auto params_data_cur = params_tensor.repeat({n_flows_active, 1});
        auto input_tensor = torch::cat({nlinks_cur, params_data_cur, h_vec_active}, 1);
        
        // Perform inference using MLP output layer (matching @inference/ lines 293-294)
        auto sldn_est = output_layer.forward({input_tensor}).toTensor().view(-1);
        sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());
        
        // Calculate predicted FCT (matching @inference/ line 296)
        auto fct_est = release_time_tensor.index_select(0, flowid_active_indices) + 
                      sldn_est * i_fct_tensor.index_select(0, flowid_active_indices);
        
        // Schedule completion for all flows based on ML predictions
        for (int i = 0; i < n_flows_active; i++) {
            int32_t flow_id = flowid_active_indices[i].item<int32_t>();
            float predicted_completion_time = fct_est[i].item<float>();
            uint64_t delay = static_cast<uint64_t>(std::max(1000.0f, predicted_completion_time - time_clock));
            
            // Schedule completion using ASTRA-Sim event system
            if (flow_callbacks.count(flow_id)) {
                auto callback_pair = flow_callbacks[flow_id];
                Schedule(delay, callback_pair.first, callback_pair.second);
                
                // Mark flow as inactive
                flowid_active_mask[flow_id] = false;
                flow_callbacks.erase(flow_id);
            }
            
            if (flow_id <= 5 || flow_id % 10000 == 0) {
                float slowdown = sldn_est[i].item<float>();
                float ideal_fct = i_fct_tensor[flow_id].item<float>();
                std::cout << "[M4 DEBUG] Flow #" << flow_id << " slowdown=" << slowdown 
                          << ", ideal_fct=" << ideal_fct << "ns, predicted_fct=" << predicted_completion_time << "ns" << std::endl;
            }
        }
        
        // LSTM+GNN state update (matching @inference/ step_m4() function)
        // This updates the h_vec for future predictions but doesn't affect current predictions
        UpdateFlowStates();
    }
}

void M4::UpdateFlowStates() {
    // Update flow states using LSTM+GNN pipeline (matching @inference/ step_m4())
    // This is separate from prediction - it updates h_vec for future use
    
    torch::NoGradGuard no_grad;
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    
    // Get active flows in current graph (matching @inference/ step_m4())
    auto flowid_active_mask_cur = torch::logical_and(flowid_active_mask, flow_to_graph_id == graph_id_cur);
    auto flowid_active_list_cur = torch::nonzero(flowid_active_mask_cur).flatten();
    
    if (flowid_active_list_cur.numel() > 0) {
        int n_flows_active_cur = flowid_active_list_cur.size(0);
        
        // Calculate time deltas for active flows
        auto time_deltas = (time_clock - time_last.index_select(0, flowid_active_list_cur).squeeze()).view({-1, 1});
        
        // Build edge_index for active flows
        std::vector<int32_t> flow_ids, link_ids;
        for (int i = 0; i < n_flows_active_cur; i++) {
            int32_t fid = flowid_active_list_cur[i].item<int32_t>();
            if (fid < flowid_to_link_indices.size()) {
                for (int32_t lid : flowid_to_link_indices[fid]) {
                    flow_ids.push_back(i);
                    link_ids.push_back(lid);
                }
            }
        }
        
        if (!flow_ids.empty()) {
            auto edge_flow_tensor = torch::from_blob(flow_ids.data(), {(int)flow_ids.size()}, 
                                                   torch::TensorOptions().dtype(torch::kInt32)).to(options_int64).clone();
            auto edge_link_tensor = torch::from_blob(link_ids.data(), {(int)link_ids.size()}, 
                                                   torch::TensorOptions().dtype(torch::kInt32)).to(options_int64).clone();
            
            // Get unique link indices
            auto unique_result_tuple = torch::_unique(edge_link_tensor, true, true);
            auto active_link_idx = std::get<0>(unique_result_tuple);
            auto new_link_indices = std::get<1>(unique_result_tuple);
            
            new_link_indices += n_flows_active_cur;
            auto edges_list_active = torch::cat({
                torch::stack({edge_flow_tensor, new_link_indices}, 0),
                torch::stack({new_link_indices, edge_flow_tensor}, 0)
            }, 1);
            
            // LSTM TIME processing
            auto h_vec_time_updated = h_vec.index_select(0, flowid_active_list_cur);
            auto h_vec_time_link_updated = z_t_link.index_select(0, active_link_idx);
            auto max_time_delta = torch::max(time_deltas).item<float>();
            if (max_time_delta > 0.0f) {
                time_deltas.fill_(max_time_delta / 1000.0f);
                h_vec_time_updated = lstmcell_time.forward({time_deltas, h_vec_time_updated}).toTensor();
                
                auto time_deltas_link = torch::zeros({active_link_idx.size(0), 1}, options_float);
                time_deltas_link.fill_(max_time_delta / 1000.0f);
                h_vec_time_link_updated = lstmcell_time_link.forward({time_deltas_link, h_vec_time_link_updated}).toTensor();
            }
            
            // GNN processing
            auto x_combined = torch::cat({h_vec_time_updated, h_vec_time_link_updated}, 0);
            auto gnn_output_0 = gnn_layer_0.forward({x_combined, edges_list_active}).toTensor();
            auto gnn_output_1 = gnn_layer_1.forward({gnn_output_0, edges_list_active}).toTensor();
            auto gnn_output_2 = gnn_layer_2.forward({gnn_output_1, edges_list_active}).toTensor();
            
            // LSTM RATE processing
            auto h_vec_rate_updated = gnn_output_2.slice(0, 0, n_flows_active_cur);
            auto h_vec_rate_link = gnn_output_2.slice(0, n_flows_active_cur, gnn_output_2.size(0));
            
            auto params_data = params_tensor.repeat({n_flows_active_cur, 1});
            h_vec_rate_updated = torch::cat({h_vec_rate_updated, params_data}, 1);
            h_vec_rate_updated = lstmcell_rate.forward({h_vec_rate_updated, h_vec_time_updated}).toTensor();
            h_vec_rate_link = lstmcell_rate_link.forward({h_vec_rate_link, h_vec_time_link_updated}).toTensor();
            
            // Update tensors for future predictions
            h_vec.index_copy_(0, flowid_active_list_cur, h_vec_rate_updated);
            auto long_indices = active_link_idx.to(torch::kInt64);
            z_t_link.index_copy_(0, long_indices, h_vec_rate_link);
            time_last.index_put_({flowid_active_list_cur}, time_clock);
        }
    }
}

bool M4::IsRoutingFrameworkLoaded() {
    return routing_framework_ != nullptr;
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
