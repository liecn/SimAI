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
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <fstream>
#include <sstream>

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
int32_t M4::hidden_size_ = 64;
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

// Flow and graph management
int32_t M4::n_flows_max = 1000000;  // Large enough for simulation
int32_t M4::n_flows_active = 0;
int32_t M4::n_flows_completed = 0;
int32_t M4::graph_id_counter = 0;
int32_t M4::graph_id_cur = 0;
float M4::time_clock = 0.0f;
int32_t M4::next_flow_id = 0;  // For assigning unique flow IDs

void M4::Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo) {

    M4::event_queue = event_queue;
    M4::topology = topo;
    M4::topology->set_event_queue(event_queue);
    
    // Setup ML models
    SetupML();
}

void M4::SetupML() {
    if (models_loaded) return;
    
    if (!torch::cuda::is_available()) {
        std::cerr << "[M4] ERROR: CUDA is not available!" << std::endl;
        return;
    }
    
    torch::NoGradGuard no_grad;
    
    // Model directory: use relative path from M4 frontend
    const std::string model_dir = "./astra-sim-alibabacloud/astra-sim/network_frontend/m4/models/";

    
    try {
    
        lstmcell_time = torch::jit::load(model_dir + "lstmcell_time.pt", device);
        lstmcell_rate = torch::jit::load(model_dir + "lstmcell_rate.pt", device);
        lstmcell_time_link = torch::jit::load(model_dir + "lstmcell_time_link.pt", device);
        lstmcell_rate_link = torch::jit::load(model_dir + "lstmcell_rate_link.pt", device);
        output_layer = torch::jit::load(model_dir + "output_layer.pt", device);
        gnn_layer_0 = torch::jit::load(model_dir + "gnn_layer_0.pt", device);
        gnn_layer_1 = torch::jit::load(model_dir + "gnn_layer_1.pt", device);
        gnn_layer_2 = torch::jit::load(model_dir + "gnn_layer_2.pt", device);

        // Set models to evaluation mode and optimize for inference
        lstmcell_time.eval();
        lstmcell_rate.eval();
        lstmcell_time_link.eval();
        lstmcell_rate_link.eval();
        output_layer.eval();
        gnn_layer_0.eval();
        gnn_layer_1.eval();
        gnn_layer_2.eval();

        lstmcell_time = torch::jit::optimize_for_inference(lstmcell_time);
        lstmcell_rate = torch::jit::optimize_for_inference(lstmcell_rate);
        lstmcell_time_link = torch::jit::optimize_for_inference(lstmcell_time_link);
        lstmcell_rate_link = torch::jit::optimize_for_inference(lstmcell_rate_link);
        output_layer = torch::jit::optimize_for_inference(output_layer);
        gnn_layer_0 = torch::jit::optimize_for_inference(gnn_layer_0);
        gnn_layer_1 = torch::jit::optimize_for_inference(gnn_layer_1);
        gnn_layer_2 = torch::jit::optimize_for_inference(gnn_layer_2);

        models_loaded = true;
    
        
    } catch (const std::exception& e) {
        std::cerr << "[M4] ERROR: Failed to load M4 models: " << e.what() << std::endl;
        std::cerr << "[M4] ERROR: Model directory: " << model_dir << std::endl;
        models_loaded = false;
    }

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
    
    // Initialize params tensor (zeros(13))
    params_tensor = torch::zeros({13}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Initialize multi-flow state tensors (from @inference/ ground truth)
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);
    
    // Initialize flow and link state tensors
    h_vec = torch::zeros({n_flows_max, hidden_size_}, options_float);
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
    
    std::cout << "[M4] Initialized multi-flow state tensors for " << n_flows_max << " flows and " << n_links_max_ << " links" << std::endl;
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
    // AS_NVLS_ENABLE behavior (FlowSim parity for minimum chunk size); affects features only
    const char* nvls_env = std::getenv("AS_NVLS_ENABLE");
    if (nvls_env && std::stoi(nvls_env) == 1) {
        if (size < 4096 && size > 0) {
            size = 4096;
        }
    }

    // Get pre-calculated path from routing framework (same as FlowSim)
    std::vector<int> node_path = routing_framework_->GetFlowSimPathByNodeIds(src, dst);
    if (node_path.empty()) {
        // Nothing to schedule; invoke callback immediately to keep app progressing
        callback(callbackArg);
        return;
    }

    // Flow ID for state management
    int flow_id = (next_flow_id++) % n_flows_max;

    // Compute ideal FCT components (from @inference/ ground truth)
    const double MTU = 1000.0;
    const double BYTES_PER_HEADER = 48.0;
    const double default_link_bandwidth_bytes_per_ns = 100e9 / 8.0 / 1e9; // 100Gbps
    const double default_link_latency_ns = 1000.0; // 1μs
    int num_hops = static_cast<int>(node_path.size());
    double size_bytes = static_cast<double>(size);
    double prop_delay = default_link_latency_ns * (num_hops - 1);
    double trans_delay = ((size_bytes + std::ceil(size_bytes / MTU) * BYTES_PER_HEADER)) / default_link_bandwidth_bytes_per_ns;
    double first_packet = (std::min(MTU, size_bytes) + BYTES_PER_HEADER) / default_link_bandwidth_bytes_per_ns * (num_hops - 2);
    double ideal_fct = trans_delay + prop_delay + first_packet;

    // Initialize/update per-flow state tensors
    time_clock = static_cast<float>(Now());
    flowid_active_mask[flow_id] = true;
    time_last[flow_id] = time_clock;
    release_time_tensor[flow_id] = static_cast<float>(Now());
    flowid_to_nlinks_tensor[flow_id] = num_hops;
    i_fct_tensor[flow_id] = static_cast<float>(ideal_fct);
    h_vec[flow_id].zero_();
    h_vec[flow_id][0] = 1.0f; // flow active
    h_vec[flow_id][2] = std::log2(size_bytes / 1000.0 + 1.0);
    h_vec[flow_id][3] = static_cast<float>(num_hops);

    // Perform single-flow inference using output_layer (batch-compatible)
    float flow_completion_time;
    if (models_loaded) {
        torch::NoGradGuard no_grad;
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);

        auto nlinks_cur = flowid_to_nlinks_tensor.index({flow_id}).unsqueeze(0).unsqueeze(1).to(options_float); // [1,1]
        auto params_data_cur = params_tensor.view({1, -1}); // [1,13]
        auto h_vec_cur = h_vec.index({flow_id}).unsqueeze(0); // [1,H]
        auto input_tensor = torch::cat({nlinks_cur, params_data_cur, h_vec_cur}, 1);

        auto sldn_est = output_layer.forward({input_tensor}).toTensor().view(-1);
        sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());

        auto fct_est = release_time_tensor.index({flow_id}) + sldn_est[0] * i_fct_tensor.index({flow_id});
        flow_completion_time = fct_est.item<float>();
    } else {
        flow_completion_time = release_time_tensor[flow_id].item<float>() + i_fct_tensor[flow_id].item<float>();
    }

    uint64_t now_ns = static_cast<uint64_t>(Now());
    uint64_t predicted_tx_time = static_cast<uint64_t>(std::max(0.0f, flow_completion_time - static_cast<float>(now_ns)));
    if (predicted_tx_time == 0) {
        predicted_tx_time = 1000; // minimum 1μs
    }

    // Schedule the provided callback (m4_completion_callback from M4Network) at predicted completion time
    Schedule(predicted_tx_time, callback, callbackArg);
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
