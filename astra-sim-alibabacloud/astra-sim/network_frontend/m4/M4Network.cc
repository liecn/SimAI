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

#include"M4Network.h"
#include"astra-sim/system/SendPacketEventHandlerData.hh"
#include"astra-sim/system/RecvPacketEventHadndlerData.hh"
#include"astra-sim/system/MockNcclLog.h"
#include"astra-sim/system/SharedBusStat.hh"
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <map>
// #include <torch/torch.h>
// #include <torch/script.h>
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <sys/stat.h>
#include <sys/types.h>

// Avoid using-directives to prevent conflicts with c10::optional/nullopt

// Static routing framework pointer
std::unique_ptr<AstraSim::RoutingFramework> M4Network::s_routing = nullptr;

// Copy FlowSim's exact globals
static uint64_t m4_current_time = 0;
static uint64_t g_fct_lines_written = 0;
static FILE* fct_output_file = nullptr;

// Copy FlowSim's exact extern declarations
extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern std::map<std::pair<int, std::pair<int, int>>, struct M4Task> expeRecvHash;
extern std::map<std::pair<int, std::pair<int, int>>, int> recvHash;
extern std::map<std::pair<int, int>, int64_t> nodeHash;
extern std::map<std::pair<int, std::pair<int, int>>, struct M4Task> sentHash;
extern int local_rank;
// Ensure a directory exists (best-effort)
static void ensure_dir(const char* path) {
  struct stat st = {};
  if (stat(path, &st) != 0) {
    mkdir(path, 0755);
  }
}


// Copy FlowSim's exact callback data structure
struct M4CallbackData {
    M4Network* network;
    int src;
    int dst; 
    uint64_t count;
    AstraSim::ncclFlowTag flowTag;
    uint64_t actual_completion_time;
    M4CallbackData* receiver_data = nullptr;
    uint64_t start_time = 0;
    void (*msg_handler)(void* fun_arg) = nullptr;
    void* fun_arg = nullptr;
};

// Copy FlowSim's exact flow tracking
static std::map<std::tuple<int, int, int, int>, uint64_t> flow_start_times;
static std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_sent_callback;
static std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_notify_receiver;

// Copy FlowSim's exact dependency checking functions
bool is_sending_finished(int src, int dst, AstraSim::ncclFlowTag flowTag) {
    int dep_cur_id = flowTag.current_flow_id;
    auto dep_key = std::make_pair(dep_cur_id, std::make_pair(src, dst));
    if (waiting_to_sent_callback.find(dep_key) != waiting_to_sent_callback.end()) {
        waiting_to_sent_callback[dep_key]--;
        if (waiting_to_sent_callback[dep_key] <= 0) {
            waiting_to_sent_callback.erase(dep_key);
            return true;
        }
        return false;
    }
    return true;
}

bool is_receive_finished(int src, int dst, AstraSim::ncclFlowTag flowTag) {
    int dep_cur_id = flowTag.current_flow_id;
    auto dep_key = std::make_pair(dep_cur_id, std::make_pair(src, dst));
    if (waiting_to_notify_receiver.find(dep_key) != waiting_to_notify_receiver.end()) {
        waiting_to_notify_receiver[dep_key]--;
        if (waiting_to_notify_receiver[dep_key] <= 0) {
            waiting_to_notify_receiver.erase(dep_key);
            return true;
        }
        return false;
    }
    return true;
}

// Copy FlowSim's exact completion callback
static void m4_completion_callback(void* arg) {
    M4CallbackData* data = static_cast<M4CallbackData*>(arg);
    
    // Record the actual completion time when callback is triggered
    data->actual_completion_time = m4_current_time;
    
    // Copy FlowSim's exact logic
    bool sender_done = is_sending_finished(data->src, data->dst, data->flowTag);
    if (sender_done) {
        data->network->notify_sender_sending_finished(data->src, data->dst, data->count, data->flowTag);
    }
    
    bool receiver_done = is_receive_finished(data->src, data->dst, data->flowTag);
    if (receiver_done) {
        // Write per-flow FCT when the receive side finishes (copy FlowSim logic)
        if (fct_output_file == nullptr) {
            fct_output_file = fopen("results/m4/m4_fct.txt", "w");
        }
        if (fct_output_file) {
            auto flow_key = std::make_tuple(data->flowTag.tag_id, data->flowTag.current_flow_id, data->src, data->dst);
            auto it = flow_start_times.find(flow_key);
            if (it != flow_start_times.end()) {
                uint64_t start_time = it->second;
                uint64_t fct_ns = data->actual_completion_time - start_time;
                flow_start_times.erase(it);

                uint32_t src_ip = 0u;
                uint32_t dst_ip = 0u;
                unsigned int src_port = 0u;
                unsigned int dst_port = 0u;
                uint64_t standalone_fct = fct_ns;

                fprintf(fct_output_file, "%08x %08x %u %u %lu %lu %lu %lu %d\n",
                        src_ip, dst_ip, src_port, dst_port, data->count, start_time, fct_ns, standalone_fct,
                        data->flowTag.current_flow_id);
                fflush(fct_output_file);
                ++g_fct_lines_written;
            }
        }

        data->network->notify_receiver_packet_arrived(data->src, data->dst, data->count, data->flowTag);
        if (data->receiver_data) {
            delete data->receiver_data;
        }
    }
    
    delete data;
}

M4Network::~M4Network() {
  if (fct_output_file) {
    fclose(fct_output_file);
    fct_output_file = nullptr;
  }
}

M4Network::M4Network(int _local_rank) : AstraSim::AstraNetworkAPI(_local_rank), device(torch::kCUDA, 0) {
  npu_offset = 0;
  local_rank = _local_rank;
  models_loaded = false;
  n_flows = 0;
  flow_id_in_prop = 0;
  n_flows_active = 0;
  flow_arrival_time = 0.0f;
  flow_completion_time = 0.0f;
  hidden_size_ = 0;
  n_links_max_ = 0;
  
  // Initialize M4 core components (similar to FlowSim)
  event_queue = std::make_shared<EventQueue>();
  // topology will be initialized in M4Astra.cc like FlowSim
}

AstraSim::timespec_t M4Network::sim_get_time() {
  AstraSim::timespec_t time;
  time.time_val = m4_current_time;
  return time;
}

void M4Network::sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) {
  // M4 fake: ONLY advance time, do NOT call the callback to avoid infinite loop
  // The application layer uses sim_get_time() to check progress, so advancing time is key
  m4_current_time += delta.time_val;
  // DO NOT call fun_ptr(fun_arg) here - this causes infinite recursion
}

int M4Network::sim_send(void* buffer, uint64_t count, int type, int dst, int tag,
                        AstraSim::sim_request* request, void (*msg_handler)(void*), void* fun_arg) {
    
    // Copy FlowSim's EXACT logic
    dst += npu_offset;
    M4Task t;
    t.src = rank;
    t.dest = dst;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    
    // Store using FlowSim key (tag_id, (src,dst))
    auto sh_key = make_pair(request->flowTag.tag_id, make_pair(t.src, t.dest));
    sentHash[sh_key] = t;
    
    // Track initial request time (actual send occurs after send latency)
    uint64_t start = m4_current_time;
    
    // Copy FlowSim's exact dependency logic
    int dep_cur_id = request->flowTag.current_flow_id;
    auto dep_key = std::make_pair(dep_cur_id, std::make_pair(rank, dst));
    waiting_to_sent_callback[dep_key]++;
    waiting_to_notify_receiver[dep_key]++;

    // Apply send latency delay like FlowSim
    int send_lat = 0;
    const char* send_lat_env = std::getenv("AS_SEND_LAT");
    if (send_lat_env) {
        try {
            send_lat = std::stoi(send_lat_env);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[M4] AS_SEND_LAT parse error" << std::endl;
        }
    }
    send_lat *= 1000;  // Convert μs to ns

    uint64_t actual_start_time = start + send_lat;

    // Record actual start time for FCT (keyed by tag_id,flow_id,src,dst)
    auto flow_key = std::make_tuple(request->flowTag.tag_id, request->flowTag.current_flow_id, rank, dst);
    flow_start_times[flow_key] = actual_start_time;

    // Advance time by send latency
    m4_current_time += send_lat;

    // Ensure models are loaded
    setup_m4();

    // Compute online features exactly like no_flowsim
    if (!link_params_initialized) {
        if (!s_routing) {
            std::cerr << "[M4] ERROR: Routing framework not initialized" << std::endl;
            throw std::runtime_error("RoutingFramework missing");
        }
        const auto& link_info_map = s_routing->GetTopology().GetLinkInfo();
        bool found = false;
        for (const auto& kv : link_info_map) {
            for (const auto& kv2 : kv.second) {
                auto info = kv2.second;
                default_link_bandwidth_bytes_per_ns = static_cast<double>(info.bandwidth) / 8.0 / 1e9;
                default_link_latency_ns = static_cast<double>(info.delay);
                found = true;
                break;
            }
            if (found) break;
        }
        if (!found || default_link_bandwidth_bytes_per_ns <= 0.0) {
            std::cerr << "[M4] ERROR: Unable to derive link parameters from topology" << std::endl;
            throw std::runtime_error("Invalid topology link parameters");
        }
        link_params_initialized = true;
    }

    int src = rank;
    int dst_node = dst;
    std::vector<int> path = {};
    if (s_routing) {
        path = s_routing->GetFlowSimPathByNodeIds(src, dst_node);
    }
    int num_hops = path.empty() ? 2 : static_cast<int>(path.size());

    const double MTU = 1000.0;
    const double BYTES_PER_HEADER = 48.0;
    double size_bytes = static_cast<double>(count);
    double prop_delay = default_link_latency_ns * (num_hops - 1);
    double trans_delay = ((size_bytes + std::ceil(size_bytes / MTU) * BYTES_PER_HEADER)) / default_link_bandwidth_bytes_per_ns;
    double first_packet = (std::min(MTU, size_bytes) + BYTES_PER_HEADER) / default_link_bandwidth_bytes_per_ns * (num_hops - 2);
    double i_fct_est = trans_delay + prop_delay + first_packet; // ns

    // Initialize tensors lazily
    if (!i_fct_tensor.defined()) {
        i_fct_tensor = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        release_time_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        h_vec = torch::zeros({1, hidden_size_}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        flowid_active_mask = torch::zeros({1}, torch::TensorOptions().dtype(torch::kBool).device(device));
        flowid_to_nlinks_tensor = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
    i_fct_tensor[0] = static_cast<float>(i_fct_est);
    release_time_tensor[0] = static_cast<float>(m4_current_time);
    h_vec.index_put_({0, 0}, 1.0f);
    h_vec.index_put_({0, 2}, static_cast<float>(std::log2(size_bytes / 1000.0 + 1.0)));
    h_vec.index_put_({0, 3}, static_cast<float>(num_hops));
    flowid_to_nlinks_tensor[0] = num_hops;
    flowid_active_mask[0] = true;
    n_flows_active = 1;
    n_flows = 1;
    // Disable arrivals in the online path; we are already active
    flow_id_in_prop = 1;

    // params zeros(13)
    if (!params_tensor.defined()) {
        params_tensor = torch::zeros({13}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Predict completion time via no_flowsim path
    update_times_m4();
    step_m4();
    uint64_t predicted_tx_time = static_cast<uint64_t>(std::max(0.0f, flow_completion_time - release_time_tensor[0].item<float>()));
    if (predicted_tx_time == 0) {
        std::cerr << "[M4] ERROR: predicted_tx_time is zero" << std::endl;
        throw std::runtime_error("M4 predicted zero time");
    }

    // Advance to completion and trigger notifications similar to FlowSim
    m4_current_time += predicted_tx_time;

    // Sender finished notification (unblocks dependency counters and may chain receiver)
    notify_sender_sending_finished(rank, dst, count, request->flowTag);
    
    // Receiver notification attempts delivery; recv side may be already registered
    notify_receiver_packet_arrived(rank, dst, count, request->flowTag);

    return 0;
}

int M4Network::sim_recv(void* buffer, uint64_t count, int type, int src, int tag,
                        AstraSim::sim_request* request, void (*msg_handler)(void*), void* fun_arg) {
    // Mirror FlowSim's recv path: if data already arrived (recvHash), satisfy immediately.
    M4Task t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;

    auto key = std::make_pair(tag, std::make_pair(t.src, t.dest));

    auto it = recvHash.find(key);
    if (it != recvHash.end()) {
        uint64_t received = static_cast<uint64_t>(it->second);
        if (received >= t.count) {
            // Immediate callback
            t.msg_handler(t.fun_arg);
            return 0;
        }
    }

    // Otherwise, register expected receive
    expeRecvHash[key] = t;
    return 0;
}

void M4Network::notify_receiver_packet_arrived(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    auto key = std::make_pair(flowTag.tag_id, std::make_pair(sender_node, receiver_node));

    // Accumulate received bytes
    if (recvHash.find(key) == recvHash.end()) {
        recvHash[key] = static_cast<int>(message_size);
    } else {
        recvHash[key] += static_cast<int>(message_size);
    }

    // If receiver registered, satisfy when full message is received
    auto it = expeRecvHash.find(key);
    if (it != expeRecvHash.end()) {
        M4Task t = it->second;
        uint64_t existing_count = static_cast<uint64_t>(recvHash[key]);
        if (existing_count >= t.count) {
            // Write per-flow FCT upon receive completion
            if (fct_output_file == nullptr) {
                ensure_dir("results");
                ensure_dir("results/m4");
                fct_output_file = fopen("results/m4/m4_fct.txt", "w");
            }
            if (fct_output_file) {
                auto flow_key = std::make_tuple(flowTag.tag_id, flowTag.current_flow_id, sender_node, receiver_node);
                auto fit = flow_start_times.find(flow_key);
                if (fit != flow_start_times.end()) {
                    uint64_t start_time = fit->second;
                    uint64_t fct_ns = m4_current_time - start_time;
                    flow_start_times.erase(fit);
                    uint32_t src_ip = 0u;
                    uint32_t dst_ip = 0u;
                    unsigned int src_port = 0u;
                    unsigned int dst_port = 0u;
                    uint64_t standalone_fct = fct_ns;
                    fprintf(fct_output_file, "%08x %08x %u %u %lu %lu %lu %lu %d\n",
                            src_ip, dst_ip, src_port, dst_port, message_size, start_time, fct_ns, standalone_fct,
                            flowTag.current_flow_id);
                    fflush(fct_output_file);
                    ++g_fct_lines_written;
                }
            }
            // Erase receiver state and invoke callback
            expeRecvHash.erase(it);
            t.msg_handler(t.fun_arg);
        }
    }
}

void M4Network::notify_sender_sending_finished(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    // Decrement dependency counters similar to FlowSim
    auto dep_key = std::make_pair(flowTag.current_flow_id, std::make_pair(sender_node, receiver_node));
    auto it = waiting_to_sent_callback.find(dep_key);
    if (it != waiting_to_sent_callback.end()) {
        it->second -= 1;
        if (it->second <= 0) {
            waiting_to_sent_callback.erase(it);
        }
    }
}

int M4Network::sim_finish() {
    if (fct_output_file) {
        fflush(fct_output_file);
    }
    std::cout << "[M4] sim_finish()" << std::endl;
    return 0;
}

// ========== M4 INFERENCE FUNCTIONS ==========
// Based on m4/inference/main_m4.cpp

void M4Network::setup_m4() {
    if (models_loaded) return;
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA is not available!" << std::endl;
        return;
    }
    
    torch::NoGradGuard no_grad;

    // Model directory: use relative path from M4 frontend
    const std::string model_dir = "./astra-sim-alibabacloud/astra-sim/network_frontend/m4/models/";
    
    try {
        // Load all 8 PyTorch models
        lstmcell_time = torch::jit::load(model_dir + "lstmcell_time.pt", device);
        lstmcell_rate = torch::jit::load(model_dir + "lstmcell_rate.pt", device);
        lstmcell_time_link = torch::jit::load(model_dir + "lstmcell_time_link.pt", device);
        lstmcell_rate_link = torch::jit::load(model_dir + "lstmcell_rate_link.pt", device);
        output_layer = torch::jit::load(model_dir + "output_layer.pt", device);
        gnn_layer_0 = torch::jit::load(model_dir + "gnn_layer_0.pt", device);
        gnn_layer_1 = torch::jit::load(model_dir + "gnn_layer_1.pt", device);
        gnn_layer_2 = torch::jit::load(model_dir + "gnn_layer_2.pt", device);

        // Set models to evaluation mode
        lstmcell_time.eval();
        lstmcell_rate.eval();
        lstmcell_time_link.eval();
        lstmcell_rate_link.eval();
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

        models_loaded = true;
        std::cout << "[M4] Successfully loaded all 8 PyTorch models" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load M4 models: " << e.what() << std::endl;
        std::cerr << "[ERROR] Model directory: " << model_dir << std::endl;
        models_loaded = false;
    }

    // Parse no_flowsim config for hidden_size and n_links_max
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
            std::cout << "[M4] Config loaded (" << cfg_path << ") hidden_size=" << hidden_size_ << ", n_links_max=" << n_links_max_ << std::endl;
        } else {
            std::cerr << "[M4] Warning: cannot open config at " << cfg_path << ". Using defaults." << std::endl;
            hidden_size_ = 64;
            n_links_max_ = 4096;
        }
    } catch (const std::exception& e) {
        std::cerr << "[M4] Config parse error: " << e.what() << std::endl;
        hidden_size_ = 64;
        n_links_max_ = 4096;
    }
}

void M4Network::update_times_m4() {
    if (!models_loaded) {
        std::cerr << "[M4] ERROR: models not loaded; cannot run inference." << std::endl;
        throw std::runtime_error("M4 models not loaded");
    }
    torch::NoGradGuard no_grad;

    // Single-flow online path
    flow_arrival_time = release_time_tensor.defined() ? release_time_tensor[0].item<float>() : 0.0f;
    flow_completion_time = std::numeric_limits<float>::infinity();

    if (n_flows_active > 0) {
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        if (!params_tensor.defined()) {
            params_tensor = torch::zeros({13}, options_float);
        }
        auto nlinks_cur = flowid_to_nlinks_tensor.index({0}).to(options_float).view({1, 1});
        auto params_data = params_tensor.view({1, 13});
        auto h_vec_active = h_vec.index({0, torch::indexing::Slice()}).view({1, hidden_size_});
        auto input_tensor = torch::cat({nlinks_cur, params_data, h_vec_active}, 1);
        auto sldn = output_layer.forward({ input_tensor }).toTensor().view(-1);
        sldn = torch::clamp(sldn, 1.0f, std::numeric_limits<float>::infinity());
        auto fct_stamp_est = release_time_tensor.index({0}) + sldn.index({0}) * i_fct_tensor.index({0});
        flow_completion_time = fct_stamp_est.item<float>();
        completed_flow_id = 0;
    }
}

void M4Network::step_m4() {
    if (!models_loaded) return;
    torch::NoGradGuard no_grad;
    if (flow_completion_time < std::numeric_limits<float>::infinity()) {
        m4_current_time = (uint64_t)flow_completion_time;
        if (n_flows_active > 0) {
            n_flows_active--;
        }
    }
}