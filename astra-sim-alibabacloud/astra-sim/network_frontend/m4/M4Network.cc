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
#include"M4.h"
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <map>
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <fstream>
#include <sstream>

// Avoid using-directives to prevent conflicts with c10::optional/nullopt

// Static routing framework pointer
std::unique_ptr<AstraSim::RoutingFramework> M4Network::s_routing = nullptr;

static uint64_t g_fct_lines_written = 0;
// Note: FCT file now handled globally like FlowSim

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


// M4CallbackData is declared in M4Network.h

// Copy FlowSim's exact flow tracking
static std::map<std::tuple<int, int, int, int>, uint64_t> flow_start_times;
// Copy FlowSim's simple global FCT file handle
static FILE* g_fct_output_file = nullptr;

// Static counters for logging (same as FlowSim)
static int m4_send_count = 0;
static int m4_callback_count = 0;


// M4 completion callback (same pattern as FlowSim)
// Lightweight counters for concise logging (FlowSim-style)

static void m4_completion_callback(void* arg) {
    M4CallbackData* data = static_cast<M4CallbackData*>(arg);

    // Record the actual completion time when callback is triggered
    data->actual_completion_time = M4::Now();

    // Unconditionally notify sender completion (works with SimAI integration)
    data->network->notify_sender_sending_finished(data->src, data->dst, data->count, data->flowTag);

    // Open FCT file on first use
    if (g_fct_output_file == nullptr) {
        std::string fct_file_path = data->network->result_dir + "m4_fct.txt";
        ensure_dir(data->network->result_dir.c_str());
        g_fct_output_file = fopen(fct_file_path.c_str(), "w");
    }
    if (g_fct_output_file) {
        auto flow_key = std::make_tuple(data->flowTag.tag_id, data->flowTag.current_flow_id, data->src, data->dst);
        auto it = flow_start_times.find(flow_key);
        uint64_t start_time = 0;
        if (it != flow_start_times.end()) {
            start_time = it->second;
            flow_start_times.erase(it);
        } else {
            // Fallback: use start_time stored in callback data if key was overwritten
            start_time = data->start_time;
        }
        if (start_time > 0 && data->actual_completion_time >= start_time) {
            uint64_t fct_ns = data->actual_completion_time - start_time;

            uint32_t src_ip = 0u;
            uint32_t dst_ip = 0u;
            unsigned int src_port = 0u;
            unsigned int dst_port = 0u;

            const AstraSim::RoutingFramework* rf = M4::GetRoutingFramework();
            if (!rf) {
                throw std::runtime_error("[M4 ERROR] RoutingFramework is null when computing standalone_fct (send)");
            }
            uint64_t base_rtt = rf->GetPairRtt(data->src, data->dst);
            uint64_t b_bps = rf->GetPairBandwidth(data->src, data->dst);
            const uint32_t packet_payload_size = 1000u;
            const uint32_t header_overhead = 52u;
            uint64_t num_pkts = (data->count + packet_payload_size - 1) / packet_payload_size;
            uint64_t total_bytes = data->count + num_pkts * header_overhead;
            uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b_bps;

            fprintf(g_fct_output_file, "%08x %08x %u %u %lu %lu %lu %lu %d\n",
                    src_ip, dst_ip, src_port, dst_port, data->count, start_time, fct_ns, standalone_fct,
                    data->flowTag.current_flow_id);
            if (g_fct_lines_written == 0) {
                std::cout << "[M4] First FCT: " << (fct_ns/1000.0) << "μs" << std::endl;
            }
            fflush(g_fct_output_file);
            ++g_fct_lines_written;
        }
    }

    // Unconditionally notify receiver arrival (ensures SimAI/NCCL dependency chain progresses)
    data->network->notify_receiver_packet_arrived(data->src, data->dst, data->count, data->flowTag);

    // Mark flow completed in M4 state
    M4::OnFlowCompleted(data->flowTag.current_flow_id);
}

M4Network::~M4Network() {
  // FCT file now handled globally like FlowSim
}

M4Network::M4Network(int _local_rank, const std::string& result_dir) : AstraSim::AstraNetworkAPI(_local_rank) {
  npu_offset = 0;
  local_rank = _local_rank;
  this->result_dir = result_dir;
  
  // Initialize M4 core components (similar to FlowSim)
  event_queue = std::make_shared<EventQueue>();
  // topology will be initialized in M4Astra.cc like FlowSim
}

AstraSim::timespec_t M4Network::sim_get_time() {
  AstraSim::timespec_t time;
  time.time_val = M4::Now();
  return time;
}

void M4Network::sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) {
  // Use M4::Schedule (same logic as FlowSim but with M4 backend)
  M4::Schedule(delta.time_val, fun_ptr, fun_arg);
  return;
}

// M4 processes sends/receives immediately like FlowSim - no event queue needed

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
    uint64_t start = static_cast<uint64_t>(M4::Now());
    
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

    // Match FlowSim's EXACT send logging format
    m4_send_count++;
    if (m4_send_count <= 5 || m4_send_count % 10000 == 0) {
        uint64_t now = static_cast<uint64_t>(M4::Now());
        std::cout << "[M4] SEND #" << m4_send_count << " at time=" << now << "ns: "
                  << "src=" << rank << " (group=" << (rank / 16) << ") -> "
                  << "dst=" << dst << " (group=" << (dst / 16) << "), "
                  << "size=" << count << ", tag=" << request->flowTag.tag_id << std::endl;
    }

    // Create callback data for M4::Send
    M4CallbackData* send_data = new M4CallbackData{this, rank, dst, count, request->flowTag, 0, nullptr, 0, msg_handler, fun_arg};
    
    // Record actual start time for FCT (keyed by tag_id,flow_id,src,dst)
    uint64_t actual_start_time = start + send_lat;
    auto flow_key = std::make_tuple(request->flowTag.tag_id, request->flowTag.current_flow_id, rank, dst);
    flow_start_times[flow_key] = actual_start_time;
    // Also store in callback data for fallback logging
    send_data->start_time = actual_start_time;

    // Schedule M4::Send with the same delay as FlowSim
    M4::Schedule(send_lat, [](void* arg) {
        M4CallbackData* data = static_cast<M4CallbackData*>(arg);
        // Update flow start time to actual send time (after delay)
        uint64_t actual_start = static_cast<uint64_t>(M4::Now());
        auto flow_key = std::make_tuple(data->flowTag.tag_id, data->flowTag.current_flow_id, data->src, data->dst);
        flow_start_times[flow_key] = actual_start;  // Update with actual start time
        data->start_time = actual_start;            // Keep in callback data as well
        
        M4::Send(data->src, data->dst, data->count, data->flowTag.tag_id, m4_completion_callback, data);
    }, send_data);
    
    return 0;
}

int M4Network::sim_recv(void* buffer, uint64_t count, int type, int src, int tag,
                        AstraSim::sim_request* request, void (*msg_handler)(void*), void* fun_arg) {
    // Mirror FlowSim's recv path with proper tag restoration and immediate delivery
    M4Task t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 1;  // receiver type
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;

    // Extract tag from handler data like FlowSim
    AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
    AstraSim::EventType event = ehd->event;
    tag = ehd->flowTag.tag_id;

    auto key = std::make_pair(tag, std::make_pair(t.src, t.dest));

    if (recvHash.find(key) != recvHash.end()) {
        // Data already arrived - immediate callback
        uint64_t existing_count = static_cast<uint64_t>(recvHash[key]);
        if (existing_count == t.count || existing_count > t.count) {
            if (existing_count == t.count) {
                recvHash.erase(key);
            } else {
                recvHash[key] = static_cast<int>(existing_count - t.count);
            }
            // Restore pending flowTag like FlowSim if queued before recv registered
            if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
                AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
                receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
                ehd->flowTag = pending_tag;
            }
            // Immediate callback
            t.msg_handler(t.fun_arg);
            return 0;
        } else {
            // Partial data, keep remaining expectation
            recvHash.erase(key);
            t.count -= existing_count;
            expeRecvHash[key] = t;
            return 0;
        }
    }

    // Otherwise, register expected receive
    if (expeRecvHash.find(key) == expeRecvHash.end()) {
        expeRecvHash[key] = t;
    }
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
            expeRecvHash.erase(it);

            // Update nodeHash for receiver accounting like FlowSim
            if (nodeHash.find(std::make_pair(receiver_node, 1)) == nodeHash.end()) {
                nodeHash[std::make_pair(receiver_node, 1)] = message_size;
            } else {
                nodeHash[std::make_pair(receiver_node, 1)] += message_size;
            }

            // Set flowTag in handler data before invoking callback
            AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
            // If a pending tag was queued before receiver registered, restore it
            int tag = flowTag.tag_id;
            if (receiver_pending_queue.count(std::make_pair(std::make_pair(receiver_node, sender_node), tag)) != 0) {
                AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node), tag)];
                receiver_pending_queue.erase(std::make_pair(std::make_pair(receiver_node, sender_node), tag));
                ehd->flowTag = pending_tag;
            } else {
                ehd->flowTag = flowTag;
            }

            t.msg_handler(t.fun_arg);
        }
    } else {
        // Receiver not yet registered: queue flowTag and accumulate received size
        receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node), flowTag.tag_id)] = flowTag;
        auto rkey = std::make_pair(flowTag.tag_id, std::make_pair(sender_node, receiver_node));
        if (recvHash.find(rkey) == recvHash.end()) {
            recvHash[rkey] = static_cast<int>(message_size);
        } else {
            recvHash[rkey] += static_cast<int>(message_size);
        }
        
        if (nodeHash.find(std::make_pair(receiver_node, 1)) == nodeHash.end()) {
            nodeHash[std::make_pair(receiver_node, 1)] = message_size;
        } else {
            nodeHash[std::make_pair(receiver_node, 1)] += message_size;
        }
    }
}

void M4Network::notify_sender_sending_finished(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    // FlowSim-style callback logging
    m4_callback_count++;
    // Concise callback logging (same gating as FlowSim)
    if (m4_callback_count <= 5 || m4_callback_count % 10000 == 0) {
        std::cout << "[M4] CALLBACK #" << m4_callback_count
                  << " at time=" << static_cast<uint64_t>(M4::Now()) << "ns: "
                  << "src=" << sender_node << " -> dst=" << receiver_node
                  << ", size=" << message_size << ", tag=" << flowTag.tag_id
                  << std::endl;
    }

    int tag = flowTag.tag_id;
    auto skey = std::make_pair(tag, std::make_pair(sender_node, receiver_node));
    if (sentHash.find(skey) != sentHash.end()) {
        M4Task t2 = sentHash[skey];
        AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
        ehd->flowTag = flowTag;
        if (t2.count == message_size) {
            sentHash.erase(skey);
            if (nodeHash.find(std::make_pair(sender_node, 0)) == nodeHash.end()) {
                nodeHash[std::make_pair(sender_node, 0)] = message_size;
            } else {
                nodeHash[std::make_pair(sender_node, 0)] += message_size;
            }
            // Call sender handler directly to continue AstraSim chain
            t2.msg_handler(t2.fun_arg);
        }
    }
}

int M4Network::sim_finish() {
    // CRITICAL FIX: Process any remaining pending flows before finishing
    M4::process_final_batch();
    
    // Match FlowSim's exact data transfer statistics
    for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
        std::pair<int, int> p = it->first;
        if (p.second == 0) {
            std::cout << "All data sent from node " << p.first << " is " << it->second << "\n";
        } else {
            std::cout << "All data received by node " << p.first << " is " << it->second << "\n";
        }
    }
    // Match FlowSim's FCT summary format exactly
    std::cout << "[FCT SUMMARY] lines=" << g_fct_lines_written << std::endl;
    
    if (g_fct_output_file) {
        fflush(g_fct_output_file);
    }
    std::cout << "[M4] sim_finish()" << std::endl;
    return 0;
}