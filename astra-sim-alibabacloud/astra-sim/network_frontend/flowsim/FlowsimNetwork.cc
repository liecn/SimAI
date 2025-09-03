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

#include"FlowsimNetwork.h"
#include"FlowSim.h"
#include"astra-sim/system/SendPacketEventHandlerData.hh"
#include"astra-sim/system/RecvPacketEventHadndlerData.hh"
#include"astra-sim/system/MockNcclLog.h"
#include"astra-sim/system/SharedBusStat.hh"
#include <iomanip>
#include <cstdlib>
#include <cstring>

// Global variable definitions needed for receiver-side event handling (like NS3)
std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
// NS3-style receiver bookkeeping keyed by (tag_id, (src,dst))
std::map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
std::map<std::pair<int, std::pair<int, int>>, int> recvHash;
std::map<std::pair<int, int>, int64_t> nodeHash;
// NS3-style sender bookkeeping keyed by (tag_id, (src,dst))
std::map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
// Track flow start times with a unique identity per flow instance to avoid collisions across batches
// key: (tag_id, current_flow_id, src, dst)
static std::map<std::tuple<int,int,int,int>, uint64_t> flow_start_times;

// NS3-style dependency counters keyed by (current_flow_id, (src,dst))
std::map<std::pair<int,std::pair<int,int>>,int> waiting_to_sent_callback;  
std::map<std::pair<int,std::pair<int,int>>,int> waiting_to_notify_receiver;
// Minimal logging for debugging (controlled by FS_TRACE_DEPS)
static bool fs_trace_deps = [](){
    const char* v = std::getenv("FS_TRACE_DEPS");
    return v && std::strcmp(v, "0") != 0;
}();

// FCT summary counter
static uint64_t g_fct_lines_written = 0;

// NS3-style: no batch-level gating, individual flow handlers trigger workload continuation


extern int local_rank;

// FlowSim callback data structure 
struct FlowSimCallbackData {
    FlowSimNetWork* network;
    int src;
    int dst; 
    uint64_t count;
    AstraSim::ncclFlowTag flowTag;
    uint64_t actual_completion_time;
    FlowSimCallbackData* receiver_data = nullptr; // For chaining receiver callback
    uint64_t start_time = 0;
    void (*msg_handler)(void* fun_arg) = nullptr;
    void* fun_arg = nullptr;
};

// DEPENDENCY CHECKING WITH SAFETY CHECKS AND CORRUPTION DETECTION
bool is_sending_finished(int src, int dst, AstraSim::ncclFlowTag flowTag) {
    // NS3-style: counters keyed by current_flow_id
    int cur_id = flowTag.current_flow_id;
    auto key = std::make_pair(cur_id, std::make_pair(src, dst));
    
    auto it = waiting_to_sent_callback.find(key);
    if (it != waiting_to_sent_callback.end()) {
        // SAFETY CHECK: Prevent underflow corruption
        if (it->second <= 0) {
            std::cerr << "[DEPENDENCY ERROR] Sender counter already at " << it->second 
                      << " for cur_id=" << cur_id << " src=" << src << " dst=" << dst 
                      << " at time=" << FlowSim::Now() << "ns" << std::endl;
            waiting_to_sent_callback.erase(it);
            return true; // Force completion to avoid deadlock
        }
        
        // Decrement counter
        it->second--;
        if (it->second == 0) {
            if (fs_trace_deps) {
                std::cout << "[DEP SEND-ZERO] cur_id=" << cur_id << " src=" << src << " dst=" << dst << std::endl;
            }
            waiting_to_sent_callback.erase(it);
            return true;
        }
    }
    return false;
}

bool is_receive_finished(int src, int dst, AstraSim::ncclFlowTag flowTag) {
    // NS3-style: counters keyed by current_flow_id
    int cur_id = flowTag.current_flow_id;
    auto key = std::make_pair(cur_id, std::make_pair(src, dst));
    
    auto it = waiting_to_notify_receiver.find(key);
    if (it != waiting_to_notify_receiver.end()) {
        // Prevent underflow corruption  
        if (it->second <= 0) {
            waiting_to_notify_receiver.erase(it);
            return true; // Force completion
        }
        
        // Decrement counter
        it->second--;
        if (it->second == 0) {
            if (fs_trace_deps) {
                std::cout << "[DEP RECV-ZERO] cur_id=" << cur_id << " src=" << src << " dst=" << dst << std::endl;
            }
            waiting_to_notify_receiver.erase(it);
            return true;
        }
    }
    return false;
}

// Global instance for callback access (simplified for single-node simulation)
static FlowSimNetWork* global_flowsim_network = nullptr;

// (FlowSimCallbackData struct already defined above with receiver_data field)

// Forward declarations of static callbacks
static void flowsim_completion_callback(void* arg);
static void flowsim_receiver_callback(void* arg);

FlowSimNetWork::FlowSimNetWork(int _local_rank) : AstraNetworkAPI(_local_rank) {
    this->npu_offset = 0;
    
    // Set global instance for callback access
    global_flowsim_network = this;
}

FlowSimNetWork::~FlowSimNetWork() {
}

int FlowSimNetWork::sim_finish() {
    for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
        std::pair<int, int> p = it->first;
        if (p.second == 0) {
            std::cout << "All data sent from node " << p.first << " is " << it->second << "\n";
        } else {
            std::cout << "All data received by node " << p.first << " is " << it->second << "\n";
        }
    }
    std::cout << "[FCT SUMMARY] lines=" << g_fct_lines_written << std::endl;
    return 0;
}

AstraSim::timespec_t FlowSimNetWork::sim_get_time() {
  AstraSim::timespec_t timeSpec;
  timeSpec.time_val = FlowSim::Now();
  return timeSpec;
}

void FlowSimNetWork::sim_schedule(
    AstraSim::timespec_t delta,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    
    // Use FlowSim::Schedule (same logic as NS3 but with FlowSim backend)
    FlowSim::Schedule(delta.time_val, fun_ptr, fun_arg);
    return;
}

int FlowSimNetWork::sim_send(
    void* buffer, uint64_t count, int type, int dst, int tag,
    AstraSim::sim_request* request, void (*msg_handler)(void* fun_arg), void* fun_arg) {
    
    static int send_count = 0;
    send_count++;
    if (send_count <= 5 || send_count % 10000 == 0) {
        // CRITICAL FIX: Log at current time (before delay) to match NS3 exactly
        uint64_t current_time = FlowSim::Now();
        std::cout << "[FLOWSIM] SEND #" << send_count << " at time=" << current_time << "ns: "
                  << "src=" << rank << " (group=" << (rank / 16) << ") -> "
                  << "dst=" << dst << " (group=" << (dst / 16) << "), "
                  << "size=" << count << ", tag=" << request->flowTag.tag_id << std::endl;
    }
    // Store callback using NS3's pattern
    dst += npu_offset;
    task1 t;
    t.src = rank;
    t.dest = dst;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    
    // Store using NS3 key (tag_id, (src,dst))
    auto sh_key = make_pair(request->flowTag.tag_id, make_pair(t.src, t.dest));
    sentHash[sh_key] = t;
    // Store for callback lookup (no logging needed)
    
    // Track initial request time (will be updated to actual start time after delay)
    uint64_t start = FlowSim::Now();
    
    // IMPLEMENT NS-3's EXACT DEPENDENCY LOGIC: Use completion counters and realistic transmission delays
    
    // Prepare callback data for sender completion  
    FlowSimCallbackData* completion_data = new FlowSimCallbackData{this, rank, dst, count, request->flowTag, 0};

    // Prepare callback data for receiver event
    FlowSimCallbackData* receiver_data = new FlowSimCallbackData{this, rank, dst, count, request->flowTag, 0};

    // NS3-STYLE GATING: dependency counters keyed by current_flow_id
    int dep_cur_id = request->flowTag.current_flow_id;
    auto dep_key = std::make_pair(dep_cur_id, std::make_pair(rank, dst));
    waiting_to_sent_callback[dep_key]++;
    waiting_to_notify_receiver[dep_key]++;

    // Apply send latency delay like NS3 does
    int send_lat = 0;  // Default 6μs like NS3
    const char* send_lat_env = std::getenv("AS_SEND_LAT");
    if (send_lat_env) {
        try {
            send_lat = std::stoi(send_lat_env);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[FLOWSIM] AS_SEND_LAT parse error" << std::endl;
        }
    }
    send_lat *= 1000;  // Convert μs to ns, exactly like NS3

    // CRITICAL FIX: Match NS3's exact logging timing
    // NS3 logs SEND time BEFORE applying send_lat delay, then schedules the actual flow
    // Note: send_count is already declared at the top of this function
    
    // Schedule the send with delay, matching NS3's appCon.Start(Time(send_lat))
    completion_data->receiver_data = receiver_data; // receiver gated on completion
    completion_data->start_time = start + send_lat;  // Record actual start time after delay
    completion_data->msg_handler = msg_handler;
    completion_data->fun_arg = fun_arg;
    
    // Schedule FlowSim::Send with the same delay as NS3
    FlowSim::Schedule(send_lat, [](void* arg) {
        FlowSimCallbackData* data = static_cast<FlowSimCallbackData*>(arg);
        // Update flow start time to actual send time (after delay)
        uint64_t actual_start = FlowSim::Now();
        auto flow_key = std::make_tuple(data->flowTag.tag_id, data->flowTag.current_flow_id, data->src, data->dst);
        flow_start_times[flow_key] = actual_start;  // Update with actual start time
        
        FlowSim::Send(data->src, data->dst, data->count, data->flowTag.tag_id, flowsim_completion_callback, data);
    }, completion_data);
    
    return 0;
}



// Static callback function for FlowSim completion
static void flowsim_completion_callback(void* arg) {
    FlowSimCallbackData* data = static_cast<FlowSimCallbackData*>(arg);
    
    // Record the actual completion time when callback is triggered
    data->actual_completion_time = FlowSim::Now();
    
    // CRITICAL FIX: Handle both sender and receiver notifications in proper sequence
    // 1. Check if sending is finished and notify sender
    bool sender_done = is_sending_finished(data->src, data->dst, data->flowTag);
    if (sender_done) {
        // Write FCT and sender-side accounting, trigger workload continuation per flow (NS3-style)
        data->network->notify_sender_sending_finished(data->src, data->dst, data->count, data->flowTag);
    }
    
    // 2. Check if receiving is finished and notify receiver (triggers NCCL dependency chain)
    bool receiver_done = is_receive_finished(data->src, data->dst, data->flowTag);
    if (receiver_done) {
        data->network->notify_receiver_packet_arrived(data->src, data->dst, data->count, data->flowTag);
        if (data->receiver_data) {
            delete data->receiver_data;  // Clean up receiver data
        }
    }

    // 3. No NI-level batch queue: gating is solely via dependency counters (NS3-style)
    
    delete data;
}

// Static callback function for FlowSim receiver events
static void flowsim_receiver_callback(void* arg) {
    FlowSimCallbackData* data = static_cast<FlowSimCallbackData*>(arg);
    
    // COPY NS-3's EXACT DEPENDENCY LOGIC: Only notify receiver if all flows in batch are ready
    if (is_receive_finished(data->src, data->dst, data->flowTag)) {
        data->network->notify_receiver_packet_arrived(data->src, data->dst, data->count, data->flowTag);
    }
    delete data;
}

void FlowSimNetWork::notify_receiver_packet_arrived(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    
    int tag = flowTag.tag_id;
    
    // Check if receiver is registered
    auto recv_key = make_pair(tag, make_pair(sender_node, receiver_node));
    // Check if receiver is registered (no logging needed)
    if (expeRecvHash.find(recv_key) != expeRecvHash.end()) {
        task1 t = expeRecvHash[recv_key];
        
        if (t.count == message_size) {
            // Remove from receiver hash
            expeRecvHash.erase(recv_key);
            
            // Update nodeHash
            if (nodeHash.find(make_pair(receiver_node, 1)) == nodeHash.end()) {
                nodeHash[make_pair(receiver_node, 1)] = message_size;
            } else {
                nodeHash[make_pair(receiver_node, 1)] += message_size;
            }
            
            // Set the flowTag in the event handler data - CRITICAL for flow continuity
            AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
            // MATCH NS3: Check for pending flowTag and restore it
            assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
            if(receiver_pending_queue.count(std::make_pair(std::make_pair(receiver_node, sender_node),tag))!= 0) {
                AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node),tag)];
                receiver_pending_queue.erase(std::make_pair(std::make_pair(receiver_node,sender_node),tag));
                ehd->flowTag = pending_tag;
            } else {
                ehd->flowTag = flowTag;
            }
            
            NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim triggering PacketReceived event for receiver cur_id=%d", ehd->flowTag.current_flow_id);
            
            // Call receiver handler directly (NS3-style) to continue AstraSim chain
            t.msg_handler(t.fun_arg);
            
        } else {
            NcclLog->writeLog(NcclLogLevel::ERROR,"FlowSim receiver size mismatch: expected=%lu actual=%lu", t.count, message_size);
        }
    } else {
        // Mirror NS3: queue flowTag and accumulate received size until receiver registers
        receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node), tag)] = flowTag;
        auto rkey = make_pair(tag, make_pair(sender_node, receiver_node));
        if (recvHash.find(rkey) == recvHash.end()) {
            recvHash[rkey] = message_size;
        } else {
            recvHash[rkey] += message_size;
        }
        NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim receiver not found yet; queued pending tag=%d src=%d dst=%d size=%lu", tag, sender_node, receiver_node, message_size);
    }
}

void FlowSimNetWork::notify_sender_sending_finished(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    static int callback_count = 0;
    callback_count++;
    
    // Add CALLBACK logging to match NS3 pattern (first 5 + every 10000th)
    if (callback_count <= 5 || callback_count % 10000 == 0) {
        std::cout << "[FLOWSIM] CALLBACK #" << callback_count << " at time=" << FlowSim::Now() << "ns: "
                  << "src=" << sender_node << " -> dst=" << receiver_node << ", "
                  << "size=" << message_size << ", tag=" << flowTag.tag_id << std::endl;
    }
    
    // Export per-flow FCT data to file (matching NS-3 format)
    static FILE* fct_output_file = nullptr;
    if (fct_output_file == nullptr) {
        fct_output_file = fopen("results/flowsim/flowsim_fct.txt", "w");
    }
    
    if (fct_output_file) {
        // Use the recorded completion time from callback data
        uint64_t completion_time = FlowSim::Now();  // This should be individual flow completion time
        uint64_t start_time = 0;
        uint64_t fct_ns = 0;
        
        // Look up actual start time
        auto flow_key = std::make_tuple(flowTag.tag_id, flowTag.current_flow_id, sender_node, receiver_node);
        auto start_time_it = flow_start_times.find(flow_key);
        if (start_time_it != flow_start_times.end()) {
            start_time = start_time_it->second;
            fct_ns = completion_time - start_time;
            
            flow_start_times.erase(start_time_it);
        } else {
            // ERROR: This should not happen with correct key matching
            std::cerr << "[FLOWSIM FCT ERROR] Flow start time not found for tag=" << flowTag.tag_id 
                      << " src=" << sender_node << " dst=" << receiver_node << std::endl;
            return; // Skip this FCT entry rather than output bad data
        }
        
        // Extended format: src_node dst_node src_port dst_port msg_size start_time fct_ns standalone_fct flow_id
        fprintf(fct_output_file, "%08x %08x %u %u %lu %lu %lu %lu %d\n", 
                sender_node, receiver_node, flowTag.tag_id, 0, 
                message_size, start_time, fct_ns, fct_ns, flowTag.current_flow_id);
        fflush(fct_output_file);
        ++g_fct_lines_written;
    }
    

    
    int tag = flowTag.tag_id;        
    auto skey = make_pair(tag, make_pair(sender_node, receiver_node));
    
    if (sentHash.find(skey) != sentHash.end()) {
      task1 t2 = sentHash[skey];
      AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
      ehd->flowTag = flowTag;   
      
      if (t2.count == message_size) {
        sentHash.erase(skey);
        if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
          nodeHash[make_pair(sender_node, 0)] = message_size;
        } else {
          nodeHash[make_pair(sender_node, 0)] += message_size;
        }
        
        // Call sender handler directly (NS3-style) to continue AstraSim chain
        t2.msg_handler(t2.fun_arg);
        
        goto sender_end_1st_section;
      } else {
        MockNcclLog* NcclLog = MockNcclLog::getInstance();
        NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash msg size != sender_node %d receiver_node %d message_size %lu flow_id ",sender_node,receiver_node,message_size);
      }
    } else {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
      NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash cann't find sender_node %d receiver_node %d message_size %lu",sender_node,receiver_node,message_size);
    }       

sender_end_1st_section:
    return;
}

int FlowSimNetWork::sim_recv(
    void* buffer,
    uint64_t count,
    int /*type*/,
    int src,
    int tag,
    AstraSim::sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
    
    // Implement proper receiver-side event handling like NS3
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    AstraSim::ncclFlowTag flowTag = request->flowTag;
    src += npu_offset;
    
    // Store receiver task
    task1 t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 1;  // receiver type
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    
    AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
    AstraSim::EventType event = ehd->event;
    tag = ehd->flowTag.tag_id;
    
    NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim sim_recv on rank %d tag_id %d channel_id %d",rank,tag,ehd->flowTag.channel_id);
    
    // EXACTLY MATCH NS3's sim_recv logic
    auto key = make_pair(tag, make_pair(t.src, t.dest));
    if (recvHash.find(key) != recvHash.end()) {
        // Data already arrived - immediate callback like NS3
        uint64_t existing_count = recvHash[key];
        if (existing_count == t.count) {
            recvHash.erase(key);
            // Restore pending flowTag like NS3
            if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
                AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
                receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
                ehd->flowTag = pending_tag;
            }
            // Immediate callback
            t.msg_handler(t.fun_arg);
            NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim sim_recv immediate callback - data already arrived");
            return 0;
        } else if (existing_count > t.count) {
            recvHash[key] = existing_count - t.count;
            // Restore pending flowTag like NS3
            if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
                AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
                receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
                ehd->flowTag = pending_tag;
            }
            // Immediate callback
            t.msg_handler(t.fun_arg);
            NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim sim_recv partial immediate callback");
            return 0;
        } else {
            recvHash.erase(key);
            t.count -= existing_count;
            expeRecvHash[key] = t;
        }
    } else {
        // Register expected receive like NS3
        if (expeRecvHash.find(key) == expeRecvHash.end()) {
            expeRecvHash[key] = t;
            NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim sim_recv registered new receiver src=%d dst=%d tag=%d count=%lu",
                             t.src, t.dest, tag, count);
        } else {
            // Multiple recv calls for same key - should not happen in normal flow
            NcclLog->writeLog(NcclLogLevel::ERROR,"FlowSim sim_recv duplicate registration src=%d dst=%d tag=%d",
                             t.src, t.dest, tag);
        }
    }
    
    return 0;
}