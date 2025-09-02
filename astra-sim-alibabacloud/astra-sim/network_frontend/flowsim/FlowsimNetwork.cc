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

// Global variable definitions needed for receiver-side event handling (like NS3)
std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
std::map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
std::map<std::pair<int, std::pair<int, int>>, int> recvHash;
std::map<std::pair<int, int>, int64_t> nodeHash;
std::map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
std::map<std::pair<int, std::pair<int, int>>, uint64_t> flow_start_times; // Track flow start times
extern int local_rank;

// Global instance for callback access (simplified for single-node simulation)
static FlowSimNetWork* global_flowsim_network = nullptr;

// Callback data structure for FlowSim completion (must appear before sim_send)
struct FlowSimCallbackData {
    FlowSimNetWork* network;
    int src;
    int dst;
    uint64_t count;
    AstraSim::ncclFlowTag flowTag;
    uint64_t actual_completion_time;  // Store the scheduled completion time
};

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
        // Apply same send latency adjustment as used in FlowSim::Send for accurate timing display
        uint64_t send_latency_ns = 0;
        const char* send_lat_env = std::getenv("AS_SEND_LAT");
        if (send_lat_env) {
            send_latency_ns = std::stoi(send_lat_env) * 1000;  // Convert μs to ns
        }
        
        // Calculate group ID based on model_parallel_NPU_group=16
        int src_group = rank / 16;
        int dst_group = dst / 16;
        std::cout << "[FLOWSIM] SEND #" << send_count << " at time=" << (FlowSim::Now() + send_latency_ns)
                  << "ns: src=" << rank << " (group=" << src_group << ") -> dst=" << dst << " (group=" << dst_group << "), size=" << count 
                  << ", tag=" << tag << std::endl;
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
    
    sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
    
    // Track flow start time for accurate FCT calculation (after send latency for fair NS-3 comparison)
    uint64_t send_latency_ns = 0;
    const char* send_lat_env = std::getenv("AS_SEND_LAT");
    if (send_lat_env) {
        send_latency_ns = std::stoi(send_lat_env) * 1000;  // Convert μs to ns
    }
    flow_start_times[make_pair(request->flowTag.tag_id, make_pair(rank, dst))] = FlowSim::Now() + send_latency_ns;
    
    // Call FlowSim's network simulation
    // Prepare callback data for sender completion  
    FlowSimCallbackData* completion_data = new FlowSimCallbackData{this, rank, dst, count, request->flowTag, 0};

    // Prepare callback data for receiver event (immediate)
    FlowSimCallbackData* receiver_data = new FlowSimCallbackData{this, rank, dst, count, request->flowTag, 0};


    
    // Invoke real FlowSim
    FlowSim::Send(rank, dst, count, tag, flowsim_completion_callback, completion_data);

    // Schedule receiver event at the same simulated time 0ns offset
    FlowSim::Schedule(0, flowsim_receiver_callback, receiver_data);
    
    return 0;
}

// Static callback function for FlowSim completion
static void flowsim_completion_callback(void* arg) {
    FlowSimCallbackData* data = static_cast<FlowSimCallbackData*>(arg);
    // Record the actual completion time when callback is triggered
    data->actual_completion_time = FlowSim::Now();
    data->network->notify_sender_sending_finished(data->src, data->dst, data->count, data->flowTag);
    delete data;
}

// Static callback function for FlowSim receiver events
static void flowsim_receiver_callback(void* arg) {
    FlowSimCallbackData* data = static_cast<FlowSimCallbackData*>(arg);
    data->network->notify_receiver_packet_arrived(data->src, data->dst, data->count, data->flowTag);
    delete data;
}

void FlowSimNetWork::notify_receiver_packet_arrived(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    
    int tag = flowTag.tag_id;
    
    // Check if receiver is registered
    if (expeRecvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != expeRecvHash.end()) {
        task1 t = expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))];
        
        if (t.count == message_size) {
            // Remove from receiver hash
            expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
            recvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
            
            // Update nodeHash
            if (nodeHash.find(make_pair(receiver_node, 1)) == nodeHash.end()) {
                nodeHash[make_pair(receiver_node, 1)] = message_size;
            } else {
                nodeHash[make_pair(receiver_node, 1)] += message_size;
            }
            
            // Set the flowTag in the event handler data
            AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
            ehd->flowTag = flowTag;
            
            NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim triggering PacketReceived event for receiver");
            
            // Call receiver callback directly like NS3 does
            t.msg_handler(t.fun_arg);
            
        } else {
            NcclLog->writeLog(NcclLogLevel::ERROR,"FlowSim receiver size mismatch: expected=%lu actual=%lu", t.count, message_size);
        }
    } else {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim receiver not found for tag=%d src=%d dst=%d", tag, sender_node, receiver_node);
    }
}

void FlowSimNetWork::notify_sender_sending_finished(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
    static int callback_count = 0;
    callback_count++;
    
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
        auto flow_key = make_pair(flowTag.tag_id, make_pair(sender_node, receiver_node));
        auto start_time_it = flow_start_times.find(flow_key);
        if (start_time_it != flow_start_times.end()) {
            start_time = start_time_it->second;
            fct_ns = completion_time - start_time;
            
            // DEBUG: Check if completion times are actually different (reduced)
            if (callback_count <= 3) {
                std::cerr << "[FCT #" << callback_count << "] " << sender_node << "->" << receiver_node
                          << " FCT=" << std::fixed << std::setprecision(1) << (fct_ns/1000.0) << "μs" << std::endl;
            }
            
            flow_start_times.erase(start_time_it);
        } else {
            // ERROR: This should not happen with correct key matching
            std::cerr << "[FLOWSIM FCT ERROR] Flow start time not found for tag=" << flowTag.tag_id 
                      << " src=" << sender_node << " dst=" << receiver_node << std::endl;
            return; // Skip this FCT entry rather than output bad data
        }
        
        // Match NS-3 format: src_node dst_node src_port dst_port msg_size start_time fct_ns standalone_fct
        fprintf(fct_output_file, "%08x %08x %u %u %lu %lu %lu %lu\n", 
                sender_node, receiver_node, flowTag.tag_id, 0, 
                message_size, start_time, fct_ns, fct_ns);
        fflush(fct_output_file);
    }
    
    // Essential logging: First 5 callbacks and every 10,000th callback
    if (callback_count <= 5 || callback_count % 10000 == 0) {
        std::cout << "[FLOWSIM] CALLBACK #" << callback_count << " at time=" << FlowSim::Now() 
                  << "ns: src=" << sender_node << " -> dst=" << receiver_node << ", size=" << message_size 
                  << ", tag=" << flowTag.tag_id << std::endl;
    }
    
    int tag = flowTag.tag_id;        
    
    if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != sentHash.end()) {
      task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
      AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
      ehd->flowTag = flowTag;   
      
      if (t2.count == message_size) {
        sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
          nodeHash[make_pair(sender_node, 0)] = message_size;
        } else {
          nodeHash[make_pair(sender_node, 0)] += message_size;
        }
        
        // CRITICAL FIX: Call with the SendPacketEventHandlerData*, NOT SharedBusStat
        // This will trigger StreamBaseline::sendcallback() -> NcclTreeFlowModel::run(PacketSentFinshed)
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
    
    // Store in recvHash like NS3 does
    if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) != recvHash.end()) {
        uint64_t existing_count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
        NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim sim_recv found existing receiver");
    } else {
        // Register new receiver
        recvHash[make_pair(tag, make_pair(t.src, t.dest))] = count;
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
        
        // Store in receiver_pending_queue like NS3 does
        receiver_pending_queue[make_pair(make_pair(t.src, t.dest), tag)] = flowTag;
        
        NcclLog->writeLog(NcclLogLevel::DEBUG,"FlowSim sim_recv registered new receiver src=%d dst=%d tag=%d count=%lu",
                         t.src, t.dest, tag, count);
    }
    
    return 0;
}