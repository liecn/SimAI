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

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "entry.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
#endif

#define RESULT_PATH "./ncclFlowModel_"

using namespace std;
using namespace ns3;
extern "C" void PrintRoutingStatsDirect();
extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern GPUType gpu_type;
extern std::vector<int>NVswitchs;

struct sim_event {
  void *buffer;
  uint64_t count;
  int type;
  int dst;
  int tag;
  string fnType;
};

class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
  int npu_offset;

public:
  queue<sim_event> sim_event_queue;
  ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
    this->npu_offset = npu_offset;
  }
  ~ASTRASimNetwork() {}
  int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }
  int sim_finish() {
    for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
      pair<int, int> p = it->first;
      if (p.second == 0) {
        std::cout << "sim_finish on sent, " << " Thread id: " << pthread_self() << std::endl;
        cout << "All data sent from node " << p.first << " is " << it->second
             << "\n";
      } else {
        std::cout << "sim_finish on received, " << " Thread id: " << pthread_self() << std::endl;
        cout << "All data received by node " << p.first << " is " << it->second
             << "\n";
      }
    }
    return 0;
  }
  double sim_time_resolution() { return 0; }
  int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }
  AstraSim::timespec_t sim_get_time() {
    AstraSim::timespec_t timeSpec;
    timeSpec.time_val = Simulator::Now().GetNanoSeconds();
    return timeSpec;
  }
  virtual void sim_schedule(AstraSim::timespec_t delta,
                            void (*fun_ptr)(void *fun_arg), void *fun_arg) {
    task1 t;
    t.type = 2;
    t.fun_arg = fun_arg;
    t.msg_handler = fun_ptr;
    t.schTime = delta.time_val;
    Simulator::Schedule(NanoSeconds(t.schTime), t.msg_handler, t.fun_arg);
    return;
  }
  virtual int sim_send(void *buffer, uint64_t count, int type, int dst, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    static int ns3_send_count = 0;
    ns3_send_count++;
    
    // Efficient logging: Only log first 20 sends and every 1000th send
    if (ns3_send_count <= 20 || ns3_send_count % 1000 == 0) {
        std::cout << "[NS3] SEND #" << ns3_send_count << " at time=" << Simulator::Now().GetNanoSeconds() 
                  << "ns: rank=" << rank << " -> dst=" << (dst + npu_offset) << ", size=" << count 
                  << " bytes, tag=" << tag << std::endl;
    }
    
    // Track timing patterns for first batch
    if (ns3_send_count <= 1000) {
        static uint64_t first_send_time = 0;
        if (ns3_send_count == 1) {
            first_send_time = Simulator::Now().GetNanoSeconds();
        }
        
        if (ns3_send_count <= 10 || ns3_send_count % 100 == 0) {
            uint64_t current_time = Simulator::Now().GetNanoSeconds();
            std::cout << "[NS3] TIMING: send #" << ns3_send_count 
                      << " at " << current_time << "ns (+" << (current_time - first_send_time) 
                      << "ns from first)" << std::endl;
        }
    }
    
    // Enhanced logging for debugging
    std::cout << "[NS3] SEND DETAILS #" << ns3_send_count 
              << " tag=" << tag 
              << " request->flowTag.tag_id=" << request->flowTag.tag_id
              << " request->flowTag.current_flow_id=" << request->flowTag.current_flow_id
              << " request->flowTag.channel_id=" << request->flowTag.channel_id << std::endl;
    
    dst += npu_offset;
    task1 t;
    t.src = rank;
    t.dest = dst;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    
    std::cout << "[NS3] Storing callback with key: tag=" << tag 
              << " src=" << t.src << " dst=" << t.dest 
              << " size=" << t.count << std::endl;
    
    {
      #ifdef NS3_MTP
      MtpInterface::explicitCriticalSection cs;
      #endif
      sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
    }
    SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
    return 0;
  }
  virtual int sim_recv(void *buffer, uint64_t count, int type, int src, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    AstraSim::ncclFlowTag flowTag = request->flowTag;
    src += npu_offset;
    task1 t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 1;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
    AstraSim::EventType event = ehd->event;
    tag = ehd->flowTag.tag_id;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"接收事件注册 src %d sim_recv on rank %d tag_id %d channdl id %d",src,rank,tag,ehd->flowTag.channel_id);
    
    if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) !=
        recvHash.end()) {
      uint64_t count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
      if (count == t.count) {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
          receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
          ehd->flowTag = pending_tag;
        } 
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else if (count > t.count) {
        recvHash[make_pair(tag, make_pair(t.src, t.dest))] = count - t.count;
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
          receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
          ehd->flowTag = pending_tag;
        } 
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        t.count -= count;
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      }
    } else {
      if (expeRecvHash.find(make_pair(tag, make_pair(t.src, t.dest))) ==
          expeRecvHash.end()) {
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
          NcclLog->writeLog(NcclLogLevel::DEBUG," 网络包后到，先进行注册 recvHash do not find expeRecvHash.new make src  %d dest  %d t.count:  %d channel_id  %d current_flow_id  %d",t.src,t.dest,t.count,tag,flowTag.current_flow_id);
          
      } else {
        uint64_t expecount =
            expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))].count;
          NcclLog->writeLog(NcclLogLevel::DEBUG," 网络包后到，重复注册 recvHash do not find expeRecvHash.add make src  %d dest  %d expecount:  %d t.count:  %d tag_id  %d current_flow_id  %d",t.src,t.dest,expecount,t.count,tag,flowTag.current_flow_id);
          
      }
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif

sim_recv_end_section:    
    return 0;
  }
  void handleEvent(int dst, int cnt) {
  }
};

struct user_param {
  int thread;
  string workload;
  string network_topo;
  string network_conf;
  bool use_custom_routing;
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    network_conf = "";
    use_custom_routing = false;
  };
  ~user_param(){};
};

static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:g:s:n:c:r"))!=-1){
    switch (opt)
    {
    case 'h':
      /* code */
      std::cout<<"-t    number of threads,default 1"<<std::endl;
      std::cout<<"-w    workloads default none "<<std::endl;
      std::cout<<"-n    network topo"<<std::endl;
      std::cout<<"-c    network_conf"<<std::endl;
      std::cout<<"-r    use custom routing (default: false)"<<std::endl;
      return 1;
      break;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'n':
      user_param->network_topo = optarg;
      break;
    case 'c':
      user_param->network_conf = optarg;
      break;
    case 'r':
      user_param->use_custom_routing = true;
      break;
    default:
      std::cerr<<"-h    help message"<<std::endl;
      return 1;
    }
  }
  return 0 ;
}

int main(int argc, char *argv[]) {
  struct user_param user_param;
  MockNcclLog::set_log_name("SimAI.log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");
  if(user_param_prase(argc,argv,&user_param)){
    return 0;
  }
  #ifdef NS3_MTP
  MtpInterface::Enable(user_param.thread);
  #endif
  
  main1(user_param.network_topo,user_param.network_conf,user_param.use_custom_routing);
  int nodes_num = node_num - switch_num;
  int gpu_num = node_num - nvswitch_num - switch_num;

  std::map<int, int> node2nvswitch; 
  for(int i = 0; i < gpu_num; ++ i) {
    node2nvswitch[i] = gpu_num + i / gpus_per_server;
  }
  for(int i = gpu_num; i < gpu_num + nvswitch_num; ++ i){
    node2nvswitch[i] = i;
    NVswitchs.push_back(i);
  } 

  LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
  LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_INFO);

  std::vector<ASTRASimNetwork *> networks(nodes_num, nullptr);
  std::vector<AstraSim::Sys *> systems(nodes_num, nullptr);

  for (int j = 0; j < nodes_num; j++) {
    networks[j] =
        new ASTRASimNetwork(j ,0);
    systems[j ] = new AstraSim::Sys(
        networks[j], 
        nullptr,                  
        j,                        
        0,               
        1,                        
        {nodes_num},        
        {1},          
        "", 
        user_param.workload, 
        1, 
        1,          
        1,          
        1,
        0,                 
        RESULT_PATH, 
        "test1",            
        true,               
        false,               
        gpu_type,
        {gpu_num},
        NVswitchs,
        gpus_per_server
    );
    systems[j ]->nvswitch_id = node2nvswitch[j];
    systems[j ]->num_gpus = nodes_num - nvswitch_num;
  }
  for (int i = 0; i < nodes_num; i++) {
    systems[i]->workload->fire();
  }
  std::cout << "simulator run " << std::endl;

  Simulator::Run();
  Simulator::Stop(Seconds(2000000000));
  
  // Print routing statistics after simulation completes
  std::cout << "\n[SIMULATION COMPLETE] Printing routing statistics..." << std::endl;
  PrintRoutingStatsDirect();
  
  // Cleanup routing framework before destroying simulator
  CleanupRoutingFramework();
  
  Simulator::Destroy();
  
  #ifdef NS3_MPI
  MpiInterface::Disable ();
  #endif
  return 0;
}

// Function definitions moved from entry.h
void notify_sender_sending_finished(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  MockNcclLog * NcclLog = MockNcclLog::getInstance();
  
  static int ns3_callback_count = 0;
  ns3_callback_count++;
  
  // Efficient logging: Only log first 20 callbacks and every 1000th callback
  if (ns3_callback_count <= 20 || ns3_callback_count % 1000 == 0) {
      std::cout << "[NS3] CALLBACK #" << ns3_callback_count << " at time=" << Simulator::Now().GetNanoSeconds() 
                << "ns: src=" << sender_node << " -> dst=" << receiver_node << ", size=" << message_size 
                << ", tag=" << flowTag.tag_id << std::endl;
  }
  
  // Track callback timing patterns for first batch
  if (ns3_callback_count <= 1000) {
      static uint64_t first_callback_time = 0;
      if (ns3_callback_count == 1) {
          first_callback_time = Simulator::Now().GetNanoSeconds();
      }
      
      if (ns3_callback_count <= 10 || ns3_callback_count % 100 == 0) {
          uint64_t current_time = Simulator::Now().GetNanoSeconds();
          std::cout << "[NS3] CALLBACK_TIMING: #" << ns3_callback_count 
                    << " at " << current_time << "ns (+" << (current_time - first_callback_time) 
                    << "ns from first callback)" << std::endl;
      }
  }
  
  // Enhanced logging for debugging
  std::cout << "[NS3] CALLBACK DETAILS #" << ns3_callback_count 
            << " at time=" << Simulator::Now().GetNanoSeconds() << "ns"
            << " src=" << sender_node << " dst=" << receiver_node 
            << " size=" << message_size 
            << " flowTag.tag_id=" << flowTag.tag_id
            << " flowTag.current_flow_id=" << flowTag.current_flow_id
            << " flowTag.channel_id=" << flowTag.channel_id << std::endl;
  
  #ifdef NS3_MTP
  MtpInterface::explicitCriticalSection cs;
  #endif    
  int tag = flowTag.tag_id;        
  
  std::cout << "[NS3] Looking for callback key: tag=" << tag 
            << " src=" << sender_node << " dst=" << receiver_node 
            << " sentHash.size()=" << sentHash.size() << std::endl;
  
  if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) !=
    sentHash.end()) {
    task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
    ehd->flowTag=flowTag;   
    
    std::cout << "[NS3] Found callback! Expected size=" << t2.count 
              << " actual size=" << message_size << std::endl;
    
    if (t2.count == message_size) {
      sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      
      std::cout << "[NS3] Calling callback handler - this should trigger more sends!" << std::endl;
      
      t2.msg_handler(t2.fun_arg);
      
      std::cout << "[NS3] Callback handler completed" << std::endl;
      goto sender_end_1st_section;
    }else{
      NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash msg size != sender_node %d receiver_node %d message_size %lu flow_id ",sender_node,receiver_node,message_size);
    }
  }else{
    std::cout << "[NS3] ERROR: Callback not found for tag=" << tag << " src=" << sender_node << " dst=" << receiver_node << std::endl;
    std::cout << "[NS3] Available keys in sentHash:" << std::endl;
    for (const auto& entry : sentHash) {
        std::cout << "[NS3]   tag=" << entry.first.first 
                  << " src=" << entry.first.second.first 
                  << " dst=" << entry.first.second.second 
                  << " size=" << entry.second.count << std::endl;
    }
    NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash cann't find sender_node %d receiver_node %d message_size %lu",sender_node,receiver_node,message_size);
  }       
  #ifdef NS3_MTP
  cs.ExitSection();
  #endif

sender_end_1st_section:
  return;
}

void notify_sender_packet_arrivered_receiver(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  int tag = flowTag.channel_id;
  if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) !=
      sentHash.end()) {
    task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
    ehd->flowTag=flowTag;
    if (t2.count == message_size) {
      sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
      t2.msg_handler(t2.fun_arg);
    }
  }
}

void qp_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
  uint32_t total_bytes =
      q->m_size +
      ((q->m_size - 1) / packet_payload_size + 1) *
          (CustomHeader::GetStaticWholeHeaderSize() -
           IntHeader::GetStaticSize()); 
  uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
  fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(),
          q->sport, q->dport, q->m_size, q->startTime.GetTimeStep(),
          (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
  fflush(fout);

  AstraSim::ncclFlowTag flowTag;
  uint64_t notify_size;
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG,"qp finish, src:  %d did:  %d port:  %d total bytes:  %d at the tick:  %d",sid,did,q->sport,q->m_size,AstraSim::Sys::boostedTick());
    if (sender_src_port_map.find(make_pair(q->sport, make_pair(sid, did))) ==
        sender_src_port_map.end()) {
      NcclLog->writeLog(NcclLogLevel::ERROR,"could not find the tag, there must be something wrong");
      exit(-1);
    }
    flowTag = sender_src_port_map[make_pair(q->sport, make_pair(sid, did))];
    sender_src_port_map.erase(make_pair(q->sport, make_pair(sid, did)));
    received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))]+=q->m_size;
    if(!is_receive_finished(sid,did,flowTag)) {
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      return; 
    }
    notify_size = received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    received_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did)));    
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
  notify_receiver_receive_data(sid, did, notify_size, flowTag);
}

void send_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  AstraSim::ncclFlowTag flowTag;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"数据包出发送网卡 send finish, src:  %d did:  %d port:  %d srcip  %d dstip  %d total bytes:  %d at the tick:  %d",sid,did,q->sport,q->sip,q->dip,q->m_size,AstraSim::Sys::boostedTick());
  int all_sent_chunksize;
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    flowTag = sender_src_port_map[make_pair(q->sport, make_pair(sid, did))];
    sent_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))]+=q->m_size;
    if(!is_sending_finished(sid,did,flowTag)) {
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      return;
    }
    all_sent_chunksize = sent_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    sent_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did)));
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
  notify_sender_sending_finished(sid, did, all_sent_chunksize, flowTag);
}

int main1(string network_topo,string network_conf, bool use_custom_routing) {
  // Set random seed BEFORE any other NS3 operations for maximum determinism
  Config::SetGlobal("RngSeed", UintegerValue(12345));
  Config::SetGlobal("RngRun", UintegerValue(1));
  
  clock_t begint, endt;
  begint = clock();

  if (!ReadConf(network_topo,network_conf))
    return -1;
  SetConfig();
  
  // Set the custom routing flag - SetupNetwork will handle the actual enabling
  EnableCustomRouting(use_custom_routing);
  if (use_custom_routing) {
    cout << "[CUSTOM ROUTING] Custom routing enabled via command line argument" << endl;
  } else {
    cout << "[CUSTOM ROUTING] Using default NS3 ECMP routing" << endl;
  }
  
  SetupNetwork(qp_finish,send_finish);

std::cout << "Running Simulation.\n";
  fflush(stdout);
  NS_LOG_INFO("Run Simulation.");

  endt = clock();
  return 0;
}
