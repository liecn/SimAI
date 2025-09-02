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

#include<unistd.h>
#include<string>
#include<iostream>
#include<vector>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/AstraComputeAPI.hh"
#include "astra-sim/system/Common.hh"  // For GPUType enum
#include "astra-sim/system/routing/include/RoutingFramework.h"

#include "FlowsimNetwork.h"
#include "FlowSim.h"
#include "TopologyBuilder.h"
#include "Topology.h"
#include "Type.h"
#include <tuple>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#define RESULT_PATH "./simai_flowsim_"
// WORKLOAD_PATH removed - using direct workload path like NS-3

using namespace std;

extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
// Global variables (similar to NS3 common.h)
GPUType gpu_type = GPUType::NONE;
std::vector<int> NVswitchs;
uint32_t gpus_per_server = 8;
extern map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
extern map<std::pair<int, std::pair<int, int>>, int> recvHash;
extern map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
extern map<std::pair<int, int>, int64_t> nodeHash;
extern int local_rank;

std::vector<string> workloads;
std::vector<std::vector<int>> physical_dims;

struct user_param {
  int thread;
  string workload;
  string network_topo;
  string result_dir;
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    result_dir = "results/flowsim/";
  };
  ~user_param(){};
};

static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:n:o:"))!=-1){
    switch (opt)
    {
    case 'h':
      std::cout<<"-t    number of threads,default 1"<<std::endl;
      std::cout<<"-w    workloads default none "<<std::endl;
      std::cout<<"-n    network topo"<<std::endl;
      std::cout<<"-o    output/result directory (default: results/flowsim/)"<<std::endl;
      return 1;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'n':
      user_param->network_topo = optarg;
      break;
    case 'o':
      user_param->result_dir = optarg;
      // Ensure the directory ends with a slash
      if (user_param->result_dir.back() != '/') {
        user_param->result_dir += '/';
      }
      break;
    default:
      std::cerr<<"-h    help message"<<std::endl;
      return 1;
    }
  }
  return 0 ;
}

int main(int argc,char *argv[]) {
  struct user_param user_param;
  if(user_param_prase(argc,argv,&user_param)){
    return -1;
  }
  
  // Create result directory once at startup
  std::string mkdir_cmd = "mkdir -p " + user_param.result_dir;
  int result = system(mkdir_cmd.c_str());
  if (result != 0) {
    std::cerr << "[FLOWSIM] Warning: Could not create result directory: " << user_param.result_dir << std::endl;
  }
  
  std::cout << "[FLOWSIM] Starting SimAI-FlowSim" << std::endl;
  std::cout << "[FLOWSIM] Workload: " << user_param.workload << std::endl;
  std::cout << "[FLOWSIM] Network: " << user_param.network_topo << std::endl;
  std::cout << "[FLOWSIM] Results: " << user_param.result_dir << std::endl;
  
  // FlowSim always uses custom routing
  std::cout << "[CUSTOM ROUTING] Custom routing enabled via command line argument" << std::endl;
  
  // Initialize routing framework early (same position as NS3)
  if (!user_param.network_topo.empty()) {
    // Add system routing logging to match NS3
    std::cout << "[SYSTEM ROUTING] Routing framework initialized with topology: " << user_param.network_topo << std::endl;
    
    // Create routing framework instance
    auto routing_framework = std::make_unique<AstraSim::RoutingFramework>();
    
    // First parse the topology file
    bool parse_result = routing_framework->ParseTopology(user_param.network_topo);
    
    if (parse_result) {
      routing_framework->PrecalculateRoutingTables();
      routing_framework->PrecalculateFlowPathsForFlowSim(user_param.network_topo, user_param.network_topo);
      
      // Add topology node count logging to match NS3 (read from topology file quickly)
      std::ifstream temp_topof(user_param.network_topo);
      if (temp_topof.is_open()) {
        uint32_t temp_node_num, temp_gpus_per_server, temp_nvswitch_num, temp_switch_num, temp_link_num;
        std::string temp_gpu_type_str;
        temp_topof >> temp_node_num >> temp_gpus_per_server >> temp_nvswitch_num >> temp_switch_num >> temp_link_num >> temp_gpu_type_str;
        std::cout << "[SYSTEM ROUTING] Topology has " << temp_node_num << " nodes" << std::endl;
        temp_topof.close();
      }
      
      // Set the routing framework in FlowSim
      FlowSim::SetRoutingFramework(std::move(routing_framework));
      
      // Add routing sync logging to match NS3 (get actual count)
      const AstraSim::RoutingFramework* rf = FlowSim::GetRoutingFramework();
      size_t actual_flow_count = rf ? rf->GetFlowPathCount() : 0;
      std::cout << "[ROUTING] Synced " << actual_flow_count << " pre-calculated flow paths with backend using routing framework" << std::endl;
    }
  }
  
  
  std::shared_ptr<Topology> topology = construct_fat_tree_topology(user_param.network_topo);

  // Read topology parameters from file (same as NS3 common.h)
  std::ifstream topof(user_param.network_topo);
  uint32_t node_num, switch_num, link_num, nvswitch_num;
  int gpu_num = 0;
  if (topof.is_open()) {
    std::string gpu_type_str;
    topof >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str;
    
    // Calculate GPU count (same as NS3: total nodes - switches - NVSwitches)
    gpu_num = node_num - nvswitch_num - switch_num;
    
    // Set GPU type (same logic as NS3)
    if(gpu_type_str == "A100"){
      gpu_type = GPUType::A100;
    } else if(gpu_type_str == "A800"){
      gpu_type = GPUType::A800;
    } else if(gpu_type_str == "H100"){
      gpu_type = GPUType::H100;
    } else if(gpu_type_str == "H800"){
      gpu_type = GPUType::H800;
    } else{
      gpu_type = GPUType::NONE;
    }
    topof.close();
    
    std::cout << "[FLOWSIM] Topology: " << gpu_num << " GPUs, " << nvswitch_num << " NVSwitches, " 
              << switch_num << " switches (" << gpu_type_str << ")" << std::endl;
  }

  // -----------------------------------------------------------------------------
  // Extract topology information for NVSwitch mapping
  // -----------------------------------------------------------------------------
  std::map<int, int> node2nvswitch; // gpu_id -> nvswitch_id
  try {
    int npus_cnt = 0;
    std::vector<int> nv_ids;
    std::vector<std::tuple<int, int, double, double, double>> dummy_links;
    std::tie(npus_cnt, nvswitch_num, nv_ids, dummy_links) =
        parse_fat_tree_topology_file(user_param.network_topo);

    // Basic deterministic mapping: group GPUs by gpus_per_server and assign to NVSwitch
    // gpus_per_server is now a global variable initialized to 8
    for (int gpu = 0; gpu < npus_cnt; ++gpu) {
      int idx = gpu / gpus_per_server;
      // Clamp idx in case of malformed topologies
      if (idx >= static_cast<int>(nv_ids.size())) idx = nv_ids.size() - 1;
      node2nvswitch[gpu] = nv_ids[idx];
    }
    
    // Populate global NVswitchs vector (same as NS3)
    NVswitchs = nv_ids;
  } catch (const std::exception &e) {
    std::cerr << "[FLOWSIM] WARNING: unable to build NVSwitch mapping: " << e.what()
              << std::endl;
  }

  // Use GPU count from topology file (same as NS3)
  int nodes_num = gpu_num;  // Same as NS3: total nodes minus switches

  // Create FlowSimNetwork and Sys instances
  std::vector<FlowSimNetWork *> networks;
  std::vector<AstraSim::Sys *> systems;
  for (int i = 0; i < nodes_num; i++) {
    FlowSimNetWork *network = new FlowSimNetWork(i);
    networks.push_back(network);
    AstraSim::Sys *system = new AstraSim::Sys(
      network,
      nullptr,
      i,
      0,
      1,
      {nodes_num},      // Use nodes_num for physical dimensions (same as NS3)
      {1},
      "",
      user_param.workload,  // SYNCHRONIZED: Use direct workload path like NS-3
      1,                    // SYNCHRONIZED: Use integer comm_scale like NS-3
      1,
      1,
      1,
      0,
      user_param.result_dir,
      "test1",             // SYNCHRONIZED: Use same test name as NS-3
      true,
      false,
      gpu_type,
      {gpu_num}, // all_gpus
      NVswitchs,
      gpus_per_server
    );
    system->nvswitch_id = (nvswitch_num > 0 && node2nvswitch.count(i)) ? node2nvswitch[i] : -1;
    system->num_gpus = nodes_num - nvswitch_num;  // Match NS3's calculation
    systems.push_back(system);
  }
  
  // FlowSim will write FCT data to the current directory
  
  // Initialize FlowSim AFTER routing framework is set up
  std::shared_ptr<EventQueue> event_queue = std::make_shared<EventQueue>();
  FlowSim::Init(event_queue, topology);
  
  for (int i = 0; i < nodes_num; i++) {  // SYNCHRONIZED: Use same loop pattern as NS-3
    systems[i]->workload->fire();
  }
  
  FlowSim::Run();
  
  // Print data summary once (nodeHash is global, shared across all networks)
  if (!networks.empty()) {
    networks[0]->sim_finish();
  }
  
  // Print routing statistics (same as NS3)
  std::cout << "\n[SIMULATION COMPLETE] Printing routing statistics..." << std::endl;
  
  // Get actual routing statistics from FlowSim
  const AstraSim::RoutingFramework* routing_framework = FlowSim::GetRoutingFramework();
  
  std::cout << "[ROUTING] Routing framework cleaned up." << std::endl;
  
  // Clean shutdown
  FlowSim::Stop();
  FlowSim::Destroy();

  std::cout << "[FLOWSIM] SimAI-FlowSim finished" << std::endl;
  return 0;
};