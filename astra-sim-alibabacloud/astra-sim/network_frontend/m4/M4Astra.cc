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

#include "M4Network.h"
#include <tuple>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#define RESULT_PATH "./simai_m4_"

using namespace std;

// Global variable definitions
std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
// Global variables (similar to NS3 common.h)
GPUType gpu_type = GPUType::NONE;
std::vector<int> NVswitchs;
uint32_t gpus_per_server = 8;
map<std::pair<int, std::pair<int, int>>, struct M4Task> expeRecvHash;
map<std::pair<int, std::pair<int, int>>, int> recvHash;
map<std::pair<int, std::pair<int, int>>, struct M4Task> sentHash;
map<std::pair<int, int>, int64_t> nodeHash;
int local_rank;

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
    result_dir = "results/m4/";
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
      std::cout<<"-o    output/result directory (default: results/m4/)"<<std::endl;
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
    std::cerr << "[M4] Warning: Could not create result directory: " << user_param.result_dir << std::endl;
  }
  
  std::cout << "[M4] Starting SimAI-M4" << std::endl;
  std::cout << "[M4] Workload: " << user_param.workload << " topo: " << user_param.network_topo << " results: " << user_param.result_dir << std::endl;
  
  // M4 always uses custom routing
  std::cout << "[CUSTOM ROUTING] Custom routing enabled via command line argument" << std::endl;
  
  // Initialize routing framework early (same position as FlowSim)
  if (!user_param.network_topo.empty()) {
    // Add system routing logging to match FlowSim
    std::cout << "[SYSTEM ROUTING] Routing framework initialized with topology: " << user_param.network_topo << std::endl;
    
    // Create routing framework instance
    auto routing_framework = std::make_unique<AstraSim::RoutingFramework>();
    
    // First parse the topology file
    bool parse_result = routing_framework->ParseTopology(user_param.network_topo);
    
    if (parse_result) {
      routing_framework->PrecalculateRoutingTables();
      routing_framework->PrecalculateFlowPathsForFlowSim(user_param.network_topo, user_param.network_topo);
      
      // Add topology node count logging to match FlowSim (read from topology file quickly)
      std::ifstream temp_topof(user_param.network_topo);
      if (temp_topof.is_open()) {
        uint32_t temp_node_num, temp_gpus_per_server, temp_nvswitch_num, temp_switch_num, temp_link_num;
        std::string temp_gpu_type_str;
        temp_topof >> temp_node_num >> temp_gpus_per_server >> temp_nvswitch_num >> temp_switch_num >> temp_link_num >> temp_gpu_type_str;
        std::cout << "[SYSTEM ROUTING] Topology has " << temp_node_num << " nodes" << std::endl;
        temp_topof.close();
      }
      
      std::cout << "[ROUTING] M4 routing framework setup completed" << std::endl;
    }
  }
  
  // Read topology parameters from file (same as FlowSim)
  std::ifstream topof(user_param.network_topo);
  uint32_t node_num, switch_num, link_num, nvswitch_num;
  int gpu_num = 0;
  if (topof.is_open()) {
    std::string gpu_type_str;
    topof >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str;
    
    // Calculate GPU count (same as FlowSim: total nodes - switches - NVSwitches)
    gpu_num = node_num - nvswitch_num - switch_num;
    
    // Set GPU type (same logic as FlowSim)
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
    
    std::cout << "[M4] Topology: " << gpu_num << " GPUs, " << nvswitch_num << " NVSwitches, " 
              << switch_num << " switches (" << gpu_type_str << ")" << std::endl;
  }

  // Extract topology information for NVSwitch mapping (same as FlowSim)
  std::map<int, int> node2nvswitch; // gpu_id -> nvswitch_id
  try {
    // For M4 stub, use simple mapping like FlowSim
    for (int i = 0; i < (int)nvswitch_num; i++) {
      NVswitchs.push_back(gpu_num + i); // NVSwitch IDs start after GPU IDs
    }
    
    // Basic deterministic mapping: group GPUs by gpus_per_server and assign to NVSwitch
    for (int gpu = 0; gpu < gpu_num; ++gpu) {
      int idx = gpu / gpus_per_server;
      // Clamp idx in case of malformed topologies
      if (idx >= static_cast<int>(NVswitchs.size())) idx = NVswitchs.size() - 1;
      if (idx < static_cast<int>(NVswitchs.size())) {
        node2nvswitch[gpu] = NVswitchs[idx];
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[M4] WARNING: unable to build NVSwitch mapping: " << e.what()
              << std::endl;
  }

  // Use FlowSim's exact node calculation: include NVSwitches in AstraSim node count
  int nodes_num = node_num - switch_num;  // FlowSim formula: includes NVSwitches

  // Create M4Network and Sys instances (same as FlowSim)
  std::vector<M4Network *> networks;
  std::vector<AstraSim::Sys *> systems;
  for (int i = 0; i < nodes_num; i++) {
    M4Network *network = new M4Network(i);
    networks.push_back(network);
    AstraSim::Sys *system = new AstraSim::Sys(
      network,
      nullptr,
      i,
      0,
      1,
      {nodes_num},      // Use nodes_num for physical dimensions (same as FlowSim)
      {1},
      "",
      user_param.workload,  // Use direct workload path like FlowSim
      1,                    // Use integer comm_scale like FlowSim
      1,
      1,
      1,
      0,
      user_param.result_dir,
      "m4",             // Use m4 test name
      true,
      false,
      gpu_type,
      {gpu_num}, // all_gpus
      NVswitchs,        // NVswitchs vector
      gpus_per_server
    );
    system->nvswitch_id = (nvswitch_num > 0 && node2nvswitch.count(i)) ? node2nvswitch[i] : -1;
    system->num_gpus = nodes_num - nvswitch_num;  // Match FlowSim's calculation
    systems.push_back(system);
  }
  
  // Fire workloads (same as FlowSim)
  for (int i = 0; i < nodes_num; i++) {
    systems[i]->workload->fire();
  }
  
  // M4 stub - no actual simulation, just immediate completion
  std::cout << "[M4] Stub execution completed" << std::endl;
  
  // Print data summary once (same as FlowSim)
  if (!networks.empty()) {
    networks[0]->sim_finish();
  }
  
  // Print routing statistics (same as FlowSim)
  std::cout << "\n[SIMULATION COMPLETE] Printing routing statistics..." << std::endl;
  std::cout << "[ROUTING] Routing framework cleaned up." << std::endl;

  std::cout << "[M4] SimAI-M4 finished" << std::endl;
  return 0;
};