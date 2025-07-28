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
#include "astra-sim/system/AstraParamParse.hh"
#include "astra-sim/system/routing/include/RoutingFramework.h"

#include "FlowsimNetwork.h"
#include "FlowSim.h"
#include "TopologyBuilder.h"
#include "Topology.h"
#include "Type.h"
#include <tuple>
#include <stdexcept>

#define RESULT_PATH "./results/"
#define WORKLOAD_PATH ""

using namespace std;

extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
//extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern std::string gpu_type;
//extern std::vector<int>NVswitchs;
extern std::vector<std::vector<int>>all_gpus;
extern int ngpus_per_node;
extern map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
extern map<std::pair<int, std::pair<int, int>>, int> recvHash;
extern map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
extern map<std::pair<int, int>, int64_t> nodeHash;
extern int local_rank;

std::vector<string> workloads;
std::vector<std::vector<int>> physical_dims;

struct user_param {
  int thread;
  int gpus;
  string workload;
  int comm_scale;
  user_param() {
    thread = 1;
    gpus = 1;
    workload = "";
    comm_scale = 1;
  };
  ~user_param(){};
  user_param(int _thread, int _gpus, string _workload, int _comm_scale = 1)
      : thread(_thread),
        gpus(_gpus),
        workload(_workload),
        comm_scale(_comm_scale){};
};

int main(int argc,char *argv[]) {
  UserParam* param = UserParam::getInstance();
  if (param->parseArg(argc,argv)) {
    std::cerr << "-h,       --help                Help message" << std::endl;
    return -1;
  }
  param->mode = ModeType::FLOWSIM;
  std::cout << "[FLOWSIM] Topology file passed to FlowSim: " << param->net_work_param.topology_file << std::endl;
  std::cout << "[FLOWSIM] Workload file: " << param->workload << std::endl;
  std::cout << "[FLOWSIM] Result file: " << param->res << std::endl;
  std::shared_ptr<Topology> topology = construct_fat_tree_topology(UserParam::getInstance()->net_work_param.topology_file);

  // -----------------------------------------------------------------------------
  // Build NVSwitch list and GPU->NVSwitch mapping from the topology description
  // -----------------------------------------------------------------------------
  std::map<int, int> node2nvswitch; // gpu_id -> nvswitch_id
  int nvswitch_num = 0;
  try {
    int npus_cnt = 0;
    std::vector<int> nv_ids;
    std::vector<std::tuple<int, int, double, double, double>> dummy_links;
    std::tie(npus_cnt, nvswitch_num, nv_ids, dummy_links) =
        parse_fat_tree_topology_file(param->net_work_param.topology_file);

    // Save the NVSwitch IDs globally so that Sys / MockNccl can consume them
    param->net_work_param.NVswitchs = nv_ids;

    // Basic deterministic mapping: group GPUs by gpus_per_server and assign to NVSwitch
    int gpus_per_server = param->net_work_param.gpus_per_server > 0
                              ? param->net_work_param.gpus_per_server
                              : 1;
    for (int gpu = 0; gpu < npus_cnt; ++gpu) {
      int idx = gpu / gpus_per_server;
      // Clamp idx in case of malformed topologies
      if (idx >= static_cast<int>(nv_ids.size())) idx = nv_ids.size() - 1;
      node2nvswitch[gpu] = nv_ids[idx];
    }
  } catch (const std::exception &e) {
    std::cerr << "[FLOWSIM] WARNING: unable to build NVSwitch mapping: " << e.what()
              << std::endl;
  }

  int gpu_num = topology->get_npus_count();

  // Create FlowSimNetwork and Sys instances
  std::vector<FlowSimNetWork *> networks;
  std::vector<AstraSim::Sys *> systems;
  for (uint32_t i = 0; i < topology->get_npus_count(); i++) {
    FlowSimNetWork *network = new FlowSimNetWork(i);
    networks.push_back(network);
    AstraSim::Sys *system = new AstraSim::Sys(
      network,
      nullptr,
      i,
      0,
      1,
      {topology->get_devices_count()},
      {1},
      "",
      WORKLOAD_PATH + param->workload,
      param->comm_scale,
      1,
      1,
      1,
      0,
      RESULT_PATH + param->res,
      "FlowSim_test",
      true,
      false,
      param->net_work_param.gpu_type,
      param->gpus,
      param->net_work_param.NVswitchs,
      param->net_work_param.gpus_per_server
    );
    system->nvswitch_id = (nvswitch_num > 0 && node2nvswitch.count(i)) ? node2nvswitch[i] : -1;
    system->num_gpus = topology->get_npus_count();
    systems.push_back(system);
  }

  std::shared_ptr<EventQueue> event_queue = std::make_shared<EventQueue>();
  FlowSim::Init(event_queue, topology);
  
  // Initialize routing framework and pre-calculate flow paths (same as NS3)
  if (!param->net_work_param.topology_file.empty()) {
    // Create routing framework instance
    auto routing_framework = std::make_unique<AstraSim::RoutingFramework>();
    
    // First parse the topology file
    bool parse_result = routing_framework->ParseTopology(param->net_work_param.topology_file);
    
    if (parse_result) {
      routing_framework->PrecalculateRoutingTables();
      routing_framework->PrecalculateFlowPathsForFlowSim(param->net_work_param.topology_file, param->net_work_param.topology_file);
      
      // Set the routing framework in FlowSim
      FlowSim::SetRoutingFramework(std::move(routing_framework));
    }
  }
  for (uint32_t i = 0; i < systems.size(); i++) {
    systems[i]->workload->fire();
  }
  
  std::cout << "SimAI begin run FlowSim" << std::endl;
  FlowSim::Run();
  FlowSim::Stop();
  FlowSim::Destroy();

  std::cout << "SimAI-FlowSim finished." << std::endl;
  return 0;
};