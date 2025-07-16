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

#include "FlowsimNetwork.h"
#include "FlowSim.h"
#include "TopologyBuilder.h"
#include "Topology.h"
#include "Type.h"

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
  std::cout << "Topology file passed to FlowSim: " << param->net_work_param.topology_file << std::endl;
  std::shared_ptr<Topology> topology = construct_fat_tree_topology(UserParam::getInstance()->net_work_param.topology_file);
  std::map<int, int> node2nvswitch; 
  int gpu_num = topology->get_npus_count();
  int nvswitch_num = 0;

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
    system->nvswitch_id = nvswitch_num > 0 ? node2nvswitch[i] : -1;
    system->num_gpus = topology->get_npus_count();
    systems.push_back(system);
  }

  for (uint32_t i = 0; i < systems.size(); i++) {
    systems[i]->workload->fire();
  }
  
  std::cout << "SimAI begin run FlowSim" << std::endl;
  FlowSim::Run(topology);
  FlowSim::Stop();
  FlowSim::Destroy();

  std::cout << "SimAI-FlowSim finished." << std::endl;
  return 0;
};