#include <unistd.h>
#include <string>
#include <iostream>
#include <vector>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/AstraComputeAPI.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/routing/include/RoutingFramework.h"

#include "M4Network.h"

using namespace std;

static int parse_args(int argc,char * argv[], string& workload, string& network_topo, string& result_dir){
  int opt; result_dir = "results/m4/";
  while ((opt = getopt(argc,argv,"hw:n:o:"))!=-1){
    switch (opt) {
      case 'h':
        std::cout<<"-w workload path\n-n network topo\n-o results dir"<<std::endl; return 1;
      case 'w': workload = optarg; break;
      case 'n': network_topo = optarg; break;
      case 'o': result_dir = optarg; if (result_dir.back()!='/') result_dir+='/'; break;
      default: return 1;
    }
  }
  return 0;
}

int main(int argc,char *argv[]) {
  string workload, network_topo, result_dir;
  if(parse_args(argc,argv,workload,network_topo,result_dir)) return -1;

  string mkdir_cmd = "mkdir -p "+result_dir; system(mkdir_cmd.c_str());
  cout<<"[M4] Workload: "<<workload<<" topo: "<<network_topo<<" results: "<<result_dir<<endl;

  // Minimal stub: skip topology and routing setup for now

  // Basic node sizing similar to flowsim
  std::ifstream topof(network_topo);
  uint32_t node_num=0, switch_num=0, link_num=0, nvswitch_num=0; uint32_t gpus_per_server=8; string gpu_type_str;
  if (topof.is_open()) { topof >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str; topof.close(); }
  int nodes_num = node_num - switch_num;

  vector<M4Network*> networks;
  vector<AstraSim::Sys*> systems;
  for (int i=0;i<nodes_num;i++) {
    auto* net = new M4Network(i); networks.push_back(net);
    auto* sys = new AstraSim::Sys(
      net, nullptr, i, 0, 1, {nodes_num}, {1}, "", workload,
      1,1,1,1,0, result_dir, "m4", true, false, GPUType::NONE, {nodes_num-nvswitch_num}, {}, gpus_per_server);
    systems.push_back(sys);
  }

  // Fire workloads (minimal stub - immediate completion via M4Network)
  for (int i=0;i<nodes_num;i++) systems[i]->workload->fire();
  
  cout<<"[M4] Stub execution completed"<<endl;
  return 0;
}


