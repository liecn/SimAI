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

#include <unistd.h>
#include "FlowSim.h"
#include "astra-sim/system/routing/include/RoutingFramework.h"
using namespace std;

queue<struct CallTask> FlowSim::call_list = {};
uint64_t FlowSim::tick = 0;

std::shared_ptr<EventQueue> FlowSim::event_queue = nullptr;
std::shared_ptr<Topology> FlowSim::topology = nullptr;
std::unique_ptr<AstraSim::RoutingFramework> FlowSim::routing_framework_ = nullptr;

void FlowSim::Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo) {
    FlowSim::event_queue = event_queue;
    FlowSim::topology = topo;
    FlowSim::topology->set_event_queue(event_queue);
}

void FlowSim::InitWithRouting(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo,
                             const std::string& topology_file, const std::string& network_config_file) {
    // Initialize basic FlowSim
    Init(event_queue, topo);
    
    // Initialize routing framework
    routing_framework_ = std::make_unique<AstraSim::RoutingFramework>();
    if (routing_framework_->PrecalculateFlowPathsForFlowSim(topology_file, network_config_file)) {
        std::cout << "[FLOWSIM] Routing framework initialized successfully" << std::endl;
    } else {
        std::cerr << "[FLOWSIM] Failed to initialize routing framework, using default routing" << std::endl;
        routing_framework_.reset();
    }
}

void FlowSim::SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework) {
    routing_framework_ = std::move(routing_framework);
}

void FlowSim::Run() {
    while (!event_queue->finished()) {
        event_queue->proceed();
        tick = event_queue->get_current_time();
    }
}

void FlowSim::Schedule(
    uint64_t delay,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    uint64_t time = event_queue->get_current_time() + delay;
    event_queue->schedule_event(time, fun_ptr, fun_arg);
}

void FlowSim::Stop(){
    return;
}

void FlowSim::Destroy(){
    // Cleanup routing framework
    routing_framework_.reset();
    return;
}

double FlowSim::Now(){
    return event_queue->get_current_time();
}

void FlowSim::Send(int src, int dst, uint64_t size, Callback callback, CallbackArg callbackArg) {
    Route route;
    
    // Try to get pre-calculated path from routing framework
    if (routing_framework_ && routing_framework_->IsTopologyLoaded()) {
        // Create FlowKey with proper IP mapping
        AstraSim::FlowKey flow_key;
        flow_key.src_ip = 0x0A000000 + src;  // 10.0.0.x format (same as NS3)
        flow_key.dst_ip = 0x0A000000 + dst;
        flow_key.protocol = 17;  // UDP default
        flow_key.src_port = 0;
        flow_key.dst_port = 0;
        
        // Get the complete path from routing framework
        std::vector<int> node_path = routing_framework_->GetFlowSimPath(flow_key);
        
        if (!node_path.empty()) {
            // Convert node IDs to Device pointers for FlowSim
            for (int node_id : node_path) {
                if (node_id >= 0 && node_id < topology->get_devices_count()) {
                    route.push_back(topology->get_device(node_id));
                }
            }
        } else {
            // Fall back to default routing
            route = topology->get_route(src, dst);
        }
    } else {
        // Routing framework not available, use default routing
        route = topology->get_route(src, dst);
    }
    
    auto chunk = std::make_unique<Chunk>(size, route, callback, callbackArg);
    topology->send(std::move(chunk));
}