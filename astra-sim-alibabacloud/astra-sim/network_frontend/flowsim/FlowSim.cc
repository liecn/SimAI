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

void FlowSim::SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework) {
    routing_framework_ = std::move(routing_framework);
}

void FlowSim::Run() {
    std::cout << "[FLOWSIM] Running FlowSim" << std::endl;
    while (!event_queue->finished()) {
        std::cout << "[FLOWSIM] FlowSim event" << std::endl;
        event_queue->proceed();
        tick = event_queue->get_current_time();
    }
    std::cout << "[FLOWSIM] FlowSim finished" << std::endl;
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
    return;
}

double FlowSim::Now(){
    return event_queue->get_current_time();
}

void FlowSim::Send(int src, int dst, uint64_t size, Callback callback, CallbackArg callbackArg) {
    std::cout << "[FLOWSIM] Sending flow " << src << "->" << dst << " with size " << size << std::endl;
    Route route;
    // Try to get pre-calculated path from routing framework
    if (routing_framework_ && routing_framework_->IsTopologyLoaded()) {
        // Create FlowKey with proper IP mapping (same as NS3)
        AstraSim::FlowKey flow_key;
        flow_key.src_ip = 0x0A000000 + src;  // 10.0.0.x format
        flow_key.dst_ip = 0x0A000000 + dst;
        flow_key.protocol = 17;  // UDP default
        flow_key.src_port = 10006;  // Default source port
        flow_key.dst_port = 100;    // Default destination port
        
        // Get the complete path from routing framework
        std::vector<int> node_path = routing_framework_->GetFlowSimPath(flow_key);
        
        if (!node_path.empty()) {
            std::cout << "[FLOWSIM] Using pre-calculated path for flow " << src << "->" << dst 
                      << " with " << node_path.size() << " hops" << std::endl;
            
            // Convert node IDs to Device pointers for FlowSim
            for (int node_id : node_path) {
                if (node_id >= 0 && node_id < topology->get_devices_count()) {
                    route.push_back(topology->get_device(node_id));
                }
            }
        } else {
            std::cout << "[FLOWSIM] No pre-calculated path found for flow " << src << "->" << dst 
                      << ", using default routing" << std::endl;
            // Fall back to default routing
            route = topology->get_route(src, dst);
        }
    } else {
        std::cout << "[FLOWSIM] Routing framework not available, using default routing for flow " 
                  << src << "->" << dst << std::endl;
        // Routing framework not available, use default routing
        route = topology->get_route(src, dst);
    }
    
    auto chunk = std::make_unique<Chunk>(size, route, callback, callbackArg);
    topology->send(std::move(chunk));
}

bool FlowSim::IsRoutingFrameworkLoaded() {
    return routing_framework_ != nullptr && routing_framework_->IsTopologyLoaded();
}