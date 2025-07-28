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
    return;
}

double FlowSim::Now(){
    return event_queue->get_current_time();
}

void FlowSim::Send(int src, int dst, uint64_t size, Callback callback, CallbackArg callbackArg) {
    Route route;
    
    // Try to get pre-calculated path from routing framework
    if (routing_framework_ && routing_framework_->IsTopologyLoaded()) {
        // Use the simplified helper function that handles FlowKey creation
        std::vector<int> node_path = routing_framework_->GetFlowSimPathByNodeIds(src, dst);
        
        if (!node_path.empty()) {
            // Convert node IDs to Device pointers for FlowSim
            for (int i = 0; i < node_path.size(); i++) {
                int node_id = node_path[i];
                if (node_id >= 0 && node_id < topology->get_devices_count()) {
                    auto device = topology->get_device(node_id);
                    route.push_back(device);
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

bool FlowSim::IsRoutingFrameworkLoaded() {
    return routing_framework_ != nullptr && routing_framework_->IsTopologyLoaded();
}