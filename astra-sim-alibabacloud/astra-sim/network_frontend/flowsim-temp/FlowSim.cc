/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <unistd.h>
#include <iostream>
#include "FlowSim.h"
using namespace std;

// Static members
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
    int iteration = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        
        iteration++;
        if (iteration > 100000) {
            std::cerr << "[FLOWSIM] ERROR: Too many iterations!" << std::endl;
            break;
        }
    }
}

void FlowSim::Schedule(
    uint64_t delay,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    uint64_t time = event_queue->get_current_time() + delay;
    event_queue->schedule_event(time, fun_ptr, fun_arg);
}

double FlowSim::Now(){
    return event_queue->get_current_time();
}

void FlowSim::Send(int src, int dst, uint64_t size, int tag, Callback callback, CallbackArg callbackArg) {
    static int flowsim_send_count = 0;
    flowsim_send_count++;
    
    // Log send events (like NS3 does)
    if (flowsim_send_count % 10 == 0) {
        std::cout << "[FLOWSIM] send #" << flowsim_send_count 
                  << " at time=" << event_queue->get_current_time() 
                  << "ns: " << src << " -> " << dst 
                  << ", size=" << size << " bytes" << std::endl;
    }
    
    // Get pre-calculated path from routing framework
    std::vector<int> node_path = routing_framework_->GetFlowSimPathByNodeIds(src, dst);
    if (node_path.empty()) return;
    
    // Convert to device route
    Route route;
    for (int node_id : node_path) {
        if (node_id >= 0 && node_id < topology->get_devices_count()) {
            route.push_back(topology->get_device(node_id));
        }
    }
    
    if (route.size() >= 2) {
        // Create chunk exactly like the old working version
        auto chunk = std::make_unique<Chunk>(size, route, callback, callbackArg);
        topology->send(std::move(chunk));
    }
}

bool FlowSim::IsRoutingFrameworkLoaded() {
    return routing_framework_ != nullptr && routing_framework_->IsTopologyLoaded();
}