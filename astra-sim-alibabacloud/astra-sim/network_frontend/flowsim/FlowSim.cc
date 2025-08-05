/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <unistd.h>
#include "FlowSim.h"
#include <cstdlib>
#include <functional>
#include <iostream>

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
    
    while (true) {
        // Process FlowSim events if available
        if (!event_queue->finished()) {
            event_queue->proceed();
            iteration++;
        } else {
            // Queue is empty - simulation is complete
            break;
        }
        
        // Safety limit
        if (iteration > 100000000) {
            std::cout << "[FLOWSIM] Reached maximum iterations - stopping" << std::endl;
            break;
        }
    }
}

void FlowSim::Schedule(
    uint64_t delay,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
    // Use FlowSim's event queue
    uint64_t time = event_queue->get_current_time() + delay;
    event_queue->schedule_event(time, fun_ptr, fun_arg);
}

double FlowSim::Now(){
    // Use FlowSim's event queue time
    return event_queue->get_current_time();
}

void FlowSim::Send(int src, int dst, uint64_t size, int tag, Callback callback, CallbackArg callbackArg) {
    // Apply AS_SEND_LAT for fair comparison with NS3
    uint64_t send_latency_ns = 0;
    const char* send_lat_env = std::getenv("AS_SEND_LAT");
    if (send_lat_env) {
        // Convert from microseconds to nanoseconds (same as NS3)
        send_latency_ns = std::stoi(send_lat_env) * 1000;
    }
    
    // Check AS_NVLS_ENABLE for hardware acceleration simulation
    bool nvls_enabled = false;
    const char* nvls_env = std::getenv("AS_NVLS_ENABLE");
    if (nvls_env && std::stoi(nvls_env) == 1) {
        nvls_enabled = true;
        // NVLS reduces effective chunk size for better pipelining
        if (size < 4096 && size > 0) {
            size = 4096; // Minimum chunk size with NVLS
        }
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
        // Create chunk
        auto chunk = std::make_unique<Chunk>(size, route, callback, callbackArg);
        
        if (send_latency_ns > 0) {
            // Schedule the actual send after the send latency
            auto delayed_send = [chunk_ptr = chunk.release()]() {
                std::unique_ptr<Chunk> delayed_chunk(chunk_ptr);
                topology->send_with_batching(std::move(delayed_chunk));
            };
            
            // Store the lambda for the event system
            auto* lambda_ptr = new std::function<void()>(delayed_send);
            
            event_queue->schedule_event(
                event_queue->get_current_time() + send_latency_ns,
                [](void* arg) {
                    auto* func = static_cast<std::function<void()>*>(arg);
                    (*func)();
                    delete func;
                },
                lambda_ptr
            );
        } else {
            // Send immediately
            topology->send_with_batching(std::move(chunk));
        }
    }
}

bool FlowSim::IsRoutingFrameworkLoaded() {
    return routing_framework_ != nullptr && routing_framework_->IsTopologyLoaded();
}

void FlowSim::Stop() {
    // Clear all remaining events without processing them to prevent callbacks during cleanup
    // Processing events during cleanup can cause infinite loops or segfaults
    if (event_queue) {
        event_queue->clear_all_events();
    }
}

void FlowSim::Destroy() {
    // Clear static resources in proper order
    routing_framework_.reset();
    topology.reset();
    event_queue.reset();
}