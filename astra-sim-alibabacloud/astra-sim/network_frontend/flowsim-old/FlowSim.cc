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

void FlowSim::Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo) {
    FlowSim::event_queue = event_queue;
    FlowSim::topology = topo;
    FlowSim::topology->set_event_queue(event_queue);
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
  Route route = topology->get_route(src, dst);
  auto chunk = std::make_unique<Chunk>(size, route, callback, callbackArg);
  topology->send(std::move(chunk));
}