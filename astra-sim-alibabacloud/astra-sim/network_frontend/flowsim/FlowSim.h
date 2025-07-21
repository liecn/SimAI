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

#ifndef __FLOWSIM_HH__
#define __FLOWSIM_HH__

#include<iostream>
#include<queue>
#include<list>
#include<cstdint>
#include <memory>
#include "Topology.h"
#include "EventQueue.h"
#include "Type.h"

// Forward declaration
namespace AstraSim {
    class RoutingFramework;
}

using namespace std;

/**
 * Callback task structure for FlowSim scheduling
 */
struct CallTask {
  uint64_t time;
  void (*fun_ptr)(void* fun_arg);
  void* fun_arg;
  
  CallTask(uint64_t _time, void (*_fun_ptr)(void* _fun_arg), void* _fun_arg)
      : time(_time), fun_ptr(_fun_ptr), fun_arg(_fun_arg) {};
  ~CallTask(){}
};

/**
 * FlowSim Core Simulation Engine
 * Manages the simulation timeline, event scheduling, and network operations
 */
class FlowSim {
 private:
  static queue<struct CallTask> call_list;
  static uint64_t tick;
  static std::shared_ptr<Topology> topology;  
  static std::unique_ptr<AstraSim::RoutingFramework> routing_framework_;

 public:
  static std::shared_ptr<EventQueue> event_queue;

  // Core simulation methods
  static double Now();
  static void Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo);
  static void InitWithRouting(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo,
                             const std::string& topology_file, const std::string& network_config_file);
  static void SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework);
  static void Run();
  static void Schedule(
      uint64_t delay,
      void (*fun_ptr)(void* fun_arg),
      void* fun_arg);
  static void Stop();
  static void Destroy();
  
  // Network communication methods
  static void Send(int src, int dst, uint64_t size, Callback callback, CallbackArg callbackArg);
};

#endif // __FLOWSIM_HH__
