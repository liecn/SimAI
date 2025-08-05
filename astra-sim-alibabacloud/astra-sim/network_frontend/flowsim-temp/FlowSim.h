/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef _FLOWSIM_
#define _FLOWSIM_

#include <memory>
#include <vector>
#include "EventQueue.h"
#include "Topology.h"
#include "Type.h"
#include "astra-sim/system/routing/include/RoutingFramework.h"

class FlowSim {
 public:
  static double Now();
  static void Init(std::shared_ptr<EventQueue> event_queue, std::shared_ptr<Topology> topo);
  static void SetRoutingFramework(std::unique_ptr<AstraSim::RoutingFramework> routing_framework);
  static void Run();
  static void Schedule(
      uint64_t delay,
      void (*fun_ptr)(void* fun_arg),
      void* fun_arg);

  static void Send(int src, int dst, uint64_t size, int tag, Callback callback, CallbackArg callbackArg);
  static bool IsRoutingFrameworkLoaded();
  
  // Cleanup methods
  static void Stop() {}  // Empty for now
  static void Destroy() {}  // Empty for now

 private:
  static std::shared_ptr<EventQueue> event_queue;
  static std::shared_ptr<Topology> topology;
  static std::unique_ptr<AstraSim::RoutingFramework> routing_framework_;
};

#endif // _FLOWSIM_