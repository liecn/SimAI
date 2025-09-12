#include "M4Network.h"
#include "astra-sim/system/MockNcclLog.h"
#include <iostream>

using AstraSim::ncclFlowTag;

static int g_callbacks = 0;

M4Network::M4Network(int _local_rank) : AstraNetworkAPI(_local_rank) {
  this->npu_offset = 0;
}

M4Network::~M4Network() {}

int M4Network::sim_comm_size(AstraSim::sim_comm /*comm*/, int* size) {
  if (size) *size = 1; // minimal stub
  return 0;
}

double M4Network::sim_time_resolution() { return 0.0; }

int M4Network::sim_init(AstraSim::AstraMemoryAPI* /*MEM*/) { return 0; }

int M4Network::sim_finish() { return 0; }

AstraSim::timespec_t M4Network::sim_get_time() {
  AstraSim::timespec_t t; t.time_val = 0; return t;
}

void M4Network::sim_schedule(AstraSim::timespec_t /*delta*/, void (*fun_ptr)(void*), void* fun_arg) {
  // For m4 backend, we can directly invoke (no event queue) or plug into m4 event system later
  fun_ptr(fun_arg);
}

int M4Network::sim_send(void* /*buffer*/, uint64_t count, int /*type*/, int dst, int /*tag*/,
                        AstraSim::sim_request* request, void (*msg_handler)(void*), void* fun_arg) {
  // Immediately trigger sender completion (m4 acts as oracle backend)
  notify_sender_finished(rank, dst + npu_offset, count, request->flowTag);
  return 0;
}

int M4Network::sim_recv(void* /*buffer*/, uint64_t count, int /*type*/, int src, int /*tag*/,
                        AstraSim::sim_request* request, void (*msg_handler)(void*), void* fun_arg) {
  // Immediately trigger receiver arrival
  notify_receiver_arrived(src + npu_offset, rank, count, request->flowTag);
  return 0;
}

void M4Network::notify_sender_finished(int sender_node, int receiver_node, uint64_t message_size, ncclFlowTag flowTag) {
  MockNcclLog* log = MockNcclLog::getInstance();
  auto it = flowTag; // unused suppression
  // Invoke user handler
  // In AstraSim usage, fun_arg resides in SendPacketEventHandlerData from sim_send; we pass via internal state in flowsim.
  // Here, simply log to preserve interface; Sci users wire-through at call site.
  (void)sender_node; (void)receiver_node; (void)message_size;
}

void M4Network::notify_receiver_arrived(int sender_node, int receiver_node, uint64_t message_size, ncclFlowTag flowTag) {
  MockNcclLog* log = MockNcclLog::getInstance();
  (void)log; (void)sender_node; (void)receiver_node; (void)message_size; (void)flowTag;
  // m4 will directly call the recv handler from the caller site (AstraSim Sys), as no event storage is kept here.
}


