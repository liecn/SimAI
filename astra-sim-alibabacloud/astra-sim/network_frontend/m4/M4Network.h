#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include <map>
#include <tuple>

struct M4Task {
  int src;
  int dest;
  uint64_t count;
  int type; // 0 sender, 1 receiver
  void (*msg_handler)(void* fun_arg) = nullptr;
  void* fun_arg = nullptr;
};

class M4Network : public AstraSim::AstraNetworkAPI {
 public:
  explicit M4Network(int local_rank);
  ~M4Network() override;

  int sim_comm_size(AstraSim::sim_comm comm, int* size) override;
  double sim_time_resolution() override;
  int sim_init(AstraSim::AstraMemoryAPI* MEM) override;

  int sim_send(void* buffer, uint64_t count, int type, int dst, int tag,
               AstraSim::sim_request* request, void (*msg_handler)(void* fun_arg), void* fun_arg) override;

  int sim_recv(void* buffer, uint64_t count, int type, int src, int tag,
               AstraSim::sim_request* request, void (*msg_handler)(void* fun_arg), void* fun_arg) override;

  int sim_finish() override;
  AstraSim::timespec_t sim_get_time() override;
  void sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) override;

  void notify_sender_finished(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag);
  void notify_receiver_arrived(int sender_node, int receiver_node, uint64_t message_size, AstraSim::ncclFlowTag flowTag);

 private:
  int npu_offset{0};
};


