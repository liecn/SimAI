#include "EventQueue.h"
#include <cassert>
#include <iostream>
#include <limits>

EventQueue::EventQueue() noexcept : current_time(0), next_event_id(0) {
  // Create empty event queue (FlowSim-style)
  event_queue = std::list<EventList>();
  // Keep M4-style single events for compatibility
  next_arrival = nullptr;
  next_completion = nullptr;
}

EventTime EventQueue::get_current_time() const noexcept {
  return current_time;
}

bool EventQueue::finished() const noexcept {
  // Check whether event queue is empty (FlowSim-style + M4-style)
  bool queue_empty = event_queue.empty();
  bool m4_events_empty = (next_arrival == nullptr && next_completion == nullptr);
  return queue_empty && m4_events_empty;
}

void EventQueue::proceed() noexcept {
  // To proceed, the next event should exist
  assert(!finished());

  // Process FlowSim-style queue events first (highest priority)
  if (!event_queue.empty()) {
    auto& current_event_list = event_queue.front();
    current_time = std::max(current_time, current_event_list.get_event_time());
    
    // Process events sequentially like FlowSim
    while (!current_event_list.empty()) {
      current_event_list.invoke_event();
    }
    
    // Drop processed event list
    event_queue.pop_front();
    return;
  }

  // Fallback to M4-style single events if queue is empty
  EventTime arrival_time = std::numeric_limits<uint64_t>::max();
  EventTime completion_time = std::numeric_limits<uint64_t>::max();

  if (next_arrival != nullptr) {
      arrival_time = next_arrival->get_time();
  }

  if (next_completion != nullptr) {
      completion_time = next_completion->get_time();
  }

  if (arrival_time < completion_time) {
      assert(next_arrival != nullptr);
      Event arrival = *next_arrival;
      delete next_arrival;
      next_arrival = nullptr;
      current_time = arrival_time;
      arrival.invoke_event();
  } else {
      assert(next_completion != nullptr);
      Event completion = *next_completion;
      delete next_completion;
      next_completion = nullptr;
      current_time = completion_time;
      completion.invoke_event();
  }
}

void EventQueue::log_events() {
    std::cout << "Event logs\n";
}

void EventQueue::schedule_arrival(
    const EventTime arrival_time,
    const Callback callback,
    const CallbackArg callback_arg) noexcept {

    assert(arrival_time >= current_time);

    delete next_arrival;

    next_arrival = new Event(arrival_time, callback, callback_arg);
}

void EventQueue::schedule_completion(
    const EventTime completion_time,
    const Callback callback,
    const CallbackArg callback_arg) noexcept {

    assert(completion_time >= current_time);

    delete next_completion;
    next_completion = new Event(completion_time, callback, callback_arg);

}

void EventQueue::cancel_completion() {
    //delete next_completion;
    //next_completion = nullptr;
}

EventId EventQueue::schedule_event(
    const EventTime event_time,
    const Callback callback,
    const CallbackArg callback_arg) noexcept {
  // Time should be at least larger than current time
  //assert(event_time >= current_time);

  // Fast-path: most events are scheduled in non-decreasing time order.
  // If the new event occurs **not earlier** than the last event_list, we can
  // directly push_back (O(1)) instead of an O(n) scan.

  std::list<EventList>::iterator event_list_it;

  if (event_queue.empty()) {
    // First event — just emplace it.
    event_queue.emplace_back(event_time);
    event_list_it = event_queue.begin();
  } else {
    EventTime last_event_time = event_queue.back().get_event_time();
    if (event_time >= last_event_time) {
      // Fast path: append to existing list or create new one
      if (event_time == last_event_time) {
        event_list_it = std::prev(event_queue.end());
      } else {
        event_queue.emplace_back(event_time);
        event_list_it = std::prev(event_queue.end());
      }
    } else {
      // Slow path: find the correct position
      event_list_it = event_queue.begin();
      while (event_list_it != event_queue.end()) {
        if (event_list_it->get_event_time() >= event_time) {
          break;
        }
        ++event_list_it;
      }

      if (event_list_it == event_queue.end() || event_list_it->get_event_time() != event_time) {
        event_list_it = event_queue.emplace(event_list_it, event_time);
      }
    }
  }

  // Now add the event to the appropriate EventList
  EventId event_id = next_event_id++;
  event_list_it->add_event(callback, callback_arg, event_id);

  return event_id;
}

