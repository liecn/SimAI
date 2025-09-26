#include "EventQueue.h"
#include <cassert>
#include <iostream>
#include <limits>

EventQueue::EventQueue() noexcept : current_time(0), next_event_id(0) {
  // Create empty event queue (FlowSim-style)
  event_queue = std::list<EventList>();
}

EventTime EventQueue::get_current_time() const noexcept {
  return current_time;
}

bool EventQueue::finished() const noexcept {
  // Check whether event queue is empty (FlowSim-style only)
  return event_queue.empty();
}

void EventQueue::proceed() noexcept {
  // To proceed, the next event should exist (FlowSim-style only)
  if (event_queue.empty()) {
    return;
  }

  // SAFETY: Move the current event list out before invoking callbacks.
  // This prevents any modifications caused by callbacks from invalidating
  // references/iterators to the front list.
  EventList current_event_list = std::move(event_queue.front());
  event_queue.pop_front();

  // Check the validity and update current time
  current_time = std::max(current_time, current_event_list.get_event_time());

  // Process events sequentially like FlowSim
  while (!current_event_list.empty()) {
    current_event_list.invoke_event();
  }
}

void EventQueue::log_events() {
    std::cout << "Event logs\n";
}

// M4-style single event methods removed - using FlowSim-style queue only

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

