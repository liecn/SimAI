#include "EventQueue.h"
#include <cassert>
#include <iostream> 

EventQueue::EventQueue() noexcept : current_time(0), next_event_id(0) {
  // Create empty event queue
  event_queue = std::list<EventList>();
}

EventTime EventQueue::get_current_time() const noexcept {
  return current_time;
}

bool EventQueue::finished() const noexcept {
  // Check whether event queue is empty
  return event_queue.empty();
}

void EventQueue::proceed() noexcept {
  // To proceed, the next event should exist
  assert(!finished());

  // Proceed to the next event time
  auto& current_event_list = event_queue.front();

  // Check the validity and update current time
  current_time = std::max(current_time, current_event_list.get_event_time());

  // CRITICAL FIX: Process events sequentially like NS-3 instead of all at once
  // This creates natural dependencies between flows instead of parallel processing
  while (!current_event_list.empty()) {
    current_event_list.invoke_event();  // Process ONE event at a time
  }

  // Drop processed event list
  event_queue.pop_front();
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
    event_list_it = std::prev(event_queue.end());
  } else if (event_time >= event_queue.back().get_event_time()) {
    // Append to the tail (either reuse last list if same time, or create new).
    if (event_time == event_queue.back().get_event_time()) {
      event_list_it = std::prev(event_queue.end());
    } else {
      event_queue.emplace_back(event_time);
      event_list_it = std::prev(event_queue.end());
    }
  } else {
    // Fallback: need to insert somewhere in the middle — keep original O(n) scan.
    event_list_it = event_queue.begin();
    while (event_list_it != event_queue.end() &&
           event_list_it->get_event_time() < event_time) {
      ++event_list_it;
    }
    if (event_list_it == event_queue.end() ||
        event_time < event_list_it->get_event_time()) {
      event_list_it = event_queue.insert(event_list_it, EventList(event_time));
    }
  }

  // Generate a new event ID
  EventId event_id = next_event_id++;

  // Now, whether (1) or (2), the entry to insert the event is found
  // Add event to event_list
  event_list_it->add_event(callback, callback_arg, event_id);

  // Store event in map for cancellation
  event_map[event_id] = event_list_it;
  return event_id;
}

void EventQueue::cancel_event(EventId event_id) noexcept {
  // std::cerr << "Cancelling event with ID " << event_id << std::endl;
  auto it = event_map.find(event_id);
  if (it != event_map.end()) {
    auto& event_list_it = it->second;
    event_list_it->remove_event(event_id);
    event_map.erase(it);
  }
}

void EventQueue::clear_all_events() noexcept {
  // Clear all events without processing them
  event_queue.clear();
  event_map.clear();
}
