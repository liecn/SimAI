#include "Event.h"
#include <cassert>

Event::Event(EventTime event_time, const Callback callback, const CallbackArg callback_arg) noexcept
    : event_time(event_time),
      callback(callback),
      callback_arg(callback_arg) {
    assert(callback != nullptr);
}

void Event::invoke_event() noexcept {
    // check the validity of the event - return safely if null
    if (callback == nullptr) {
        return;
    }

    // invoke the callback function
    (*callback)(callback_arg);
}

EventTime Event::get_time() noexcept {
    return event_time;
}

std::pair<Callback, CallbackArg> Event::get_handler_arg() const noexcept {
    // check the validity of the event - return null pair if invalid
    if (callback == nullptr) {
        return {nullptr, nullptr};
    }

    return {callback, callback_arg};
}
