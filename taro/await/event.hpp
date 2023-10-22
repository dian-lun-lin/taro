#pragma once

#include "../core/taro.hpp"

namespace taro { // begin of namespace taro ===================================

// ==========================================================================
//
// Declaration/Definition of class Event
//
// ==========================================================================

class Event {

  struct EventData {
    Taro* taro;
    Worker* worker;
    size_t task_id;
  };

  friend class EventAwait;

  private:

    enum class STAT {
      UNSET,
      WAIT,
      SET
    };

    std::atomic<STAT> _state{STAT::UNSET};
    EventData _data;
};

// ==========================================================================
//
// Declaration of class EventAwait
//
// ==========================================================================

class EventAwait {

  public:

    EventAwait(Taro& taro, size_t num_events);

    auto until(size_t eid);

    void set(size_t eid);
    
    bool is_set(size_t eid) const noexcept;

    // TODO: until_all, until_any
    //template <typename... E>
    //auto until_all(E... events);

    //template <typename ...E>
    //auto until_any(E... events);

  private:

    Taro& _taro;
    std::vector<Event> _events;
};

// ==========================================================================
//
// Definition of class EventAwait
//
// ==========================================================================

inline
EventAwait::EventAwait(Taro& taro, size_t num_events): _taro{taro}, _events{num_events} {
}

auto EventAwait::until(size_t eid) {
  struct event_awaiter {
    EventAwait& events;
    size_t eid;
    event_awaiter(EventAwait& events, size_t eid): events{events}, eid{eid} {}

    bool await_ready() const noexcept {
      return events.is_set(eid);
    }

    bool await_suspend(std::coroutine_handle<>) {
      events._events[eid]._data.taro = &events._taro;
      events._events[eid]._data.worker = events._taro._this_worker();
      events._events[eid]._data.task_id = events._taro._this_worker()->_work_on_task_id;

      auto old_state = Event::STAT::UNSET;
      return events._events[eid]._state.compare_exchange_strong(old_state, Event::STAT::WAIT);
    }

    void await_resume() noexcept {}
    
  };

  return event_awaiter{*this, eid};
}

void EventAwait::set(size_t eid) {
  auto old_state = _events[eid]._state.exchange(Event::STAT::SET);
  if(old_state == Event::STAT::WAIT) {
    auto* taro = _events[eid]._data.taro;
    auto* worker = _events[eid]._data.worker;
    auto task_id = _events[eid]._data.task_id;
    taro->_enqueue_back(*worker, task_id);
  }
}

bool EventAwait::is_set(size_t eid) const noexcept {
  return _events[eid]._state.load() == Event::STAT::SET;
}


// ==========================================================================
//
// Definition of event_await in Taro
//
// ==========================================================================

inline
EventAwait Taro::event_await(size_t num_events) {
  return EventAwait(*this, num_events);
}

} // end of namespace taro ==============================================
