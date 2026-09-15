#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>

namespace xgc2_math {

struct PlanarVelocity {
    double linear_m_s{0.0};
    double yaw_rad_s{0.0};
};

struct DelayedPlanarVelocityParameters {
    double delay_s{0.005};
    double linear_time_constant_s{0.005};
    double yaw_time_constant_s{0.005};
};

// Two independent unity-DC-gain first-order channels after ONE shared pure delay.
// Inputs are zero-order held. Split integration at each delayed command event;
// a 5 ms delay is not rounded to 8 ms by a 4 ms caller update period.
// No ROS clock, wall clock, pose integration, command gain, or wheel controller.
// The caller serializes command(), advance(), and reset(), and applies limits.
class DelayedPlanarVelocity {
  public:
    explicit DelayedPlanarVelocity(DelayedPlanarVelocityParameters parameters = {}) : parameters_(parameters) {
        requireNonnegative(parameters_.delay_s);
        requireNonnegative(parameters_.linear_time_constant_s);
        requireNonnegative(parameters_.yaw_time_constant_s);
    }

    void reset(double now_s) {
        requireFinite(now_s);
        time_s_ = now_s;
        state_ = {};
        target_ = {};
        pending_.clear();
    }

    // received_at_s is in the SAME monotonic simulation-time domain as advance.
    // A stale timestamp is rejected, not used to rewrite already evolved state.
    void command(double received_at_s, PlanarVelocity input) {
        requireFinite(received_at_s);
        requireFinite(input.linear_m_s);
        requireFinite(input.yaw_rad_s);
        const double release = received_at_s + parameters_.delay_s;
        requireFinite(release);
        if (received_at_s < time_s_ || (!pending_.empty() && release < pending_.back().release_s)) {
            throw std::invalid_argument("planar command timestamp is stale or out of order");
        }
        // While physics is paused, commands have the same simulation timestamp.
        // Only the last value at that instant has nonzero duration in a ZOH input.
        if (!pending_.empty() && release == pending_.back().release_s) {
            pending_.back().input = input;
            return;
        }
        if (pending_.size() == kMaxPendingCommands) {
            throw std::length_error("planar command delay queue capacity exceeded");
        }
        pending_.push_back({release, input});
    }

    PlanarVelocity advance(double now_s) {
        requireFinite(now_s);
        if (now_s < time_s_) {
            // World reset / rewind: do not replay commands from the old epoch.
            reset(now_s);
            return state_;
        }
        while (!pending_.empty() && pending_.front().release_s <= now_s) {
            integrate(pending_.front().release_s);
            target_ = pending_.front().input;
            pending_.pop_front();
            // Explicit zero-tau semantics; positive tau has no direct feedthrough.
            if (parameters_.linear_time_constant_s == 0.0)
                state_.linear_m_s = target_.linear_m_s;
            if (parameters_.yaw_time_constant_s == 0.0)
                state_.yaw_rad_s = target_.yaw_rad_s;
        }
        integrate(now_s);
        return state_;
    }

    PlanarVelocity velocity() const { return state_; }
    double time() const { return time_s_; }
    std::size_t pendingCommands() const { return pending_.size(); }

  private:
    struct Event {
        double release_s;
        PlanarVelocity input;
    };
    static constexpr std::size_t kMaxPendingCommands = 4096;
    static void requireFinite(double value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("planar model input must be finite");
    }
    static void requireNonnegative(double value) {
        requireFinite(value);
        if (value < 0.0)
            throw std::invalid_argument("planar delay and time constants must be nonnegative");
    }
    static double lag(double state, double target, double dt_s, double tau_s) {
        if (tau_s == 0.0)
            return target;
        // Exact ZOH discretization, stable for any nonnegative dt/tau.
        const double alpha = -std::expm1(-dt_s / tau_s);
        // Convex form avoids overflow in (target-state) for finite extremes.
        return (1.0 - alpha) * state + alpha * target;
    }
    void integrate(double until_s) {
        const double dt = until_s - time_s_;
        state_.linear_m_s = lag(state_.linear_m_s, target_.linear_m_s, dt, parameters_.linear_time_constant_s);
        state_.yaw_rad_s = lag(state_.yaw_rad_s, target_.yaw_rad_s, dt, parameters_.yaw_time_constant_s);
        time_s_ = until_s;
    }
    DelayedPlanarVelocityParameters parameters_;
    double time_s_{0.0};
    PlanarVelocity state_{};
    PlanarVelocity target_{};
    std::deque<Event> pending_;
};

} // namespace xgc2_math
