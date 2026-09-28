#pragma once

#include <cmath>

#include <Eigen/Core>

#include "xgc2_math/geometry/se2.hpp"

namespace xgc2_math {

struct TranslationalState {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
};

// Exact ZOH plant for an ideal acceleration inner loop. Position, velocity and
// acceleration are world-frame SI values; acceleration already excludes gravity.
// The controller supplies acceleration: no reference tracking law is added here.
// Caller supplies finite values and a nonnegative elapsed time in seconds.
inline TranslationalState stepWorldAcceleration(const TranslationalState& state, const Eigen::Vector3d& acceleration,
                                                double dt) {
    return {state.position + dt * state.velocity + (0.5 * dt * dt) * acceleration, state.velocity + dt * acceleration};
}

// Exact planar rigid-body motion with constant body velocity and yaw rate.
// Scout uses (forward, 0); a mecanum base also has body-left velocity. An existing
// actuator/delay model runs before this plant, so these are actual velocities.
// No collision forces, hidden clamps, command gains or clock are introduced.
inline Pose2 stepBodyVelocity(const Pose2& pose, const Eigen::Vector2d& body_velocity, double yaw_rate, double dt) {
    const double half_turn = 0.5 * yaw_rate * dt;
    const double sinc =
        std::abs(half_turn) < 1e-6 ? 1.0 - half_turn * half_turn / 6.0 : std::sin(half_turn) / half_turn;
    Pose2 result;
    result.position = pose.position + (dt * sinc) * rotationMatrix2(pose.yaw + half_turn) * body_velocity;
    result.yaw = normalizeAngle(pose.yaw + 2.0 * half_turn);
    return result;
}

} // namespace xgc2_math
