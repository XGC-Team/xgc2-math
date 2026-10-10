#pragma once

// Placement and motion of scene obstacles: the pose algebra of scene assembly and the motion intents of a scene
// document (xgc2.scene.v1).
//
// Orientations are quaternions in the document convention (x, y, z, w), i.e. Eigen::Quaterniond::coeffs(). The
// functions evaluate the same expressions in the same order as the scene runtime they replace and, unlike
// xgc2_math::compose, neither normalise nor flip the sign of a result, so a placed document carries the same numbers.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

#include "xgc2_math/geometry/se3.hpp"

namespace xgc2_math::scene {

// Hamilton product a * b.
inline Eigen::Quaterniond multiply(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
    const double x = a.x(), y = a.y(), z = a.z(), w = a.w();
    const double X = b.x(), Y = b.y(), Z = b.z(), W = b.w();
    return Eigen::Quaterniond(w * W - x * X - y * Y - z * Z, w * X + x * W + y * Z - z * Y,
                              w * Y - x * Z + y * W + z * X, w * Z + x * Y - y * X + z * W);
}

// q * v * conj(q) for a unit quaternion q.
inline Eigen::Vector3d rotate(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {
    const Eigen::Quaterniond product = multiply(multiply(q, Eigen::Quaterniond(0.0, v.x(), v.y(), v.z())),
                                                Eigen::Quaterniond(q.w(), -q.x(), -q.y(), -q.z()));
    return Eigen::Vector3d(product.x(), product.y(), product.z());
}

// The pose `b` (given in the frame of `a`) in the frame `a` is given in.
inline Pose3 composePlacement(const Pose3& a, const Pose3& b) {
    Pose3 result;
    result.position = a.position + rotate(a.orientation, b.position);
    result.orientation = multiply(a.orientation, b.orientation);
    return result;
}

// Where a sealed element goes in an assembled world: translated by (x, y) on the ground plane, then turned by `yaw`
// about +z.
struct ElementPlacement {
    double x = 0.0, y = 0.0, yaw = 0.0;

    Pose3 pose() const {
        Pose3 result;
        result.position = Eigen::Vector3d(x, y, 0.0);
        result.orientation = Eigen::Quaterniond(std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2));
        return result;
    }
};

// The motion intent of an obstacle. hold: stays; constant_twist: moves with `linear` and turns with `angular` (world
// frame, rad/s); ping_pong: shuttles between two points at `speed`; circle: runs a circle of `radius` about `center` in
// the x-y plane at `angular_speed` from `phase`.
struct Motion {
    enum Type { kHold, kConstantTwist, kPingPong, kCircle };
    Type type = kHold;
    Eigen::Vector3d linear = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_b = Eigen::Vector3d::Zero();
    double speed = 0.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double radius = 0.0;
    double angular_speed = 0.0;
    double phase = 0.0;
};

// The motion of an obstacle once its element is placed: points of the motion move with the element, velocities turn
// with it, and a circle starts `yaw` further round.
inline Motion placeMotion(const ElementPlacement& element, const Motion& motion) {
    const Pose3 placement = element.pose();
    const auto place = [&placement](const Eigen::Vector3d& point) {
        return Eigen::Vector3d(placement.position + rotate(placement.orientation, point));
    };
    Motion placed = motion;
    switch (motion.type) {
    case Motion::kHold:
        break;
    case Motion::kConstantTwist:
        placed.linear = rotate(placement.orientation, motion.linear);
        placed.angular = rotate(placement.orientation, motion.angular);
        break;
    case Motion::kPingPong:
        placed.point_a = place(motion.point_a);
        placed.point_b = place(motion.point_b);
        break;
    case Motion::kCircle:
        placed.center = place(motion.center);
        placed.phase = motion.phase + element.yaw;
        break;
    }
    return placed;
}

// Where the obstacle must be at scene time zero for the motion: the start of the path of a ping_pong or circle, the
// initial position otherwise.
inline Eigen::Vector3d motionStart(const Eigen::Vector3d& initial_position, const Motion& motion) {
    if (motion.type == Motion::kPingPong) {
        return motion.point_a;
    }
    if (motion.type == Motion::kCircle) {
        return Eigen::Vector3d(motion.center.x() + motion.radius * std::cos(motion.phase),
                               motion.center.y() + motion.radius * std::sin(motion.phase), motion.center.z());
    }
    return initial_position;
}

struct MotionState {
    Pose3 pose;
    Eigen::Vector3d linear = Eigen::Vector3d::Zero();  // world frame
    Eigen::Vector3d angular = Eigen::Vector3d::Zero(); // world frame
};

// The state of an obstacle with `initial` pose after `elapsed` seconds of scene time. While the playback clock is
// paused (`playing` false) the pose still follows the clock but the twist reads zero.
inline MotionState evaluateMotion(const Pose3& initial, const Motion& motion, double elapsed, bool playing) {
    MotionState state;
    state.pose = initial;
    switch (motion.type) {
    case Motion::kHold:
        break;
    case Motion::kConstantTwist: {
        state.linear = motion.linear;
        state.angular = motion.angular;
        state.pose.position = Eigen::Vector3d(initial.position.x() + elapsed * motion.linear.x(),
                                              initial.position.y() + elapsed * motion.linear.y(),
                                              initial.position.z() + elapsed * motion.linear.z());
        const double norm =
            std::sqrt(motion.angular.x() * motion.angular.x() + motion.angular.y() * motion.angular.y() +
                      motion.angular.z() * motion.angular.z());
        if (norm > 1e-12) {
            const double half = 0.5 * norm * elapsed;
            const Eigen::Quaterniond rotation(std::cos(half), motion.angular.x() / norm * std::sin(half),
                                              motion.angular.y() / norm * std::sin(half),
                                              motion.angular.z() / norm * std::sin(half));
            state.pose.orientation = multiply(rotation, initial.orientation);
        }
        break;
    }
    case Motion::kPingPong: {
        const Eigen::Vector3d delta = motion.point_b - motion.point_a;
        const double length = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y() + delta.z() * delta.z());
        // The remainder has the sign of the divisor, so the clock may run backwards.
        double phase = std::fmod(elapsed * motion.speed / length, 2.0);
        if (phase < 0.0) {
            phase += 2.0;
        }
        const double fraction = phase < 1 ? phase : 2 - phase;
        const double direction = phase < 1 ? 1 : -1;
        state.pose.position = motion.point_a + fraction * delta;
        state.linear = Eigen::Vector3d(direction * motion.speed * delta.x() / length,
                                       direction * motion.speed * delta.y() / length,
                                       direction * motion.speed * delta.z() / length);
        break;
    }
    case Motion::kCircle: {
        const double phase = motion.phase + elapsed * motion.angular_speed;
        state.pose.position = Eigen::Vector3d(motion.center.x() + motion.radius * std::cos(phase),
                                              motion.center.y() + motion.radius * std::sin(phase), motion.center.z());
        state.linear = Eigen::Vector3d(-motion.radius * motion.angular_speed * std::sin(phase),
                                       motion.radius * motion.angular_speed * std::cos(phase), 0.0);
        break;
    }
    }
    if (!playing) {
        state.linear.setZero();
        state.angular.setZero();
    }
    return state;
}

} // namespace xgc2_math::scene
