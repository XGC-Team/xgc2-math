#pragma once

#include "xgc2_math/trajectory/trajectory2.hpp"
#include "xgc2_math/trajectory/trajectory3.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace xgc2_math::trajectory::analytic_detail {

constexpr double kMinDuration = 1.0e-6;

inline bool finiteScalar(double value) {
    return std::isfinite(value);
}

inline double clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(max_value, value));
}

inline double safeRadius(double radius) {
    return std::max(1.0e-3, std::abs(radius));
}

inline double angularRate(double radius, double line_speed) {
    return line_speed / safeRadius(radius);
}

inline double wrapAngle(double value) {
    return std::atan2(std::sin(value), std::cos(value));
}

inline double unwrapAngleNear(double value, double reference) {
    return reference + wrapAngle(value - reference);
}

inline double smoothstep01(double value) {
    const double u = clamp(value, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

inline double smoothstep01Derivative(double value) {
    const double u = clamp(value, 0.0, 1.0);
    if (value <= 0.0 || value >= 1.0) {
        return 0.0;
    }
    return 6.0 * u * (1.0 - u);
}

inline double smoothstep01SecondDerivative(double value) {
    const double u = clamp(value, 0.0, 1.0);
    if (value <= 0.0 || value >= 1.0) {
        return 0.0;
    }
    return 6.0 - 12.0 * u;
}

inline double polyValue(const std::vector<double>& coeffs, double t, int derivative) {
    if (derivative < 0 || coeffs.empty() || derivative >= static_cast<int>(coeffs.size())) {
        return 0.0;
    }
    double value = 0.0;
    for (int i = static_cast<int>(coeffs.size()) - 1; i >= derivative; --i) {
        double factor = 1.0;
        for (int k = 0; k < derivative; ++k) {
            factor *= static_cast<double>(i - k);
        }
        value = value * t + factor * coeffs[static_cast<size_t>(i)];
    }
    return value;
}

inline std::vector<double> septicBoundary(double p0, double v0, double a0, double j0, double p1, double v1, double a1,
                                          double j1, double duration) {
    return trajectory2_detail::septicBoundary(p0, v0, a0, j0, p1, v1, a1, j1, duration);
}

inline void evalSeptic(const std::vector<double>& coeffs, double t, double& p, double& v, double& a, double& j,
                       double& s) {
    p = polyValue(coeffs, t, 0);
    v = polyValue(coeffs, t, 1);
    a = polyValue(coeffs, t, 2);
    j = polyValue(coeffs, t, 3);
    s = polyValue(coeffs, t, 4);
}

inline void fillYawFromVelocity(FlatOutput3& output) {
    const double vx = output.velocity.x();
    const double vy = output.velocity.y();
    const double speed_sq = vx * vx + vy * vy;
    if (speed_sq < 1.0e-8) {
        output.yaw = 0.0;
        output.yaw_rate = 0.0;
        output.yaw_accel = 0.0;
        return;
    }

    const double ax = output.acceleration.x();
    const double ay = output.acceleration.y();
    const double jx = output.jerk.x();
    const double jy = output.jerk.y();
    const double numerator = vx * ay - vy * ax;
    const double numerator_dot = vx * jy - vy * jx;
    const double denominator_dot = 2.0 * (vx * ax + vy * ay);
    output.yaw = std::atan2(vy, vx);
    output.yaw_rate = numerator / speed_sq;
    output.yaw_accel = (numerator_dot * speed_sq - numerator * denominator_dot) / (speed_sq * speed_sq);
}

inline void fillYawHold(FlatOutput3& output, double yaw = 0.0) {
    output.yaw = wrapAngle(yaw);
    output.yaw_rate = 0.0;
    output.yaw_accel = 0.0;
}

inline void velocityYawWithOriginBlend(const Eigen::Vector2d& velocity, const Eigen::Vector2d& acceleration,
                                       const Eigen::Vector2d& jerk, double origin_yaw, double& yaw, double& yaw_rate,
                                       double& yaw_accel) {
    constexpr double kHoldSpeed = 0.05;
    constexpr double kFullSpeed = 0.50;
    const double speed_sq = velocity.squaredNorm();
    const double speed = std::sqrt(speed_sq);
    if (speed < 1.0e-9) {
        yaw = wrapAngle(origin_yaw);
        yaw_rate = 0.0;
        yaw_accel = 0.0;
        return;
    }

    const double vx = velocity.x();
    const double vy = velocity.y();
    const double ax = acceleration.x();
    const double ay = acceleration.y();
    const double jx = jerk.x();
    const double jy = jerk.y();
    const double numerator = vx * ay - vy * ax;
    const double numerator_dot = vx * jy - vy * jx;
    const double speed_sq_dot = 2.0 * (vx * ax + vy * ay);
    const double heading = unwrapAngleNear(std::atan2(vy, vx), origin_yaw);
    const double heading_rate = speed_sq > 1.0e-12 ? numerator / speed_sq : 0.0;
    const double heading_accel =
        speed_sq > 1.0e-12 ? (numerator_dot * speed_sq - numerator * speed_sq_dot) / (speed_sq * speed_sq) : 0.0;

    const double blend_arg = (speed - kHoldSpeed) / (kFullSpeed - kHoldSpeed);
    const double blend = smoothstep01(blend_arg);
    const double blend_du = smoothstep01Derivative(blend_arg);
    const double blend_ddu = smoothstep01SecondDerivative(blend_arg);
    const double speed_dot = (vx * ax + vy * ay) / speed;
    const double speed_ddot = (acceleration.squaredNorm() + velocity.dot(jerk)) / speed -
                              std::pow(velocity.dot(acceleration), 2.0) / (speed_sq * speed);
    const double blend_dot = blend_du * speed_dot / (kFullSpeed - kHoldSpeed);
    const double blend_ddot = blend_ddu * std::pow(speed_dot / (kFullSpeed - kHoldSpeed), 2.0) +
                              blend_du * speed_ddot / (kFullSpeed - kHoldSpeed);
    const double delta = heading - origin_yaw;

    yaw = wrapAngle(origin_yaw + blend * delta);
    yaw_rate = blend_dot * delta + blend * heading_rate;
    yaw_accel = blend_ddot * delta + 2.0 * blend_dot * heading_rate + blend * heading_accel;
}

inline void velocityYawWithBlend(const Eigen::Vector2d& velocity, const Eigen::Vector2d& acceleration,
                                 const Eigen::Vector2d& jerk, double origin_yaw, double blend, double blend_dot,
                                 double blend_ddot, double& yaw, double& yaw_rate, double& yaw_accel) {
    const double speed_sq = velocity.squaredNorm();
    if (speed_sq < 1.0e-12) {
        yaw = wrapAngle(origin_yaw);
        yaw_rate = 0.0;
        yaw_accel = 0.0;
        return;
    }

    const double vx = velocity.x();
    const double vy = velocity.y();
    const double ax = acceleration.x();
    const double ay = acceleration.y();
    const double jx = jerk.x();
    const double jy = jerk.y();
    const double numerator = vx * ay - vy * ax;
    const double numerator_dot = vx * jy - vy * jx;
    const double speed_sq_dot = 2.0 * (vx * ax + vy * ay);
    const double heading = unwrapAngleNear(std::atan2(vy, vx), origin_yaw);
    const double heading_rate = numerator / speed_sq;
    const double heading_accel = (numerator_dot * speed_sq - numerator * speed_sq_dot) / (speed_sq * speed_sq);
    const double w = clamp(blend, 0.0, 1.0);
    const double delta = heading - origin_yaw;

    yaw = wrapAngle(origin_yaw + w * delta);
    yaw_rate = blend_dot * delta + w * heading_rate;
    yaw_accel = blend_ddot * delta + 2.0 * blend_dot * heading_rate + w * heading_accel;
}

inline void fillYawFromVelocityWithOriginBlend(PlanarReference2& output, double origin_yaw) {
    velocityYawWithOriginBlend(output.velocity, output.acceleration, output.jerk, origin_yaw, output.yaw,
                               output.yaw_rate, output.yaw_acceleration);
    output.curvature = output.speed > 1.0e-6 ? output.yaw_rate / output.speed : 0.0;
}

inline void fillYawFromVelocityWithOriginBlend(FlatOutput3& output, double origin_yaw) {
    velocityYawWithOriginBlend(output.velocity.head<2>(), output.acceleration.head<2>(), output.jerk.head<2>(),
                               origin_yaw, output.yaw, output.yaw_rate, output.yaw_accel);
}

} // namespace xgc2_math::trajectory::analytic_detail
