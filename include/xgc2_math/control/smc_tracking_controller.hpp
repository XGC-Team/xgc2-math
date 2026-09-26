#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace xgc2_math::control {

// Boundary-layer SMC for a double-integrator reference, in world coordinates.
// The nominal gains reproduce the manuscript's parameter illustration; they
// do not establish a disturbance bound or physical-vehicle tracking guarantee.
struct SmcTrackingConfig {
    double k1{0.5};
    double k2{0.5};
    double boundary_layer{0.01};
};

struct SmcTrackingOutput {
    bool success{false};
    Eigen::Vector3d position_error{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity_error{Eigen::Vector3d::Zero()};
    Eigen::Vector3d sliding{Eigen::Vector3d::Zero()};
    Eigen::Vector3d feedback{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
};

// e = measured - reference; z = e_v |e_v| + k2 atan(e_p),
// s = e_v + sign(z) sqrt(|z|), u = a_ref - k1 sat(s / rho).
// Gravity, thrust normalization and attitude/rate control belong to the
// vehicle's existing inner loop, not this translational feedback law.
inline SmcTrackingOutput computeSmcTracking(const SmcTrackingConfig& config, const Eigen::Vector3d& measured_position,
                                            const Eigen::Vector3d& measured_velocity,
                                            const Eigen::Vector3d& reference_position,
                                            const Eigen::Vector3d& reference_velocity,
                                            const Eigen::Vector3d& reference_acceleration) {
    SmcTrackingOutput out;
    if (!std::isfinite(config.k1) || config.k1 <= 0.0 || !std::isfinite(config.k2) || config.k2 <= 0.0 ||
        !std::isfinite(config.boundary_layer) || config.boundary_layer <= 0.0 || !measured_position.allFinite() ||
        !measured_velocity.allFinite() || !reference_position.allFinite() || !reference_velocity.allFinite() ||
        !reference_acceleration.allFinite()) {
        return out;
    }
    out.position_error = measured_position - reference_position;
    out.velocity_error = measured_velocity - reference_velocity;
    for (int axis = 0; axis < 3; ++axis) {
        const double velocity = out.velocity_error[axis];
        const double z = velocity * std::abs(velocity) + config.k2 * std::atan(out.position_error[axis]);
        out.sliding[axis] = velocity + std::copysign(std::sqrt(std::abs(z)), z);
        out.feedback[axis] = -config.k1 * std::clamp(out.sliding[axis] / config.boundary_layer, -1.0, 1.0);
    }
    out.acceleration = reference_acceleration + out.feedback;
    out.success = out.position_error.allFinite() && out.velocity_error.allFinite() && out.sliding.allFinite() &&
                  out.acceleration.allFinite();
    return out;
}

} // namespace xgc2_math::control
