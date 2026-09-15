#pragma once

#include <algorithm>

namespace xgc2_math {

struct DifferentialDriveParameters {
    double wheel_radius_m;
    double wheel_separation_m;
    double command_gain{1.0};
    double angular_command_gain{1.0};
};

struct WheelAngularVelocity {
    double left_rad_s;
    double right_rad_s;
};

// Inverse differential-drive kinematics, with dimensionless command calibration.
// Positive longitudinal velocity is forward; positive yaw is counterclockwise.
// Inputs must be finite and geometry positive. The adapter validates configuration.
// This allocation does not model tire slip or predict achieved chassis velocity.
inline WheelAngularVelocity differentialDriveWheelVelocity(double longitudinal_m_s, double yaw_rad_s,
                                                           const DifferentialDriveParameters& parameters) {
    const double steering = yaw_rad_s * parameters.angular_command_gain;
    const double half_track = parameters.wheel_separation_m * 0.5;
    const double left = (longitudinal_m_s - steering * half_track) / parameters.wheel_radius_m;
    const double right = (longitudinal_m_s + steering * half_track) / parameters.wheel_radius_m;
    return {left * parameters.command_gain, right * parameters.command_gain};
}

struct WheelVelocityIPParameters {
    double proportional_nm_s_per_rad;
    double integral_nm_per_rad;
    double effort_limit_nm;
};

struct WheelVelocityIPState {
    double effort_nm{0.0};
    double previous_velocity_rad_s{0.0};
};

// Incremental I-P: integral action on error, proportional action on measurement.
// The clamped output is also the next effort state; there is no hidden integrator.
// tau[k] = clamp(tau[k-1] + Ki (r[k]-w[k]) dt - Kp (w[k]-w[k-1]), +/- limit).
// Inputs are finite, Kp >= 0, Ki > 0, effort limit > 0. Command velocity limits
// belong to the adapter. Initialize previous_velocity from the measured wheel.
// dt is supplied by the caller in seconds; this function owns no clock or timer.
inline WheelVelocityIPState wheelVelocityIPStep(WheelVelocityIPState state, double target_rad_s, double measured_rad_s,
                                                double dt_s, const WheelVelocityIPParameters& parameters) {
    if (dt_s <= 0.0) {
        return state;
    }
    state.effort_nm += parameters.integral_nm_per_rad * (target_rad_s - measured_rad_s) * dt_s -
                       parameters.proportional_nm_s_per_rad * (measured_rad_s - state.previous_velocity_rad_s);
    state.effort_nm = std::clamp(state.effort_nm, -parameters.effort_limit_nm, parameters.effort_limit_nm);
    state.previous_velocity_rad_s = measured_rad_s;
    return state;
}

} // namespace xgc2_math
