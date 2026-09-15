#pragma once

namespace xgc2_math {

struct WheelContactKinematics {
    double longitudinal_ground_m_s;
    double lateral_ground_m_s;
    double longitudinal_slip_m_s;
    double lateral_slip_m_s;
};

// All vectors are resolved in the same planar body frame and at the same time.
// (x,y) is the wheel CONTACT centre relative to the velocity reference point,
// not necessarily the mounting joint or centre of mass. rolling_speed = R*Omega
// is signed positive along body +x. Slip velocities remain defined at standstill.
inline WheelContactKinematics wheelContactKinematics(double forward_m_s, double lateral_m_s, double yaw_rad_s,
                                                     double x_m, double y_m, double rolling_speed_m_s) {
    const double longitudinal = forward_m_s - yaw_rad_s * y_m;
    const double lateral = lateral_m_s + yaw_rad_s * x_m;
    return {longitudinal, lateral, rolling_speed_m_s - longitudinal, lateral};
}

// Wheel/ground power after cancellation of equal/opposite traction torque.
// Dissipative contact satisfies this <= 0. Force signs: +x forward, +y left.
inline double wheelContactPower(const WheelContactKinematics& contact, double force_x_n, double force_y_n) {
    return -force_x_n * contact.longitudinal_slip_m_s + force_y_n * contact.lateral_slip_m_s;
}

} // namespace xgc2_math
