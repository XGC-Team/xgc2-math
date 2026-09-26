#include "xgc2_math/control/smc_tracking_controller.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

using namespace xgc2_math::control;
using Eigen::Vector3d;

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

int main() {
    const SmcTrackingConfig config;
    const Vector3d zero = Vector3d::Zero();
    const Vector3d ff(0.2, -0.3, 0.1);
    const auto equilibrium = computeSmcTracking(config, zero, zero, zero, zero, ff);
    require(equilibrium.success && equilibrium.feedback.isZero() &&
            equilibrium.acceleration.isApprox(ff), "reference feedforward at zero error");

    // Scalar values evaluated independently from the manuscript's atan surface.
    const auto points = computeSmcTracking(config, Vector3d(1.0, -0.1, 1e-6),
                                           Vector3d(-0.5, 0.01, 0.0), zero, zero, ff);
    require(points.success && points.sliding.isApprox(
        Vector3d(-0.12224468011856648, -0.21301194193491302, 0.0007071067811864297), 1e-12),
        "sliding surface sign, atan and signed square root");
    require(points.feedback.isApprox(Vector3d(0.5, 0.5, -0.035355339059321485), 1e-12),
            "saturated and boundary-layer feedback");
    const auto opposite = computeSmcTracking(config, Vector3d(-1.0, 0.1, -1e-6),
                                             Vector3d(0.5, -0.01, 0.0), zero, zero, zero);
    require(opposite.feedback.isApprox(-points.feedback), "odd feedback symmetry");

    // Actual sampled controller driving the exact constant-acceleration plant.
    // The reference moves and accelerates; no state is assigned to its target.
    for (double dt : {0.002, 0.01}) {
        Vector3d p(1.0, -1.0, 0.4), v(-0.5, 0.5, 0.0);
        const Vector3d acceleration(0.2, -0.1, 0.03);
        const Vector3d disturbance(0.1, -0.05, 0.02);
        const int steps = static_cast<int>(30.0 / dt);
        for (int k = 0; k < steps; ++k) {
            const double t = k * dt;
            const auto u = computeSmcTracking(config, p, v, 0.5 * t * t * acceleration,
                                               t * acceleration, acceleration);
            require(u.success && u.feedback.cwiseAbs().maxCoeff() <= config.k1,
                    "bounded finite sampled feedback");
            const Vector3d applied = u.acceleration + disturbance;
            p += dt * v + 0.5 * dt * dt * applied;
            v += dt * applied;
        }
        const double position_error = (p - 0.5 * 30.0 * 30.0 * acceleration).norm();
        const double velocity_error = (v - 30.0 * acceleration).norm();
        std::cout << "dt=" << dt << " final position error=" << position_error
                  << " velocity error=" << velocity_error << '\n';
        require(position_error < 0.002 && velocity_error < 0.01,
                "sampled SMC tracks accelerating reference with bounded disturbance");
    }
    SmcTrackingConfig invalid = config;
    invalid.boundary_layer = 0;
    require(!computeSmcTracking(invalid, zero, zero, zero, zero, zero).success,
            "zero boundary layer is not a valid controller");
    Vector3d bad = zero;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    require(!computeSmcTracking(config, bad, zero, zero, zero, zero).success,
            "invalid measurement cannot produce a control command");
    std::cout << "SMC formula and sampled closed loop passed\n";
}
