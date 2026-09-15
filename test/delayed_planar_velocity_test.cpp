#include <xgc2_math/control/delayed_planar_velocity.hpp>
#include <xgc2_math/control/wheel_contact_kinematics.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks = 0;
void near(double actual, double expected, double tolerance = 1e-12) {
    ++checks;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}
template <typename F> void rejects(F operation) {
    ++checks;
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
}  // namespace

int main() {
    using xgc2_math::DelayedPlanarVelocity;
    try {
        DelayedPlanarVelocity model;
        model.command(0.0, {1.0, -2.0});
        near(model.advance(0.004).linear_m_s, 0.0);
        near(model.advance(0.005).yaw_rad_s, 0.0);
        near(model.advance(0.008).linear_m_s, 1.0 - std::exp(-0.003 / 0.005));
        near(model.advance(0.010).linear_m_s, 1.0 - std::exp(-1.0));
        near(model.velocity().yaw_rad_s, -2.0 * (1.0 - std::exp(-1.0)));
        near(model.advance(1.0).linear_m_s, 1.0);
        near(model.velocity().yaw_rad_s, -2.0);
        // An isolated Stop is delayed and then decays without another heartbeat.
        model.command(1.0, {});
        near(model.advance(1.004).linear_m_s, 1.0);
        near(model.advance(1.010).linear_m_s, std::exp(-1.0));
        near(model.advance(1.1).linear_m_s, std::exp(-19.0));

        // Different channel time constants, with no cross-coupling or hidden gain.
        DelayedPlanarVelocity separate({0.005, 0.005, 0.010});
        separate.command(0.0, {0.8, 0.3});
        near(separate.advance(0.010).linear_m_s, 0.8 * (1.0 - std::exp(-1.0)));
        near(separate.velocity().yaw_rad_s, 0.3 * (1.0 - std::exp(-0.5)));

        // Event-splitting: one large step equals many small steps, including
        // multiple releases inside a step and non-grid-aligned event times.
        DelayedPlanarVelocity large, small;
        const double times[] = {0.0, 0.003, 0.0061, 0.019, 0.040};
        for (int i = 0; i != 5; ++i) {
            const xgc2_math::PlanarVelocity u{0.1 * i - 0.2, 0.5 - 0.23 * i};
            large.command(times[i], u);
            small.command(times[i], u);
        }
        large.advance(0.080);
        for (int i = 1; i <= 80; ++i) small.advance(0.001 * i);
        near(large.velocity().linear_m_s, small.velocity().linear_m_s);
        near(large.velocity().yaw_rad_s, small.velocity().yaw_rad_s);

        for (double h : {0.001, 0.002, 0.004, 0.010, 0.100}) {
            DelayedPlanarVelocity stepped;
            stepped.command(0.0, {1.0, 1.0});
            for (int k = 1; k * h < 0.100; ++k) stepped.advance(k * h);
            near(stepped.advance(0.100).linear_m_s, 1.0 - std::exp(-19.0));
        }
        DelayedPlanarVelocity paused;
        for (int i = 0; i < 10000; ++i) paused.command(0.0, {i * 0.0001, 0.0});
        near(static_cast<double>(paused.pendingCommands()), 1.0);
        near(paused.advance(0.0).linear_m_s, 0.0);
        near(paused.advance(0.010).linear_m_s, 0.9999 * (1.0 - std::exp(-1.0)));
        // Hold clears BOTH lag states and ALL delayed commands immediately.
        paused.command(0.010, {1.0, 1.0});
        paused.reset(0.010);
        near(paused.velocity().linear_m_s, 0.0);
        near(static_cast<double>(paused.pendingCommands()), 0.0);
        near(paused.advance(2.0).yaw_rad_s, 0.0);
        paused.command(2.0, {1.0, 1.0});
        near(paused.advance(0.0).linear_m_s, 0.0);  // clock rollback
        near(paused.advance(2.1).linear_m_s, 0.0);
        // Explicit zero-delay/zero-lag limit, never used by Scout defaults.
        DelayedPlanarVelocity ideal({0.0, 0.0, 0.0});
        ideal.command(0.0, {0.4, -0.2});
        near(ideal.advance(0.0).linear_m_s, 0.4);
        near(ideal.velocity().yaw_rad_s, -0.2);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        rejects([&] { DelayedPlanarVelocity bad({-0.001, 0.005, 0.005}); });
        rejects([&] { DelayedPlanarVelocity bad({0.005, nan, 0.005}); });
        rejects([&] { ideal.command(0.0, {nan, 0.0}); });
        rejects([&] { ideal.advance(nan); });
        rejects([&] { ideal.command(-0.001, {}); });
        rejects([&] { ideal.reset(nan); });

        DelayedPlanarVelocity extremes({0.0, 0.005, 0.005});
        extremes.command(0.0, {1e308, 0.0});
        near(extremes.advance(1.0).linear_m_s, 1e308, 1e293);
        extremes.command(1.0, {-1e308, 0.0});
        near(extremes.advance(2.0).linear_m_s, -1e308, 1e293);
        rejects([&] { DelayedPlanarVelocity bad({0.005, -1.0, 0.005}); });
        rejects([&] { extremes.command(2.0, {std::numeric_limits<double>::infinity(), 0.0}); });

        using xgc2_math::wheelContactKinematics;
        const auto straight = wheelContactKinematics(0.4, 0.0, 0.0, 0.2255, 0.245, 0.4);
        near(straight.longitudinal_slip_m_s, 0.0);
        near(straight.lateral_slip_m_s, 0.0);
        const auto fl = wheelContactKinematics(0.4, 0.0, 0.5, 0.2255, 0.245, 0.4 - 0.5 * 0.245);
        const auto rl = wheelContactKinematics(0.4, 0.0, 0.5, -0.2255, 0.245, 0.4 - 0.5 * 0.245);
        near(fl.longitudinal_slip_m_s, 0.0);
        near(fl.lateral_slip_m_s - rl.lateral_slip_m_s, 0.5 * 0.451);
        const auto slipping = wheelContactKinematics(0.4, 0.0, 0.5, 0.2255, 0.245, 0.5);
        near(xgc2_math::wheelContactPower(slipping, 2.0, -1.0),
             -2.0 * slipping.longitudinal_slip_m_s - slipping.lateral_slip_m_s);
        const auto rest = wheelContactKinematics(0.0, 0.0, 0.0, 0.2255, 0.245, 0.0);
        near(rest.longitudinal_slip_m_s, 0.0);  // no division by |v| at standstill
        std::cout << checks << " checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
