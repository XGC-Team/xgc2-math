#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <xgc2_math/control/implicit_wheel_contact.hpp>

namespace {
int checks = 0;
void check(bool ok, const std::string& label) {
    ++checks;
    if (!ok)
        throw std::runtime_error(label);
}
void near(double a, double b, double tol, const std::string& label) {
    check(std::isfinite(a) && std::abs(a - b) <= tol, label);
}
xgc2_math::ImplicitWheelInput fixture() {
    xgc2_math::ImplicitWheelInput in;
    in.contact_offset_world_m = {
        {{.2255, .247889844}, {.2255, -.247889844}, {-.2255, .247889844}, {-.2255, -.247889844}}};
    in.normal_force_n.fill(28.01 * 9.81 / 4);
    return in;
}
void bounds(const xgc2_math::ImplicitWheelResult& r, const xgc2_math::ImplicitWheelInput& in,
            const xgc2_math::ImplicitWheelParameters& p) {
    for (int i = 0; i < 4; ++i) {
        check(std::hypot(r.tire_force_world_n[i][0], r.tire_force_world_n[i][1]) <=
                  p.friction[i] * in.normal_force_n[i] + 1e-8,
              "combined friction circle");
        check(std::abs(r.motor_torque_nm[i]) <= p.motor_limit_nm + 1e-10, "torque bound");
    }
}
} // namespace
int main() {
    using namespace xgc2_math;
    try {
        ImplicitWheelParameters p;
        auto in = fixture();
        ImplicitWheelMemory s;
        auto r = implicitWheelContactStep(in, s, p);
        for (double v : r.predicted_velocity)
            near(v, 0, 1e-14, "rest equilibrium");
        // All four tires at combined longitudinal/lateral slip; force is a
        // coupled circle, not two simultaneously available full friction bounds.
        in.velocity = {1.7, .8, .6, -4, 8, -5, 7};
        in.target_rad_s = {5, 6, 5, 6};
        r = implicitWheelContactStep(in, s, p);
        bounds(r, in, p);
        check(std::abs(r.tire_force_world_n[0][0]) > 1, "traction is active, not just diagnostics");
        check(std::abs(r.tire_force_world_n[0][1]) > 1, "lateral skid force is active");
        // Contact Jacobian and applied wrench include the same wheel reaction.
        for (int i = 0; i < 4; ++i) {
            const auto f = r.tire_force_world_n[i];
            const auto q = r.predicted_velocity;
            const auto pos = in.contact_offset_world_m[i];
            const auto dir = in.rolling_direction_world[i];
            const double generalized = f[0] * q[0] + f[1] * q[1] + (-pos[1] * f[0] + pos[0] * f[1]) * q[2] -
                                       in.radius_m[i] * (dir[0] * f[0] + dir[1] * f[1]) * q[3 + i];
            near(generalized, f[0] * r.predicted_surface_slip_m_s[i][0] + f[1] * r.predicted_surface_slip_m_s[i][1],
                 1e-10, "wheel/chassis power identity");
        }
        // Nonzero stored tread force can survive zero slip: compliant static
        // equilibrium with an 8 N external load, not F=0 at zero sliding speed.
        auto staticp = p;
        staticp.renewal_length_m.fill(0);
        in = fixture();
        in.external_force[0] = -8;
        for (int i = 0; i < 4; ++i) {
            s.tread_m[i][0] = -2 / p.stiffness_n_m[i];
            s.drive_error_rad[i] = -.16 / p.motor_i_nm;
        }
        r = implicitWheelContactStep(in, s, staticp);
        near(r.predicted_velocity[0], 0, 1e-9, "static support without creeping");
        for (int i = 0; i < 4; ++i) {
            near(r.tire_force_world_n[i][0], 2, 1e-7, "preloaded static traction");
            near(r.motor_torque_nm[i], .16, 1e-7, "static axle balance");
        }
        // Liftoff clears deformation and yields exactly zero tire force.
        in.normal_force_n.fill(0);
        r = implicitWheelContactStep(in, s, p);
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 2; ++k) {
                near(r.tire_force_world_n[i][k], 0, 0, "no ghost traction");
                near(r.memory.tread_m[i][k], 0, 0, "liftoff forgets tread");
            }
        // Global rotation invariance of force, velocity and stored tread state.
        in = fixture();
        s = {};
        in.velocity = {.7, -.2, .3, 4, 6, 4, 6};
        in.target_rad_s = {5, 7, 5, 7};
        s.tread_m[0] = {.0002, -.0004};
        auto base = implicitWheelContactStep(in, s, p);
        const double angle = .713, c = std::cos(angle), sn = std::sin(angle);
        const auto rot = [&](WheelPair a) {
            return WheelPair{c * a[0] - sn * a[1], sn * a[0] + c * a[1]};
        };
        auto rotated = in;
        auto rotateds = s;
        auto vrot = rot({in.velocity[0], in.velocity[1]});
        rotated.velocity[0] = vrot[0];
        rotated.velocity[1] = vrot[1];
        for (int i = 0; i < 4; ++i) {
            rotated.contact_offset_world_m[i] = rot(in.contact_offset_world_m[i]);
            rotated.rolling_direction_world[i] = rot(in.rolling_direction_world[i]);
            rotateds.tread_m[i] = rot(s.tread_m[i]);
        }
        auto rr = implicitWheelContactStep(rotated, rotateds, p);
        auto expected = rot({base.predicted_velocity[0], base.predicted_velocity[1]});
        near(rr.predicted_velocity[0], expected[0], 2e-9, "rotation vx");
        near(rr.predicted_velocity[1], expected[1], 2e-9, "rotation vy");
        for (int i = 2; i < 7; ++i)
            near(rr.predicted_velocity[i], base.predicted_velocity[i], 2e-8, "rotation scalar");
        // Mirror symmetry, including wheel permutation and yaw sign.
        auto mirror = in;
        auto mirrors = s;
        const int perm[4] = {1, 0, 3, 2};
        mirror.velocity[1] *= -1;
        mirror.velocity[2] *= -1;
        for (int i = 0; i < 4; ++i) {
            const int j = perm[i];
            mirror.velocity[3 + i] = in.velocity[3 + j];
            mirror.target_rad_s[i] = in.target_rad_s[j];
            mirrors.tread_m[i] = {s.tread_m[j][0], -s.tread_m[j][1]};
            mirrors.drive_error_rad[i] = s.drive_error_rad[j];
        }
        rr = implicitWheelContactStep(mirror, mirrors, p);
        near(rr.predicted_velocity[0], base.predicted_velocity[0], 1e-8, "left/right forward symmetry");
        near(rr.predicted_velocity[2], -base.predicted_velocity[2], 1e-8, "left/right yaw symmetry");
        // Discrete energy/work inequality, including actuator saturation and
        // anti-windup, random preload, changing normal load, and renewal.
        std::mt19937 rng(93);
        std::uniform_real_distribution<double> unif(-1, 1);
        double worst = 0;
        for (int trial = 0; trial < 1200; ++trial) {
            in = fixture();
            in.dt_s = trial % 3 == 0 ? .001 : (trial % 3 == 1 ? .004 : .02);
            for (double& v : in.velocity)
                v = unif(rng) * 8;
            for (double& f : in.external_force)
                f = unif(rng) * 3;
            for (int i = 0; i < 4; ++i) {
                s.drive_error_rad[i] = unif(rng);
                s.tread_m[i] = {unif(rng) * .02, unif(rng) * .02};
                in.normal_force_n[i] = std::abs(unif(rng)) * 150;
                in.target_rad_s[i] = trial % 2 == 0 ? 0 : unif(rng) * 12;
            }
            r = implicitWheelContactStep(in, s, p);
            bounds(r, in, p);
            double work = 0;
            for (int i = 0; i < 4; ++i)
                work += in.dt_s * r.motor_torque_nm[i] * in.target_rad_s[i];
            for (int i = 0; i < 7; ++i)
                work += in.dt_s * in.external_force[i] * r.predicted_velocity[i];
            const double excess =
                implicitWheelEnergy(r.predicted_velocity, r.memory, p) - implicitWheelEnergy(in.velocity, s, p) - work;
            worst = std::max(worst, excess);
            check(excess <= 1e-7, "discrete passivity/work bound");
        }
        // High stiffness, small inertia and large servo gain must stay bounded
        // at 4 ms without retuning an Euler stability condition.
        auto stiff = p;
        stiff.stiffness_n_m.fill(1e6);
        stiff.motor_p_nm_s = 30;
        stiff.max_sweeps = 2000;
        in = fixture();
        s = {};
        in.target_rad_s.fill(12);
        for (int k = 0; k < 200; ++k) {
            r = implicitWheelContactStep(in, s, stiff);
            bounds(r, in, stiff);
            in.velocity = r.predicted_velocity;
            s = r.memory;
        }
        check(std::abs(in.velocity[0]) < 2, "stiff coupled solve bounded");
        // PI steady tracking under finite load; compare integration steps.
        std::array<double, 3> terminal{};
        int index = 0;
        for (double h : {.001, .002, .004}) {
            in = fixture();
            in.dt_s = h;
            in.target_rad_s.fill(5);
            in.external_force[0] = -8;
            s = {};
            for (int k = 0; k < static_cast<int>(3 / h); ++k) {
                r = implicitWheelContactStep(in, s, p);
                in.velocity = r.predicted_velocity;
                s = r.memory;
            }
            for (int i = 0; i < 4; ++i)
                near(in.velocity[3 + i], 5, 1e-5, "loaded wheel servo has unity steady gain");
            terminal[index++] = in.velocity[0];
            // A stalled wheel with bounded torque must not accumulate unlimited
            // hidden integral state; velocity held externally for this unit test.
        }
        near(terminal[0], terminal[2], 2e-5, "steady step-size agreement");
        in = fixture();
        in.normal_force_n.fill(0);
        in.target_rad_s.fill(25);
        s = {};
        for (int k = 0; k < 3000; ++k) {
            r = implicitWheelContactStep(in, s, p);
            s = r.memory;
        }
        for (double e : s.drive_error_rad)
            check(std::abs(e) <= p.motor_limit_nm / p.motor_i_nm + 1e-8,
                  "dissipative antiwindup bounds stalled drive state");
        // Invalid input and forced nonconvergence never return force writes.
        auto rejects = [&](auto fn) {
            bool failed = false;
            try {
                fn();
            } catch (const std::exception&) {
                failed = true;
            }
            check(failed, "fail closed");
        };
        in = fixture();
        in.dt_s = 0;
        rejects([&] {
            implicitWheelContactStep(in, s, p);
        });
        in = fixture();
        in.velocity[0] = std::numeric_limits<double>::quiet_NaN();
        rejects([&] {
            implicitWheelContactStep(in, s, p);
        });
        in = fixture();
        in.normal_force_n[0] = -1;
        rejects([&] {
            implicitWheelContactStep(in, s, p);
        });
        in = fixture();
        in.rolling_direction_world[0] = {2, 0};
        rejects([&] {
            implicitWheelContactStep(in, s, p);
        });
        in = fixture();
        in.target_rad_s.fill(7);
        auto one = p;
        one.max_sweeps = 1;
        rejects([&] {
            implicitWheelContactStep(in, s, one);
        });
        std::cout << checks << " checks passed; random energy excess <= " << worst << " J\n";
        std::cout << "loaded steady forward speeds (1/2/4 ms): " << terminal[0] << ", " << terminal[1] << ", "
                  << terminal[2] << " m/s\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << "\n";
        return 1;
    }
}
