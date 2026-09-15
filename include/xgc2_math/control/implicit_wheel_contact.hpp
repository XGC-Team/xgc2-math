#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace xgc2_math {

// Reduced planar four-wheel dynamics, in a fixed WORLD tangent frame for each
// step. Velocities: world vx, world vy, yaw rate, four signed wheel speeds.
// Positive wheel speed rolls along rolling_x/y. Forces are NOT chassis commands.
using WheelVector7 = std::array<double, 7>;
using WheelVector12 = std::array<double, 12>;
using WheelPair = std::array<double, 2>;

struct ImplicitWheelParameters {
    double mass_kg{28.01};
    double yaw_inertia_kg_m2{1.0};
    std::array<double, 4> wheel_inertia{{.0096, .0096, .0096, .0096}};
    // Illustrative constitutive parameters, NOT identified Scout tire data.
    std::array<double, 4> stiffness_n_m{{12000, 12000, 12000, 12000}};
    std::array<double, 4> friction{{.6, .6, .6, .6}};
    std::array<double, 4> renewal_length_m{{.04, .04, .04, .04}};
    double motor_p_nm_s{1.8};
    double motor_i_nm{10.0};
    double motor_limit_nm{6.0};
    double passive_drag_nm_s{0.0};
    double kkt_tolerance{1e-9};
    unsigned max_sweeps{500};
};

struct ImplicitWheelMemory {
    // Mean tread elastic deformation in world XY (m). No division by speed.
    std::array<WheelPair, 4> tread_m{};
    // Elastic motor tracking error (rad), with dissipative yield anti-windup.
    std::array<double, 4> drive_error_rad{};
};

struct ImplicitWheelInput {
    double dt_s{.004};
    WheelVector7 velocity{};
    WheelVector7 external_force{}; // optional known generalized external load
    std::array<WheelPair, 4> contact_offset_world_m{};
    std::array<WheelPair, 4> rolling_direction_world{{{1, 0}, {1, 0}, {1, 0}, {1, 0}}};
    std::array<double, 4> radius_m{{.08, .08, .08, .08}};
    std::array<double, 4> normal_force_n{};
    std::array<double, 4> target_rad_s{};
};

struct ImplicitWheelResult {
    WheelVector7 predicted_velocity{}; // prediction, never a Gazebo velocity write
    std::array<WheelPair, 4> tire_force_world_n{};
    std::array<WheelPair, 4> predicted_surface_slip_m_s{};
    std::array<double, 4> motor_torque_nm{};
    std::array<double, 4> joint_torque_nm{}; // motor minus passive drag
    ImplicitWheelMemory memory{};
    unsigned sweeps{0};
    double kkt_residual{0};
};

namespace implicit_wheel_detail {
inline void finite(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite wheel model input");
}
inline void positive(double value) {
    finite(value);
    if (value <= 0)
        throw std::invalid_argument("wheel model parameter must be positive");
}
inline void nonnegative(double value) {
    finite(value);
    if (value < 0)
        throw std::invalid_argument("negative wheel model parameter");
}
inline WheelPair disk(WheelPair x, double radius) {
    const double norm = std::hypot(x[0], x[1]);
    if (norm > radius && norm > 0) {
        x[0] *= radius / norm;
        x[1] *= radius / norm;
    }
    return x;
}
// Exact 2D block minimizer on a Euclidean disk. A radial clamp of the
// unconstrained solution would be WRONG for a nondiagonal effective inertia.
inline WheelPair minimizeDisk(double a, double b, double c, double gx, double gy, double radius) {
    if (radius == 0)
        return {};
    const auto solve = [&](double lambda) {
        const double aa = a + lambda, cc = c + lambda;
        const double determinant = aa * cc - b * b;
        if (!(determinant > 0) || !std::isfinite(determinant))
            throw std::runtime_error("invalid contact block conditioning");
        return WheelPair{(-cc * gx + b * gy) / determinant, (b * gx - aa * gy) / determinant};
    };
    WheelPair x = solve(0);
    if (std::hypot(x[0], x[1]) <= radius)
        return x;
    double lo = 0, hi = std::max({a, c, 1.0});
    for (unsigned k = 0; std::hypot(solve(hi)[0], solve(hi)[1]) > radius; ++k) {
        if (k == 80)
            throw std::runtime_error("contact projection bracket failed");
        hi *= 2;
    }
    for (unsigned k = 0; k < 70; ++k) {
        const double mid = .5 * (lo + hi);
        x = solve(mid);
        if (std::hypot(x[0], x[1]) > radius)
            lo = mid;
        else
            hi = mid;
    }
    return disk(solve(hi), radius);
}
} // namespace implicit_wheel_detail

// Simultaneously solve 8 tangential tire impulses and 4 bounded motor impulses.
// Strictly convex, compliant maximum-dissipation problem; no explicit stiff
// spring, independent friction pyramid, set-velocity, or wheel-rate Euler step.
inline ImplicitWheelResult implicitWheelContactStep(const ImplicitWheelInput& in, const ImplicitWheelMemory& old,
                                                    const ImplicitWheelParameters& p = {}) {
    using namespace implicit_wheel_detail;
    positive(in.dt_s);
    positive(p.mass_kg);
    positive(p.yaw_inertia_kg_m2);
    positive(p.motor_p_nm_s);
    positive(p.motor_i_nm);
    positive(p.motor_limit_nm);
    positive(p.kkt_tolerance);
    nonnegative(p.passive_drag_nm_s);
    if (p.max_sweeps == 0)
        throw std::invalid_argument("zero contact iterations");
    const double h = in.dt_s, damping = p.motor_p_nm_s + h * p.motor_i_nm;
    positive(damping);
    WheelVector7 mass{p.mass_kg, p.mass_kg, p.yaw_inertia_kg_m2, 0, 0, 0, 0};
    WheelVector7 inv{}, free{};
    std::array<WheelVector7, 12> columns{};
    WheelVector12 diagonal{}, linear{}, impulse{}, limit{};
    for (std::size_t i = 0; i < 4; ++i) {
        positive(p.wheel_inertia[i]);
        positive(p.stiffness_n_m[i]);
        nonnegative(p.friction[i]);
        nonnegative(p.renewal_length_m[i]);
        nonnegative(in.normal_force_n[i]);
        positive(in.radius_m[i]);
        finite(in.target_rad_s[i]);
        finite(old.drive_error_rad[i]);
        const auto r = in.contact_offset_world_m[i];
        const auto t = in.rolling_direction_world[i];
        for (double v : r)
            finite(v);
        for (double v : t)
            finite(v);
        for (double v : old.tread_m[i])
            finite(v);
        if (std::abs(std::hypot(t[0], t[1]) - 1) > 1e-8)
            throw std::invalid_argument("rolling direction must be a unit vector");
        mass[3 + i] = p.wheel_inertia[i];
        columns[2 * i] = {1, 0, -r[1], 0, 0, 0, 0};
        columns[2 * i + 1] = {0, 1, r[0], 0, 0, 0, 0};
        columns[2 * i][3 + i] = -in.radius_m[i] * t[0];
        columns[2 * i + 1][3 + i] = -in.radius_m[i] * t[1];
        columns[8 + i][3 + i] = 1;
        const double gamma =
            p.renewal_length_m[i] > 0 ? std::abs(in.radius_m[i] * in.velocity[3 + i]) / p.renewal_length_m[i] : 0;
        diagonal[2 * i] = diagonal[2 * i + 1] = (1 + h * gamma) / (h * h * p.stiffness_n_m[i]);
        // Loss of contact forgets old tread preload; never creates ghost traction.
        const auto tread = in.normal_force_n[i] > 0 ? old.tread_m[i] : WheelPair{};
        linear[2 * i] = tread[0] / h;
        linear[2 * i + 1] = tread[1] / h;
        limit[2 * i] = limit[2 * i + 1] = h * p.friction[i] * in.normal_force_n[i];
        diagonal[8 + i] = 1 / (h * damping);
        linear[8 + i] = -in.target_rad_s[i] + p.motor_i_nm * old.drive_error_rad[i] / damping;
        limit[8 + i] = h * p.motor_limit_nm;
    }
    for (std::size_t j = 0; j < 7; ++j) {
        finite(in.velocity[j]);
        finite(in.external_force[j]);
        inv[j] = 1 / (mass[j] + (j >= 3 ? h * p.passive_drag_nm_s : 0));
        free[j] = inv[j] * (mass[j] * in.velocity[j] + h * in.external_force[j]);
        finite(free[j]);
    }
    std::array<WheelVector12, 12> matrix{};
    double lipschitz = 0, scale = 1;
    for (std::size_t a = 0; a < 12; ++a) {
        for (std::size_t j = 0; j < 7; ++j)
            linear[a] += columns[a][j] * free[j];
        finite(linear[a]);
        positive(diagonal[a]);
        finite(limit[a]);
        scale = std::max(scale, std::abs(linear[a]));
        double row_sum = 0;
        for (std::size_t b = 0; b < 12; ++b) {
            matrix[a][b] = a == b ? diagonal[a] : 0;
            for (std::size_t j = 0; j < 7; ++j)
                matrix[a][b] += columns[a][j] * inv[j] * columns[b][j];
            finite(matrix[a][b]);
            row_sum += std::abs(matrix[a][b]);
        }
        lipschitz = std::max(lipschitz, row_sum);
    }
    positive(lipschitz);
    ImplicitWheelResult result;
    for (unsigned sweep = 1; sweep <= p.max_sweeps; ++sweep) {
        for (std::size_t i = 0; i < 4; ++i) {
            const std::size_t a = 2 * i;
            double gx = linear[a], gy = linear[a + 1];
            for (std::size_t j = 0; j < 12; ++j)
                if (j != a && j != a + 1) {
                    gx += matrix[a][j] * impulse[j];
                    gy += matrix[a + 1][j] * impulse[j];
                }
            const auto x = minimizeDisk(matrix[a][a], matrix[a][a + 1], matrix[a + 1][a + 1], gx, gy, limit[a]);
            impulse[a] = x[0];
            impulse[a + 1] = x[1];
        }
        for (std::size_t a = 8; a < 12; ++a) {
            double g = linear[a];
            for (std::size_t j = 0; j < 12; ++j)
                if (j != a)
                    g += matrix[a][j] * impulse[j];
            impulse[a] = std::clamp(-g / matrix[a][a], -limit[a], limit[a]);
        }
        WheelVector12 projected{};
        for (std::size_t a = 0; a < 12; ++a) {
            double gradient = linear[a];
            for (std::size_t j = 0; j < 12; ++j)
                gradient += matrix[a][j] * impulse[j];
            projected[a] = impulse[a] - gradient / lipschitz;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            const auto x = disk({projected[2 * i], projected[2 * i + 1]}, limit[2 * i]);
            projected[2 * i] = x[0];
            projected[2 * i + 1] = x[1];
            projected[8 + i] = std::clamp(projected[8 + i], -limit[8 + i], limit[8 + i]);
        }
        result.kkt_residual = 0;
        for (std::size_t a = 0; a < 12; ++a)
            result.kkt_residual = std::max(result.kkt_residual, lipschitz * std::abs(projected[a] - impulse[a]));
        result.sweeps = sweep;
        if (result.kkt_residual <= p.kkt_tolerance * scale)
            break;
        if (sweep == p.max_sweeps)
            throw std::runtime_error("coupled wheel solve did not converge");
    }
    result.predicted_velocity = free;
    for (std::size_t j = 0; j < 7; ++j)
        for (std::size_t a = 0; a < 12; ++a)
            result.predicted_velocity[j] += inv[j] * columns[a][j] * impulse[a];
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t k = 0; k < 2; ++k) {
            result.tire_force_world_n[i][k] = impulse[2 * i + k] / h;
            result.memory.tread_m[i][k] = -result.tire_force_world_n[i][k] / p.stiffness_n_m[i];
            for (std::size_t j = 0; j < 7; ++j)
                result.predicted_surface_slip_m_s[i][k] += columns[2 * i + k][j] * result.predicted_velocity[j];
        }
        result.motor_torque_nm[i] = impulse[8 + i] / h;
        result.joint_torque_nm[i] = result.motor_torque_nm[i] - p.passive_drag_nm_s * result.predicted_velocity[3 + i];
        result.memory.drive_error_rad[i] =
            (p.motor_p_nm_s * old.drive_error_rad[i] - h * result.motor_torque_nm[i]) / damping;
    }
    for (double v : result.predicted_velocity)
        finite(v);
    return result;
}

// Storage function for the FROZEN planar numerical system, not the whole Gazebo
// model. Under zero target/external work its discrete increase must be <= solver
// tolerance. Nonzero reference supplies motor port work h*sum(tau*Omega_ref).
inline double implicitWheelEnergy(const WheelVector7& v, const ImplicitWheelMemory& s,
                                  const ImplicitWheelParameters& p) {
    double energy = .5 * p.mass_kg * (v[0] * v[0] + v[1] * v[1]) + .5 * p.yaw_inertia_kg_m2 * v[2] * v[2];
    for (std::size_t i = 0; i < 4; ++i) {
        energy += .5 * p.wheel_inertia[i] * v[3 + i] * v[3 + i];
        energy += .5 * p.stiffness_n_m[i] * (s.tread_m[i][0] * s.tread_m[i][0] + s.tread_m[i][1] * s.tread_m[i][1]);
        energy += .5 * p.motor_i_nm * s.drive_error_rad[i] * s.drive_error_rad[i];
    }
    return energy;
}

} // namespace xgc2_math
