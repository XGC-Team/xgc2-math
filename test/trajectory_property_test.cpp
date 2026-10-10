#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xgc2_math/optimization/minco.hpp"
#include "xgc2_math/trajectory/analytic.hpp"

namespace {

namespace tr = xgc2_math::trajectory;
namespace minco = xgc2_math::optimization::minco;
constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, const std::string& context) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << std::setprecision(17) << context << ": actual=" << actual << ", expected=" << expected
            << ", tolerance=" << tolerance;
        throw std::runtime_error(out.str());
    }
}

template <typename A, typename B>
void matrixNear(const Eigen::MatrixBase<A>& actual, const Eigen::MatrixBase<B>& expected, double tolerance,
                const std::string& context) {
    require(actual.allFinite() && expected.allFinite(), context + ": nonfinite matrix");
    near((actual - expected).norm(), 0.0, tolerance, context + " (norm of difference)");
}

double wrap(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

tr::PlanarReference2 evaluate(const tr::TrajectoryEvaluator2& evaluator, double time) {
    tr::PlanarReference2 output;
    require(evaluator.evaluate(time, output), "2D evaluation failed at t=" + std::to_string(time));
    require(tr::TrajectoryValidator2::finite(output), "2D output not finite at t=" + std::to_string(time));
    return output;
}

tr::FlatOutput3 evaluate(const tr::TrajectoryEvaluator3& evaluator, double time) {
    tr::FlatOutput3 output;
    require(evaluator.evaluate(time, output), "3D evaluation failed at t=" + std::to_string(time));
    require(tr::TrajectoryValidator3::finite(output), "3D output not finite at t=" + std::to_string(time));
    return output;
}

using Named2 = std::pair<std::string, std::unique_ptr<tr::TrajectoryEvaluator2>>;
using Named3 = std::pair<std::string, std::unique_ptr<tr::TrajectoryEvaluator3>>;

std::vector<Named2> analytic2() {
    std::vector<Named2> result;
    tr::HoldCurveParameters2 hold;
    hold.position = Eigen::Vector2d(1.2, -0.4);
    hold.yaw = -0.7;
    hold.duration = 8.0;
    result.emplace_back("hold2", std::make_unique<tr::HoldCurveEvaluator2>(hold));
    tr::CircleCurveParameters2 circle;
    circle.center = Eigen::Vector2d(1.1, -0.6);
    circle.radius = 1.4;
    circle.line_speed = 1.1;
    circle.duration = 8.0;
    result.emplace_back("circle2", std::make_unique<tr::CircleCurveEvaluator2>(circle));
    tr::FigureEightCurveParameters2 eight;
    eight.center = Eigen::Vector2d(-0.4, 0.7);
    eight.radius = 1.3;
    eight.line_speed = 0.9;
    eight.duration = 8.0;
    result.emplace_back("figure_eight2", std::make_unique<tr::FigureEightCurveEvaluator2>(eight));
    tr::CircleEntryCurveParameters2 entry;
    entry.origin = Eigen::Vector2d(-0.3, 0.5);
    entry.origin_yaw = 0.37;
    entry.entry_duration = 2.3;
    entry.duration = 8.0;
    entry.circle = circle;
    result.emplace_back("circle_entry2", std::make_unique<tr::CircleEntryCurveEvaluator2>(entry));
    return result;
}

std::vector<Named3> analytic3() {
    std::vector<Named3> result;
    tr::HoldCurveParameters3 hold;
    hold.position = Eigen::Vector3d(0.5, -1.2, 2.3);
    hold.yaw = 0.4;
    hold.duration = 8.0;
    result.emplace_back("hold3", std::make_unique<tr::HoldCurveEvaluator3>(hold));
    tr::CircleCurveParameters3 circle;
    circle.center = Eigen::Vector2d(0.4, -0.8);
    circle.radius = 1.3;
    circle.line_speed = 1.1;
    circle.height = 1.2;
    circle.z_amplitude = 0.3;
    circle.z_frequency = 0.7;
    circle.yaw = -0.35;
    circle.duration = 8.0;
    result.emplace_back("circle3", std::make_unique<tr::CircleCurveEvaluator3>(circle));
    tr::FigureEightCurveParameters3 eight;
    eight.origin = Eigen::Vector3d(0.2, -0.4, 0.0);
    eight.radius = 1.7;
    eight.line_speed = 1.2;
    eight.height = 1.4;
    eight.yaw = -0.6;
    eight.duration = 8.0;
    result.emplace_back("figure_eight3", std::make_unique<tr::FigureEightCurveEvaluator3>(eight));
    tr::LineCurveParameters3 line;
    line.start = Eigen::Vector3d(-0.7, 0.3, 0.8);
    line.target = Eigen::Vector3d(1.5, -1.1, 2.0);
    line.start_velocity = Eigen::Vector3d(0.1, -0.2, 0.3);
    line.target_velocity = Eigen::Vector3d(-0.2, 0.1, 0.15);
    line.yaw = 0.9;
    line.duration = 8.0;
    result.emplace_back("line3", std::make_unique<tr::LineCurveEvaluator3>(line));
    tr::HelixXyCurveParameters3 xy;
    xy.origin = Eigen::Vector3d(0.7, -0.5, 1.2);
    xy.radius = 0.7;
    xy.omega = 1.1;
    xy.linear_scale = 2.5;
    xy.yaw = 0.7;
    xy.duration = 8.0;
    result.emplace_back("helix_xy3", std::make_unique<tr::HelixXyCurveEvaluator3>(xy));
    tr::HelixYzCurveParameters3 yz;
    yz.origin = Eigen::Vector3d(-0.7, 0.5, 1.7);
    yz.radius = 0.8;
    yz.omega = 1.2;
    yz.linear_scale = 3.5;
    yz.yaw = 0.25;
    yz.duration = 8.0;
    result.emplace_back("helix_yz3", std::make_unique<tr::HelixYzCurveEvaluator3>(yz));
    tr::LemniscateCurveParameters3 lemniscate;
    lemniscate.origin = Eigen::Vector3d(-0.2, 0.9, 0.5);
    lemniscate.radius = 1.2;
    lemniscate.omega = 0.85;
    lemniscate.height = 1.7;
    lemniscate.yaw = -0.55;
    lemniscate.duration = 8.0;
    result.emplace_back("lemniscate3", std::make_unique<tr::LemniscateCurveEvaluator3>(lemniscate));
    tr::TorusKnotCurveParameters3 torus;
    torus.origin = Eigen::Vector3d(0.7, -0.8, 0.3);
    torus.omega = 0.6;
    torus.scale = 0.5;
    torus.yaw = 0.9;
    torus.duration = 8.0;
    result.emplace_back("torus_knot3", std::make_unique<tr::TorusKnotCurveEvaluator3>(torus));
    tr::CircleEntryCurveParameters3 circle_entry;
    circle_entry.origin = Eigen::Vector3d(-0.4, 0.2, circle.height);
    circle_entry.origin_yaw = 0.37;
    circle_entry.entry_duration = 2.3;
    circle_entry.duration = 8.0;
    circle_entry.circle = circle;
    result.emplace_back("circle_entry3", std::make_unique<tr::CircleEntryCurveEvaluator3>(circle_entry));
    tr::TorusKnotEntryCurveParameters3 torus_entry;
    torus_entry.start = Eigen::Vector3d(0.3, 0.1, 1.5);
    torus_entry.origin_yaw = -0.63;
    torus_entry.entry_duration = 2.7;
    torus_entry.duration = 8.0;
    torus_entry.torus = torus;
    result.emplace_back("torus_knot_entry3", std::make_unique<tr::TorusKnotEntryCurveEvaluator3>(torus_entry));
    return result;
}

// Five-point finite differences test the relation between independently returned
// fields. No derivative formulas from the implementation are copied here.
void checkDerivatives(const tr::TrajectoryEvaluator2& evaluator, double t, const std::string& context,
                      bool check_kinematics = true) {
    constexpr double h = 1.0e-4;
    const auto m2 = evaluate(evaluator, t - 2.0 * h);
    const auto m1 = evaluate(evaluator, t - h);
    const auto c = evaluate(evaluator, t);
    const auto p1 = evaluate(evaluator, t + h);
    const auto p2 = evaluate(evaluator, t + 2.0 * h);
    const auto derivative = [&](Eigen::Vector2d tr::PlanarReference2::*field) -> Eigen::Vector2d {
        return (m2.*field - 8.0 * m1.*field + 8.0 * p1.*field - p2.*field) / (12.0 * h);
    };
    matrixNear(c.velocity, derivative(&tr::PlanarReference2::position), 2.0e-7 * (1.0 + c.velocity.norm()),
               context + " dp/dt");
    matrixNear(c.acceleration, derivative(&tr::PlanarReference2::velocity), 2.0e-7 * (1.0 + c.acceleration.norm()),
               context + " dv/dt");
    matrixNear(c.jerk, derivative(&tr::PlanarReference2::acceleration), 2.0e-7 * (1.0 + c.jerk.norm()),
               context + " da/dt");
    const double yaw_derivative =
        (wrap(m2.yaw - c.yaw) - 8.0 * wrap(m1.yaw - c.yaw) + 8.0 * wrap(p1.yaw - c.yaw) - wrap(p2.yaw - c.yaw)) /
        (12.0 * h);
    near(c.yaw_rate, yaw_derivative, 2.0e-7 * (1.0 + std::abs(c.yaw_rate)), context + " dyaw/dt");
    near(c.yaw_acceleration, (m2.yaw_rate - 8.0 * m1.yaw_rate + 8.0 * p1.yaw_rate - p2.yaw_rate) / (12.0 * h),
         2.0e-7 * (1.0 + std::abs(c.yaw_acceleration)), context + " dyaw_rate/dt");
    if (check_kinematics) {
        near(c.linear_acceleration, (m2.speed - 8.0 * m1.speed + 8.0 * p1.speed - p2.speed) / (12.0 * h),
             2.0e-7 * (1.0 + std::abs(c.linear_acceleration)), context + " dspeed/dt");
        near(c.curvature * c.speed, c.yaw_rate, 2.0e-10, context + " curvature/yaw-rate relation");
    }
}

void checkDerivatives(const tr::TrajectoryEvaluator3& evaluator, double t, const std::string& context) {
    constexpr double h = 1.0e-4;
    const auto m2 = evaluate(evaluator, t - 2.0 * h);
    const auto m1 = evaluate(evaluator, t - h);
    const auto c = evaluate(evaluator, t);
    const auto p1 = evaluate(evaluator, t + h);
    const auto p2 = evaluate(evaluator, t + 2.0 * h);
    const auto derivative = [&](Eigen::Vector3d tr::FlatOutput3::*field) -> Eigen::Vector3d {
        return (m2.*field - 8.0 * m1.*field + 8.0 * p1.*field - p2.*field) / (12.0 * h);
    };
    matrixNear(c.velocity, derivative(&tr::FlatOutput3::position), 2.0e-7 * (1.0 + c.velocity.norm()),
               context + " dp/dt");
    matrixNear(c.acceleration, derivative(&tr::FlatOutput3::velocity), 2.0e-7 * (1.0 + c.acceleration.norm()),
               context + " dv/dt");
    matrixNear(c.jerk, derivative(&tr::FlatOutput3::acceleration), 2.0e-7 * (1.0 + c.jerk.norm()), context + " da/dt");
    matrixNear(c.snap, derivative(&tr::FlatOutput3::jerk), 2.0e-7 * (1.0 + c.snap.norm()), context + " djerk/dt");
    const double yaw_derivative =
        (wrap(m2.yaw - c.yaw) - 8.0 * wrap(m1.yaw - c.yaw) + 8.0 * wrap(p1.yaw - c.yaw) - wrap(p2.yaw - c.yaw)) /
        (12.0 * h);
    near(c.yaw_rate, yaw_derivative, 2.0e-7, context + " dyaw/dt");
    near(c.yaw_accel, (m2.yaw_rate - 8.0 * m1.yaw_rate + 8.0 * p1.yaw_rate - p2.yaw_rate) / (12.0 * h), 2.0e-7,
         context + " dyaw_rate/dt");
}

void analytic2Derivatives() {
    for (const auto& named : analytic2()) {
        for (double fraction : {0.071, 0.133, 0.231, 0.353, 0.487, 0.613, 0.739, 0.857, 0.937}) {
            const double t = fraction * named.second->duration();
            checkDerivatives(*named.second, t, named.first + " t=" + std::to_string(t));
        }
    }
}

void analytic3Derivatives() {
    for (const auto& named : analytic3()) {
        for (double fraction : {0.071, 0.133, 0.231, 0.353, 0.487, 0.613, 0.739, 0.857, 0.937}) {
            const double t = fraction * named.second->duration();
            checkDerivatives(*named.second, t, named.first + " t=" + std::to_string(t));
        }
    }
}

void sameOutput(const tr::PlanarReference2& a, const tr::PlanarReference2& b, double tolerance,
                const std::string& context) {
    matrixNear(a.position, b.position, tolerance, context + " position");
    matrixNear(a.velocity, b.velocity, tolerance, context + " velocity");
    matrixNear(a.acceleration, b.acceleration, tolerance, context + " acceleration");
    matrixNear(a.jerk, b.jerk, tolerance, context + " jerk");
    near(wrap(a.yaw - b.yaw), 0.0, tolerance, context + " yaw");
    near(a.speed, b.speed, tolerance, context + " speed");
    near(a.linear_acceleration, b.linear_acceleration, tolerance, context + " linear acceleration");
    near(a.yaw_rate, b.yaw_rate, tolerance, context + " yaw rate");
    near(a.yaw_acceleration, b.yaw_acceleration, tolerance, context + " yaw acceleration");
    near(a.curvature, b.curvature, tolerance, context + " curvature");
}

void sameOutput(const tr::FlatOutput3& a, const tr::FlatOutput3& b, double tolerance, const std::string& context) {
    matrixNear(a.position, b.position, tolerance, context + " position");
    matrixNear(a.velocity, b.velocity, tolerance, context + " velocity");
    matrixNear(a.acceleration, b.acceleration, tolerance, context + " acceleration");
    matrixNear(a.jerk, b.jerk, tolerance, context + " jerk");
    matrixNear(a.snap, b.snap, tolerance, context + " snap");
    near(wrap(a.yaw - b.yaw), 0.0, tolerance, context + " yaw");
    near(a.yaw_rate, b.yaw_rate, tolerance, context + " yaw rate");
    near(a.yaw_accel, b.yaw_accel, tolerance, context + " yaw acceleration");
}

void analyticTimeDomains() {
    for (const auto& named : analytic2()) {
        const auto& curve = *named.second;
        sameOutput(evaluate(curve, -10.0), evaluate(curve, 0.0), 0.0, named.first + " start clamp");
        sameOutput(evaluate(curve, curve.duration() + 10.0), evaluate(curve, curve.duration()), 0.0,
                   named.first + " end clamp");
        for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
            tr::PlanarReference2 output;
            require(!curve.evaluate(bad, output) && (output.flags & tr::kFlagInvalidInput),
                    named.first + " invalid time");
        }
    }
    for (const auto& named : analytic3()) {
        const auto& curve = *named.second;
        sameOutput(evaluate(curve, -10.0), evaluate(curve, 0.0), 0.0, named.first + " start clamp");
        sameOutput(evaluate(curve, curve.duration() + 10.0), evaluate(curve, curve.duration()), 0.0,
                   named.first + " end clamp");
        for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
            tr::FlatOutput3 output;
            require(!curve.evaluate(bad, output) && (output.flags & tr::kFlagInvalidInput),
                    named.first + " invalid time");
        }
    }
}

void analyticPeriodicity() {
    tr::CircleCurveParameters2 p2;
    p2.center = Eigen::Vector2d(0.7, -1.4);
    p2.radius = 1.8;
    p2.line_speed = 1.3;
    const tr::CircleCurveEvaluator2 circle2(p2);
    const double period2 = 2.0 * kPi * p2.radius / p2.line_speed;
    for (double t : {0.0, 0.13, 1.7, 4.1}) {
        const auto output = evaluate(circle2, t);
        sameOutput(output, evaluate(circle2, t + period2), 2.0e-14, "2D circle periodicity");
        near((output.position - p2.center).norm(), p2.radius, 1.0e-14, "circle radius");
        near(output.speed, p2.line_speed, 1.0e-14, "circle speed");
        near(output.curvature, 1.0 / p2.radius, 1.0e-14, "circle curvature");
    }
    const tr::TorusKnotCurveEvaluator3 torus;
    const double torus_period = 2.0 * kPi / torus.params().omega;
    sameOutput(evaluate(torus, 0.37), evaluate(torus, 0.37 + torus_period), 2.0e-14, "torus periodicity");
    const tr::HelixXyCurveEvaluator3 helix;
    const double helix_period = 2.0 * kPi / helix.params().omega;
    const auto start = evaluate(helix, 0.31);
    auto end = evaluate(helix, 0.31 + helix_period);
    end.position.z() -= helix_period / helix.params().linear_scale;
    sameOutput(start, end, 1.0e-14, "helix advances one pitch per revolution");
}

void lineBoundaries() {
    for (double duration : {0.05, 0.4, 3.0, 25.0}) {
        tr::LineCurveParameters3 params;
        params.duration = duration;
        params.start = Eigen::Vector3d(-0.7, 0.6, 1.1);
        params.target = Eigen::Vector3d(1.6, -1.2, 2.0);
        params.start_velocity = Eigen::Vector3d(0.1, -0.3, 0.2) / duration;
        params.target_velocity = Eigen::Vector3d(-0.2, 0.1, 0.3) / duration;
        const tr::LineCurveEvaluator3 curve(params);
        const auto start = evaluate(curve, 0.0), end = evaluate(curve, duration);
        matrixNear(start.position, params.start, 2.0e-13, "line start position");
        matrixNear(end.position, params.target, 2.0e-13, "line target position");
        matrixNear(duration * start.velocity, duration * params.start_velocity, 2.0e-12, "line start velocity");
        matrixNear(duration * end.velocity, duration * params.target_velocity, 2.0e-12, "line target velocity");
        matrixNear(duration * duration * start.acceleration, Eigen::Vector3d::Zero(), 2.0e-12,
                   "line start acceleration");
        matrixNear(duration * duration * end.acceleration, Eigen::Vector3d::Zero(), 2.0e-12,
                   "line target acceleration");
    }
}

void entryContinuity() {
    for (const auto& named : analytic2()) {
        const auto* entry = dynamic_cast<const tr::CircleEntryCurveEvaluator2*>(named.second.get());
        if (!entry) {
            continue;
        }
        const double t = entry->params().entry_duration;
        const auto before = evaluate(*entry, std::nextafter(t, 0.0)), after = evaluate(*entry, t);
        matrixNear(before.position, after.position, 3.0e-10, "2D circle-entry C0");
        matrixNear(before.velocity, after.velocity, 3.0e-10, "2D circle-entry C1");
        matrixNear(before.acceleration, after.acceleration, 3.0e-10, "2D circle-entry C2");
        matrixNear(before.jerk, after.jerk, 3.0e-9, "2D circle-entry C3");
        near(wrap(before.yaw - after.yaw), 0.0, 3.0e-10, "2D entry yaw continuous");
        near(before.yaw_rate, after.yaw_rate, 3.0e-10, "2D entry yaw rate continuous");
        // The cubic yaw blend promises C1; it does not promise continuous yaw
        // acceleration. The septic position blend promises C3, not C4.
    }
    for (const auto& named : analytic3()) {
        double entry_time = 0.0;
        if (const auto* c = dynamic_cast<const tr::CircleEntryCurveEvaluator3*>(named.second.get())) {
            entry_time = c->params().entry_duration;
        } else if (const auto* t = dynamic_cast<const tr::TorusKnotEntryCurveEvaluator3*>(named.second.get())) {
            entry_time = t->params().entry_duration;
        } else {
            continue;
        }
        const auto before = evaluate(*named.second, std::nextafter(entry_time, 0.0));
        const auto after = evaluate(*named.second, entry_time);
        matrixNear(before.position, after.position, 3.0e-10, named.first + " C0");
        matrixNear(before.velocity, after.velocity, 3.0e-10, named.first + " C1");
        matrixNear(before.acceleration, after.acceleration, 3.0e-10, named.first + " C2");
        matrixNear(before.jerk, after.jerk, 3.0e-9, named.first + " C3");
        near(wrap(before.yaw - after.yaw), 0.0, 3.0e-10, named.first + " yaw continuity");
        near(before.yaw_rate, after.yaw_rate, 0.0, named.first + " yaw-rate continuity");
        near(before.yaw_accel, after.yaw_accel, 0.0, named.first + " yaw-acceleration continuity");
        const auto initial = evaluate(*named.second, 0.0);
        matrixNear(initial.velocity, Eigen::Vector3d::Zero(), 0.0, named.first + " starts at rest");
        matrixNear(initial.acceleration, Eigen::Vector3d::Zero(), 0.0, named.first + " starts with zero acceleration");
        matrixNear(initial.jerk, Eigen::Vector3d::Zero(), 0.0, named.first + " starts with zero jerk");
    }
}

void torusEntryYaw() {
    for (double origin_yaw : {-3.12, -0.63, 0.0, 2.41, 7.2}) {
        tr::TorusKnotEntryCurveParameters3 params;
        params.origin_yaw = origin_yaw;
        params.torus.yaw = origin_yaw + 1.1;
        params.entry_duration = 1.7;
        const tr::TorusKnotEntryCurveEvaluator3 curve(params);
        near(curve.params().torus.yaw, origin_yaw, 0.0, "stored torus yaw must describe effective yaw hold");
        for (double time : {0.0, 0.2, 1.699999, 1.7, 1.700001, 4.0, curve.duration()}) {
            const auto output = evaluate(curve, time);
            near(wrap(output.yaw - origin_yaw), 0.0, 2.0e-15, "torus entry must hold origin yaw before and after join");
            near(output.yaw_rate, 0.0, 0.0, "torus entry yaw rate");
            near(output.yaw_accel, 0.0, 0.0, "torus entry yaw acceleration");
        }
        // Constructing the evaluator must not mutate the caller's parameter value.
        near(params.torus.yaw, origin_yaw + 1.1, 0.0, "caller-owned torus yaw");
    }
}

void septicWaypointContinuity() {
    tr::WaypointProblem2 problem;
    problem.constraints = {{Eigen::Vector2d(-0.5, 0.3), -0.2},
                           {Eigen::Vector2d(0.7, 0.9), 0.4},
                           {Eigen::Vector2d(1.3, -0.2), -0.3},
                           {Eigen::Vector2d(2.1, 0.2), 0.1}};
    problem.segment_times = {0.8, 1.7, 1.2};
    problem.start_velocity = Eigen::Vector2d(0.1, -0.2);
    problem.end_velocity = Eigen::Vector2d(0.3, 0.1);
    problem.start_acceleration = Eigen::Vector2d(-0.2, 0.1);
    problem.end_acceleration = Eigen::Vector2d(0.15, -0.1);
    tr::PiecewisePolynomialEvaluator2 curve;
    require(tr::SepticWaypointInterpolator2().solve(problem, curve), "septic waypoint solve");
    double t = 0.0;
    for (std::size_t i = 0; i < problem.constraints.size(); ++i) {
        matrixNear(evaluate(curve, t).position, problem.constraints[i].position, 1.0e-10, "waypoint interpolation");
        if (i < problem.segment_times.size()) {
            t += problem.segment_times[i];
        }
    }
    matrixNear(evaluate(curve, 0.0).velocity, problem.start_velocity, 1.0e-11, "waypoint initial velocity");
    matrixNear(evaluate(curve, 0.0).acceleration, problem.start_acceleration, 1.0e-11, "waypoint initial acceleration");
    matrixNear(evaluate(curve, t).velocity, problem.end_velocity, 1.0e-10, "waypoint terminal velocity");
    matrixNear(evaluate(curve, t).acceleration, problem.end_acceleration, 1.0e-10, "waypoint terminal acceleration");
    for (std::size_t i = 0; i < curve.segments().size(); ++i) {
        tr::PiecewisePolynomialEvaluator2 piece;
        require(piece.setSegments({curve.segments()[i]}, curve.order()), "isolate septic piece");
        checkDerivatives(piece, 0.37 * piece.duration(), "septic piece " + std::to_string(i), false);
        if (i + 1 == curve.segments().size()) {
            continue;
        }
        tr::PiecewisePolynomialEvaluator2 next;
        require(next.setSegments({curve.segments()[i + 1]}, curve.order()), "isolate next septic piece");
        const auto left = evaluate(piece, piece.duration()), right = evaluate(next, 0.0);
        matrixNear(left.position, right.position, 2.0e-10, "septic waypoint C0");
        matrixNear(left.velocity, right.velocity, 2.0e-10, "septic waypoint C1");
        matrixNear(left.acceleration, right.acceleration, 2.0e-9, "septic waypoint C2");
        matrixNear(left.jerk, right.jerk, 2.0e-8, "septic waypoint C3");
    }
    const auto clone = tr::cloneEvaluator(curve);
    require(clone != nullptr, "polynomial clone");
    sameOutput(evaluate(curve, 1.2), evaluate(*clone, 1.2), 0.0, "polynomial clone preserves reference");
    sameOutput(evaluate(curve, -1.0), evaluate(curve, 0.0), 0.0, "polynomial lower clamp");
    sameOutput(evaluate(curve, t + 1.0), evaluate(curve, t), 0.0, "polynomial upper clamp");
}

void sampledInterpolation() {
    tr::SampledPoint2 a2, b2;
    a2.t = 0.2;
    b2.t = 2.6;
    a2.reference.position = Eigen::Vector2d(1.0, -2.0);
    b2.reference.position = Eigen::Vector2d(4.0, 1.0);
    a2.reference.velocity = Eigen::Vector2d(0.3, 0.4);
    b2.reference.velocity = Eigen::Vector2d(-0.5, 0.7);
    a2.reference.yaw = kPi - 0.1;
    b2.reference.yaw = -kPi + 0.1;
    a2.reference.speed = -0.4;
    b2.reference.speed = -0.8;
    a2.reference.yaw_rate = 0.2;
    b2.reference.yaw_rate = 0.6;
    tr::SampledEvaluator2 sampled2;
    require(sampled2.setSamples({b2, a2}, true), "2D samples accept unsorted input and preserve explicit signed speed");
    const auto middle2 = evaluate(sampled2, 1.4);
    matrixNear(middle2.position, Eigen::Vector2d(2.5, -0.5), 2.0e-15,
               "2D interpolation with non-unit timestamp spacing");
    matrixNear(middle2.velocity, Eigen::Vector2d(-0.1, 0.55), 2.0e-15, "2D velocity interpolation");
    near(wrap(middle2.yaw - kPi), 0.0, 2.0e-15, "2D interpolation takes shortest yaw arc across pi");
    near(middle2.speed, -0.6, 2.0e-15, "2D interpolation retains reverse branch");
    near(middle2.yaw_rate, 0.4, 2.0e-15, "2D explicit yaw-rate interpolation");
    sameOutput(evaluate(sampled2, -1.0), a2.reference, 0.0, "2D sample lower clamp");
    sameOutput(evaluate(sampled2, 10.0), b2.reference, 0.0, "2D sample upper clamp");
    const auto clone2 = tr::cloneEvaluator(sampled2);
    require(clone2 != nullptr, "2D sampled clone");
    require(sampled2.setSamples({a2}, true), "2D replace with single sample");
    sameOutput(evaluate(*clone2, 1.4), middle2, 0.0, "2D clone independent of subsequent source replacement");
    require(!sampled2.setSamples({a2, a2}, true), "2D duplicate timestamps must be rejected");

    tr::SampledPoint3 a3, b3;
    a3.t = 0.2;
    b3.t = 2.6;
    a3.flat.position = Eigen::Vector3d(1.0, -2.0, 0.5);
    b3.flat.position = Eigen::Vector3d(4.0, 1.0, 1.5);
    a3.flat.velocity = Eigen::Vector3d(0.3, -0.4, 0.8);
    b3.flat.velocity = Eigen::Vector3d(-0.5, 0.7, 0.2);
    a3.flat.acceleration = Eigen::Vector3d(0.2, 0.4, -0.1);
    b3.flat.acceleration = Eigen::Vector3d(0.6, -0.2, 0.3);
    a3.flat.jerk = Eigen::Vector3d(0.1, -0.3, 0.5);
    b3.flat.jerk = Eigen::Vector3d(0.7, 0.1, 0.3);
    a3.flat.snap = Eigen::Vector3d(-0.1, 0.2, 0.7);
    b3.flat.snap = Eigen::Vector3d(0.3, 0.8, -0.1);
    a3.flat.yaw = kPi - 0.1;
    b3.flat.yaw = -kPi + 0.1;
    a3.flat.yaw_rate = 0.2;
    b3.flat.yaw_rate = 0.6;
    a3.flat.yaw_accel = -0.3;
    b3.flat.yaw_accel = 0.7;
    a3.flat.flags = tr::kFlagVelocityLimit;
    b3.flat.flags = tr::kFlagAccelerationLimit;
    tr::SampledEvaluator3 sampled3;
    require(sampled3.setSamples({a3, b3}), "3D sample setup");
    const auto middle3 = evaluate(sampled3, 1.4);
    matrixNear(middle3.position, Eigen::Vector3d(2.5, -0.5, 1.0), 2.0e-15, "3D position interpolation");
    matrixNear(middle3.velocity, Eigen::Vector3d(-0.1, 0.15, 0.5), 2.0e-15, "3D velocity interpolation");
    matrixNear(middle3.acceleration, Eigen::Vector3d(0.4, 0.1, 0.1), 2.0e-15, "3D acceleration interpolation");
    matrixNear(middle3.jerk, Eigen::Vector3d(0.4, -0.1, 0.4), 2.0e-15, "3D jerk interpolation");
    matrixNear(middle3.snap, Eigen::Vector3d(0.1, 0.5, 0.3), 2.0e-15, "3D snap interpolation");
    near(wrap(middle3.yaw - kPi), 0.0, 2.0e-15, "3D shortest yaw arc");
    near(middle3.yaw_rate, 0.4, 2.0e-15, "3D yaw-rate interpolation");
    near(middle3.yaw_accel, 0.2, 2.0e-15, "3D yaw-acceleration interpolation");
    require(middle3.flags == (a3.flat.flags | b3.flat.flags), "3D interpolation unions source flags");
    sameOutput(evaluate(sampled3, -1.0), a3.flat, 0.0, "3D sample lower clamp");
    sameOutput(evaluate(sampled3, 10.0), b3.flat, 0.0, "3D sample upper clamp");
    const auto clone3 = tr::cloneEvaluator(sampled3);
    require(clone3 != nullptr && sampled3.setSamples({a3}), "3D sampled clone and replacement");
    sameOutput(evaluate(*clone3, 1.4), middle3, 0.0, "3D clone independent of source replacement");
    tr::SampledEvaluator3 invalid3;
    require(!invalid3.setSamples({b3, a3}), "3D decreasing timestamps must be rejected");
    b3.flat.jerk.x() = std::numeric_limits<double>::quiet_NaN();
    require(!invalid3.setSamples({a3, b3}), "nonfinite sampled derivative must be rejected");
}

void flatnessDynamics() {
    constexpr double gravity = 9.8066;
    constexpr double h = 2.0e-5;
    const tr::FlatnessMapper3 mapper(gravity);
    for (const auto& named : analytic3()) {
        for (double t : {0.41, 1.13, 1.91, 3.17, 4.63, 6.29, 7.31}) {
            const auto flat = evaluate(*named.second, t);
            const auto state = mapper.map(flat);
            const auto before = mapper.map(evaluate(*named.second, t - h));
            const auto after = mapper.map(evaluate(*named.second, t + h));
            const auto rotation = state.attitude.toRotationMatrix();
            require((state.flags & (tr::kFlagNonFinite | tr::kFlagLowThrust | tr::kFlagYawSingularity)) == 0U,
                    named.first + " unexpected flatness singularity");
            matrixNear(state.position, flat.position, 0.0, named.first + " flatness position");
            matrixNear(state.velocity, flat.velocity, 0.0, named.first + " flatness velocity");
            near(state.attitude.norm(), 1.0, 2.0e-15, named.first + " unit attitude");
            require(state.attitude.w() >= 0.0, named.first + " quaternion canonical sign");
            matrixNear(rotation.transpose() * rotation, Eigen::Matrix3d::Identity(), 3.0e-15,
                       named.first + " orthogonal flatness frame");
            near(rotation.determinant(), 1.0, 3.0e-15, named.first + " right-handed flatness frame");
            matrixNear(state.specific_thrust * rotation.col(2) - gravity * Eigen::Vector3d::UnitZ(), flat.acceleration,
                       1.0e-13, named.first + " reconstructed translational dynamics");
            const Eigen::Matrix3d rotation_dot =
                (after.attitude.toRotationMatrix() - before.attitude.toRotationMatrix()) / (2.0 * h);
            const Eigen::Matrix3d omega_hat = rotation.transpose() * rotation_dot;
            const Eigen::Vector3d numerical_rate(omega_hat(2, 1), omega_hat(0, 2), omega_hat(1, 0));
            matrixNear(state.body_rate, numerical_rate, 3.0e-6 * (1.0 + state.body_rate.norm()),
                       named.first + " body rate differentiates returned rotation");
            matrixNear(state.angular_acceleration, (after.body_rate - before.body_rate) / (2.0 * h),
                       3.0e-6 * (1.0 + state.angular_acceleration.norm()), named.first + " angular acceleration");
        }
    }
    tr::FlatOutput3 freefall{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, -gravity)};
    require((mapper.map(freefall).flags & tr::kFlagLowThrust) != 0U, "free fall must signal low thrust");
    freefall.jerk.x() = std::numeric_limits<double>::quiet_NaN();
    require((mapper.map(freefall).flags & tr::kFlagNonFinite) != 0U,
            "invalid flatness derivative must signal nonfinite");
}

void trajectoryValidation() {
    tr::CircleCurveParameters2 params2;
    params2.radius = 2.0;
    params2.line_speed = 1.2;
    params2.duration = 3.0;
    const tr::CircleCurveEvaluator2 curve2(params2);
    tr::TrajectoryLimits2 loose2;
    loose2.max_velocity = 2.0;
    loose2.max_acceleration = 2.0;
    loose2.max_yaw_rate = 2.0;
    require(tr::TrajectoryValidator2::validate(curve2, loose2, 0.07) == tr::kFlagNone, "feasible 2D limits");
    const tr::TrajectoryLimits2 tight2{0.5, 0.3, 0.2};
    const auto flags2 = tr::TrajectoryValidator2::validate(curve2, tight2, 0.07);
    require(flags2 == (tr::kFlagVelocityLimit | tr::kFlagYawRateLimit),
            "2D circle limits apply to longitudinal acceleration, not centripetal acceleration; flags=" +
                std::to_string(flags2));
    tr::PolynomialSegment2 accelerating_line;
    accelerating_line.duration = 1.0;
    accelerating_line.x = {0.0, 1.0, 0.5};
    accelerating_line.y = {0.0, 0.0, 0.0};
    tr::PiecewisePolynomialEvaluator2 line2;
    require(line2.setSegments({accelerating_line}, 2U), "constant longitudinal acceleration setup");
    const tr::TrajectoryLimits2 acceleration_limit{10.0, 0.3, 2.0};
    const auto acceleration_flags = tr::TrajectoryValidator2::validate(line2, acceleration_limit, 0.07);
    require(acceleration_flags == tr::kFlagAccelerationLimit,
            "constant longitudinal acceleration must violate the 2D acceleration limit; flags=" +
                std::to_string(acceleration_flags));

    tr::CircleCurveParameters3 params3;
    params3.radius = 1.3;
    params3.line_speed = 1.1;
    params3.duration = 2.7;
    const tr::CircleCurveEvaluator3 curve3(params3);
    tr::TrajectoryLimits3 limits3;
    const auto check_limit = [&](double tr::TrajectoryLimits3::*field, uint32_t flag, const std::string& name) {
        tr::TrajectoryLimits3 tight = limits3;
        tight.*field = 1.0e-3;
        const auto result = tr::TrajectoryValidator3::validate(curve3, tight, 0.07);
        require(result == flag, name + ": wrong limit flags=" + std::to_string(result));
    };
    require(tr::TrajectoryValidator3::validate(curve3, limits3, 0.07) == tr::kFlagNone, "feasible 3D limits");
    check_limit(&tr::TrajectoryLimits3::max_velocity, tr::kFlagVelocityLimit, "velocity");
    check_limit(&tr::TrajectoryLimits3::max_acceleration, tr::kFlagAccelerationLimit, "acceleration");
    check_limit(&tr::TrajectoryLimits3::max_jerk, tr::kFlagJerkLimit, "jerk");
    check_limit(&tr::TrajectoryLimits3::max_snap, tr::kFlagSnapLimit, "snap");
    check_limit(&tr::TrajectoryLimits3::max_body_rate, tr::kFlagBodyRateLimit, "body rate");
    check_limit(&tr::TrajectoryLimits3::max_tilt, tr::kFlagTiltLimit, "tilt");
    check_limit(&tr::TrajectoryLimits3::max_specific_thrust, tr::kFlagThrustLimit, "thrust");
}

double fallingFactorial(int degree, int derivative) {
    double value = 1.0;
    for (int d = 0; d < derivative; ++d) {
        value *= degree - d;
    }
    return value;
}

// Independent dense constrained minimization: minimize the integrated squared
// s-th derivative subject only to endpoint derivatives, waypoint interpolation,
// and C^(s-1) continuity. MINCO's banded optimality/continuity system is not used.
template <int S>
Eigen::MatrixX3d minimumDerivativeOracle(const Eigen::Matrix<double, 3, S>& head,
                                         const Eigen::Matrix<double, 3, S>& tail, const Eigen::Matrix3Xd& points,
                                         const Eigen::VectorXd& times) {
    constexpr int coefficients = 2 * S;
    const int pieces = static_cast<int>(times.size());
    const int variables = coefficients * pieces;
    const int constraints = 2 * S + (pieces - 1) * (S + 1);
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(variables, variables);
    Eigen::MatrixXd rows = Eigen::MatrixXd::Zero(constraints, variables);
    Eigen::MatrixX3d values = Eigen::MatrixX3d::Zero(constraints, 3);
    for (int piece = 0; piece < pieces; ++piece) {
        for (int i = S; i < coefficients; ++i) {
            for (int j = S; j < coefficients; ++j) {
                const int power = i + j - 2 * S + 1;
                hessian(coefficients * piece + i, coefficients * piece + j) =
                    2.0 * fallingFactorial(i, S) * fallingFactorial(j, S) * std::pow(times(piece), power) / power;
            }
        }
    }
    const auto add_evaluation = [&](int row, int piece, int derivative, double t, double sign) {
        for (int degree = derivative; degree < coefficients; ++degree) {
            rows(row, piece * coefficients + degree) +=
                sign * fallingFactorial(degree, derivative) * std::pow(t, degree - derivative);
        }
    };
    int row = 0;
    for (int d = 0; d < S; ++d) {
        add_evaluation(row, 0, d, 0.0, 1.0);
        values.row(row++) = head.col(d).transpose();
        add_evaluation(row, pieces - 1, d, times(pieces - 1), 1.0);
        values.row(row++) = tail.col(d).transpose();
    }
    for (int piece = 0; piece + 1 < pieces; ++piece) {
        add_evaluation(row, piece, 0, times(piece), 1.0);
        values.row(row++) = points.col(piece).transpose();
        add_evaluation(row, piece + 1, 0, 0.0, 1.0);
        values.row(row++) = points.col(piece).transpose();
        for (int d = 1; d < S; ++d) {
            add_evaluation(row, piece, d, times(piece), 1.0);
            add_evaluation(row++, piece + 1, d, 0.0, -1.0);
        }
    }
    require(row == constraints, "oracle constraint count");
    Eigen::MatrixXd kkt = Eigen::MatrixXd::Zero(variables + constraints, variables + constraints);
    kkt.topLeftCorner(variables, variables) = hessian;
    kkt.topRightCorner(variables, constraints) = rows.transpose();
    kkt.bottomLeftCorner(constraints, variables) = rows;
    Eigen::MatrixX3d rhs = Eigen::MatrixX3d::Zero(variables + constraints, 3);
    rhs.bottomRows(constraints) = values;
    const Eigen::MatrixX3d solution = kkt.fullPivLu().solve(rhs);
    matrixNear(kkt * solution, rhs, 2.0e-8 * (1.0 + rhs.norm()), "independent dense KKT residual");
    return solution.topRows(variables);
}

template <int S> using MincoSolver = std::conditional_t<S == 2, minco::MINCO_S2NU, minco::MINCO_S3NU>;

template <int S> double quadratureEnergy(const xgc2_math::optimization::Trajectory<2 * S - 1>& trajectory) {
    // Three Gauss-Legendre nodes integrate squared linear acceleration (S2) or
    // squared quadratic jerk (S3) exactly; this bypasses getEnergy's formula.
    const double node = std::sqrt(3.0 / 5.0);
    const std::array<double, 3> nodes{{-node, 0.0, node}};
    const std::array<double, 3> weights{{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}};
    double energy = 0.0;
    for (int i = 0; i < trajectory.getPieceNum(); ++i) {
        const auto& piece = trajectory[i];
        for (std::size_t n = 0; n < nodes.size(); ++n) {
            const double t = 0.5 * piece.getDuration() * (nodes[n] + 1.0);
            Eigen::Vector3d derivative;
            if constexpr (S == 2) {
                derivative = piece.getAcc(t);
            } else {
                derivative = piece.getJer(t);
            }
            energy += 0.5 * piece.getDuration() * weights[n] * derivative.squaredNorm();
        }
    }
    return energy;
}

template <int S> void mincoOracle() {
    for (int count : {1, 2, 4}) {
        Eigen::Matrix<double, 3, S> head = Eigen::Matrix<double, 3, S>::Zero();
        Eigen::Matrix<double, 3, S> tail = Eigen::Matrix<double, 3, S>::Zero();
        head.col(0) = Eigen::Vector3d(-0.2, 0.4, 1.1);
        head.col(1) = Eigen::Vector3d(0.1, -0.2, 0.15);
        tail.col(0) = Eigen::Vector3d(2.3, -1.2, 0.7);
        tail.col(1) = Eigen::Vector3d(-0.2, 0.1, 0.25);
        if constexpr (S == 3) {
            head.col(2) = Eigen::Vector3d(0.05, 0.03, -0.04);
            tail.col(2) = Eigen::Vector3d(-0.03, 0.08, 0.02);
        }
        Eigen::VectorXd times(count);
        Eigen::Matrix3Xd points(3, count - 1);
        for (int i = 0; i < count; ++i) {
            times(i) = 0.8 + 0.3 * i;
        }
        for (int i = 0; i + 1 < count; ++i) {
            points.col(i) = Eigen::Vector3d(0.5 + 0.4 * i, std::sin(0.8 * i + 0.3), 1.3 - 0.3 * i);
        }
        MincoSolver<S> solver;
        solver.setConditions(head, tail, count);
        solver.setParameters(points, times);
        const auto oracle = minimumDerivativeOracle<S>(head, tail, points, times);
        matrixNear(solver.getCoeffs(), oracle, 2.0e-8 * (1.0 + oracle.norm()),
                   "MINCO S" + std::to_string(S) + " dense variational oracle, pieces=" + std::to_string(count));
        xgc2_math::optimization::Trajectory<2 * S - 1> trajectory;
        solver.getTrajectory(trajectory);
        require(trajectory.getPieceNum() == count, "MINCO output piece count");
        matrixNear(trajectory.getDurations(), times, 0.0, "MINCO duration preservation");
        matrixNear(trajectory.getPos(0.0), head.col(0), 2.0e-11, "MINCO initial position");
        matrixNear(trajectory.getVel(0.0), head.col(1), 2.0e-11, "MINCO initial velocity");
        matrixNear(trajectory.getPos(times.sum()), tail.col(0), 2.0e-10, "MINCO terminal position");
        matrixNear(trajectory.getVel(times.sum()), tail.col(1), 2.0e-10, "MINCO terminal velocity");
        for (int i = 0; i + 1 < count; ++i) {
            matrixNear(trajectory[i].getPos(times(i)), points.col(i), 2.0e-10, "MINCO left waypoint");
            matrixNear(trajectory[i + 1].getPos(0.0), points.col(i), 2.0e-10, "MINCO right waypoint");
            matrixNear(trajectory[i].getVel(times(i)), trajectory[i + 1].getVel(0.0), 2.0e-10,
                       "MINCO velocity continuity");
            matrixNear(trajectory[i].getAcc(times(i)), trajectory[i + 1].getAcc(0.0), 2.0e-10,
                       "MINCO acceleration continuity");
            if constexpr (S == 3) {
                matrixNear(trajectory[i].getJer(times(i)), trajectory[i + 1].getJer(0.0), 2.0e-9,
                           "MINCO jerk continuity");
            }
        }
        double energy = 0.0;
        solver.getEnergy(energy);
        near(energy, quadratureEnergy<S>(trajectory), 2.0e-10 * (1.0 + energy), "MINCO energy versus Gauss quadrature");
        require(energy >= 0.0, "minimum-derivative energy cannot be negative");
    }
}

template <int S> void mincoGradients() {
    constexpr int count = 4;
    Eigen::Matrix<double, 3, S> head = Eigen::Matrix<double, 3, S>::Zero();
    Eigen::Matrix<double, 3, S> tail = Eigen::Matrix<double, 3, S>::Zero();
    head.col(0) = Eigen::Vector3d(-0.2, 0.4, 0.7);
    tail.col(0) = Eigen::Vector3d(2.3, -1.2, 1.1);
    head.col(1) = Eigen::Vector3d(0.1, -0.1, 0.03);
    tail.col(1) = Eigen::Vector3d(-0.05, 0.1, 0.2);
    if constexpr (S == 3) {
        head.col(2) = Eigen::Vector3d(0.1, -0.02, 0.05);
        tail.col(2) = Eigen::Vector3d(-0.03, 0.06, -0.1);
    }
    Eigen::VectorXd times(count);
    times(0) = 0.9;
    times(1) = 1.4;
    times(2) = 1.1;
    times(3) = 1.8;
    Eigen::Matrix3Xd points(3, count - 1);
    points(0, 0) = 0.3;
    points(0, 1) = 1.1;
    points(0, 2) = 1.7;
    points(1, 0) = 0.8;
    points(1, 1) = -0.2;
    points(1, 2) = 0.3;
    points(2, 0) = 1.0;
    points(2, 1) = 1.6;
    points(2, 2) = 0.9;
    MincoSolver<S> solver;
    solver.setConditions(head, tail, count);
    solver.setParameters(points, times);
    Eigen::MatrixX3d partial_c;
    Eigen::VectorXd partial_t, gradient_t;
    Eigen::Matrix3Xd gradient_p;
    solver.getEnergyPartialGradByCoeffs(partial_c);
    solver.getEnergyPartialGradByTimes(partial_t);
    solver.propogateGrad(partial_c, partial_t, gradient_p, gradient_t);
    const auto objective = [&](const Eigen::Matrix3Xd& p, const Eigen::VectorXd& t) {
        MincoSolver<S> perturbed;
        perturbed.setConditions(head, tail, count);
        perturbed.setParameters(p, t);
        xgc2_math::optimization::Trajectory<2 * S - 1> trajectory;
        perturbed.getTrajectory(trajectory);
        return quadratureEnergy<S>(trajectory);
    };
    constexpr double h = 5.0e-6;
    for (int col = 0; col < count - 1; ++col) {
        for (int row = 0; row < 3; ++row) {
            Eigen::Matrix3Xd plus = points, minus = points;
            plus(row, col) += h;
            minus(row, col) -= h;
            const double numerical = (objective(plus, times) - objective(minus, times)) / (2.0 * h);
            near(gradient_p(row, col), numerical, 2.0e-5 + 3.0e-6 * std::abs(numerical),
                 "MINCO waypoint adjoint S" + std::to_string(S) + " row=" + std::to_string(row) +
                     " col=" + std::to_string(col));
        }
    }
    for (int piece = 0; piece < count; ++piece) {
        Eigen::VectorXd plus = times, minus = times;
        plus(piece) += h;
        minus(piece) -= h;
        const double numerical = (objective(points, plus) - objective(points, minus)) / (2.0 * h);
        near(gradient_t(piece), numerical, 2.0e-5 + 3.0e-6 * std::abs(numerical),
             "MINCO total time adjoint S" + std::to_string(S) + " piece=" + std::to_string(piece));
    }
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 15> tests{{
        {"analytic_2d_derivatives", analytic2Derivatives},
        {"analytic_3d_derivatives", analytic3Derivatives},
        {"analytic_time_domains", analyticTimeDomains},
        {"analytic_periodicity", analyticPeriodicity},
        {"line_boundaries", lineBoundaries},
        {"entry_continuity", entryContinuity},
        {"torus_entry_yaw", torusEntryYaw},
        {"septic_waypoint_continuity", septicWaypointContinuity},
        {"sampled_interpolation", sampledInterpolation},
        {"flatness_dynamics", flatnessDynamics},
        {"trajectory_validation", trajectoryValidation},
        {"minco_s2_oracle", mincoOracle<2>},
        {"minco_s3_oracle", mincoOracle<3>},
        {"minco_s2_gradients", mincoGradients<2>},
        {"minco_s3_gradients", mincoGradients<3>},
    }};
    if (argc > 2) {
        std::cerr << "Usage: trajectory_property_test [case_name|--list]\n";
        return 2;
    }
    const std::string requested = argc == 2 ? argv[1] : "";
    if (requested == "--list") {
        for (const auto& test : tests) {
            std::cout << test.name << '\n';
        }
        return 0;
    }
    int failures = 0;
    int executed = 0;
    for (const auto& test : tests) {
        if (!requested.empty() && requested != test.name) {
            continue;
        }
        ++executed;
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    if (executed == 0) {
        std::cerr << "Unknown case: " << requested << " (use --list)\n";
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
