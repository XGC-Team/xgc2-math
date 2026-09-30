#pragma once

#include "xgc2_math/optimization/lbfgs.hpp"
#include "xgc2_math/optimization/minco.hpp"
#include "xgc2_math/trajectory/trajectory2.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace xgc2_math::trajectory {

struct Se2TargetState2 {
    Eigen::Vector2d position{Eigen::Vector2d::Zero()};
    double yaw{0.0};
    double speed{0.0};
};

struct Se2TargetTrajectoryOptions2 {
    int piece_count{4};
    double sample_dt{0.01};
    double desired_speed{1.0};
    double min_boundary_speed{0.15};
    double approach_speed{0.4};
    double hold_duration{0.5};
    double max_duration{20.0};
    double min_segment_time{0.1};
    double max_segment_time{8.0};
    double time_weight{1.0};
    double dynamic_penalty_weight{1000.0};
    int max_iterations{80};
    double rel_cost_tol{1.0e-5};
    double validation_sample_dt{0.02};
    double position_tolerance{0.03};
    double max_velocity{3.0};
    double max_acceleration{2.0};
    double max_yaw_rate{2.5};
};

struct Se2TargetTrajectoryResult2 {
    std::vector<SampledPoint2> samples;
    uint32_t flags{kFlagNone};
    double duration{0.0};
    bool optimized{false};
    bool fallback_hold{false};
};

class Se2MincoTargetPlanner2 final {
  public:
    // Pose-to-pose sampled reference. Long, well-aligned hops may use MINCO; otherwise
    // a complete differential-drive TST (turn in place, straight or reverse, turn)
    // so any relative SE2 pose in the workspace stays kinematically feasible.
    bool plan(const Se2TargetState2& start, const Se2TargetState2& target, const Se2TargetTrajectoryOptions2& options,
              Se2TargetTrajectoryResult2& result) const;
};

} // namespace xgc2_math::trajectory

namespace xgc2_math::trajectory::se2_target_detail {

constexpr double kMinDuration = 1.0e-6;
constexpr double kMinNorm = 1.0e-9;

inline bool finiteScalar(double value) {
    return std::isfinite(value);
}

inline bool finiteVector(const Eigen::Vector2d& value) {
    return value.array().isFinite().all();
}

inline double clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(max_value, value));
}

inline double wrapAngle(double value) {
    return std::atan2(std::sin(value), std::cos(value));
}

inline Eigen::Vector2d unitFromYaw(double yaw) {
    return Eigen::Vector2d(std::cos(yaw), std::sin(yaw));
}

inline void forwardT(const Eigen::VectorXd& tau, Eigen::VectorXd& times) {
    times.resize(tau.size());
    for (int i = 0; i < tau.size(); ++i) {
        times(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0) : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
}

inline void backwardT(const Eigen::VectorXd& times, Eigen::VectorXd& tau) {
    tau.resize(times.size());
    for (int i = 0; i < times.size(); ++i) {
        const double time = std::max(kMinDuration, times(i));
        tau(i) = time > 1.0 ? std::sqrt(2.0 * time - 1.0) - 1.0 : 1.0 - std::sqrt(2.0 / time - 1.0);
    }
}

inline double positiveLimit(double configured, double fallback) {
    return finiteScalar(configured) && configured > 0.0 ? configured : fallback;
}

inline double softLimitPenalty(double value, double limit) {
    if (limit <= 0.0 || value <= limit) {
        return 0.0;
    }
    const double violation = value - limit;
    return violation * violation;
}

inline double polyValue(const Eigen::MatrixX3d& coeffs, int piece, int dim, double t, int derivative) {
    const int row = piece * 4;
    if (derivative == 0) {
        return coeffs(row, dim) + coeffs(row + 1, dim) * t + coeffs(row + 2, dim) * t * t +
               coeffs(row + 3, dim) * t * t * t;
    }
    if (derivative == 1) {
        return coeffs(row + 1, dim) + 2.0 * coeffs(row + 2, dim) * t + 3.0 * coeffs(row + 3, dim) * t * t;
    }
    if (derivative == 2) {
        return 2.0 * coeffs(row + 2, dim) + 6.0 * coeffs(row + 3, dim) * t;
    }
    if (derivative == 3) {
        return 6.0 * coeffs(row + 3, dim);
    }
    return 0.0;
}

inline bool appendSample(std::vector<SampledPoint2>& samples, SampledPoint2 sample) {
    if (!finiteScalar(sample.t) || !TrajectoryValidator2::finite(sample.reference)) {
        return false;
    }
    if (!samples.empty() && sample.t <= samples.back().t + 1.0e-9) {
        sample.t = samples.back().t + 1.0e-6;
    }
    samples.push_back(std::move(sample));
    return true;
}

inline uint32_t validateSamples(const std::vector<SampledPoint2>& samples, const Se2TargetTrajectoryOptions2& options) {
    SampledEvaluator2 evaluator;
    const bool preserve = std::any_of(samples.begin(), samples.end(), [](const SampledPoint2& sample) {
        return (sample.reference.flags & kFlagExplicitPlanarKinematics) != 0U || sample.reference.speed < 0.0;
    });
    if (!evaluator.setSamples(samples, preserve)) {
        return kFlagInvalidInput;
    }
    TrajectoryLimits2 limits;
    limits.max_velocity = options.max_velocity;
    limits.max_acceleration = options.max_acceleration;
    limits.max_yaw_rate = options.max_yaw_rate;
    uint32_t flags = kFlagNone;
    for (const auto& sample : samples) {
        if (!TrajectoryValidator2::finite(sample.reference)) {
            flags |= kFlagNonFinite;
            continue;
        }
        flags |= trajectory2_detail::limitFlags(sample.reference, limits);
    }
    const double validation_dt =
        std::min(positiveLimit(options.sample_dt, 0.01), positiveLimit(options.validation_sample_dt, 0.02));
    return flags | TrajectoryValidator2::validate(evaluator, limits, validation_dt);
}

inline void appendHold(const Se2TargetState2& state, double start_t, double duration, double sample_dt,
                       std::vector<SampledPoint2>& samples) {
    const double dt = positiveLimit(sample_dt, 0.01);
    const double total = std::max(dt, duration);
    const int count = std::max(1, static_cast<int>(std::ceil(total / dt)));
    for (int i = 0; i <= count; ++i) {
        if (!samples.empty() && i == 0) {
            continue;
        }
        SampledPoint2 sample;
        sample.t = start_t + total * static_cast<double>(i) / static_cast<double>(count);
        sample.reference.position = state.position;
        sample.reference.yaw = wrapAngle(state.yaw);
        sample.reference.speed = 0.0;
        sample.reference.flags |= kFlagLowSpeedSingularity;
        appendSample(samples, std::move(sample));
    }
}

inline double sampleStartT(const std::vector<SampledPoint2>& samples) {
    return samples.empty() ? 0.0 : samples.back().t;
}

inline bool appendYawSegment(const Eigen::Vector2d& position, double start_yaw, double target_yaw,
                             const Se2TargetTrajectoryOptions2& options, std::vector<SampledPoint2>& samples) {
    const double max_yaw_rate = positiveLimit(options.max_yaw_rate, 1.0);
    const double dt = positiveLimit(options.sample_dt, 0.01);
    const double yaw_delta = wrapAngle(target_yaw - start_yaw);
    if (std::abs(yaw_delta) <= 0.02) {
        return true;
    }
    const double duration = std::max(0.5, 2.0 * std::abs(yaw_delta) / max_yaw_rate);
    const int count = std::max(1, static_cast<int>(std::ceil(duration / dt)));
    const double t_offset = sampleStartT(samples);
    for (int i = 0; i <= count; ++i) {
        if (!samples.empty() && i == 0) {
            continue;
        }
        const double t = duration * static_cast<double>(i) / static_cast<double>(count);
        const double s = t / duration;
        const double s2 = s * s;
        const double s3 = s2 * s;
        const double s4 = s3 * s;
        const double s5 = s4 * s;
        const double blend = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
        const double blend_dot = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / duration;
        const double blend_ddot = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (duration * duration);
        SampledPoint2 sample;
        sample.t = t_offset + t;
        sample.reference.position = position;
        sample.reference.yaw = wrapAngle(start_yaw + yaw_delta * blend);
        sample.reference.yaw_rate = yaw_delta * blend_dot;
        sample.reference.yaw_acceleration = yaw_delta * blend_ddot;
        sample.reference.speed = 0.0;
        sample.reference.flags |= kFlagLowSpeedSingularity | kFlagExplicitPlanarKinematics;
        if (!appendSample(samples, std::move(sample))) {
            return false;
        }
    }
    return true;
}

inline void appendYawHold(const Se2TargetState2& start, const Se2TargetState2& target,
                          const Se2TargetTrajectoryOptions2& options, std::vector<SampledPoint2>& samples) {
    const double dt = positiveLimit(options.sample_dt, 0.01);
    (void)appendYawSegment(start.position, start.yaw, target.yaw, options, samples);
    Se2TargetState2 hold = target;
    hold.position = start.position;
    appendHold(hold, samples.empty() ? 0.0 : samples.back().t, options.hold_duration, dt, samples);
}

inline bool appendTrapezoidChord(const Eigen::Vector2d& from, const Eigen::Vector2d& to, double body_yaw,
                                 const Se2TargetTrajectoryOptions2& options, bool reverse,
                                 std::vector<SampledPoint2>& samples) {
    const Eigen::Vector2d delta = to - from;
    const double dist = delta.norm();
    if (dist <= std::max(1.0e-4, options.position_tolerance)) {
        return true;
    }
    const Eigen::Vector2d dir = delta / dist;
    const double dt = positiveLimit(options.sample_dt, 0.01);
    const double max_velocity = positiveLimit(options.max_velocity, 3.0);
    const double a_max = 0.95 * positiveLimit(options.max_acceleration, 2.0);
    const double v_des = clamp(positiveLimit(options.desired_speed, 1.0), 0.05, max_velocity);
    double t_acc = v_des / a_max;
    double v_peak = v_des;
    double t_cruise = 0.0;
    const double d_acc = 0.5 * a_max * t_acc * t_acc;
    if (2.0 * d_acc >= dist) {
        t_acc = std::sqrt(std::max(dist / a_max, 0.0));
        v_peak = a_max * t_acc;
        t_cruise = 0.0;
    } else {
        t_cruise = (dist - 2.0 * d_acc) / v_des;
    }
    const double t_move = 2.0 * t_acc + t_cruise;
    const int count = std::max(1, static_cast<int>(std::ceil(t_move / dt)));
    const double t_offset = sampleStartT(samples);
    const double speed_sign = reverse ? -1.0 : 1.0;
    for (int i = 0; i <= count; ++i) {
        if (!samples.empty() && i == 0) {
            continue;
        }
        const double t = t_move * static_cast<double>(i) / static_cast<double>(count);
        double s;
        double v = 0.0;
        double a = 0.0;
        if (t < t_acc) {
            v = a_max * t;
            s = 0.5 * a_max * t * t;
            a = a_max;
        } else if (t < t_acc + t_cruise) {
            v = v_peak;
            s = d_acc + v_peak * (t - t_acc);
            a = 0.0;
        } else {
            const double tau = t - t_acc - t_cruise;
            v = std::max(0.0, v_peak - a_max * tau);
            const double remaining = std::max(0.0, t_acc - tau);
            s = dist - 0.5 * a_max * remaining * remaining;
            a = -a_max;
        }
        SampledPoint2 sample;
        sample.t = t_offset + t;
        sample.reference.position = from + dir * s;
        sample.reference.velocity = dir * v;
        sample.reference.acceleration = dir * a;
        sample.reference.yaw = wrapAngle(body_yaw);
        sample.reference.speed = speed_sign * v;
        sample.reference.linear_acceleration = speed_sign * a;
        sample.reference.flags |= kFlagExplicitPlanarKinematics;
        if (v < 1.0e-6) {
            sample.reference.flags |= kFlagLowSpeedSingularity;
        }
        if (!appendSample(samples, std::move(sample))) {
            return false;
        }
    }
    return true;
}

inline bool appendDecelToStop(const Se2TargetState2& start, const Se2TargetTrajectoryOptions2& options,
                              Se2TargetState2& stopped, std::vector<SampledPoint2>& samples) {
    stopped = start;
    if (!finiteScalar(start.speed) || std::abs(start.speed) <= 0.05) {
        stopped.speed = 0.0;
        return true;
    }
    const double a_max = 0.95 * positiveLimit(options.max_acceleration, 2.0);
    const double dt = positiveLimit(options.sample_dt, 0.01);
    const double v0 = start.speed;
    const double duration = std::abs(v0) / a_max;
    const int count = std::max(1, static_cast<int>(std::ceil(duration / dt)));
    const Eigen::Vector2d heading = unitFromYaw(start.yaw);
    const double t_offset = sampleStartT(samples);
    for (int i = 0; i <= count; ++i) {
        if (!samples.empty() && i == 0) {
            continue;
        }
        const double t = duration * static_cast<double>(i) / static_cast<double>(count);
        const double accel = (v0 >= 0.0 ? -a_max : a_max);
        const double v = v0 + accel * t;
        const double s = v0 * t + 0.5 * accel * t * t;
        SampledPoint2 sample;
        sample.t = t_offset + t;
        sample.reference.position = start.position + heading * s;
        sample.reference.velocity = heading * v;
        sample.reference.acceleration = heading * accel;
        sample.reference.yaw = wrapAngle(start.yaw);
        sample.reference.speed = v;
        sample.reference.linear_acceleration = accel;
        sample.reference.flags |= kFlagExplicitPlanarKinematics;
        if (std::abs(v) < 1.0e-6) {
            sample.reference.flags |= kFlagLowSpeedSingularity;
        }
        if (!appendSample(samples, std::move(sample))) {
            return false;
        }
    }
    const double s_end = v0 * duration + 0.5 * (v0 >= 0.0 ? -a_max : a_max) * duration * duration;
    stopped.position = start.position + heading * s_end;
    stopped.speed = 0.0;
    return true;
}

inline bool mincoWellPosedForUnicycle(const Se2TargetState2& start, double travel_yaw, double distance,
                                      const Se2TargetTrajectoryOptions2& options) {
    const double max_velocity = positiveLimit(options.max_velocity, 3.0);
    const double desired = clamp(positiveLimit(options.desired_speed, 1.0), 0.05, max_velocity);
    const double max_yaw_rate = positiveLimit(options.max_yaw_rate, 2.5);
    const double r_min = desired / std::max(max_yaw_rate, 1.0e-3);
    const double yaw_to_travel = std::abs(wrapAngle(start.yaw - travel_yaw));
    if (yaw_to_travel > 0.6) {
        return false;
    }
    if (distance < std::max(1.0, 2.5 * r_min)) {
        return false;
    }
    return true;
}

inline bool composeFeasibleUnicycle(const Se2TargetState2& start, const Se2TargetState2& target,
                                    const Se2TargetTrajectoryOptions2& options, std::vector<SampledPoint2>& samples) {
    samples.clear();
    Se2TargetState2 cursor;
    if (!appendDecelToStop(start, options, cursor, samples)) {
        return false;
    }
    cursor.speed = 0.0;
    const Eigen::Vector2d delta = target.position - cursor.position;
    const double dist = delta.norm();
    const double travel_yaw = dist > 1.0e-4 ? std::atan2(delta.y(), delta.x()) : cursor.yaw;
    const double reverse_yaw = wrapAngle(travel_yaw + 3.141592653589793);
    const double d_fwd = std::abs(wrapAngle(cursor.yaw - travel_yaw));
    const double d_rev = std::abs(wrapAngle(cursor.yaw - reverse_yaw));
    const bool reverse = d_rev + 0.15 < d_fwd;
    const double move_yaw = reverse ? reverse_yaw : travel_yaw;
    if (!appendYawSegment(cursor.position, cursor.yaw, move_yaw, options, samples)) {
        return false;
    }
    cursor.yaw = move_yaw;
    if (!appendTrapezoidChord(cursor.position, target.position, cursor.yaw, options, reverse, samples)) {
        return false;
    }
    cursor.position = target.position;
    if (!appendYawSegment(cursor.position, cursor.yaw, target.yaw, options, samples)) {
        return false;
    }
    appendHold(target, sampleStartT(samples), options.hold_duration, positiveLimit(options.sample_dt, 0.01), samples);
    return !samples.empty();
}

class MincoSe2TargetOptimizer {
  public:
    MincoSe2TargetOptimizer(const Se2TargetState2& start, const Se2TargetState2& target,
                            const Se2TargetTrajectoryOptions2& options, const Eigen::Vector2d& motion_target,
                            const Eigen::Vector2d& tail_velocity)
        : start_(start), target_(target), options_(options), motion_target_(motion_target),
          tail_velocity_(tail_velocity) {
        piece_count_ = std::max(1, options_.piece_count);
        const double max_velocity = positiveLimit(options_.max_velocity, 3.0);
        const double start_speed =
            clamp(std::abs(start_.speed), positiveLimit(options_.min_boundary_speed, 0.15), max_velocity);
        head_pv_.setZero();
        tail_pv_.setZero();
        head_pv_.col(0) = Eigen::Vector3d(start_.position.x(), start_.position.y(), 0.0);
        head_pv_.col(1) = Eigen::Vector3d(start_speed * std::cos(start_.yaw), start_speed * std::sin(start_.yaw), 0.0);
        tail_pv_.col(0) = Eigen::Vector3d(motion_target_.x(), motion_target_.y(), 0.0);
        tail_pv_.col(1) = Eigen::Vector3d(tail_velocity_.x(), tail_velocity_.y(), 0.0);
        minco_.setConditions(head_pv_, tail_pv_, piece_count_);
    }

    Eigen::VectorXd initialGuess() const {
        Eigen::VectorXd x(piece_count_ + 2 * std::max(0, piece_count_ - 1));
        const double distance = (motion_target_ - start_.position).norm();
        const double desired_speed = positiveLimit(options_.desired_speed, 1.0);
        Eigen::VectorXd times(piece_count_);
        times.setConstant(clamp(distance / desired_speed / static_cast<double>(piece_count_), options_.min_segment_time,
                                positiveLimit(options_.max_segment_time, 8.0)));
        Eigen::VectorXd tau;
        backwardT(times, tau);
        x.head(piece_count_) = tau;

        const Eigen::Vector2d start_tangent = unitFromYaw(start_.yaw) * distance * 0.35;
        const Eigen::Vector2d end_tangent = tail_velocity_.squaredNorm() > kMinNorm
                                                ? tail_velocity_.normalized() * distance * 0.35
                                                : unitFromYaw(target_.yaw) * distance * 0.35;
        for (int i = 1; i < piece_count_; ++i) {
            const double s = static_cast<double>(i) / static_cast<double>(piece_count_);
            const double s2 = s * s;
            const double s3 = s2 * s;
            const Eigen::Vector2d point = (2.0 * s3 - 3.0 * s2 + 1.0) * start_.position +
                                          (s3 - 2.0 * s2 + s) * start_tangent +
                                          (-2.0 * s3 + 3.0 * s2) * motion_target_ + (s3 - s2) * end_tangent;
            const int offset = piece_count_ + 2 * (i - 1);
            x(offset) = point.x();
            x(offset + 1) = point.y();
        }
        return x;
    }

    bool optimize(Eigen::VectorXd& x, uint32_t& flags) {
        x = initialGuess();
        xgc2_math::optimization::lbfgs::lbfgs_parameter_t params;
        params.mem_size = 32;
        params.past = 3;
        params.g_epsilon = 0.0;
        params.delta = std::max(1.0e-9, options_.rel_cost_tol);
        params.max_iterations = std::max(1, options_.max_iterations);
        params.min_step = 1.0e-24;
        double min_cost = 0.0;
        const int ret = xgc2_math::optimization::lbfgs::lbfgs_optimize(
            x, min_cost, &MincoSe2TargetOptimizer::costFunction, nullptr, nullptr, this, params);
        if (ret < 0 || !finiteScalar(min_cost)) {
            if (!finiteScalar(scalarCost(x))) {
                flags |= kFlagOptimizationFailure;
                return false;
            }
            flags |= kFlagOptimizationFailure;
        }
        return true;
    }

    bool decode(const Eigen::VectorXd& x, Eigen::VectorXd& times, Eigen::Matrix3Xd& points) const {
        if (x.size() != piece_count_ + 2 * std::max(0, piece_count_ - 1)) {
            return false;
        }
        forwardT(x.head(piece_count_), times);
        const double max_segment_time = positiveLimit(options_.max_segment_time, 8.0);
        for (int i = 0; i < times.size(); ++i) {
            times(i) = clamp(times(i), options_.min_segment_time, max_segment_time);
        }
        points.resize(3, std::max(0, piece_count_ - 1));
        for (int i = 1; i < piece_count_; ++i) {
            const int offset = piece_count_ + 2 * (i - 1);
            points.col(i - 1) = Eigen::Vector3d(x(offset), x(offset + 1), 0.0);
        }
        return points.array().isFinite().all() && times.array().isFinite().all();
    }

    bool buildCoefficients(const Eigen::VectorXd& times, const Eigen::Matrix3Xd& points, Eigen::MatrixX3d& coeffs) {
        if (times.size() != piece_count_ || points.cols() != std::max(0, piece_count_ - 1)) {
            return false;
        }
        minco_.setParameters(points, times);
        coeffs = minco_.getCoeffs();
        return coeffs.array().isFinite().all();
    }

  private:
    static double costFunction(void* ptr, const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        return static_cast<MincoSe2TargetOptimizer*>(ptr)->costWithFiniteGradient(x, grad);
    }

    double costWithFiniteGradient(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        const double base = scalarCost(x);
        grad.setZero(x.size());
        if (!finiteScalar(base)) {
            return std::numeric_limits<double>::infinity();
        }
        for (int i = 0; i < x.size(); ++i) {
            const double step = 1.0e-5 * std::max(1.0, std::abs(x(i)));
            Eigen::VectorXd xp = x;
            Eigen::VectorXd xm = x;
            xp(i) += step;
            xm(i) -= step;
            const double fp = scalarCost(xp);
            const double fm = scalarCost(xm);
            if (finiteScalar(fp) && finiteScalar(fm)) {
                grad(i) = (fp - fm) / (2.0 * step);
            } else if (finiteScalar(fp)) {
                grad(i) = (fp - base) / step;
            } else if (finiteScalar(fm)) {
                grad(i) = (base - fm) / step;
            }
        }
        if (!grad.array().isFinite().all()) {
            grad.setZero(x.size());
        }
        return base;
    }

    double scalarCost(const Eigen::VectorXd& x) {
        Eigen::VectorXd times;
        Eigen::Matrix3Xd points;
        if (!decode(x, times, points)) {
            return std::numeric_limits<double>::infinity();
        }
        Eigen::MatrixX3d coeffs;
        if (!buildCoefficients(times, points, coeffs)) {
            return std::numeric_limits<double>::infinity();
        }

        double energy = 0.0;
        minco_.getEnergy(energy);
        double cost = energy + std::max(0.0, options_.time_weight) * times.sum();
        const double weight = std::max(0.0, options_.dynamic_penalty_weight);
        if (weight <= 0.0) {
            return cost;
        }
        const double dt = positiveLimit(options_.validation_sample_dt, 0.02);
        for (int i = 0; i < piece_count_; ++i) {
            const int count = std::max(2, static_cast<int>(std::ceil(times(i) / dt)));
            const double step = times(i) / static_cast<double>(count);
            for (int j = 0; j <= count; ++j) {
                const double t = step * static_cast<double>(j);
                PlanarReference2 ref;
                ref.position << polyValue(coeffs, i, 0, t, 0), polyValue(coeffs, i, 1, t, 0);
                ref.velocity << polyValue(coeffs, i, 0, t, 1), polyValue(coeffs, i, 1, t, 1);
                ref.acceleration << polyValue(coeffs, i, 0, t, 2), polyValue(coeffs, i, 1, t, 2);
                ref.jerk << polyValue(coeffs, i, 0, t, 3), polyValue(coeffs, i, 1, t, 3);
                completePlanarReference2(ref);
                if (!TrajectoryValidator2::finite(ref)) {
                    return std::numeric_limits<double>::infinity();
                }
                const double node_weight = (j == 0 || j == count) ? 0.5 : 1.0;
                const double accel_norm = ref.acceleration.norm();
                const double penalty = softLimitPenalty(ref.speed, options_.max_velocity) +
                                       softLimitPenalty(accel_norm, options_.max_acceleration) +
                                       softLimitPenalty(std::abs(ref.yaw_rate), options_.max_yaw_rate);
                cost += weight * penalty * node_weight * step;
            }
        }
        cost += weight * softLimitPenalty(times.sum(), options_.max_duration);
        return finiteScalar(cost) ? cost : std::numeric_limits<double>::infinity();
    }

    Se2TargetState2 start_;
    Se2TargetState2 target_;
    Se2TargetTrajectoryOptions2 options_;
    Eigen::Vector2d motion_target_{Eigen::Vector2d::Zero()};
    Eigen::Vector2d tail_velocity_{Eigen::Vector2d::Zero()};
    int piece_count_{1};
    Eigen::Matrix<double, 3, 2> head_pv_{Eigen::Matrix<double, 3, 2>::Zero()};
    Eigen::Matrix<double, 3, 2> tail_pv_{Eigen::Matrix<double, 3, 2>::Zero()};
    xgc2_math::optimization::minco::MINCO_S2NU minco_;
};

inline bool appendMincoSamples(const Eigen::VectorXd& times, const Eigen::MatrixX3d& coeffs,
                               const Se2TargetTrajectoryOptions2& options, std::vector<SampledPoint2>& samples) {
    const double dt = positiveLimit(options.sample_dt, 0.01);
    double t_offset = samples.empty() ? 0.0 : samples.back().t;
    for (int i = 0; i < times.size(); ++i) {
        const int count = std::max(1, static_cast<int>(std::ceil(times(i) / dt)));
        for (int j = 0; j <= count; ++j) {
            if ((!samples.empty() || i > 0) && j == 0) {
                continue;
            }
            const double local_t = times(i) * static_cast<double>(j) / static_cast<double>(count);
            SampledPoint2 sample;
            sample.t = t_offset + local_t;
            sample.reference.position << polyValue(coeffs, i, 0, local_t, 0), polyValue(coeffs, i, 1, local_t, 0);
            sample.reference.velocity << polyValue(coeffs, i, 0, local_t, 1), polyValue(coeffs, i, 1, local_t, 1);
            sample.reference.acceleration << polyValue(coeffs, i, 0, local_t, 2), polyValue(coeffs, i, 1, local_t, 2);
            sample.reference.jerk << polyValue(coeffs, i, 0, local_t, 3), polyValue(coeffs, i, 1, local_t, 3);
            completePlanarReference2(sample.reference);
            if (!appendSample(samples, std::move(sample))) {
                return false;
            }
        }
        t_offset = samples.back().t;
    }
    return true;
}

inline bool appendStopSamples(const Eigen::Vector2d& start_position, const Eigen::Vector2d& start_velocity,
                              const Se2TargetState2& target, double duration,
                              const Se2TargetTrajectoryOptions2& options, std::vector<SampledPoint2>& samples) {
    const double dt = positiveLimit(options.sample_dt, 0.01);
    const double T = std::max(options.min_segment_time, duration);
    const auto cx = trajectory2_detail::septicBoundary(start_position.x(), start_velocity.x(), 0.0, 0.0,
                                                       target.position.x(), 0.0, 0.0, 0.0, T);
    const auto cy = trajectory2_detail::septicBoundary(start_position.y(), start_velocity.y(), 0.0, 0.0,
                                                       target.position.y(), 0.0, 0.0, 0.0, T);
    const double t_offset = samples.empty() ? 0.0 : samples.back().t;
    const int count = std::max(1, static_cast<int>(std::ceil(T / dt)));
    for (int i = 0; i <= count; ++i) {
        if (!samples.empty() && i == 0) {
            continue;
        }
        const double local_t = T * static_cast<double>(i) / static_cast<double>(count);
        SampledPoint2 sample;
        sample.t = t_offset + local_t;
        sample.reference.position << trajectory2_detail::polyValue(cx, local_t, 0),
            trajectory2_detail::polyValue(cy, local_t, 0);
        sample.reference.velocity << trajectory2_detail::polyValue(cx, local_t, 1),
            trajectory2_detail::polyValue(cy, local_t, 1);
        sample.reference.acceleration << trajectory2_detail::polyValue(cx, local_t, 2),
            trajectory2_detail::polyValue(cy, local_t, 2);
        sample.reference.jerk << trajectory2_detail::polyValue(cx, local_t, 3),
            trajectory2_detail::polyValue(cy, local_t, 3);
        sample.reference.yaw = target.yaw;
        completePlanarReference2(sample.reference);
        if (sample.reference.speed < trajectory2_detail::kMinSpeed) {
            sample.reference.yaw = target.yaw;
            sample.reference.yaw_rate = 0.0;
            sample.reference.yaw_acceleration = 0.0;
        }
        if (!appendSample(samples, std::move(sample))) {
            return false;
        }
    }
    return true;
}

} // namespace xgc2_math::trajectory::se2_target_detail

namespace xgc2_math::trajectory {

inline bool Se2MincoTargetPlanner2::plan(const Se2TargetState2& start, const Se2TargetState2& target,
                                         const Se2TargetTrajectoryOptions2& options,
                                         Se2TargetTrajectoryResult2& result) const {
    using namespace se2_target_detail;
    result = Se2TargetTrajectoryResult2{};
    if (!finiteVector(start.position) || !finiteVector(target.position) || !finiteScalar(start.yaw) ||
        !finiteScalar(target.yaw)) {
        result.flags |= kFlagInvalidInput;
        return false;
    }

    const double dt = positiveLimit(options.sample_dt, 0.01);
    const double distance = (target.position - start.position).norm();
    if (distance <= std::max(1.0e-4, options.position_tolerance)) {
        appendYawHold(start, target, options, result.samples);
        result.flags |= validateSamples(result.samples, options);
        result.duration = result.samples.empty() ? 0.0 : result.samples.back().t;
        result.optimized = false;
        return !result.samples.empty();
    }

    const Eigen::Vector2d travel = target.position - start.position;
    const double travel_yaw = std::atan2(travel.y(), travel.x());

    auto fill_feasible = [&]() -> bool {
        std::vector<SampledPoint2> samples;
        if (!composeFeasibleUnicycle(start, target, options, samples)) {
            result.fallback_hold = true;
            result.samples.clear();
            result.duration = 0.0;
            result.flags |= kFlagOptimizationFailure;
            return false;
        }
        const uint32_t validation_flags = validateSamples(samples, options);
        constexpr uint32_t kHardLimitFlags =
            kFlagInvalidInput | kFlagNonFinite | kFlagVelocityLimit | kFlagAccelerationLimit | kFlagYawRateLimit;
        if ((validation_flags & kHardLimitFlags) != 0U) {
            result.flags |= validation_flags | kFlagOptimizationFailure;
            result.fallback_hold = true;
            result.samples.clear();
            result.duration = 0.0;
            return false;
        }
        result.samples = std::move(samples);
        result.flags |= validation_flags | kFlagExplicitPlanarKinematics;
        result.duration = result.samples.empty() ? 0.0 : result.samples.back().t;
        result.optimized = false;
        result.fallback_hold = false;
        return !result.samples.empty();
    };

    const bool try_minco = mincoWellPosedForUnicycle(start, travel_yaw, distance, options);
    if (!try_minco) {
        return fill_feasible();
    }

    const double max_velocity = positiveLimit(options.max_velocity, 3.0);
    const double max_acceleration = positiveLimit(options.max_acceleration, 2.0);
    const double desired_speed = clamp(positiveLimit(options.desired_speed, 1.0), 0.05, max_velocity);
    const double approach_speed = clamp(positiveLimit(options.approach_speed, 0.4), 0.05, desired_speed);
    const Eigen::Vector2d target_heading = unitFromYaw(target.yaw);
    const double nominal_stop_distance = approach_speed * approach_speed / (2.0 * max_acceleration);
    const double stop_distance = clamp(std::max(0.2, nominal_stop_distance), 0.0, 0.45 * distance);
    const Eigen::Vector2d motion_target = target.position - stop_distance * target_heading;
    const Eigen::Vector2d tail_velocity = approach_speed * target_heading;

    Se2TargetTrajectoryOptions2 local_options = options;
    local_options.piece_count = std::max(1, options.piece_count);
    local_options.desired_speed = desired_speed;
    local_options.approach_speed = approach_speed;
    local_options.sample_dt = dt;
    local_options.max_velocity = max_velocity;
    local_options.max_acceleration = max_acceleration;
    local_options.max_yaw_rate = positiveLimit(options.max_yaw_rate, 2.5);
    local_options.min_segment_time = positiveLimit(options.min_segment_time, 0.1);

    MincoSe2TargetOptimizer optimizer(start, target, local_options, motion_target, tail_velocity);
    Eigen::VectorXd x;
    uint32_t flags = kFlagNone;
    const bool optimized = optimizer.optimize(x, flags);
    if (!optimized) {
        return fill_feasible();
    }

    Eigen::VectorXd times;
    Eigen::Matrix3Xd points;
    if (!optimizer.decode(x, times, points)) {
        return fill_feasible();
    }

    double stop_duration = std::max(local_options.min_segment_time, 2.0 * stop_distance / approach_speed);
    bool valid = false;
    const double max_duration = positiveLimit(options.max_duration, 20.0);
    for (int attempt = 0; attempt < 8; ++attempt) {
        Eigen::MatrixX3d coeffs;
        if (!optimizer.buildCoefficients(times, points, coeffs)) {
            return fill_feasible();
        }
        std::vector<SampledPoint2> samples;
        if (!appendMincoSamples(times, coeffs, local_options, samples) ||
            !appendStopSamples(motion_target, tail_velocity, target, stop_duration, local_options, samples)) {
            return fill_feasible();
        }
        Se2TargetState2 hold = target;
        appendHold(hold, samples.back().t, options.hold_duration, dt, samples);
        const uint32_t validation_flags = validateSamples(samples, local_options);
        constexpr uint32_t kHardLimitFlags =
            kFlagInvalidInput | kFlagNonFinite | kFlagVelocityLimit | kFlagAccelerationLimit | kFlagYawRateLimit;
        if ((validation_flags & kHardLimitFlags) == 0U) {
            result.samples = std::move(samples);
            result.flags |= validation_flags;
            valid = true;
            break;
        }
        const double duration = samples.empty() ? 0.0 : samples.back().t;
        if (duration >= max_duration) {
            result.flags |= validation_flags;
            break;
        }
        times *= 1.25;
        stop_duration *= 1.25;
    }

    if (!valid) {
        return fill_feasible();
    }

    result.duration = result.samples.empty() ? 0.0 : result.samples.back().t;
    result.optimized = true;
    return true;
}

} // namespace xgc2_math::trajectory
