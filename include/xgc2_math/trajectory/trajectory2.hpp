#pragma once

#include "xgc2_math/trajectory/types.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace xgc2_math::trajectory {

struct PlanarReference2 {
    Eigen::Vector2d position{Eigen::Vector2d::Zero()};
    Eigen::Vector2d velocity{Eigen::Vector2d::Zero()};
    Eigen::Vector2d acceleration{Eigen::Vector2d::Zero()};
    Eigen::Vector2d jerk{Eigen::Vector2d::Zero()};
    double yaw{0.0};
    double speed{0.0};
    double linear_acceleration{0.0};
    double yaw_rate{0.0};
    double yaw_acceleration{0.0};
    double curvature{0.0};
    uint32_t flags{kFlagNone};
};

struct TrajectoryLimits2 {
    double max_velocity{0.0};
    double max_acceleration{0.0};
    double max_yaw_rate{0.0};
};

enum class WaypointConstraintType2 : uint8_t {
    kPoint = 0,
    kSphere = 1,
    kBox = 2,
    kGate = 3,
};

struct WaypointConstraint2 {
    WaypointConstraintType2 type{WaypointConstraintType2::kPoint};
    Eigen::Vector2d position{Eigen::Vector2d::Zero()};
    Eigen::Vector2d size{Eigen::Vector2d::Zero()};
    double yaw{0.0};
};

class TrajectoryEvaluator2 {
  public:
    virtual ~TrajectoryEvaluator2() = default;
    virtual bool evaluate(double t, PlanarReference2& output) const = 0;
    virtual double duration() const = 0;
    virtual TrajectoryModelType type() const = 0;
    virtual uint32_t flags() const = 0;
};

struct PolynomialSegment2 {
    double duration{0.0};
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> yaw;
};

class PiecewisePolynomialEvaluator2 final : public TrajectoryEvaluator2 {
  public:
    bool setSegments(std::vector<PolynomialSegment2> segments, uint8_t order);
    bool evaluate(double t, PlanarReference2& output) const override;
    double duration() const override { return total_duration_; }
    TrajectoryModelType type() const override { return TrajectoryModelType::kPolynomial; }
    uint32_t flags() const override { return flags_; }
    uint8_t order() const { return order_; }
    const std::vector<PolynomialSegment2>& segments() const { return segments_; }

  private:
    std::vector<PolynomialSegment2> segments_;
    double total_duration_{0.0};
    uint8_t order_{0U};
    uint32_t flags_{kFlagNone};
};

struct WaypointProblem2 {
    std::vector<WaypointConstraint2> constraints;
    std::vector<double> segment_times;
    Eigen::Vector2d start_velocity{Eigen::Vector2d::Zero()};
    Eigen::Vector2d start_acceleration{Eigen::Vector2d::Zero()};
    Eigen::Vector2d end_velocity{Eigen::Vector2d::Zero()};
    Eigen::Vector2d end_acceleration{Eigen::Vector2d::Zero()};
    TrajectoryLimits2 limits{};
    double desired_speed{1.0};
    double time_weight{1.0};
    double dynamic_penalty_weight{1000.0};
    int max_iterations{80};
    double rel_cost_tol{1.0e-5};
    double min_segment_time{0.1};
    double validation_sample_dt{0.02};
    uint32_t flags{kFlagNone};
};

class MincoWaypointSolver2 final {
  public:
    bool solve(const WaypointProblem2& problem, PiecewisePolynomialEvaluator2& evaluator,
               uint32_t* flags = nullptr) const;
};

struct SampledPoint2 {
    double t{0.0};
    PlanarReference2 reference{};
};

class SampledEvaluator2 final : public TrajectoryEvaluator2 {
  public:
    bool setSamples(std::vector<SampledPoint2> samples, bool preserve_explicit_planar_kinematics = false);
    bool evaluate(double t, PlanarReference2& output) const override;
    double duration() const override { return duration_; }
    TrajectoryModelType type() const override { return TrajectoryModelType::kSampled; }
    uint32_t flags() const override { return flags_; }
    const std::vector<SampledPoint2>& samples() const { return samples_; }
    bool preservesExplicitPlanarKinematics() const { return preserve_explicit_planar_kinematics_; }

  private:
    std::vector<SampledPoint2> samples_;
    double duration_{0.0};
    uint32_t flags_{kFlagNone};
    bool preserve_explicit_planar_kinematics_{false};
};

class TrajectoryValidator2 final {
  public:
    static uint32_t validate(const TrajectoryEvaluator2& evaluator, const TrajectoryLimits2& limits, double sample_dt);
    static bool finite(const PlanarReference2& output);
};

inline void completePlanarReference2(PlanarReference2& output);
std::unique_ptr<TrajectoryEvaluator2> cloneEvaluator(const TrajectoryEvaluator2& evaluator);

} // namespace xgc2_math::trajectory

#include <algorithm>
#include <cmath>

namespace xgc2_math::trajectory {
namespace trajectory2_detail {

constexpr double kMinDuration = 1.0e-6;
constexpr double kMinSpeed = 1.0e-4;

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
    const double T = std::max(kMinDuration, duration);
    std::vector<double> c(8, 0.0);
    c[0] = p0;
    c[1] = v0;
    c[2] = 0.5 * a0;
    c[3] = j0 / 6.0;

    Eigen::Matrix4d A;
    Eigen::Vector4d b;
    const double T2 = T * T;
    const double T3 = T2 * T;
    const double T4 = T3 * T;
    const double T5 = T4 * T;
    const double T6 = T5 * T;
    const double T7 = T6 * T;
    A << T4, T5, T6, T7, 4.0 * T3, 5.0 * T4, 6.0 * T5, 7.0 * T6, 12.0 * T2, 20.0 * T3, 30.0 * T4, 42.0 * T5, 24.0 * T,
        60.0 * T2, 120.0 * T3, 210.0 * T4;
    b << p1 - (c[0] + c[1] * T + c[2] * T2 + c[3] * T3), v1 - (c[1] + 2.0 * c[2] * T + 3.0 * c[3] * T2),
        a1 - (2.0 * c[2] + 6.0 * c[3] * T), j1 - (6.0 * c[3]);
    const Eigen::Vector4d tail = A.colPivHouseholderQr().solve(b);
    for (int i = 0; i < 4; ++i) {
        c[static_cast<size_t>(i) + 4U] = tail(i);
    }
    return c;
}

inline bool segmentAt(const std::vector<PolynomialSegment2>& segments, double t, size_t& index, double& local_t) {
    if (segments.empty() || !finiteScalar(t)) {
        return false;
    }
    double remaining = std::max(0.0, t);
    for (size_t i = 0; i < segments.size(); ++i) {
        const double duration = std::max(0.0, segments[i].duration);
        if (remaining <= duration || i + 1U == segments.size()) {
            index = i;
            local_t = clamp(remaining, 0.0, duration);
            return true;
        }
        remaining -= duration;
    }
    return false;
}

inline double yawFromVelocityOr(double vx, double vy, double fallback) {
    if (vx * vx + vy * vy < kMinSpeed * kMinSpeed) {
        return fallback;
    }
    return std::atan2(vy, vx);
}

inline Eigen::Vector2d unitFromYaw(double yaw) {
    return Eigen::Vector2d(std::cos(yaw), std::sin(yaw));
}

inline uint32_t limitFlags(const PlanarReference2& output, const TrajectoryLimits2& limits) {
    uint32_t flags = kFlagNone;
    if (limits.max_velocity > 0.0 && std::abs(output.speed) > limits.max_velocity) {
        flags |= kFlagVelocityLimit;
    }
    if (limits.max_acceleration > 0.0 && std::abs(output.linear_acceleration) > limits.max_acceleration) {
        flags |= kFlagAccelerationLimit;
    }
    if (limits.max_yaw_rate > 0.0 && std::abs(output.yaw_rate) > limits.max_yaw_rate) {
        flags |= kFlagYawRateLimit;
    }
    return flags;
}

} // namespace trajectory2_detail

inline void completePlanarReference2(PlanarReference2& output) {
    const double speed = output.velocity.norm();
    if (!trajectory2_detail::finiteScalar(speed) || speed < trajectory2_detail::kMinSpeed) {
        output.flags |= kFlagLowSpeedSingularity;
        output.speed = 0.0;
        output.linear_acceleration = 0.0;
        return;
    }

    const double vx = output.velocity.x();
    const double vy = output.velocity.y();
    const double ax = output.acceleration.x();
    const double ay = output.acceleration.y();
    const double jx = output.jerk.x();
    const double jy = output.jerk.y();
    const double speed_sq = speed * speed;
    const double cross_va = vx * ay - vy * ax;
    const double cross_vj = vx * jy - vy * jx;
    const double speed_sq_dot = 2.0 * (vx * ax + vy * ay);

    output.yaw = trajectory2_detail::yawFromVelocityOr(vx, vy, output.yaw);
    output.speed = speed;
    output.linear_acceleration = (vx * ax + vy * ay) / speed;
    output.yaw_rate = cross_va / speed_sq;
    output.yaw_acceleration = (cross_vj * speed_sq - cross_va * speed_sq_dot) / (speed_sq * speed_sq);
    output.curvature = output.yaw_rate / speed;
}

inline bool PiecewisePolynomialEvaluator2::setSegments(std::vector<PolynomialSegment2> segments, uint8_t order) {
    if (segments.empty() || order == 0U) {
        flags_ |= kFlagInvalidInput;
        return false;
    }
    total_duration_ = 0.0;
    for (const auto& segment : segments) {
        if (!trajectory2_detail::finiteScalar(segment.duration) ||
            segment.duration <= trajectory2_detail::kMinDuration ||
            segment.x.size() != static_cast<size_t>(order) + 1U ||
            segment.y.size() != static_cast<size_t>(order) + 1U ||
            (!segment.yaw.empty() && segment.yaw.size() != static_cast<size_t>(order) + 1U)) {
            flags_ |= kFlagInvalidInput;
            return false;
        }
        total_duration_ += segment.duration;
    }
    order_ = order;
    segments_ = std::move(segments);
    return true;
}

inline bool PiecewisePolynomialEvaluator2::evaluate(double t, PlanarReference2& output) const {
    size_t index = 0U;
    double local_t = 0.0;
    if (!trajectory2_detail::segmentAt(segments_, t, index, local_t)) {
        output.flags |= kFlagInvalidInput;
        return false;
    }
    const auto& segment = segments_[index];
    output = PlanarReference2{};
    output.position << trajectory2_detail::polyValue(segment.x, local_t, 0),
        trajectory2_detail::polyValue(segment.y, local_t, 0);
    output.velocity << trajectory2_detail::polyValue(segment.x, local_t, 1),
        trajectory2_detail::polyValue(segment.y, local_t, 1);
    output.acceleration << trajectory2_detail::polyValue(segment.x, local_t, 2),
        trajectory2_detail::polyValue(segment.y, local_t, 2);
    output.jerk << trajectory2_detail::polyValue(segment.x, local_t, 3),
        trajectory2_detail::polyValue(segment.y, local_t, 3);
    if (!segment.yaw.empty()) {
        output.yaw = trajectory2_detail::wrapAngle(trajectory2_detail::polyValue(segment.yaw, local_t, 0));
        output.yaw_rate = trajectory2_detail::polyValue(segment.yaw, local_t, 1);
        output.yaw_acceleration = trajectory2_detail::polyValue(segment.yaw, local_t, 2);
        output.speed = output.velocity.norm();
        output.linear_acceleration = output.speed > trajectory2_detail::kMinSpeed
                                         ? output.velocity.dot(output.acceleration) / output.speed
                                         : 0.0;
        output.curvature =
            std::abs(output.speed) > trajectory2_detail::kMinSpeed ? output.yaw_rate / output.speed : 0.0;
    } else {
        completePlanarReference2(output);
    }
    output.flags |= flags_;
    return TrajectoryValidator2::finite(output);
}

inline bool MincoWaypointSolver2::solve(const WaypointProblem2& problem, PiecewisePolynomialEvaluator2& evaluator,
                                        uint32_t* flags) const {
    uint32_t local_flags = problem.flags;
    if (problem.constraints.size() < 2U) {
        local_flags |= kFlagInvalidInput;
        if (flags) {
            *flags |= local_flags;
        }
        return false;
    }

    std::vector<double> times = problem.segment_times;
    if (times.empty()) {
        times.reserve(problem.constraints.size() - 1U);
        for (size_t i = 0; i + 1U < problem.constraints.size(); ++i) {
            const double distance = (problem.constraints[i + 1U].position - problem.constraints[i].position).norm();
            const double speed = std::max(0.1, problem.desired_speed);
            times.push_back(std::max(problem.min_segment_time, distance / speed));
        }
    }
    if (times.size() + 1U != problem.constraints.size()) {
        local_flags |= kFlagInvalidInput;
        if (flags) {
            *flags |= local_flags;
        }
        return false;
    }

    const size_t count = problem.constraints.size();
    std::vector<Eigen::Vector2d> velocities(count, Eigen::Vector2d::Zero());
    std::vector<Eigen::Vector2d> accelerations(count, Eigen::Vector2d::Zero());
    velocities.front() = problem.start_velocity;
    velocities.back() = problem.end_velocity;
    accelerations.front() = problem.start_acceleration;
    accelerations.back() = problem.end_acceleration;
    for (size_t i = 1; i + 1U < count; ++i) {
        const double dt = std::max(trajectory2_detail::kMinDuration, times[i - 1U] + times[i]);
        velocities[i] = (problem.constraints[i + 1U].position - problem.constraints[i - 1U].position) / dt;
    }

    std::vector<PolynomialSegment2> segments;
    segments.reserve(times.size());
    for (size_t i = 0; i < times.size(); ++i) {
        const double T = std::max(problem.min_segment_time, times[i]);
        PolynomialSegment2 segment;
        segment.duration = T;
        const auto& p0 = problem.constraints[i].position;
        const auto& p1 = problem.constraints[i + 1U].position;
        const auto& v0 = velocities[i];
        const auto& v1 = velocities[i + 1U];
        const auto& a0 = accelerations[i];
        const auto& a1 = accelerations[i + 1U];
        segment.x = trajectory2_detail::septicBoundary(p0.x(), v0.x(), a0.x(), 0.0, p1.x(), v1.x(), a1.x(), 0.0, T);
        segment.y = trajectory2_detail::septicBoundary(p0.y(), v0.y(), a0.y(), 0.0, p1.y(), v1.y(), a1.y(), 0.0, T);
        segment.yaw = trajectory2_detail::septicBoundary(problem.constraints[i].yaw, 0.0, 0.0, 0.0,
                                                         problem.constraints[i + 1U].yaw, 0.0, 0.0, 0.0, T);
        segments.push_back(std::move(segment));
    }
    const bool ok = evaluator.setSegments(std::move(segments), 7U);
    local_flags |= TrajectoryValidator2::validate(evaluator, problem.limits, 0.02);
    if (flags) {
        *flags |= local_flags;
    }
    return ok && (local_flags & (kFlagInvalidInput | kFlagNonFinite)) == 0U;
}

inline bool SampledEvaluator2::setSamples(std::vector<SampledPoint2> samples,
                                          bool preserve_explicit_planar_kinematics) {
    if (samples.empty()) {
        flags_ |= kFlagInvalidInput;
        return false;
    }
    std::sort(samples.begin(), samples.end(), [](const SampledPoint2& lhs, const SampledPoint2& rhs) {
        return lhs.t < rhs.t;
    });
    double last_t = -1.0;
    for (auto& sample : samples) {
        if (!preserve_explicit_planar_kinematics) {
            completePlanarReference2(sample.reference);
        }
        if (!trajectory2_detail::finiteScalar(sample.t) || sample.t <= last_t ||
            !TrajectoryValidator2::finite(sample.reference)) {
            flags_ |= kFlagInvalidInput;
            return false;
        }
        last_t = sample.t;
    }
    duration_ = samples.back().t;
    samples_ = std::move(samples);
    preserve_explicit_planar_kinematics_ = preserve_explicit_planar_kinematics;
    if (preserve_explicit_planar_kinematics_) {
        flags_ |= kFlagExplicitPlanarKinematics;
    }
    return true;
}

inline bool SampledEvaluator2::evaluate(double t, PlanarReference2& output) const {
    if (samples_.empty() || !trajectory2_detail::finiteScalar(t)) {
        output.flags |= kFlagInvalidInput;
        return false;
    }
    if (t <= samples_.front().t) {
        output = samples_.front().reference;
        return true;
    }
    if (t >= samples_.back().t) {
        output = samples_.back().reference;
        return true;
    }
    for (size_t i = 0; i + 1U < samples_.size(); ++i) {
        const auto& a = samples_[i];
        const auto& b = samples_[i + 1U];
        if (t < a.t || t > b.t) {
            continue;
        }
        const double ratio = (t - a.t) / std::max(trajectory2_detail::kMinDuration, b.t - a.t);
        output.position = (1.0 - ratio) * a.reference.position + ratio * b.reference.position;
        output.velocity = (1.0 - ratio) * a.reference.velocity + ratio * b.reference.velocity;
        output.acceleration = (1.0 - ratio) * a.reference.acceleration + ratio * b.reference.acceleration;
        output.jerk = (1.0 - ratio) * a.reference.jerk + ratio * b.reference.jerk;
        output.yaw = a.reference.yaw + trajectory2_detail::wrapAngle(b.reference.yaw - a.reference.yaw) * ratio;
        if (preserve_explicit_planar_kinematics_) {
            output.speed = (1.0 - ratio) * a.reference.speed + ratio * b.reference.speed;
            output.linear_acceleration =
                (1.0 - ratio) * a.reference.linear_acceleration + ratio * b.reference.linear_acceleration;
            output.yaw_rate = (1.0 - ratio) * a.reference.yaw_rate + ratio * b.reference.yaw_rate;
            output.yaw_acceleration =
                (1.0 - ratio) * a.reference.yaw_acceleration + ratio * b.reference.yaw_acceleration;
            output.curvature = (1.0 - ratio) * a.reference.curvature + ratio * b.reference.curvature;
        } else {
            completePlanarReference2(output);
        }
        output.flags |= flags_;
        return true;
    }
    output.flags |= kFlagTimeDomain;
    return false;
}

inline uint32_t TrajectoryValidator2::validate(const TrajectoryEvaluator2& evaluator, const TrajectoryLimits2& limits,
                                               double sample_dt) {
    uint32_t flags = evaluator.flags();
    const double duration = evaluator.duration();
    if (!trajectory2_detail::finiteScalar(duration) || duration < 0.0) {
        return flags | kFlagInvalidInput;
    }
    sample_dt = std::max(1.0e-3, sample_dt);
    const int count = std::max(1, static_cast<int>(std::ceil(duration / sample_dt)));
    for (int i = 0; i <= count; ++i) {
        const double t = duration * static_cast<double>(i) / static_cast<double>(count);
        PlanarReference2 output;
        if (!evaluator.evaluate(t, output) || !TrajectoryValidator2::finite(output)) {
            flags |= kFlagNonFinite;
            continue;
        }
        flags |= trajectory2_detail::limitFlags(output, limits);
    }
    return flags;
}

inline bool TrajectoryValidator2::finite(const PlanarReference2& output) {
    return trajectory2_detail::finiteVector(output.position) && trajectory2_detail::finiteVector(output.velocity) &&
           trajectory2_detail::finiteVector(output.acceleration) && trajectory2_detail::finiteVector(output.jerk) &&
           trajectory2_detail::finiteScalar(output.yaw) && trajectory2_detail::finiteScalar(output.speed) &&
           trajectory2_detail::finiteScalar(output.linear_acceleration) &&
           trajectory2_detail::finiteScalar(output.yaw_rate) &&
           trajectory2_detail::finiteScalar(output.yaw_acceleration) &&
           trajectory2_detail::finiteScalar(output.curvature);
}

inline std::unique_ptr<TrajectoryEvaluator2> cloneEvaluator(const TrajectoryEvaluator2& evaluator) {
    if (evaluator.type() == TrajectoryModelType::kPolynomial) {
        const auto* polynomial = dynamic_cast<const PiecewisePolynomialEvaluator2*>(&evaluator);
        if (polynomial) {
            auto clone = std::make_unique<PiecewisePolynomialEvaluator2>();
            clone->setSegments(polynomial->segments(), polynomial->order());
            return clone;
        }
    }
    if (evaluator.type() == TrajectoryModelType::kSampled) {
        const auto* sampled = dynamic_cast<const SampledEvaluator2*>(&evaluator);
        if (sampled) {
            auto clone = std::make_unique<SampledEvaluator2>();
            clone->setSamples(sampled->samples(), sampled->preservesExplicitPlanarKinematics());
            return clone;
        }
    }
    return nullptr;
}

} // namespace xgc2_math::trajectory
