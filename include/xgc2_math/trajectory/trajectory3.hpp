#pragma once

#include "xgc2_math/trajectory/flatness.hpp"
#include "xgc2_math/trajectory/types.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace xgc2_math::trajectory {

struct FlatOutput3 {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    Eigen::Vector3d jerk{Eigen::Vector3d::Zero()};
    Eigen::Vector3d snap{Eigen::Vector3d::Zero()};
    double yaw{0.0};
    double yaw_rate{0.0};
    double yaw_accel{0.0};
    uint32_t flags{kFlagNone};
};

struct FullStateReference3 {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond attitude{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d body_rate{Eigen::Vector3d::Zero()};
    Eigen::Vector3d angular_acceleration{Eigen::Vector3d::Zero()};
    double specific_thrust{0.0};
    uint32_t flags{kFlagNone};
};

struct TrajectoryLimits3 {
    double max_velocity{0.0};
    double max_acceleration{0.0};
    double max_jerk{0.0};
    double max_snap{0.0};
    double min_specific_thrust{0.1};
    double max_specific_thrust{0.0};
    double max_body_rate{0.0};
    double max_tilt{0.0};
};

class TrajectoryEvaluator3 {
  public:
    virtual ~TrajectoryEvaluator3() = default;
    virtual bool evaluate(double t, FlatOutput3& output) const = 0;
    virtual double duration() const = 0;
    virtual TrajectoryModelType type() const = 0;
    virtual uint32_t flags() const = 0;
};

struct PolynomialSegment3 {
    double duration{0.0};
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> yaw;
};

class PiecewisePolynomialEvaluator3 final : public TrajectoryEvaluator3 {
  public:
    bool setSegments(std::vector<PolynomialSegment3> segments, uint8_t order);
    bool evaluate(double t, FlatOutput3& output) const override;
    double duration() const override { return total_duration_; }
    TrajectoryModelType type() const override { return TrajectoryModelType::kPolynomial; }
    uint32_t flags() const override { return flags_; }
    uint8_t order() const { return order_; }
    const std::vector<PolynomialSegment3>& segments() const { return segments_; }

  private:
    std::vector<PolynomialSegment3> segments_;
    double total_duration_{0.0};
    uint8_t order_{0U};
    uint32_t flags_{kFlagNone};
};

struct SampledPoint3 {
    double t{0.0};
    FlatOutput3 flat{};
};

class SampledEvaluator3 final : public TrajectoryEvaluator3 {
  public:
    bool setSamples(std::vector<SampledPoint3> samples);
    bool evaluate(double t, FlatOutput3& output) const override;
    double duration() const override { return duration_; }
    TrajectoryModelType type() const override { return TrajectoryModelType::kSampled; }
    uint32_t flags() const override { return flags_; }
    const std::vector<SampledPoint3>& samples() const { return samples_; }

  private:
    std::vector<SampledPoint3> samples_;
    double duration_{0.0};
    uint32_t flags_{kFlagNone};
};

class TrajectoryValidator3 final {
  public:
    static uint32_t validate(const TrajectoryEvaluator3& evaluator, const TrajectoryLimits3& limits, double sample_dt);
    static bool finite(const FlatOutput3& output);
};

class FlatnessMapper3 final {
  public:
    explicit FlatnessMapper3(double gravity = 9.8066, double min_specific_thrust = 0.1);
    FullStateReference3 map(const FlatOutput3& flat) const;

  private:
    double gravity_;
    double min_specific_thrust_;
};

std::unique_ptr<TrajectoryEvaluator3> cloneEvaluator(const TrajectoryEvaluator3& evaluator);

} // namespace xgc2_math::trajectory

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>

namespace xgc2_math::trajectory {
namespace trajectory3_detail {

constexpr double kMinDuration = 1.0e-6;
constexpr double kMinNorm = 1.0e-9;

inline bool finiteScalar(double value) {
    return std::isfinite(value);
}

inline bool finiteVector(const Eigen::Vector3d& value) {
    return value.array().isFinite().all();
}

inline double clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(max_value, value));
}

inline double wrapAngle(double value) {
    return std::atan2(std::sin(value), std::cos(value));
}

inline double unwrapAngleNear(double value, double reference) {
    return reference + wrapAngle(value - reference);
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

inline void fillYawFromVelocity(FlatOutput3& output) {
    const double vx = output.velocity.x();
    const double vy = output.velocity.y();
    const double speed_sq = vx * vx + vy * vy;
    if (speed_sq < 1.0e-8) {
        output.yaw = 0.0;
        output.yaw_rate = 0.0;
        output.yaw_accel = 0.0;
        return;
    }

    const double ax = output.acceleration.x();
    const double ay = output.acceleration.y();
    const double jx = output.jerk.x();
    const double jy = output.jerk.y();
    const double numerator = vx * ay - vy * ax;
    const double numerator_dot = vx * jy - vy * jx;
    const double denominator_dot = 2.0 * (vx * ax + vy * ay);
    output.yaw = std::atan2(vy, vx);
    output.yaw_rate = numerator / speed_sq;
    output.yaw_accel = (numerator_dot * speed_sq - numerator * denominator_dot) / (speed_sq * speed_sq);
}

inline Eigen::Vector3d unitDerivative(const Eigen::Vector3d& value, const Eigen::Vector3d& value_dot,
                                      const Eigen::Vector3d& unit_value) {
    const double norm = value.norm();
    if (!finiteScalar(norm) || norm < kMinNorm) {
        return Eigen::Vector3d::Zero();
    }
    const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() - unit_value * unit_value.transpose();
    return projector * value_dot / norm;
}

inline Eigen::Vector3d unitSecondDerivative(const Eigen::Vector3d& value, const Eigen::Vector3d& value_dot,
                                            const Eigen::Vector3d& value_ddot, const Eigen::Vector3d& unit_value,
                                            const Eigen::Vector3d& unit_dot) {
    const double norm = value.norm();
    if (!finiteScalar(norm) || norm < kMinNorm) {
        return Eigen::Vector3d::Zero();
    }
    const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() - unit_value * unit_value.transpose();
    const Eigen::Matrix3d projector_dot = -(unit_dot * unit_value.transpose() + unit_value * unit_dot.transpose());
    const double norm_dot = unit_value.dot(value_dot);
    return projector_dot * value_dot / norm + projector * value_ddot / norm -
           projector * value_dot * norm_dot / (norm * norm);
}

inline Eigen::Vector3d normalizeOr(const Eigen::Vector3d& value, const Eigen::Vector3d& fallback) {
    const double norm = value.norm();
    if (!finiteScalar(norm) || norm < kMinNorm) {
        return fallback;
    }
    return value / norm;
}

inline Eigen::Vector3d vee(const Eigen::Matrix3d& skew) {
    return Eigen::Vector3d(skew(2, 1), skew(0, 2), skew(1, 0));
}

inline bool segmentAt(const std::vector<PolynomialSegment3>& segments, double t, size_t& index, double& local_t) {
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

inline uint32_t limitFlags(const FlatOutput3& output, const TrajectoryLimits3& limits) {
    uint32_t flags = kFlagNone;
    if (limits.max_velocity > 0.0 && output.velocity.norm() > limits.max_velocity) {
        flags |= kFlagVelocityLimit;
    }
    if (limits.max_acceleration > 0.0 && output.acceleration.norm() > limits.max_acceleration) {
        flags |= kFlagAccelerationLimit;
    }
    if (limits.max_jerk > 0.0 && output.jerk.norm() > limits.max_jerk) {
        flags |= kFlagJerkLimit;
    }
    if (limits.max_snap > 0.0 && output.snap.norm() > limits.max_snap) {
        flags |= kFlagSnapLimit;
    }
    const auto full = FlatnessMapper3(9.8066, limits.min_specific_thrust).map(output);
    if (limits.max_specific_thrust > 0.0 && full.specific_thrust > limits.max_specific_thrust) {
        flags |= kFlagThrustLimit;
    }
    if (limits.max_body_rate > 0.0 && full.body_rate.norm() > limits.max_body_rate) {
        flags |= kFlagBodyRateLimit;
    }
    if (limits.max_tilt > 0.0) {
        const Eigen::Vector3d z_body = full.attitude * Eigen::Vector3d::UnitZ();
        const double tilt = std::acos(clamp(z_body.z(), -1.0, 1.0));
        if (std::isfinite(tilt) && tilt > limits.max_tilt) {
            flags |= kFlagTiltLimit;
        }
    }
    flags |= full.flags & (kFlagLowThrust | kFlagYawSingularity | kFlagNonFinite);
    return flags;
}

} // namespace trajectory3_detail

inline bool PiecewisePolynomialEvaluator3::setSegments(std::vector<PolynomialSegment3> segments, uint8_t order) {
    segments_ = std::move(segments);
    order_ = order;
    total_duration_ = 0.0;
    flags_ = kFlagNone;
    if (segments_.empty() || order_ < 1U) {
        flags_ |= kFlagInvalidInput;
        return false;
    }
    for (const auto& segment : segments_) {
        if (!trajectory3_detail::finiteScalar(segment.duration) || segment.duration <= 0.0 ||
            segment.x.size() != static_cast<size_t>(order_) + 1U || segment.y.size() != segment.x.size() ||
            segment.z.size() != segment.x.size()) {
            flags_ |= kFlagInvalidInput;
            return false;
        }
        total_duration_ += segment.duration;
    }
    return true;
}

inline bool PiecewisePolynomialEvaluator3::evaluate(double t, FlatOutput3& output) const {
    output = FlatOutput3{};
    size_t index = 0U;
    double local_t = 0.0;
    if (!trajectory3_detail::segmentAt(segments_, t, index, local_t)) {
        output.flags |= kFlagTimeDomain;
        return false;
    }
    const auto& segment = segments_[index];
    for (int derivative = 0; derivative <= 4; ++derivative) {
        Eigen::Vector3d value(trajectory3_detail::polyValue(segment.x, local_t, derivative),
                              trajectory3_detail::polyValue(segment.y, local_t, derivative),
                              trajectory3_detail::polyValue(segment.z, local_t, derivative));
        switch (derivative) {
        case 0:
            output.position = value;
            break;
        case 1:
            output.velocity = value;
            break;
        case 2:
            output.acceleration = value;
            break;
        case 3:
            output.jerk = value;
            break;
        case 4:
            output.snap = value;
            break;
        }
    }
    if (!segment.yaw.empty()) {
        output.yaw = trajectory3_detail::polyValue(segment.yaw, local_t, 0);
        output.yaw_rate = trajectory3_detail::polyValue(segment.yaw, local_t, 1);
        output.yaw_accel = trajectory3_detail::polyValue(segment.yaw, local_t, 2);
    } else {
        trajectory3_detail::fillYawFromVelocity(output);
    }
    output.flags |= flags_;
    if (!TrajectoryValidator3::finite(output)) {
        output.flags |= kFlagNonFinite;
        return false;
    }
    return true;
}

inline bool SampledEvaluator3::setSamples(std::vector<SampledPoint3> samples) {
    samples_ = std::move(samples);
    flags_ = kFlagNone;
    duration_ = 0.0;
    if (samples_.empty()) {
        flags_ |= kFlagInvalidInput;
        return false;
    }
    double last_t = -1.0;
    for (const auto& sample : samples_) {
        if (!trajectory3_detail::finiteScalar(sample.t) || sample.t < last_t ||
            !TrajectoryValidator3::finite(sample.flat)) {
            flags_ |= kFlagInvalidInput;
            return false;
        }
        last_t = sample.t;
    }
    duration_ = samples_.back().t;
    return true;
}

inline bool SampledEvaluator3::evaluate(double t, FlatOutput3& output) const {
    output = FlatOutput3{};
    if (samples_.empty() || !trajectory3_detail::finiteScalar(t)) {
        output.flags |= kFlagTimeDomain;
        return false;
    }
    if (samples_.size() == 1U || t <= samples_.front().t) {
        output = samples_.front().flat;
        return true;
    }
    if (t >= samples_.back().t) {
        output = samples_.back().flat;
        return true;
    }

    const auto upper = std::upper_bound(samples_.begin(), samples_.end(), t, [](double lhs, const SampledPoint3& rhs) {
        return lhs < rhs.t;
    });
    const auto prev = upper - 1;
    const auto next = upper;
    const double dt = std::max(trajectory3_detail::kMinDuration, next->t - prev->t);
    const double alpha = (t - prev->t) / dt;
    output.position = (1.0 - alpha) * prev->flat.position + alpha * next->flat.position;
    output.velocity = (1.0 - alpha) * prev->flat.velocity + alpha * next->flat.velocity;
    output.acceleration = (1.0 - alpha) * prev->flat.acceleration + alpha * next->flat.acceleration;
    output.jerk = (1.0 - alpha) * prev->flat.jerk + alpha * next->flat.jerk;
    output.snap = (1.0 - alpha) * prev->flat.snap + alpha * next->flat.snap;
    output.yaw = prev->flat.yaw + alpha * trajectory3_detail::wrapAngle(next->flat.yaw - prev->flat.yaw);
    output.yaw_rate = (1.0 - alpha) * prev->flat.yaw_rate + alpha * next->flat.yaw_rate;
    output.yaw_accel = (1.0 - alpha) * prev->flat.yaw_accel + alpha * next->flat.yaw_accel;
    output.flags = prev->flat.flags | next->flat.flags | flags_;
    if (!TrajectoryValidator3::finite(output)) {
        output.flags |= kFlagNonFinite;
        return false;
    }
    return true;
}

inline uint32_t TrajectoryValidator3::validate(const TrajectoryEvaluator3& evaluator, const TrajectoryLimits3& limits,
                                               double sample_dt) {
    uint32_t flags = evaluator.flags();
    if (!trajectory3_detail::finiteScalar(evaluator.duration()) || evaluator.duration() < 0.0) {
        return flags | kFlagInvalidInput;
    }
    const double dt = std::max(1.0e-3, sample_dt);
    const int count = std::max(1, static_cast<int>(std::ceil(evaluator.duration() / dt)));
    for (int i = 0; i <= count; ++i) {
        const double t = std::min(evaluator.duration(), static_cast<double>(i) * dt);
        FlatOutput3 output;
        if (!evaluator.evaluate(t, output) || !TrajectoryValidator3::finite(output)) {
            flags |= kFlagNonFinite;
            continue;
        }
        flags |= output.flags;
        flags |= trajectory3_detail::limitFlags(output, limits);
    }
    return flags;
}

inline bool TrajectoryValidator3::finite(const FlatOutput3& output) {
    return trajectory3_detail::finiteVector(output.position) && trajectory3_detail::finiteVector(output.velocity) &&
           trajectory3_detail::finiteVector(output.acceleration) && trajectory3_detail::finiteVector(output.jerk) &&
           trajectory3_detail::finiteVector(output.snap) && trajectory3_detail::finiteScalar(output.yaw) &&
           trajectory3_detail::finiteScalar(output.yaw_rate) && trajectory3_detail::finiteScalar(output.yaw_accel);
}

inline FlatnessMapper3::FlatnessMapper3(double gravity, double min_specific_thrust)
    : gravity_(gravity), min_specific_thrust_(min_specific_thrust) {}

inline FullStateReference3 FlatnessMapper3::map(const FlatOutput3& flat) const {
    FullStateReference3 output;
    output.position = flat.position;
    output.velocity = flat.velocity;
    output.flags = flat.flags;
    if (!TrajectoryValidator3::finite(flat)) {
        output.flags |= kFlagNonFinite;
        return output;
    }

    const Eigen::Vector3d thrust = flat.acceleration + gravity_ * Eigen::Vector3d::UnitZ();
    const double thrust_norm = thrust.norm();
    output.specific_thrust = thrust_norm;
    if (!trajectory3_detail::finiteScalar(thrust_norm) || thrust_norm < min_specific_thrust_) {
        output.flags |= kFlagLowThrust;
        return output;
    }

    const Eigen::Vector3d b3 = thrust / thrust_norm;
    const Eigen::Vector3d b3_dot = trajectory3_detail::unitDerivative(thrust, flat.jerk, b3);
    const Eigen::Vector3d b3_ddot = trajectory3_detail::unitSecondDerivative(thrust, flat.jerk, flat.snap, b3, b3_dot);

    const double yaw = flat.yaw;
    const double yaw_rate = flat.yaw_rate;
    const double yaw_accel = flat.yaw_accel;
    const Eigen::Vector3d xc(std::cos(yaw), std::sin(yaw), 0.0);
    const Eigen::Vector3d xc_perp(-std::sin(yaw), std::cos(yaw), 0.0);
    const Eigen::Vector3d xc_dot = yaw_rate * xc_perp;
    const Eigen::Vector3d xc_ddot = yaw_accel * xc_perp - yaw_rate * yaw_rate * xc;

    Eigen::Vector3d y_raw = b3.cross(xc);
    Eigen::Vector3d y_raw_dot = b3_dot.cross(xc) + b3.cross(xc_dot);
    Eigen::Vector3d y_raw_ddot = b3_ddot.cross(xc) + 2.0 * b3_dot.cross(xc_dot) + b3.cross(xc_ddot);
    if (!trajectory3_detail::finiteScalar(y_raw.norm()) || y_raw.norm() < 1.0e-7) {
        output.flags |= kFlagYawSingularity;
        const Eigen::Vector3d fallback =
            std::abs(b3.dot(Eigen::Vector3d::UnitY())) > 0.95 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
        y_raw = b3.cross(fallback);
        // The fallback axis is locally constant, but b3 still moves.  Its
        // derivatives must also be used to differentiate the returned frame.
        y_raw_dot = b3_dot.cross(fallback);
        y_raw_ddot = b3_ddot.cross(fallback);
    }

    const Eigen::Vector3d yb = trajectory3_detail::normalizeOr(y_raw, Eigen::Vector3d::UnitY());
    const Eigen::Vector3d yb_dot = trajectory3_detail::unitDerivative(y_raw, y_raw_dot, yb);
    const Eigen::Vector3d yb_ddot = trajectory3_detail::unitSecondDerivative(y_raw, y_raw_dot, y_raw_ddot, yb, yb_dot);
    const Eigen::Vector3d xb = trajectory3_detail::normalizeOr(yb.cross(b3), Eigen::Vector3d::UnitX());
    const Eigen::Vector3d xb_dot = yb_dot.cross(b3) + yb.cross(b3_dot);
    const Eigen::Vector3d xb_ddot = yb_ddot.cross(b3) + 2.0 * yb_dot.cross(b3_dot) + yb.cross(b3_ddot);

    Eigen::Matrix3d rotation;
    rotation.col(0) = xb;
    rotation.col(1) = yb;
    rotation.col(2) = b3;
    Eigen::Matrix3d rotation_dot;
    rotation_dot.col(0) = xb_dot;
    rotation_dot.col(1) = yb_dot;
    rotation_dot.col(2) = b3_dot;
    Eigen::Matrix3d rotation_ddot;
    rotation_ddot.col(0) = xb_ddot;
    rotation_ddot.col(1) = yb_ddot;
    rotation_ddot.col(2) = b3_ddot;

    const Eigen::Matrix3d omega_hat = rotation.transpose() * rotation_dot;
    const Eigen::Matrix3d alpha_hat = rotation_dot.transpose() * rotation_dot + rotation.transpose() * rotation_ddot;
    output.attitude = Eigen::Quaterniond(rotation);
    output.attitude.normalize();
    if (output.attitude.w() < 0.0) {
        output.attitude.coeffs() *= -1.0;
    }
    output.body_rate = trajectory3_detail::vee(0.5 * (omega_hat - omega_hat.transpose()));
    output.angular_acceleration = trajectory3_detail::vee(0.5 * (alpha_hat - alpha_hat.transpose()));
    if (!trajectory3_detail::finiteVector(output.body_rate) ||
        !trajectory3_detail::finiteVector(output.angular_acceleration) ||
        !trajectory3_detail::finiteScalar(output.attitude.norm())) {
        output.flags |= kFlagNonFinite;
    }
    return output;
}

inline std::unique_ptr<TrajectoryEvaluator3> cloneEvaluator(const TrajectoryEvaluator3& evaluator) {
    switch (evaluator.type()) {
    case TrajectoryModelType::kAnalytic:
        break;
    case TrajectoryModelType::kPolynomial: {
        const auto* polynomial = dynamic_cast<const PiecewisePolynomialEvaluator3*>(&evaluator);
        if (!polynomial) {
            return nullptr;
        }
        auto clone = std::make_unique<PiecewisePolynomialEvaluator3>();
        (void)clone->setSegments(polynomial->segments(), polynomial->order());
        return clone;
    }
    case TrajectoryModelType::kSampled: {
        const auto* sampled = dynamic_cast<const SampledEvaluator3*>(&evaluator);
        if (!sampled) {
            return nullptr;
        }
        auto clone = std::make_unique<SampledEvaluator3>();
        (void)clone->setSamples(sampled->samples());
        return clone;
    }
    case TrajectoryModelType::kNone:
        break;
    }
    return nullptr;
}

} // namespace xgc2_math::trajectory
