#pragma once

#include "xgc2_math/optimization/lbfgs.hpp"
#include "xgc2_math/optimization/minco.hpp"
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

enum class WaypointConstraintType3 : uint8_t {
    kPoint = 0,
    kSphere = 1,
    kBox = 2,
    kGate = 3,
};

struct WaypointConstraint3 {
    WaypointConstraintType3 type{WaypointConstraintType3::kPoint};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d size{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
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

struct WaypointProblem3 {
    std::vector<WaypointConstraint3> constraints;
    std::vector<double> segment_times;
    Eigen::Vector3d start_velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d start_acceleration{Eigen::Vector3d::Zero()};
    Eigen::Vector3d end_velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d end_acceleration{Eigen::Vector3d::Zero()};
    TrajectoryLimits3 limits{};
    double desired_speed{1.0};
    double time_weight{1.0};
    double dynamic_penalty_weight{1000.0};
    int max_iterations{80};
    int integral_resolution{12};
    double rel_cost_tol{1.0e-5};
    double min_segment_time{0.1};
    double max_segment_time{30.0};
    double validation_sample_dt{0.02};
    uint32_t flags{kFlagNone};
};

class MincoWaypointSolver3 final {
  public:
    bool solve(const WaypointProblem3& problem, PiecewisePolynomialEvaluator3& evaluator,
               uint32_t* flags = nullptr) const;
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

inline bool smoothedL1(double x, double mu, double& value, double& derivative) {
    if (x < 0.0) {
        value = 0.0;
        derivative = 0.0;
        return false;
    }
    const double smooth = std::max(1.0e-6, mu);
    if (x > smooth) {
        value = x - 0.5 * smooth;
        derivative = 1.0;
        return true;
    }
    const double ratio = x / smooth;
    const double ratio2 = ratio * ratio;
    const double term = smooth - 0.5 * x;
    value = term * ratio2 * ratio;
    derivative = ratio2 * (-0.5 * ratio + 3.0 * term / smooth);
    return true;
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

inline void backwardGradT(const Eigen::VectorXd& tau, const Eigen::VectorXd& grad_times, Eigen::VectorXd& grad_tau) {
    grad_tau.resize(tau.size());
    for (int i = 0; i < tau.size(); ++i) {
        if (tau(i) > 0.0) {
            grad_tau(i) = grad_times(i) * (tau(i) + 1.0);
        } else {
            const double den = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
            grad_tau(i) = grad_times(i) * (1.0 - tau(i)) / (den * den);
        }
    }
}

inline int spatialDim(const WaypointConstraint3& constraint) {
    return constraint.type == WaypointConstraintType3::kPoint ? 0 : 3;
}

inline double sphereRadius(const WaypointConstraint3& constraint) {
    if (finiteScalar(constraint.size.x()) && constraint.size.x() > 0.0) {
        return constraint.size.x();
    }
    return std::max(
        {std::abs(constraint.size.x()), std::abs(constraint.size.y()), std::abs(constraint.size.z()), 1.0e-3});
}

inline Eigen::Vector3d halfExtents(const WaypointConstraint3& constraint) {
    return Eigen::Vector3d(std::max(0.0, std::abs(constraint.size.x())), std::max(0.0, std::abs(constraint.size.y())),
                           std::max(0.0, std::abs(constraint.size.z())));
}

inline Eigen::Matrix3d rotationOf(const WaypointConstraint3& constraint) {
    Eigen::Quaterniond q = constraint.orientation;
    if (!finiteScalar(q.norm()) || q.norm() < kMinNorm) {
        return Eigen::Matrix3d::Identity();
    }
    q.normalize();
    return q.toRotationMatrix();
}

inline Eigen::Vector3d mapConstraintPoint(const WaypointConstraint3& constraint, const Eigen::Vector3d& xi) {
    if (constraint.type == WaypointConstraintType3::kPoint) {
        return constraint.position;
    }
    if (constraint.type == WaypointConstraintType3::kSphere) {
        const double radius = sphereRadius(constraint);
        const double denom = std::sqrt(1.0 + xi.squaredNorm());
        return constraint.position + radius * xi / denom;
    }
    Eigen::Vector3d half = halfExtents(constraint);
    if (constraint.type == WaypointConstraintType3::kGate && half.x() <= 0.0) {
        half.x() = 0.0;
    }
    const Eigen::Vector3d local(half.x() * std::tanh(xi.x()), half.y() * std::tanh(xi.y()),
                                half.z() * std::tanh(xi.z()));
    return constraint.position + rotationOf(constraint) * local;
}

inline Eigen::Vector3d constraintGradXi(const WaypointConstraint3& constraint, const Eigen::Vector3d& xi,
                                        const Eigen::Vector3d& grad_point) {
    if (constraint.type == WaypointConstraintType3::kPoint) {
        return Eigen::Vector3d::Zero();
    }
    if (constraint.type == WaypointConstraintType3::kSphere) {
        const double radius = sphereRadius(constraint);
        const double denom2 = 1.0 + xi.squaredNorm();
        const double denom = std::sqrt(denom2);
        const Eigen::Matrix3d jac =
            radius * (Eigen::Matrix3d::Identity() / denom - (xi * xi.transpose()) / (denom2 * denom));
        return jac.transpose() * grad_point;
    }
    Eigen::Vector3d half = halfExtents(constraint);
    if (constraint.type == WaypointConstraintType3::kGate && half.x() <= 0.0) {
        half.x() = 0.0;
    }
    const Eigen::Vector3d tanh_xi(std::tanh(xi.x()), std::tanh(xi.y()), std::tanh(xi.z()));
    const Eigen::Vector3d scale = half.cwiseProduct(Eigen::Vector3d::Ones() - tanh_xi.cwiseProduct(tanh_xi));
    return scale.cwiseProduct(rotationOf(constraint).transpose() * grad_point);
}

inline void dynamicPenalty(const Eigen::VectorXd& times, const Eigen::MatrixX3d& coeffs,
                           const WaypointProblem3& problem, double& cost, Eigen::VectorXd& grad_times,
                           Eigen::MatrixX3d& grad_coeffs) {
    const int piece_count = static_cast<int>(times.size());
    const int integral_resolution = std::max(2, problem.integral_resolution);
    const double integral_frac = 1.0 / static_cast<double>(integral_resolution);
    const double weight = std::max(0.0, problem.dynamic_penalty_weight);
    if (piece_count <= 0 || weight <= 0.0) {
        return;
    }

    xgc2_math::trajectory::flatness::FlatnessMap flatmap;
    flatmap.reset(1.0, 9.8066, 0.0, 0.0, 0.0, 1.0e-6);
    const auto& limits = problem.limits;
    const double smooth = 1.0e-3;
    const double thrust_min = std::max(0.0, limits.min_specific_thrust);
    const double thrust_max = std::max(thrust_min, limits.max_specific_thrust);
    const bool check_thrust = thrust_max > thrust_min + 1.0e-6;

    for (int i = 0; i < piece_count; ++i) {
        const Eigen::Matrix<double, 6, 3> c = coeffs.block<6, 3>(6 * i, 0);
        const double step = times(i) * integral_frac;
        for (int j = 0; j <= integral_resolution; ++j) {
            const double t = static_cast<double>(j) * step;
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t2 * t2;
            const double t5 = t4 * t;
            Eigen::Matrix<double, 6, 1> beta0;
            Eigen::Matrix<double, 6, 1> beta1;
            Eigen::Matrix<double, 6, 1> beta2;
            Eigen::Matrix<double, 6, 1> beta3;
            Eigen::Matrix<double, 6, 1> beta4;
            Eigen::Matrix<double, 6, 1> beta5;
            beta0 << 1.0, t, t2, t3, t4, t5;
            beta1 << 0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4;
            beta2 << 0.0, 0.0, 2.0, 6.0 * t, 12.0 * t2, 20.0 * t3;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * t, 60.0 * t2;
            beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * t;
            beta5 << 0.0, 0.0, 0.0, 0.0, 0.0, 120.0;

            const Eigen::Vector3d vel = c.transpose() * beta1;
            const Eigen::Vector3d acc = c.transpose() * beta2;
            const Eigen::Vector3d jerk = c.transpose() * beta3;
            const Eigen::Vector3d snap = c.transpose() * beta4;
            const Eigen::Vector3d crackle = c.transpose() * beta5;

            Eigen::Vector3d grad_vel = Eigen::Vector3d::Zero();
            Eigen::Vector3d grad_acc = Eigen::Vector3d::Zero();
            Eigen::Vector3d grad_jerk = Eigen::Vector3d::Zero();
            Eigen::Vector3d grad_snap = Eigen::Vector3d::Zero();
            double penalty = 0.0;
            double value = 0.0;
            double derivative = 0.0;
            const auto add_norm_penalty = [&](const Eigen::Vector3d& vec, double limit, Eigen::Vector3d& grad) {
                if (limit <= 0.0) {
                    return;
                }
                if (smoothedL1(vec.squaredNorm() - limit * limit, smooth, value, derivative)) {
                    penalty += weight * value;
                    grad += weight * derivative * 2.0 * vec;
                }
            };

            add_norm_penalty(vel, limits.max_velocity, grad_vel);
            add_norm_penalty(acc, limits.max_acceleration, grad_acc);
            add_norm_penalty(jerk, limits.max_jerk, grad_jerk);
            add_norm_penalty(snap, limits.max_snap, grad_snap);

            double thrust = 0.0;
            Eigen::Vector4d quat = Eigen::Vector4d::Zero();
            Eigen::Vector3d omega = Eigen::Vector3d::Zero();
            flatmap.forward(vel, acc, jerk, 0.0, 0.0, thrust, quat, omega);
            double grad_thrust = 0.0;
            Eigen::Vector4d grad_quat = Eigen::Vector4d::Zero();
            Eigen::Vector3d grad_omega = Eigen::Vector3d::Zero();
            if (limits.max_body_rate > 0.0 &&
                smoothedL1(omega.squaredNorm() - limits.max_body_rate * limits.max_body_rate, smooth, value,
                           derivative)) {
                penalty += weight * value;
                grad_omega += weight * derivative * 2.0 * omega;
            }
            if (limits.max_tilt > 0.0) {
                const double cos_tilt =
                    clamp(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)), -1.0 + 1.0e-6, 1.0 - 1.0e-6);
                if (smoothedL1(std::acos(cos_tilt) - limits.max_tilt, smooth, value, derivative)) {
                    penalty += weight * value;
                    grad_quat += weight * derivative / std::sqrt(1.0 - cos_tilt * cos_tilt) * 4.0 *
                                 Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
                }
            }
            if (check_thrust) {
                const double thrust_mean = 0.5 * (thrust_min + thrust_max);
                const double thrust_radius = 0.5 * (thrust_max - thrust_min);
                if (smoothedL1((thrust - thrust_mean) * (thrust - thrust_mean) - thrust_radius * thrust_radius, smooth,
                               value, derivative)) {
                    penalty += weight * value;
                    grad_thrust += weight * derivative * 2.0 * (thrust - thrust_mean);
                }
            }

            Eigen::Vector3d flat_grad_pos;
            Eigen::Vector3d flat_grad_vel;
            Eigen::Vector3d flat_grad_acc;
            Eigen::Vector3d flat_grad_jerk;
            double yaw_grad = 0.0;
            double yaw_rate_grad = 0.0;
            flatmap.backward(Eigen::Vector3d::Zero(), grad_vel, grad_thrust, grad_quat, grad_omega, flat_grad_pos,
                             flat_grad_vel, flat_grad_acc, flat_grad_jerk, yaw_grad, yaw_rate_grad);
            // backward already includes the direct velocity gradient.
            grad_vel = flat_grad_vel;
            grad_acc += flat_grad_acc;
            grad_jerk += flat_grad_jerk;

            const double node = (j == 0 || j == integral_resolution) ? 0.5 : 1.0;
            const double alpha = static_cast<double>(j) * integral_frac;
            grad_coeffs.block<6, 3>(i * 6, 0) += (beta1 * grad_vel.transpose() + beta2 * grad_acc.transpose() +
                                                  beta3 * grad_jerk.transpose() + beta4 * grad_snap.transpose()) *
                                                 node * step;
            grad_times(i) += (grad_vel.dot(acc) + grad_acc.dot(jerk) + grad_jerk.dot(snap) + grad_snap.dot(crackle)) *
                                 alpha * node * step +
                             node * integral_frac * penalty;
            cost += node * step * penalty;
        }
    }
}

class MincoWaypointOptimizer {
  public:
    explicit MincoWaypointOptimizer(const WaypointProblem3& problem) : problem_(problem) {
        piece_count_ = static_cast<int>(problem_.constraints.size()) - 1;
        spatial_offsets_.assign(problem_.constraints.size(), -1);
        for (size_t i = 1; i + 1U < problem_.constraints.size(); ++i) {
            const int dim = spatialDim(problem_.constraints[i]);
            if (dim > 0) {
                spatial_offsets_[i] = spatial_dim_;
                spatial_dim_ += dim;
            }
        }
        head_pva_.setZero();
        tail_pva_.setZero();
        head_pva_.col(0) = problem_.constraints.front().position;
        head_pva_.col(1) = problem_.start_velocity;
        head_pva_.col(2) = problem_.start_acceleration;
        tail_pva_.col(0) = problem_.constraints.back().position;
        tail_pva_.col(1) = problem_.end_velocity;
        tail_pva_.col(2) = problem_.end_acceleration;
        minco_.setConditions(head_pva_, tail_pva_, piece_count_);
    }

    bool optimize(PiecewisePolynomialEvaluator3& evaluator, uint32_t& flags) {
        if (piece_count_ <= 0) {
            flags |= kFlagInvalidInput;
            return false;
        }
        Eigen::VectorXd initial_times = initialSegmentTimes();
        if (!initial_times.array().isFinite().all()) {
            flags |= kFlagInvalidInput;
            return false;
        }
        Eigen::VectorXd x(piece_count_ + spatial_dim_);
        Eigen::VectorXd tau;
        backwardT(initial_times, tau);
        x.head(piece_count_) = tau;
        if (spatial_dim_ > 0) {
            x.tail(spatial_dim_).setZero();
        }

        xgc2_math::optimization::lbfgs::lbfgs_parameter_t params;
        params.mem_size = 128;
        params.past = 3;
        params.g_epsilon = 0.0;
        params.delta = std::max(1.0e-9, problem_.rel_cost_tol);
        params.max_iterations = std::max(1, problem_.max_iterations);
        params.min_step = 1.0e-32;

        double min_cost = 0.0;
        const int ret = xgc2_math::optimization::lbfgs::lbfgs_optimize(
            x, min_cost, &MincoWaypointOptimizer::costFunction, nullptr, nullptr, this, params);
        if (ret < 0 || !finiteScalar(min_cost)) {
            flags |= kFlagOptimizationFailure;
            return false;
        }

        Eigen::VectorXd times;
        Eigen::Matrix3Xd points;
        decode(x, times, points);
        minco_.setParameters(points, times);
        return buildEvaluator(times, minco_.getCoeffs(), evaluator, flags);
    }

  private:
    static double costFunction(void* ptr, const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        return static_cast<MincoWaypointOptimizer*>(ptr)->cost(x, grad);
    }

    double cost(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Eigen::VectorXd times;
        Eigen::Matrix3Xd points;
        decode(x, times, points);
        minco_.setParameters(points, times);

        double cost = 0.0;
        minco_.getEnergy(cost);
        Eigen::MatrixX3d partial_grad_coeffs;
        Eigen::VectorXd partial_grad_times;
        minco_.getEnergyPartialGradByCoeffs(partial_grad_coeffs);
        minco_.getEnergyPartialGradByTimes(partial_grad_times);
        dynamicPenalty(times, minco_.getCoeffs(), problem_, cost, partial_grad_times, partial_grad_coeffs);

        Eigen::Matrix3Xd grad_points;
        Eigen::VectorXd grad_times;
        minco_.propogateGrad(partial_grad_coeffs, partial_grad_times, grad_points, grad_times);
        cost += std::max(0.0, problem_.time_weight) * times.sum();
        grad_times.array() += std::max(0.0, problem_.time_weight);

        grad.setZero(x.size());
        Eigen::VectorXd grad_tau;
        backwardGradT(x.head(piece_count_), grad_times, grad_tau);
        grad.head(piece_count_) = grad_tau;
        if (spatial_dim_ > 0) {
            for (size_t i = 1; i + 1U < problem_.constraints.size(); ++i) {
                const int offset = spatial_offsets_[i];
                if (offset < 0) {
                    continue;
                }
                const Eigen::Vector3d xi = x.segment(piece_count_ + offset, 3);
                const Eigen::Vector3d grad_xi =
                    constraintGradXi(problem_.constraints[i], xi, grad_points.col(static_cast<int>(i) - 1));
                grad.segment(piece_count_ + offset, 3) = grad_xi;
            }
        }
        if (!finiteScalar(cost) || !grad.array().isFinite().all()) {
            grad.setZero(x.size());
            return std::numeric_limits<double>::infinity();
        }
        return cost;
    }

    Eigen::VectorXd initialSegmentTimes() const {
        Eigen::VectorXd times(piece_count_);
        for (int i = 0; i < piece_count_; ++i) {
            double duration = 0.0;
            if (static_cast<size_t>(i) < problem_.segment_times.size()) {
                duration = problem_.segment_times[static_cast<size_t>(i)];
            }
            if (!finiteScalar(duration) || duration <= 0.0) {
                const double distance = (problem_.constraints[static_cast<size_t>(i) + 1U].position -
                                         problem_.constraints[static_cast<size_t>(i)].position)
                                            .norm();
                duration = distance / std::max(0.1, problem_.desired_speed);
            }
            times(i) = clamp(duration, problem_.min_segment_time, problem_.max_segment_time);
        }
        return times;
    }

    void decode(const Eigen::VectorXd& x, Eigen::VectorXd& times, Eigen::Matrix3Xd& points) const {
        forwardT(x.head(piece_count_), times);
        for (int i = 0; i < times.size(); ++i) {
            times(i) = clamp(times(i), problem_.min_segment_time, problem_.max_segment_time);
        }
        points.resize(3, std::max(0, piece_count_ - 1));
        for (size_t i = 1; i + 1U < problem_.constraints.size(); ++i) {
            const int offset = spatial_offsets_[i];
            Eigen::Vector3d xi = Eigen::Vector3d::Zero();
            if (offset >= 0) {
                xi = x.segment(piece_count_ + offset, 3);
            }
            points.col(static_cast<int>(i) - 1) = mapConstraintPoint(problem_.constraints[i], xi);
        }
    }

    bool buildEvaluator(const Eigen::VectorXd& times, const Eigen::MatrixX3d& coefficients,
                        PiecewisePolynomialEvaluator3& evaluator, uint32_t& flags) const {
        std::vector<PolynomialSegment3> segments;
        segments.reserve(static_cast<size_t>(piece_count_));
        for (int i = 0; i < piece_count_; ++i) {
            PolynomialSegment3 segment;
            segment.duration = times(i);
            segment.x.resize(6U);
            segment.y.resize(6U);
            segment.z.resize(6U);
            for (size_t order = 0U; order < 6U; ++order) {
                const int row = 6 * i + static_cast<int>(order);
                segment.x[order] = coefficients(row, 0);
                segment.y[order] = coefficients(row, 1);
                segment.z[order] = coefficients(row, 2);
            }
            segments.push_back(std::move(segment));
        }
        if (!evaluator.setSegments(std::move(segments), 5U)) {
            flags |= kFlagInvalidInput;
            return false;
        }
        flags |= TrajectoryValidator3::validate(evaluator, problem_.limits, problem_.validation_sample_dt);
        return (flags & (kFlagInvalidInput | kFlagNonFinite | kFlagOptimizationFailure)) == 0U;
    }

    const WaypointProblem3& problem_;
    int piece_count_{0};
    int spatial_dim_{0};
    std::vector<int> spatial_offsets_;
    Eigen::Matrix3d head_pva_{Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d tail_pva_{Eigen::Matrix3d::Zero()};
    xgc2_math::optimization::minco::MINCO_S3NU minco_;
};

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

inline bool MincoWaypointSolver3::solve(const WaypointProblem3& problem, PiecewisePolynomialEvaluator3& evaluator,
                                        uint32_t* flags) const {
    uint32_t local_flags = kFlagNone;
    if (problem.constraints.size() < 2U) {
        local_flags |= kFlagInvalidInput;
        if (flags) {
            *flags = local_flags;
        }
        return false;
    }

    for (const auto& constraint : problem.constraints) {
        if (!trajectory3_detail::finiteVector(constraint.position) ||
            !trajectory3_detail::finiteVector(constraint.size) ||
            !trajectory3_detail::finiteScalar(constraint.orientation.norm()) ||
            constraint.orientation.norm() < trajectory3_detail::kMinNorm) {
            local_flags |= kFlagInvalidInput;
        }
    }
    if (!problem.segment_times.empty() && problem.segment_times.size() + 1U != problem.constraints.size()) {
        local_flags |= kFlagInvalidInput;
    }
    for (const double duration : problem.segment_times) {
        if (!trajectory3_detail::finiteScalar(duration) || duration <= trajectory3_detail::kMinDuration) {
            local_flags |= kFlagInvalidInput;
        }
    }
    if (!trajectory3_detail::finiteVector(problem.start_velocity) ||
        !trajectory3_detail::finiteVector(problem.start_acceleration) ||
        !trajectory3_detail::finiteVector(problem.end_velocity) ||
        !trajectory3_detail::finiteVector(problem.end_acceleration) ||
        !trajectory3_detail::finiteScalar(problem.desired_speed) ||
        !trajectory3_detail::finiteScalar(problem.time_weight) ||
        !trajectory3_detail::finiteScalar(problem.dynamic_penalty_weight) ||
        !trajectory3_detail::finiteScalar(problem.rel_cost_tol)) {
        local_flags |= kFlagInvalidInput;
    }
    if ((local_flags & kFlagInvalidInput) != 0U) {
        if (flags) {
            *flags = local_flags;
        }
        return false;
    }

    trajectory3_detail::MincoWaypointOptimizer optimizer(problem);
    const bool solved = optimizer.optimize(evaluator, local_flags);
    if (flags) {
        *flags = local_flags;
    }
    return solved && (local_flags & (kFlagInvalidInput | kFlagNonFinite | kFlagOptimizationFailure)) == 0U;
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
