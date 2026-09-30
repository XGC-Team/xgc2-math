#pragma once

#include "xgc2_math/trajectory/trajectory3.hpp"
#include "xgc2_math/optimization/lbfgs.hpp"
#include "xgc2_math/optimization/minco.hpp"

// Explicit planning layer. Trajectory evaluation and analytic entry curves
// do not include this optimizer or its MINCO/LBFGS dependencies.
namespace xgc2_math::trajectory {

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

namespace trajectory3_detail {

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

} // namespace xgc2_math::trajectory
