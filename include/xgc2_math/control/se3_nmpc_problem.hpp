#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace xgc2_math::control {

using Se3StateVector = Eigen::Matrix<double, 13, 1>;
using Se3ControlVector = Eigen::Matrix<double, 4, 1>;

struct Se3State {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond attitude{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d body_rate{Eigen::Vector3d::Zero()};
};

struct Se3Control {
    double body_z_specific_force{0.0};
    Eigen::Vector3d angular_acceleration{Eigen::Vector3d::Zero()};
};

struct Se3Reference {
    Se3State state{};
    Se3Control control{};
    uint32_t flags{0U};
};

struct Se3NmpcProblemHorizon {
    std::vector<Se3Reference> references;
    double stage_dt_s{0.0};

    bool validForSteps(int horizon_steps) const {
        return horizon_steps > 0 && stage_dt_s > 0.0 && references.size() >= static_cast<size_t>(horizon_steps + 1);
    }
};

struct Se3NmpcProblemConfig {
    int horizon_steps{10};
    double prediction_horizon_s{1.0};
    double control_period_s{0.1};
    double gravity_mps2{9.8066};
    double min_body_z_specific_force{0.1};
    double max_body_z_specific_force{0.0};
    double max_body_rate{0.0};
};

struct Se3NmpcProblemResult {
    bool success{false};
    int status{0};
    double solve_time_ms{0.0};
    double max_quaternion_norm_error{0.0};
    Se3Control first_control{};
    std::vector<Se3State> predicted_states;
    std::vector<Se3Control> predicted_controls;
};

class Se3NmpcProblemBackend {
  public:
    virtual ~Se3NmpcProblemBackend() = default;
    virtual bool initialize() = 0;
    virtual void resetWarmStart() = 0;
    virtual bool solve(const Se3State& initial_state, const Se3NmpcProblemHorizon& horizon) = 0;
    virtual Se3NmpcProblemResult result() const = 0;
    virtual int horizonSteps() const = 0;
};

inline bool isFinite(const Eigen::VectorXd& value) {
    return value.array().isFinite().all();
}

inline Eigen::Quaterniond normalizedQuaternionOrIdentity(Eigen::Quaterniond quat) {
    if (!std::isfinite(quat.norm()) || quat.norm() < 1.0e-9) {
        return Eigen::Quaterniond::Identity();
    }
    quat.normalize();
    if (quat.w() < 0.0) {
        quat.coeffs() *= -1.0;
    }
    return quat;
}

inline Eigen::Matrix<double, 4, 1> quaternionToVectorWxyz(const Eigen::Quaterniond& quat_in) {
    const Eigen::Quaterniond quat = normalizedQuaternionOrIdentity(quat_in);
    Eigen::Matrix<double, 4, 1> value;
    value << quat.w(), quat.x(), quat.y(), quat.z();
    return value;
}

inline Eigen::Quaterniond vectorWxyzToQuaternion(const Eigen::Matrix<double, 4, 1>& value) {
    return normalizedQuaternionOrIdentity(Eigen::Quaterniond(value(0), value(1), value(2), value(3)));
}

inline Eigen::Matrix3d projectRotationToSo3(const Eigen::Matrix3d& value) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(value, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d projected = svd.matrixU() * svd.matrixV().transpose();
    if (projected.determinant() < 0.0) {
        Eigen::Matrix3d u = svd.matrixU();
        u.col(2) *= -1.0;
        projected = u * svd.matrixV().transpose();
    }
    return projected;
}

inline Eigen::Vector3d normalizedVectorOr(const Eigen::Vector3d& value, const Eigen::Vector3d& fallback) {
    const double norm = value.norm();
    if (!std::isfinite(norm) || norm < 1.0e-9) {
        return fallback;
    }
    return value / norm;
}

inline Eigen::Matrix3d rotationFromBodyZAndYaw(const Eigen::Vector3d& body_z, double yaw) {
    const Eigen::Vector3d z_b = normalizedVectorOr(body_z, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d x_c(std::cos(yaw), std::sin(yaw), 0.0);
    Eigen::Vector3d y_b = z_b.cross(x_c);
    if (!std::isfinite(y_b.norm()) || y_b.norm() < 1.0e-8) {
        // UnitY is also singular when body_z is parallel to the Y axis.
        // Choose a reference axis that is guaranteed not to be parallel.
        x_c = std::abs(z_b.y()) > 0.95 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
        y_b = z_b.cross(x_c);
    }
    y_b = normalizedVectorOr(y_b, Eigen::Vector3d::UnitY());
    const Eigen::Vector3d x_b = normalizedVectorOr(y_b.cross(z_b), Eigen::Vector3d::UnitX());
    Eigen::Matrix3d rotation;
    rotation.col(0) = x_b;
    rotation.col(1) = y_b;
    rotation.col(2) = z_b;
    return rotation;
}

inline Se3StateVector packState(const Se3State& state) {
    Se3StateVector value = Se3StateVector::Zero();
    value.segment<3>(0) = state.position;
    value.segment<3>(3) = state.velocity;
    value.segment<4>(6) = quaternionToVectorWxyz(state.attitude);
    value.segment<3>(10) = state.body_rate;
    return value;
}

inline Se3State unpackState(const Se3StateVector& value) {
    Se3State state;
    state.position = value.segment<3>(0);
    state.velocity = value.segment<3>(3);
    state.attitude = vectorWxyzToQuaternion(value.segment<4>(6));
    state.body_rate = value.segment<3>(10);
    return state;
}

inline Se3ControlVector packControl(const Se3Control& control) {
    Se3ControlVector value = Se3ControlVector::Zero();
    value(0) = control.body_z_specific_force;
    value.segment<3>(1) = control.angular_acceleration;
    return value;
}

inline Se3Control unpackControl(const Se3ControlVector& value) {
    Se3Control control;
    control.body_z_specific_force = value(0);
    control.angular_acceleration = value.segment<3>(1);
    return control;
}

} // namespace xgc2_math::control
