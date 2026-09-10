#pragma once

#include <cmath>

#include <Eigen/Geometry>

namespace xgc2_math {

struct Pose3 {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

inline bool isFinite(double value) {
    return std::isfinite(value);
}

inline bool isFinite(const Eigen::Vector3d& value) {
    return value.allFinite();
}

inline bool isFinite(const Eigen::Quaterniond& value) {
    return std::isfinite(value.w()) && std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

inline bool isFinite(const Pose3& pose) {
    return isFinite(pose.position) && isFinite(pose.orientation);
}

inline Eigen::Quaterniond normalizedQuaternion(Eigen::Quaterniond q) {
    if (!isFinite(q)) {
        return Eigen::Quaterniond::Identity();
    }
    // Scale before taking the norm so finite coefficients cannot overflow
    // the sum of squares and normalize to an all-zero quaternion.
    const double scale = q.coeffs().cwiseAbs().maxCoeff();
    if (scale == 0.0) {
        return Eigen::Quaterniond::Identity();
    }
    q.coeffs() /= scale;
    const double scaled_norm = q.norm();
    if (scale <= 1.0e-12 && scale * scaled_norm <= 1.0e-12) {
        return Eigen::Quaterniond::Identity();
    }
    q.coeffs() /= scaled_norm;
    if (q.w() < 0.0) {
        q.coeffs() *= -1.0;
    }
    return q;
}

inline Eigen::Quaterniond rpyToQuaternion(const Eigen::Vector3d& rpy) {
    const Eigen::AngleAxisd roll(rpy.x(), Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch(rpy.y(), Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw(rpy.z(), Eigen::Vector3d::UnitZ());
    return normalizedQuaternion(yaw * pitch * roll);
}

inline Eigen::Quaterniond expMap(const Eigen::Vector3d& rotation_vector) {
    const double angle = rotation_vector.norm();
    if (!std::isfinite(angle) || angle <= 1.0e-12) {
        return Eigen::Quaterniond::Identity();
    }
    return normalizedQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle)));
}

inline Eigen::Vector3d logMap(const Eigen::Quaterniond& rotation) {
    const Eigen::Quaterniond q = normalizedQuaternion(rotation);
    const Eigen::Vector3d v(q.x(), q.y(), q.z());
    const double sin_half_angle = v.norm();
    if (sin_half_angle <= 1.0e-12) {
        return Eigen::Vector3d::Zero();
    }
    const double angle = 2.0 * std::atan2(sin_half_angle, q.w());
    return angle * v / sin_half_angle;
}

inline Pose3 compose(const Pose3& lhs, const Pose3& rhs) {
    Pose3 result;
    const Eigen::Quaterniond lhs_q = normalizedQuaternion(lhs.orientation);
    result.position = lhs.position + lhs_q * rhs.position;
    result.orientation = normalizedQuaternion(lhs_q * rhs.orientation);
    return result;
}

inline Pose3 inverse(const Pose3& pose) {
    Pose3 result;
    result.orientation = normalizedQuaternion(pose.orientation).conjugate();
    result.position = -(result.orientation * pose.position);
    return result;
}

inline Eigen::Vector3d clampNorm(const Eigen::Vector3d& value, double max_norm) {
    if (!isFinite(value) || !std::isfinite(max_norm) || max_norm <= 0.0) {
        return Eigen::Vector3d::Zero();
    }
    const double norm = value.norm();
    if (norm <= max_norm) {
        return value;
    }
    return value * (max_norm / norm);
}

inline Eigen::Matrix<double, 6, 1> se3Error(const Pose3& reference, const Pose3& measured) {
    const Pose3 delta = compose(inverse(reference), measured);
    Eigen::Matrix<double, 6, 1> result;
    result.head<3>() = delta.position;
    result.tail<3>() = logMap(delta.orientation);
    return result;
}

} // namespace xgc2_math
