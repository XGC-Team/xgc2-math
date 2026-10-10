#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgc2_math/geometry/se3.hpp"

namespace xgc2_math::scene {

// A sensor observes bodies through world-frame poses. A frame label is not a transform, so a pose is accepted only if
// it is labelled with the expected frame and holds a finite position and a non-degenerate quaternion.
inline bool validObservationPose(const Pose3& pose, const std::string& frame, const std::string& expected_frame) {
    return frame == expected_frame && pose.position.allFinite() && pose.orientation.coeffs().allFinite() &&
           std::isfinite(pose.orientation.squaredNorm()) && pose.orientation.squaredNorm() > 1e-12;
}

// Zero is a legitimate simulation time, not permission to stamp stale data now. The sample is fresh when it is not from
// the future and at most `timeout` seconds old.
inline bool freshObservationTime(double stamp, double now, double timeout) {
    return std::isfinite(stamp) && std::isfinite(now) && std::isfinite(timeout) && stamp >= 0.0 && now >= stamp &&
           timeout > 0.0 && now - stamp <= timeout;
}

// Mounting pose of a sensor on its body from (x, y, z, roll, pitch, yaw) in metres and radians; the rotation is
// Rz(yaw) * Ry(pitch) * Rx(roll). Throws std::invalid_argument for any other length or a non-finite value.
inline Pose3 mountingPose(const std::vector<double>& xyz_rpy) {
    if (xyz_rpy.size() != 6) {
        throw std::invalid_argument("sensor_pose needs x y z roll pitch yaw (metres/radians)");
    }
    for (double value : xyz_rpy) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("sensor_pose must be finite");
        }
    }
    Pose3 pose;
    pose.position = Eigen::Vector3d(xyz_rpy[0], xyz_rpy[1], xyz_rpy[2]);
    pose.orientation = Eigen::AngleAxisd(xyz_rpy[5], Eigen::Vector3d::UnitZ()) *
                       Eigen::AngleAxisd(xyz_rpy[4], Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(xyz_rpy[3], Eigen::Vector3d::UnitX());
    return pose;
}

// World pose of a sensor from the world pose of its body and its mounting pose. Throws std::invalid_argument if either
// pose is not valid.
inline Pose3 sensorWorldPose(const Pose3& body, const Pose3& mount) {
    if (!validObservationPose(body, "world", "world") || !validObservationPose(mount, "world", "world")) {
        throw std::invalid_argument("invalid body or sensor mounting pose");
    }
    const Eigen::Quaterniond rotation = body.orientation.normalized();
    Pose3 result;
    result.position = body.position + rotation * mount.position;
    result.orientation = rotation * mount.orientation.normalized();
    return result;
}

} // namespace xgc2_math::scene
