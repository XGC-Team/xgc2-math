#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace xgc2_math::scene {

// The sensor's caller-owned configuration, not a wire schema or paper preset.
struct SensorMetadata {
    std::string observation_model; // crop_through, pinhole_depth, lidar_scan
    std::string backend;           // cpu or gpu, independent execution choice
    std::array<float, 3> prevoxel_leaf_m{{0, 0, 0}};
    double range_m = 0;
    std::optional<double> heading_cos_min;   // absent: no body-dot predicate
    std::optional<double> vertical_slab_tan; // absent: no fixed world-Z slab
    double publish_rate_hz = 0;
    std::string frame_id, stamp_policy; // explicit caller map/world and zero/pose
    std::string input_cloud_topic, pose_topic, output_topic, pose_type;
    // lidar_scan/gpu only: original spherical-nearest grid and point-cover input. CPU crop does not read these fields
    // or acquire camera/pinhole semantics.
    double min_range_m = 0, h_fov_deg = 0, v_fov_deg = 0;
    int h_res = 0, v_res = 0;
    double point_cover_spacing_m = 0;
};

// The CPU crop-through slice: a point cloud cropped around the sensor by radius, heading cone and vertical slab.
// Projection and occlusion follow that model; camera declarations remain caller provenance and do not enter this crop.
// Other combinations of backend and observation model are refused rather than silently substituted. Throws
// std::invalid_argument.
inline void validateSensorMetadata(const SensorMetadata& m) {
    if (m.backend != "cpu" || m.observation_model != "crop_through") {
        throw std::invalid_argument(
            "first shared CPU slice supports crop_through/cpu; other combinations are not silently substituted");
    }
    if (!(m.range_m > 0 && std::isfinite(m.range_m) && m.publish_rate_hz > 0 && std::isfinite(m.publish_rate_hz)) ||
        m.frame_id.empty() || (m.stamp_policy != "zero" && m.stamp_policy != "pose")) {
        throw std::invalid_argument("actual radius/rate/frame/stamp required");
    }
    for (auto leaf : m.prevoxel_leaf_m) {
        if (!(leaf > 0 && std::isfinite(leaf))) {
            throw std::invalid_argument("actual PCL voxel XYZ required");
        }
    }
    if (m.heading_cos_min &&
        (!std::isfinite(*m.heading_cos_min) || *m.heading_cos_min < -1 || *m.heading_cos_min > 1)) {
        throw std::invalid_argument("invalid body-dot predicate");
    }
    if (m.vertical_slab_tan && (!std::isfinite(*m.vertical_slab_tan) || *m.vertical_slab_tan < 0)) {
        throw std::invalid_argument("invalid world-Z predicate");
    }
}

// The spherical GPU projection (lidar_scan/gpu): validates the native float parameter domain of the original
// implementation, not demo values. Throws std::invalid_argument.
inline void validateGpuSensorMetadata(const SensorMetadata& m) {
    if (m.observation_model != "lidar_scan" || m.backend != "gpu") {
        throw std::invalid_argument("GPU requires explicit lidar_scan/gpu (spherical nearest)");
    }
    const float near = static_cast<float>(m.min_range_m), far = static_cast<float>(m.range_m);
    const float spacing = static_cast<float>(m.point_cover_spacing_m);
    if (!std::isfinite(near) || !std::isfinite(far) || near <= 0 || far <= near || !std::isfinite(spacing) ||
        spacing <= 0 || !std::isfinite(m.publish_rate_hz) || m.publish_rate_hz <= 0 ||
        m.publish_rate_hz > std::numeric_limits<int>::max() || m.h_res <= 0 || m.v_res <= 0 ||
        !std::isfinite(m.h_fov_deg) || !std::isfinite(m.v_fov_deg) || m.h_fov_deg <= 0 || m.h_fov_deg > 360 ||
        m.v_fov_deg <= 0 || m.v_fov_deg > 180 || m.h_res > std::numeric_limits<int>::max() / m.v_res) {
        throw std::invalid_argument("finite positive GPU range/spacing/rate and spherical grid required");
    }
    const float h_step = static_cast<float>(m.h_fov_deg) / static_cast<float>(m.h_res);
    const float v_step = static_cast<float>(m.v_fov_deg) / static_cast<float>(m.v_res);
    if (!std::isfinite(h_step) || h_step <= 0 || !std::isfinite(v_step) || v_step <= 0) {
        throw std::invalid_argument("finite positive GPU horizontal and vertical angular steps required");
    }
    // Original shader asin(cover_dis/depth), with depth >= near.
    const float shader_cover = 0.55f * 1.7321f * spacing;
    const float host_cover = static_cast<float>(0.55 * 1.7321 * spacing);
    if (!std::isfinite(shader_cover) || !std::isfinite(host_cover) || shader_cover > near || host_cover > near) {
        throw std::invalid_argument("original GPU point-cover/near ratio must be <=1 for asin");
    }
    if (m.frame_id.empty() || (m.stamp_policy != "zero" && m.stamp_policy != "pose") ||
        (m.pose_type != "nav_msgs/Odometry" && m.pose_type != "geometry_msgs/PoseStamped")) {
        throw std::invalid_argument("explicit GPU frame/pose-type/stamp required");
    }
    if (m.heading_cos_min || m.vertical_slab_tan) {
        throw std::invalid_argument("GPU does not implement CPU crop predicates");
    }
    if (std::any_of(m.prevoxel_leaf_m.begin(), m.prevoxel_leaf_m.end(), [](float leaf) {
            return leaf != 0;
        })) {
        throw std::invalid_argument("GPU does not implement CPU prevoxel");
    }
}

} // namespace xgc2_math::scene
