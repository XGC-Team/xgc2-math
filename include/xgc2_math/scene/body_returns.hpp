#pragma once

#include "xgc2_math/scene/sensor_model.hpp"
#include "xgc2_math/scene/shape_table.hpp"

#include <limits>

namespace xgc2_math::scene {

// Callers select eligible targets and exclude the scanner. Sensor enablement,
// dynamics and publication layout are outside these geometry queries.
struct BodyRayHit {
    double range = std::numeric_limits<double>::infinity();
    int vehicle_id = -1;
};

// origin/direction are finite, direction is unit, and bodies are valid spheres.
// A ray starting inside a body returns its exit surface, as WorldLidar does.
inline BodyRayHit nearestBody(const Eigen::Vector3d& origin, const Eigen::Vector3d& unit_direction, double max_range,
                              const std::vector<VehicleBody>& bodies) {
    BodyRayHit result;
    for (const auto& body : bodies) {
        double first, last;
        if (!detail::sphereInterval(origin, unit_direction, body.position, body.radius, &first, &last))
            continue;
        const double distance = first >= 0 ? first : last;
        if (distance >= 0 && distance <= max_range && distance < result.range)
            result = {distance, body.id};
    }
    return result;
}

inline void validateBodyReturns(const std::vector<VehicleBody>& bodies) {
    for (const auto& body : bodies)
        if (!body.position.allFinite() || !std::isfinite(body.radius) || body.radius <= 0)
            throw std::invalid_argument("invalid vehicle body");
}

// Append geometric sphere samples in native WorldLidar order. Sink supplies
// push(Vector3d, int). No static map, occlusion, transport layout or render flag
// enters this function. Callers retain their existing range-noise policy.
template <class Sink>
void sampleBodyReturns(const SensorConfig& config, const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                       const std::vector<VehicleBody>& bodies, Sink& sink) {
    if (bodies.empty())
        return;
    validateBodyReturns(bodies);
    const auto rotation = detail::rotationOf(attitude);
    const FieldOfView fov(config.h_fov_deg, config.v_fov_deg);
    const HeadingCrop crop(config.penetrating_heading_crop, config.heading_cos_min, config.vertical_slab_tan,
                           config.range);
    const double maximum = config.range * config.range;
    const double minimum = config.min_range * config.min_range;
    for (const auto& body : bodies) {
        if ((body.position - position).norm() - body.radius > config.range)
            continue;
        const int count = detail::sphereSampleCount(body.radius, config.surface_spacing);
        for (int i = 0; i < count; ++i) {
            const Eigen::Vector3d point = body.position + body.radius * detail::fibonacciPoint(i, count);
            const Eigen::Vector3d offset = point - position;
            const double squared = offset.squaredNorm();
            if (squared == 0 || squared > maximum || squared < minimum ||
                !fov.contains(rotation.transpose() * offset) || !crop.contains(offset, rotation.col(0)))
                continue;
            sink.push(point, body.id);
        }
    }
}

// Merge into an existing static BeamScan, including its no-hit rays. Never
// queries static geometry. Equal-distance ties retain the static hit. Query
// the closest surface before min-range filtering, matching WorldLidar.
inline void mergeBodyRays(const SensorConfig& config, const std::vector<VehicleBody>& bodies, BeamScan* scan) {
    if (bodies.empty())
        return;
    if (!scan || scan->direction.size() != scan->size() || scan->hit.size() != scan->size() ||
        scan->vehicle_id.size() != scan->size())
        throw std::invalid_argument("invalid body merge beam scan");
    validateBodyReturns(bodies);
    for (std::size_t i = 0; i < scan->size(); ++i) {
        const auto hit = nearestBody(scan->origin, scan->direction[i], config.range, bodies);
        if (hit.vehicle_id < 0 || (scan->hit[i] && hit.range >= scan->range[i]))
            continue;
        if (hit.range < config.min_range) {
            scan->hit[i] = 0;
            scan->range[i] = config.range;
            scan->vehicle_id[i] = -1;
        } else {
            scan->hit[i] = 1;
            scan->range[i] = hit.range;
            scan->vehicle_id[i] = hit.vehicle_id;
        }
    }
}

// The original spherical GPU depth buffer: column/row centres at resolution/2,
// steps FOV/resolution, +y/+z opposite the image axes. This is not the native
// CPU endpoint grid. It includes empty rays for dynamic-body-only returns.
inline std::vector<Eigen::Vector3d> gpuBodyBeamPattern(const SensorConfig& config) {
    std::vector<Eigen::Vector3d> result;
    result.reserve(std::size_t(config.h_res) * config.v_res);
    const double horizontal = config.h_fov_deg * detail::kPi / (180 * config.h_res);
    const double vertical = config.v_fov_deg * detail::kPi / (180 * config.v_res);
    for (int row = 0; row < config.v_res; ++row) {
        const double elevation = (0.5 * config.v_res - row) * vertical;
        for (int column = 0; column < config.h_res; ++column) {
            const double azimuth = (0.5 * config.h_res - column) * horizontal;
            result.emplace_back(std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth),
                                std::sin(elevation));
        }
    }
    return result;
}

struct GpuBodyMergeScratch {
    SensorConfig pattern_config;
    std::vector<Eigen::Vector3d> local_directions;
    BeamScan scan;
    std::vector<int> static_indices;
};

// Compose already sampled GPU static returns with dynamic bodies. Static
// points are world-space TaggedPoints (-1 source); no static-map CPU scan or
// upload occurs. Optional scratch preserves capacities across observations.
// Penetrating keeps every static return and appends native sphere samples.
inline void mergeGpuBodyCloud(const SensorConfig& config, const Eigen::Vector3d& position,
                              const Eigen::Quaterniond& attitude, const std::vector<VehicleBody>& bodies,
                              const std::vector<TaggedPoint>& static_points, std::vector<TaggedPoint>* output,
                              GpuBodyMergeScratch* scratch = nullptr) {
    if (!output || output == &static_points)
        throw std::invalid_argument("distinct body merge output required");
    output->clear();
    if (bodies.empty() || config.mode == SensorConfig::kPenetrating) {
        output->assign(static_points.begin(), static_points.end());
        if (bodies.empty())
            return;
        struct Sink {
            std::vector<TaggedPoint>* points;
            void push(const Eigen::Vector3d& point, int id) { points->push_back({point, id}); }
        } sink{output};
        sampleBodyReturns(config, position, attitude, bodies, sink);
        return;
    }
    GpuBodyMergeScratch local;
    auto& work = scratch ? *scratch : local;
    const auto& prior = work.pattern_config;
    if (work.local_directions.empty() || prior.h_res != config.h_res || prior.v_res != config.v_res ||
        prior.h_fov_deg != config.h_fov_deg || prior.v_fov_deg != config.v_fov_deg) {
        work.local_directions = gpuBodyBeamPattern(config);
        work.pattern_config = config;
    }
    const auto rotation = detail::rotationOf(attitude);
    auto& scan = work.scan;
    scan.origin = position;
    const auto count = work.local_directions.size();
    scan.direction.resize(count);
    scan.range.assign(count, config.range);
    scan.hit.assign(count, 0);
    scan.vehicle_id.assign(count, -1);
    work.static_indices.assign(count, -1);
    for (std::size_t i = 0; i < count; ++i)
        scan.direction[i] = rotation * work.local_directions[i];
    const double horizontal = config.h_fov_deg * detail::kPi / (180 * config.h_res);
    const double vertical = config.v_fov_deg * detail::kPi / (180 * config.v_res);
    for (std::size_t i = 0; i < static_points.size(); ++i) {
        const Eigen::Vector3d offset = rotation.transpose() * (static_points[i].point - position);
        const double distance = offset.norm();
        if (!(distance > 0) || !std::isfinite(distance))
            continue;
        long column = std::lround(0.5 * config.h_res - std::atan2(offset.y(), offset.x()) / horizontal);
        const long row =
            std::lround(0.5 * config.v_res - std::atan2(offset.z(), std::hypot(offset.x(), offset.y())) / vertical);
        if (config.h_fov_deg == 360)
            column = (column % config.h_res + config.h_res) % config.h_res;
        if (column < 0 || column >= config.h_res || row < 0 || row >= config.v_res)
            continue;
        const auto at = std::size_t(row) * config.h_res + column;
        if (!scan.hit[at] || distance < scan.range[at]) {
            scan.range[at] = distance;
            scan.hit[at] = 1;
            scan.vehicle_id[at] = static_points[i].vehicle_id;
            work.static_indices[at] = int(i);
        }
    }
    mergeBodyRays(config, bodies, &scan);
    output->reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!scan.hit[i])
            continue;
        if (scan.vehicle_id[i] >= 0)
            output->push_back({scan.point(i), scan.vehicle_id[i]});
        else if (work.static_indices[i] >= 0)
            output->push_back(static_points[work.static_indices[i]]);
    }
}
} // namespace xgc2_math::scene
