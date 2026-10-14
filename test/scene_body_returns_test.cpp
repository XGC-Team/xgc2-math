#include "xgc2_math/scene/body_returns.hpp"
#include "xgc2_math/scene/world_lidar.hpp"
#include <stdexcept>
#include <iostream>

using namespace xgc2_math::scene;
namespace {
void require(bool valid) {
    if (!valid) throw std::runtime_error("body return geometry invariant failed");
}
}
int main() {
    const Eigen::Vector3d origin = Eigen::Vector3d::Zero(), direction(1, 0, 0);
    const Eigen::Quaterniond attitude = Eigen::Quaterniond::Identity();
    std::vector<VehicleBody> bodies{{7, {3, 0, 0}, .3}, {8, {5, 0, 0}, .3}};
    auto hit = nearestBody(origin, direction, 20, bodies);
    require(hit.vehicle_id == 7 && std::abs(hit.range - 2.7) < 1e-12);
    require(nearestBody(origin, {0, 1, 0}, 20, bodies).vehicle_id == -1);
    require(std::abs(nearestBody({3, 0, 0}, direction, 20, bodies).range - .3) < 1e-12);
    SensorConfig config;
    config.h_res = 360; config.v_res = 32; config.v_fov_deg = 60;
    BeamScan scan;
    scan.origin = origin; scan.direction = {direction, direction, {0, 1, 0}};
    scan.range = {2, 9, 20}; scan.hit = {1, 1, 0}; scan.vehicle_id = {-1, -1, -1};
    mergeBodyRays(config, bodies, &scan);
    require(scan.range[0] == 2 && scan.vehicle_id[0] == -1);
    require(std::abs(scan.range[1] - 2.7) < 1e-12 && scan.vehicle_id[1] == 7);
    require(!scan.hit[2]);
    scan.direction[2] = direction;
    mergeBodyRays(config, bodies, &scan);
    require(scan.hit[2] && scan.vehicle_id[2] == 7);

    config.mode = SensorConfig::kPenetrating;
    config.h_fov_deg = 120; config.v_fov_deg = 45;
    config.penetrating_heading_crop = true; config.heading_cos_min = .5; config.vertical_slab_tan = .2;
    WorldLidar reference(config);
    const auto expected = reference.scanTagged(origin, attitude, bodies);
    struct Sink {
        std::vector<TaggedPoint> points;
        void push(const Eigen::Vector3d& p, int id) { points.push_back({p, id}); }
    } sink;
    sampleBodyReturns(config, origin, attitude, bodies, sink);
    require(sink.points.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(sink.points[i].point == expected[i].point && sink.points[i].vehicle_id == expected[i].vehicle_id);

    config.mode = SensorConfig::kRaycast;
    config.h_fov_deg = 360; config.v_fov_deg = 60;
    const auto pattern = gpuBodyBeamPattern(config);
    const auto center = std::size_t(config.v_res / 2) * config.h_res + config.h_res / 2;
    require((pattern[center] - direction).norm() < 1e-12);
    std::vector<TaggedPoint> map{{{9, 0, 0}, -1}}, merged;
    GpuBodyMergeScratch scratch;
    mergeGpuBodyCloud(config, origin, attitude, bodies, map, &merged, &scratch);
    bool body = false, behind = false;
    for (const auto& point : merged) { body |= point.vehicle_id == 7; behind |= point.point.x() == 9; }
    require(body && !behind);
    bodies[0].position = {30, 0, 0}; bodies[1].position = {35, 0, 0};
    mergeGpuBodyCloud(config, origin, attitude, bodies, map, &merged, &scratch);
    require(merged.size() == 1 && merged[0].point == map[0].point);
    mergeGpuBodyCloud(config, origin, attitude, {}, map, &merged, &scratch);
    require(merged.size() == 1 && merged[0].point == map[0].point);
    std::cout << "PASS native sphere queries, empty-ray fill, nearest/static occlusion, exact penetrating FOV/crop, GPU grid\n";
}
