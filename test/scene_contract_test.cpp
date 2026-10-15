// Descriptions, contracts and construction math of xgc2_math::scene: scene documents and Gazebo obstacle truth to
// obstacles, the observation contract (pose validity, mounting), sensor parameter domains, placement and motion of
// obstacles, the closed-convex-surface check of scene documents, and the primitives and polytopes of cluttered
// environments. Expected values are analytic or come from an independent computation, and the cases use fixed seeds.

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgc2_math/geometry/collision/distance_gjk_query.h"
#include "xgc2_math/scene.hpp"

namespace {

using namespace xgc2_math::scene;
using xgc2_math::Pose3;

constexpr std::uint64_t kSeed = 0x58474332434f4eULL;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Cases {
    std::mt19937_64 engine{kSeed};
    double unit() { return static_cast<double>(engine() >> 11) / 9007199254740992.0; }
    double scalar() { return 2.0 * unit() - 1.0; }
    double between(double lo, double hi) { return lo + (hi - lo) * unit(); }
    double gaussian() {
        const double u1 = 1.0 - unit(), u2 = unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
    Eigen::Vector3d vector3(double half_extent = 1.0) {
        const double x = scalar(), y = scalar(), z = scalar();
        return half_extent * Eigen::Vector3d(x, y, z);
    }
    Eigen::Quaterniond rotation() {
        const double w = gaussian(), x = gaussian(), y = gaussian(), z = gaussian();
        return Eigen::Quaterniond(w, x, y, z).normalized();
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, const std::string& context) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << std::setprecision(17) << context << ": actual=" << actual << ", expected=" << expected
            << ", tolerance=" << tolerance;
        throw std::runtime_error(out.str());
    }
}

void vectorNear(const Eigen::Vector3d& actual, const Eigen::Vector3d& expected, double tolerance,
                const std::string& context) {
    require(actual.allFinite() && (actual - expected).norm() <= tolerance,
            context + ": (" + std::to_string(actual.x()) + ", " + std::to_string(actual.y()) + ", " +
                std::to_string(actual.z()) + ") vs (" + std::to_string(expected.x()) + ", " +
                std::to_string(expected.y()) + ", " + std::to_string(expected.z()) + ")");
}

template <class F> std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return std::string();
}

template <class F> bool refuses(F&& f, const std::string& needle) {
    const std::string message = refusal(f);
    return !message.empty() && message.find(needle) != std::string::npos;
}

Pose3 pose(double x, double y, double z, double yaw = 0.0) {
    Pose3 p;
    p.position = Eigen::Vector3d(x, y, z);
    p.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    return p;
}

// obstacle * part
Pose3 place(const Pose3& a, const Pose3& b) {
    Pose3 out;
    out.position = a.position + a.orientation * b.position;
    out.orientation = a.orientation * b.orientation;
    return out;
}

Eigen::Matrix4d homogeneous(const Pose3& pose) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m.topLeftCorner<3, 3>() = pose.orientation.normalized().toRotationMatrix();
    m.topRightCorner<3, 1>() = pose.position;
    return m;
}

// ---- scene documents and Gazebo obstacle truth ---------------------------------------------------------------------

// Local-frame vertices of a triangular prism (convex part).
std::vector<Eigen::Vector3d> prism() {
    return {{-0.5, -0.4, 0.0}, {0.5, -0.4, 0.0}, {0.0, 0.5, 0.0}, {-0.5, -0.4, 2.0}, {0.5, -0.4, 2.0}, {0.0, 0.5, 2.0}};
}

// Scene document: a box, a cylinder, a sphere, a convex prism and a two-post gate (one obstacle, two parts) with an
// open middle.
using xgc2_math::geometry::encodeGeometry;
using xgc2_math::geometry::Geometry;
std::vector<SceneObstacleDescription> documentScene() {
    const auto convex = [] {
        std::vector<xgc2_math::geometry::GeometryVertex> v;
        const auto points = prism();
        std::transform(points.begin(), points.end(), std::back_inserter(v), [](const auto& p) {
            return xgc2_math::geometry::GeometryVertex{p.x(), p.y(), p.z()};
        });
        return Geometry::convex(v);
    }();
    return {{pose(6, 0, 1, .3), {{encodeGeometry(Geometry::box({1, 2, 2})), pose(0, 0, 0)}}},
            {pose(0, 5, 1.5), {{encodeGeometry(Geometry::cylinder(.4, 3)), pose(0, 0, 0)}}},
            {pose(-4, -3, 1), {{encodeGeometry(Geometry::sphere(.7)), pose(0, 0, 0)}}},
            {pose(-5, 2, 0, -.7), {{encodeGeometry(convex), pose(0, 0, 0)}}},
            {pose(0, -6, 0, .2),
             {{encodeGeometry(Geometry::box({.3, .3, 2})), pose(0, -1, 1)},
              {encodeGeometry(Geometry::box({.3, .3, 2})), pose(0, 1, 1)}}},
            {pose(3, 3, 0), {{encodeGeometry(Geometry::capsule(.25, 1.5)), pose(0, 0, 1)}}}};
}
void documentConversion() {
    SceneObstacleDescription o;
    o.pose = pose(0, 0, 1, kPi / 2);
    ScenePartDescription part{encodeGeometry(Geometry::box({1, 2, 2})), pose(4, 0, 0)};
    o.parts.push_back(part);
    auto result = toObstacles({o});
    require(result.size() == 1 && result[0].type == Obstacle::kBox, "one numerical shape per part");
    vectorNear(result[0].position, {0, 4, 1}, 1e-12, "composed part position");
    WorldLidar lidar(SensorConfig{});
    lidar.setScene(result);
    near(lidar.castRay({0, 0, 1}, Eigen::Vector3d::UnitY(), 20), 3.5, 1e-9, "rotated box ray");
    o.pose.orientation.coeffs() *= 4;
    vectorNear(toObstacles({o})[0].position, {0, 4, 1}, 1e-12, "normalized quaternion");
    o.parts[0].geometry.type = 42;
    require(refuses(
                [&] {
                    toObstacles({o});
                },
                "unknown geometry"),
            "unknown payload type");
    o.parts[0] = part;
    o.parts[0].pose.orientation = Eigen::Quaterniond(0, 0, 0, 0);
    require(refuses(
                [&] {
                    toObstacles({o});
                },
                "orientation"),
            "invalid orientation");
    require(toObstacles({}).empty(), "empty numerical input");
}
void gazeboConversion() {
    const auto box = obstacleFromGeometry(encodeGeometry(Geometry::box({1, 2, 4})), {1, 2, 3});
    vectorNear(box.size, {1, 2, 4}, 1e-12, "full size");
    vectorNear(box.position, {1, 2, 3}, 1e-12, "position");
    const auto cylinder = obstacleFromGeometry(encodeGeometry(Geometry::cylinder(.5, 3)));
    require(cylinder.radius == .5 && cylinder.height == 3, "full cylinder height");
    const auto capsule = obstacleFromGeometry(encodeGeometry(Geometry::capsule(.25, 1.5)));
    require(capsule.radius == .25 && capsule.height == 1.5, "straight capsule height");
    const auto sphere = obstacleFromGeometry(encodeGeometry(Geometry::sphere(.25)));
    require(sphere.radius == .25, "sphere radius");
    require(refuses(
                [&] {
                    obstacleFromGeometry(encodeGeometry(Geometry::sphere(1)), {kInf, 0, 0});
                },
                "position"),
            "invalid position");
    require(refuses(
                [&] {
                    obstacleFromGeometry(encodeGeometry(Geometry::sphere(1)), {}, Eigen::Quaterniond(0, 0, 0, 0));
                },
                "orientation"),
            "invalid quaternion");
}

void sameScansFromBothSources() {
    // The same toy scene described once as a scene document and once as the Gazebo obstacle truth gives the same
    // solids, hence the same scans.
    const std::vector<Obstacle> from_document = toObstacles(documentScene());
    std::vector<Obstacle> from_gazebo;
    for (const auto& o : documentScene())
        for (const auto& p : o.parts) {
            const auto world = place(o.pose, p.pose);
            from_gazebo.push_back(obstacleFromGeometry(p.geometry, world.position, world.orientation));
        }
    require(from_document.size() == 7, "seven solids in the document");
    require(from_gazebo.size() == from_document.size(), "as many from Gazebo");
    const std::vector<Pose3> sensors = {pose(0, 0, 1.0), pose(1.5, -2.0, 1.2, 0.8), pose(-2.0, 1.0, 0.5, -2.0)};
    for (const auto mode : {SensorConfig::kRaycast, SensorConfig::kDepthFrustum, SensorConfig::kPenetrating}) {
        SensorConfig c;
        c.mode = mode;
        c.range = 15.0;
        c.h_res = 180;
        c.v_res = 16;
        c.v_fov_deg = mode == SensorConfig::kPenetrating ? 180.0 : 30.0;
        c.h_fov_deg = mode == SensorConfig::kDepthFrustum ? 90.0 : 360.0;
        c.surface_spacing = 0.2;
        WorldLidar document_lidar(c);
        WorldLidar gazebo_lidar(c);
        document_lidar.setScene(from_document);
        gazebo_lidar.setScene(from_gazebo);
        for (const Pose3& s : sensors) {
            const auto a = document_lidar.scan(s.position, s.orientation);
            const auto b = gazebo_lidar.scan(s.position, s.orientation);
            require(!a.empty(), "mode " + std::to_string(mode) + ": nothing seen");
            require(a.size() == b.size(), "mode " + std::to_string(mode) + ": " + std::to_string(a.size()) + " vs " +
                                              std::to_string(b.size()) + " points");
            double worst = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                worst = std::max(worst, (a[i] - b[i]).norm());
            }
            require(worst < 1e-6, "mode " + std::to_string(mode) + ": worst point difference " + std::to_string(worst));
        }
    }
    // The gate stays open: a beam through its middle passes.
    WorldLidar lidar(SensorConfig{});
    lidar.setScene(from_gazebo);
    const Eigen::Vector3d origin = pose(0.0, -6.0, 0.0, 0.2).position + Eigen::Vector3d(0, 0, 1.0);
    const Eigen::Vector3d through = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()) * Eigen::Vector3d::UnitX();
    require(std::isinf(lidar.castRay(origin - 3.0 * through, through, 6.0)), "a beam through the gate");
}

// ---- observation contract and sensor parameters
// -----------------------------------------------------------------------

void observationContract() {
    Pose3 body;
    require(validObservationPose(body, "world", "world"), "identity pose in the expected frame");
    require(!validObservationPose(body, "map", "world"), "a frame label is not a transform");
    body.orientation.coeffs().setZero();
    require(!validObservationPose(body, "world", "world"), "zero quaternion");
    body.orientation = Eigen::Quaterniond(kNaN, 0, 0, 0);
    require(!validObservationPose(body, "world", "world"), "NaN quaternion");
    body.orientation = Eigen::Quaterniond::Identity();
    body.position.x() = kNaN;
    require(!validObservationPose(body, "world", "world"), "NaN position");
    body.position.x() = kInf;
    require(!validObservationPose(body, "world", "world"), "infinite position");
    // Zero is a legitimate simulation time, and so is the edge of the timeout.
    require(freshObservationTime(0, 0, 0.5) && freshObservationTime(0, 0.5, 0.5), "stamp zero");
    require(!freshObservationTime(0, 0.6, 0.5), "stale");
    require(!freshObservationTime(10, 1, 0.5), "from the future");
    require(!freshObservationTime(1, 10, 0.5), "long ago");
    require(!freshObservationTime(-1, 0, 0.5), "negative stamp");
    require(!freshObservationTime(1, 1, 0), "no timeout");
    require(!freshObservationTime(1, 1, kInf), "infinite timeout");
    require(!freshObservationTime(kNaN, 1, 1) && !freshObservationTime(1, kNaN, 1) && !freshObservationTime(1, 1, kNaN),
            "NaN time");

    // Mounting: x y z roll pitch yaw, rotation Rz(yaw) Ry(pitch) Rx(roll).
    Cases cases;
    for (int i = 0; i < 20; ++i) {
        const double r = cases.between(-kPi, kPi), p = cases.between(-1.5, 1.5), y = cases.between(-kPi, kPi);
        const Pose3 mount = mountingPose({1.0, -2.0, 3.0, r, p, y});
        vectorNear(mount.position, Eigen::Vector3d(1, -2, 3), 0.0, "mount position");
        const Eigen::Matrix3d expected =
            (Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()))
                .toRotationMatrix();
        require((mount.orientation.toRotationMatrix() - expected).norm() < 1e-14, "mount rotation");
    }
    for (const auto& mount : {std::vector<double>{0, 0}, std::vector<double>{0, 0, 0, 0, 0, kInf},
                              std::vector<double>{kNaN, 0, 0, 0, 0, 0}, std::vector<double>{0, 0, 0, 0, 0, 0, 0}}) {
        require(!refusal([&] {
                     mountingPose(mount);
                 }).empty(),
                "a bad mounting is refused");
    }
    require(refuses(
                [&] {
                    mountingPose({0, 0});
                },
                "x y z roll pitch yaw"),
            "the refusal names the fields");

    // Sensor pose in the world: body pose composed with the mounting.
    const Pose3 yawed = mountingPose({10, 20, 30, 0, 0, kPi / 2});
    const Pose3 sensor = sensorWorldPose(yawed, mountingPose({1, 0, 2, 0, 0, kPi / 2}));
    vectorNear(sensor.position, Eigen::Vector3d(10, 21, 32), 1e-12, "sensor position");
    vectorNear(sensor.orientation * Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX(), 1e-12, "sensor heading");
    // It is the composition of homogeneous transforms, also for a body quaternion that is not a unit.
    for (int i = 0; i < 20; ++i) {
        Pose3 b;
        b.position = cases.vector3(5.0);
        b.orientation = cases.rotation();
        const Pose3 m = mountingPose(
            {cases.scalar(), cases.scalar(), cases.scalar(), cases.scalar(), cases.scalar(), cases.scalar()});
        Pose3 scaled_body = b;
        scaled_body.orientation.coeffs() *= 2.5;
        const Pose3 world = sensorWorldPose(scaled_body, m);
        require((homogeneous(world) - homogeneous(b) * homogeneous(m)).norm() < 1e-13, "composition of transforms");
    }
    Pose3 invalid;
    invalid.orientation.coeffs().setZero();
    require(refuses(
                [&] {
                    sensorWorldPose(invalid, Pose3());
                },
                "invalid body or sensor mounting pose"),
            "invalid body");
    require(refuses(
                [&] {
                    sensorWorldPose(Pose3(), invalid);
                },
                "invalid body or sensor mounting pose"),
            "invalid mounting");
}

SensorMetadata gpuMetadata() {
    SensorMetadata m;
    m.backend = "gpu";
    m.observation_model = "lidar_scan";
    m.range_m = 5;
    m.min_range_m = .1;
    m.point_cover_spacing_m = .1;
    m.publish_rate_hz = 12;
    m.h_fov_deg = 120;
    m.v_fov_deg = 60;
    m.h_res = 240;
    m.v_res = 120;
    m.frame_id = "world";
    m.stamp_policy = "pose";
    m.pose_type = "geometry_msgs/PoseStamped";
    return m;
}

SensorMetadata cpuMetadata() {
    SensorMetadata m;
    m.backend = "cpu";
    m.observation_model = "crop_through";
    m.range_m = 8;
    m.publish_rate_hz = 20;
    m.frame_id = "world";
    m.stamp_policy = "zero";
    m.prevoxel_leaf_m = {{0.1f, 0.1f, 0.2f}};
    m.heading_cos_min = 0.5;
    m.vertical_slab_tan = 0.577;
    return m;
}

void sensorMetadata() {
    validateGpuSensorMetadata(gpuMetadata());
    validateSensorMetadata(cpuMetadata());
    const auto gpu_refused = [](auto&& edit, const std::string& needle) {
        SensorMetadata m = gpuMetadata();
        edit(&m);
        return refuses(
            [&] {
                validateGpuSensorMetadata(m);
            },
            needle);
    };
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->backend = "cpu";
                },
                "lidar_scan/gpu"),
            "cpu backend");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->observation_model = "crop_through";
                },
                "lidar_scan/gpu"),
            "model");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->h_res = 0;
                },
                "spherical grid"),
            "no columns");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->h_res = 1 << 20;
                    m->v_res = 1 << 20;
                },
                "spherical grid"),
            "grid overflow");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->range_m = m->min_range_m;
                },
                "range/spacing/rate"),
            "range at near");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->min_range_m = 0;
                },
                "range/spacing/rate"),
            "zero near");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->publish_rate_hz = 0;
                },
                "range/spacing/rate"),
            "zero rate");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->publish_rate_hz = kInf;
                },
                "range/spacing/rate"),
            "infinite rate");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->h_fov_deg = 361;
                },
                "spherical grid"),
            "wide fov");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->v_fov_deg = kNaN;
                },
                "spherical grid"),
            "NaN fov");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->point_cover_spacing_m = 1;
                },
                "point-cover"),
            "cover over near");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->point_cover_spacing_m = 0;
                },
                "range/spacing/rate"),
            "no cover");
    auto rectangular = gpuMetadata();
    rectangular.h_fov_deg = 360;
    rectangular.v_fov_deg = 180;
    rectangular.h_res = 360;
    rectangular.v_res = 32;
    validateGpuSensorMetadata(rectangular);
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->frame_id.clear();
                },
                "frame/pose-type/stamp"),
            "no frame");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->stamp_policy = "now";
                },
                "frame/pose-type/stamp"),
            "stamp policy");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->pose_type = "tf";
                },
                "frame/pose-type/stamp"),
            "pose type");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->heading_cos_min = 0.0;
                },
                "CPU crop"),
            "crop predicate");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->vertical_slab_tan = 1.0;
                },
                "CPU crop"),
            "slab predicate");
    require(gpu_refused(
                [](SensorMetadata* m) {
                    m->prevoxel_leaf_m = {{0.1f, 0.0f, 0.0f}};
                },
                "prevoxel"),
            "prevoxel");
    SensorMetadata odometry = gpuMetadata();
    odometry.pose_type = "nav_msgs/Odometry";
    validateGpuSensorMetadata(odometry);

    const auto cpu_refused = [](auto&& edit, const std::string& needle) {
        SensorMetadata m = cpuMetadata();
        edit(&m);
        return refuses(
            [&] {
                validateSensorMetadata(m);
            },
            needle);
    };
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->backend = "gpu";
                },
                "crop_through/cpu"),
            "gpu backend");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->observation_model = "pinhole_depth";
                },
                "crop_through/cpu"),
            "model");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->range_m = 0;
                },
                "radius/rate/frame/stamp"),
            "no radius");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->publish_rate_hz = kNaN;
                },
                "radius/rate/frame/stamp"),
            "NaN rate");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->frame_id.clear();
                },
                "radius/rate/frame/stamp"),
            "no frame");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->stamp_policy = "never";
                },
                "radius/rate/frame/stamp"),
            "stamp policy");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->prevoxel_leaf_m[1] = 0.0f;
                },
                "voxel"),
            "voxel leaf");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->heading_cos_min = 1.5;
                },
                "body-dot"),
            "heading cosine");
    require(cpu_refused(
                [](SensorMetadata* m) {
                    m->vertical_slab_tan = -1.0;
                },
                "world-Z"),
            "slab");
    SensorMetadata open_crop = cpuMetadata();
    open_crop.heading_cos_min.reset();
    open_crop.vertical_slab_tan.reset();
    validateSensorMetadata(open_crop);
}

// ---- placement and motion
// ---------------------------------------------------------------------------------------------

void placement() {
    Cases cases;
    // The quaternion algebra against Eigen's.
    for (int i = 0; i < 100; ++i) {
        const Eigen::Quaterniond a = cases.rotation(), b = cases.rotation();
        const Eigen::Vector3d v = cases.vector3(4.0);
        require((multiply(a, b).coeffs() - (a * b).coeffs()).norm() < 1e-15, "quaternion product");
        require((rotate(a, v) - a * v).norm() < 1e-14, "rotation of a vector");
        Pose3 pa, pb;
        pa.position = cases.vector3(3.0);
        pa.orientation = a;
        pb.position = cases.vector3(3.0);
        pb.orientation = b;
        const Pose3 c = composePlacement(pa, pb);
        require((homogeneous(c) - homogeneous(pa) * homogeneous(pb)).norm() < 1e-14, "composition of poses");
    }
    // The product keeps the sign it comes out with; xgc2_math::compose flips a result to w >= 0, which is the same
    // rotation but not the same numbers in a placed document.
    Pose3 three_quarter_turn;
    three_quarter_turn.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(1.5 * kPi, Eigen::Vector3d::UnitZ()));
    require(three_quarter_turn.orientation.w() < 0.0, "the test rotation has a negative w");
    require(composePlacement(three_quarter_turn, Pose3()).orientation.w() < 0.0, "the placement keeps the sign");
    require(xgc2_math::compose(three_quarter_turn, Pose3()).orientation.w() > 0.0,
            "the library composition canonicalises it");
    require(xgc2_math::isFinite(three_quarter_turn) && validObservationPose(three_quarter_turn, "w", "w"),
            "Pose3 is the pose type of the scene component");

    // An element placed at (10, 20) turned by 90 degrees puts the point (1, 0, 0) of its frame at (10, 21, 0).
    const ElementPlacement element{10.0, 20.0, kPi / 2};
    Pose3 inner;
    inner.position = Eigen::Vector3d(1, 0, 0);
    inner.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
    const Pose3 placed = composePlacement(element.pose(), inner);
    vectorNear(placed.position, Eigen::Vector3d(10, 21, 0), 1e-14, "placed position");
    vectorNear(placed.orientation * Eigen::Vector3d::UnitX(),
               Eigen::Vector3d(std::cos(kPi / 2 + 0.3), std::sin(kPi / 2 + 0.3), 0), 1e-14, "placed heading");
    near(element.pose().orientation.w(), std::cos(kPi / 4), 1e-15, "yaw quaternion w");
    near(element.pose().orientation.z(), std::sin(kPi / 4), 1e-15, "yaw quaternion z");

    // The motion moves with its element.
    Motion pingpong;
    pingpong.type = Motion::kPingPong;
    pingpong.point_a = Eigen::Vector3d(0, 0, 1);
    pingpong.point_b = Eigen::Vector3d(4, 0, 1);
    pingpong.speed = 2.0;
    const Motion moved = placeMotion(element, pingpong);
    vectorNear(moved.point_a, Eigen::Vector3d(10, 20, 1), 1e-14, "ping-pong start");
    vectorNear(moved.point_b, Eigen::Vector3d(10, 24, 1), 1e-14, "ping-pong end");
    near(moved.speed, 2.0, 0.0, "speed");
    Motion circle;
    circle.type = Motion::kCircle;
    circle.center = Eigen::Vector3d(1, 0, 0);
    circle.radius = 2.0;
    circle.angular_speed = 0.5;
    circle.phase = 0.25;
    const Motion circled = placeMotion(element, circle);
    vectorNear(circled.center, Eigen::Vector3d(10, 21, 0), 1e-14, "circle centre");
    near(circled.phase, 0.25 + kPi / 2, 1e-15, "the circle starts a yaw further round");
    Motion twist;
    twist.type = Motion::kConstantTwist;
    twist.linear = Eigen::Vector3d(1, 0, 0);
    twist.angular = Eigen::Vector3d(0, 0, 1);
    const Motion twisted = placeMotion(element, twist);
    vectorNear(twisted.linear, Eigen::Vector3d(0, 1, 0), 1e-14, "velocity turns with the element");
    vectorNear(twisted.angular, Eigen::Vector3d(0, 0, 1), 1e-14, "angular velocity about the axis");
    require(placeMotion(element, Motion()).type == Motion::kHold, "a hold stays a hold");
    // The start of the path is where the obstacle has to be at time zero, before and after placing.
    const Eigen::Vector3d initial(7, 8, 9);
    vectorNear(motionStart(initial, Motion()), initial, 0.0, "hold start");
    vectorNear(motionStart(initial, twist), initial, 0.0, "twist start");
    vectorNear(motionStart(initial, pingpong), pingpong.point_a, 0.0, "ping-pong start");
    vectorNear(motionStart(initial, circle), Eigen::Vector3d(1 + 2 * std::cos(0.25), 2 * std::sin(0.25), 0), 1e-15,
               "circle start");
    Pose3 start_pose;
    start_pose.position = Eigen::Vector3d(1 + 2 * std::cos(0.25), 2 * std::sin(0.25), 0);
    const Pose3 placed_start = composePlacement(element.pose(), start_pose);
    vectorNear(motionStart(placed_start.position, circled), placed_start.position, 1e-13,
               "a placed circle still starts at its position");
}

void motion() {
    Pose3 initial;
    initial.position = Eigen::Vector3d(1, 2, 3);
    initial.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()));
    // hold
    const MotionState held = evaluateMotion(initial, Motion(), 12.0, true);
    vectorNear(held.pose.position, initial.position, 0.0, "hold position");
    require(held.pose.orientation.coeffs() == initial.orientation.coeffs() && held.linear.isZero() &&
                held.angular.isZero(),
            "hold");
    // constant twist: straight line, steady turn
    Motion twist;
    twist.type = Motion::kConstantTwist;
    twist.linear = Eigen::Vector3d(0.5, -1.0, 0.25);
    twist.angular = Eigen::Vector3d(0, 0, kPi / 2);
    const MotionState moving = evaluateMotion(initial, twist, 2.0, true);
    vectorNear(moving.pose.position, Eigen::Vector3d(2, 0, 3.5), 1e-15, "twist position");
    vectorNear(moving.linear, twist.linear, 0.0, "twist velocity");
    vectorNear(moving.angular, twist.angular, 0.0, "twist angular velocity");
    // two seconds at 90 deg/s about z is a half turn about z, applied in the world frame before the initial attitude
    const Eigen::Quaterniond expected =
        Eigen::Quaterniond(Eigen::AngleAxisd(kPi, Eigen::Vector3d::UnitZ())) * initial.orientation;
    require((moving.pose.orientation.coeffs() - expected.coeffs()).norm() < 1e-15 ||
                (moving.pose.orientation.coeffs() + expected.coeffs()).norm() < 1e-15,
            "twist attitude");
    const MotionState paused = evaluateMotion(initial, twist, 2.0, false);
    vectorNear(paused.pose.position, moving.pose.position, 0.0, "a paused clock still places the obstacle");
    require(paused.linear.isZero() && paused.angular.isZero(), "a paused clock has no twist");
    Motion still = twist;
    still.angular.setZero();
    require(evaluateMotion(initial, still, 5.0, true).pose.orientation.coeffs() == initial.orientation.coeffs(),
            "no turn without an angular velocity");
    // ping-pong: out and back at constant speed, 4 m at 2 m/s
    Motion pingpong;
    pingpong.type = Motion::kPingPong;
    pingpong.point_a = Eigen::Vector3d(0, 0, 1);
    pingpong.point_b = Eigen::Vector3d(4, 0, 1);
    pingpong.speed = 2.0;
    const double times[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0};
    const double xs[] = {0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 2.0};
    const double vxs[] = {2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0};
    for (std::size_t i = 0; i < 7; ++i) {
        const MotionState s = evaluateMotion(initial, pingpong, times[i], true);
        near(s.pose.position.x(), xs[i], 1e-15, "ping-pong x at " + std::to_string(times[i]));
        near(s.linear.x(), vxs[i], 1e-15, "ping-pong velocity at " + std::to_string(times[i]));
        near(s.pose.position.z(), 1.0, 0.0, "ping-pong z");
        require(s.pose.orientation.coeffs() == initial.orientation.coeffs(), "a shuttle does not turn");
    }
    // circle: counter-clockwise for a positive speed, the velocity is the derivative of the position
    Motion circle;
    circle.type = Motion::kCircle;
    circle.center = Eigen::Vector3d(1, 1, 0.5);
    circle.radius = 2.0;
    circle.angular_speed = 0.5;
    circle.phase = 0.1;
    for (const double t : {0.0, 1.0, 3.7}) {
        const MotionState s = evaluateMotion(initial, circle, t, true);
        const double phase = 0.1 + 0.5 * t;
        vectorNear(s.pose.position, Eigen::Vector3d(1 + 2 * std::cos(phase), 1 + 2 * std::sin(phase), 0.5), 1e-15,
                   "circle position");
        const double h = 1e-6;
        const Eigen::Vector3d derivative = (evaluateMotion(initial, circle, t + h, true).pose.position -
                                            evaluateMotion(initial, circle, t - h, true).pose.position) /
                                           (2 * h);
        vectorNear(s.linear, derivative, 1e-8, "circle velocity is the derivative");
    }
    vectorNear(evaluateMotion(initial, circle, 0.0, true).pose.position, motionStart(initial.position, circle), 0.0,
               "start of the circle");
}

// ---- closed convex surfaces of scene documents
// --------------------------------------------------------------------------

struct Mesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::int64_t> triangles;
};

Mesh cubeMesh() {
    Mesh m;
    m.vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    m.triangles = {0, 2, 1, 0, 3, 2,  // z = 0, outward -z
                   4, 5, 6, 4, 6, 7,  // z = 1
                   0, 1, 5, 0, 5, 4,  // y = 0
                   2, 3, 7, 2, 7, 6,  // y = 1
                   1, 2, 6, 1, 6, 5,  // x = 1
                   3, 0, 4, 3, 4, 7}; // x = 0
    return m;
}

Mesh octahedron() {
    Mesh m;
    m.vertices = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    m.triangles = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
    return m;
}

double signedVolume(const std::vector<Eigen::Vector3d>& v, const std::vector<std::uint32_t>& t) {
    double volume = 0.0;
    for (std::size_t i = 0; i < t.size(); i += 3) {
        volume += v[t[i]].dot(v[t[i + 1]].cross(v[t[i + 2]])) / 6.0;
    }
    return volume;
}

void convexDocument() {
    // Valid surfaces: the triangles come back counter-clockwise seen from outside, whichever way they were given.
    for (const Mesh& m : {cubeMesh(), octahedron()}) {
        const std::vector<std::uint32_t> triangles = validateConvexMesh(m.vertices, m.triangles);
        require(triangles.size() == m.triangles.size(), "all triangles are kept");
        require(signedVolume(m.vertices, triangles) > 0.1, "outward orientation");
        Mesh flipped = m;
        for (std::size_t i = 0; i < flipped.triangles.size(); i += 3) {
            std::swap(flipped.triangles[i + 1], flipped.triangles[i + 2]);
        }
        const std::vector<std::uint32_t> fixed = validateConvexMesh(flipped.vertices, flipped.triangles);
        require(signedVolume(flipped.vertices, fixed) > 0.1, "inward triangles are turned outward");
        Mesh mixed = m;
        std::swap(mixed.triangles[1], mixed.triangles[2]);
        require(signedVolume(mixed.vertices, validateConvexMesh(mixed.vertices, mixed.triangles)) > 0.1,
                "one flipped triangle");
    }
    near(signedVolume(cubeMesh().vertices, validateConvexMesh(cubeMesh().vertices, cubeMesh().triangles)), 1.0, 1e-12,
         "cube volume");
    near(signedVolume(octahedron().vertices, validateConvexMesh(octahedron().vertices, octahedron().triangles)),
         4.0 / 3.0, 1e-12, "octahedron volume");

    const auto invalid = [](Mesh m, auto&& edit) {
        edit(&m);
        return refusal([&] {
            validateConvexMesh(m.vertices, m.triangles);
        });
    };
    using M = Mesh*;
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices.resize(3);
                        m->triangles.assign({0, 1, 2});
                    }) == "Convex geometry requires 4 to 4096 vertices",
            "three vertices");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices.resize(4097, Eigen::Vector3d::Zero());
                    }) == "Convex geometry requires 4 to 4096 vertices",
            "too many vertices");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices[3].y() = kNaN;
                    }) == "vertex must be finite",
            "NaN coordinate");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices[3].y() = 2e6;
                    }) == "vertex is outside the supported range",
            "huge coordinate");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices[7] = m->vertices[0];
                    }) == "Convex geometry has duplicate vertices",
            "duplicate");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
                        m->vertices[7] = Eigen::Vector3d(-0.0, -0.0, -0.0);
                    }) == "Convex geometry has duplicate vertices",
            "negative zero is zero");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->triangles.resize(9);
                    }) == "Convex geometry requires triangle index triplets",
            "few triangles");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->triangles.push_back(1);
                    }) == "Convex geometry requires triangle index triplets",
            "indices that are not triplets");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->triangles[4] = 8;
                    }) == "Convex triangle index is invalid",
            "index out of range");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->triangles[4] = -1;
                    }) == "Convex triangle index is invalid",
            "negative index");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices.emplace_back(0.5, 0.5, 0.5);
                    }) == "Every convex vertex must belong to its surface",
            "a vertex off the surface");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices[2] = Eigen::Vector3d(0.5, 0.0, 0.0);
                        m->vertices[1] = Eigen::Vector3d(0.25, 0.0, 0.0);
                    }) == "Convex geometry has a degenerate face",
            "collinear corners");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices[6] = Eigen::Vector3d(0.6, 0.6, 0.6);
                    }) == "Geometry is not convex; represent it as separate convex parts",
            "a dented corner");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->triangles.resize(33);
                    }) == "Convex surface must be a closed nonzero-volume manifold",
            "a missing triangle");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->triangles.insert(m->triangles.end(), {0, 2, 1});
                    }) == "Convex surface must be a closed nonzero-volume manifold",
            "a doubled triangle");
    require(invalid(cubeMesh(),
                    [](M m) {
                        m->vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
                        m->triangles = {0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3};
                    }) == "Convex surface must be a closed nonzero-volume manifold",
            "a flat double-sided sheet has no volume");

    // The hull of any point set is a valid document surface.
    Cases cases;
    for (int trial = 0; trial < 12; ++trial) {
        std::vector<Eigen::Vector3d> cloud;
        const int count = 6 + 5 * trial;
        cloud.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            cloud.emplace_back(Eigen::Vector3d(cases.gaussian(), cases.gaussian(), cases.gaussian()).normalized() *
                               cases.between(0.5, 1.0));
        }
        const ConvexMesh mesh = convexMeshOf(cloud, 1e-9);
        const std::vector<std::int64_t> indices(mesh.triangles.begin(), mesh.triangles.end());
        const std::vector<std::uint32_t> checked = validateConvexMesh(mesh.vertices, indices);
        require(checked == mesh.triangles,
                "the hull triangles are already outward (trial " + std::to_string(trial) + ")");
        require(signedVolume(mesh.vertices, checked) > 0.01, "volume of the hull");
        require(mesh.triangles.size() == 3 * (2 * mesh.vertices.size() - 4),
                "a triangulated sphere has 2V - 4 triangles");
    }
    require(refuses(
                [&] {
                    convexMeshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
                },
                "needs >= 4 points"),
            "three points");
    require(refuses(
                [&] {
                    convexMeshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0.5, 0.5, 0}});
                },
                "flat"),
            "coplanar points");
    require(refuses(
                [&] {
                    convexMeshOf({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, kNaN}});
                },
                "non-finite"),
            "NaN point");
}

// ---- obstacles as collision sets

xgc2_math::gjk::Result gap(const Obstacle& a, const Obstacle& b) {
    return xgc2_math::gjk::DistanceGjkQuery::query(*occupiedSet(a), *occupiedSet(b), nullptr, 1.0e-9, 256, 1.0e-12);
}

void occupiedSets() {
    Cases cases;
    // The support function of the set is that of the solid: the samples of the surface reach it from below.
    const std::vector<Obstacle> solids{Obstacle::box({1, -2, 0.5}, {1, 2, 3}, cases.rotation()),
                                       Obstacle::sphere({1, -2, 0.5}, 0.7),
                                       Obstacle::cylinder({1, -2, 0.5}, 0.5, 2.0, cases.rotation()),
                                       Obstacle::capsule({1, -2, 0.5}, 0.4, 1.2, cases.rotation()),
                                       Obstacle::convexHull({{-0.45, -0.40, -0.50},
                                                             {0.50, -0.35, -0.50},
                                                             {0.45, 0.48, -0.50},
                                                             {-0.38, 0.42, -0.50},
                                                             {-0.25, -0.15, 0.80},
                                                             {0.30, -0.20, 0.75},
                                                             {0.28, 0.25, 0.85},
                                                             {-0.22, 0.30, 0.70},
                                                             {0.0, 0.0, 0.0}},
                                                            {1, -2, 0.5}, cases.rotation())};
    for (const Obstacle& solid : solids) {
        const std::shared_ptr<const xgc2_math::ConvexSet3D> set = occupiedSet(solid);
        const std::vector<Eigen::Vector3d> samples = sampleSurface(solid, 0.05);
        for (int i = 0; i < 16; ++i) {
            const Eigen::Vector3d d = cases.vector3().normalized();
            double best = -kInf;
            for (const Eigen::Vector3d& p : samples) {
                best = std::max(best, p.dot(d));
            }
            const double support = set->support(d).support_value;
            require(best <= support + 1e-9 && best >= support - 0.06,
                    "support of obstacle type " + std::to_string(solid.type) + ": samples reach " +
                        std::to_string(best) + " of " + std::to_string(support));
        }
        vectorNear(set->center(), solid.position, 0.0, "centre of the set");
    }
    // GJK distances between obstacles with known gaps.
    const auto expect_gap = [](const Obstacle& a, const Obstacle& b, double expected, const std::string& context) {
        const xgc2_math::gjk::Result result = gap(a, b);
        require(result.status == xgc2_math::gjk::Result::Status::kSuccess, context + ": no separation found");
        near(result.separator.margin, expected, 1e-9, context);
    };
    expect_gap(Obstacle::sphere({0, 0, 0}, 1.0), Obstacle::sphere({5, 0, 0}, 0.5), 3.5, "two spheres");
    expect_gap(Obstacle::box({0, 0, 0}, {2, 2, 2}), Obstacle::box({5, 0, 0}, {2, 2, 2}), 3.0, "two boxes");
    expect_gap(Obstacle::box({0, 0, 0}, {2, 2, 2}), Obstacle::sphere({4, 0, 0}, 1.0), 2.0, "box and sphere");
    expect_gap(Obstacle::capsule({0, 0, 0}, 0.5, 2.0), Obstacle::cylinder({4, 0, 0}, 1.0, 2.0), 2.5,
               "capsule and cylinder");
    expect_gap(Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), Obstacle::sphere({3, 0, 0}, 0.5),
               1.5, "polytope and sphere");
    // Rigid motion of both bodies keeps the gap; a pair that overlaps is reported as such.
    for (int i = 0; i < 10; ++i) {
        const Eigen::Quaterniond q = cases.rotation();
        const Eigen::Vector3d shift = cases.vector3(4.0);
        const Obstacle a = Obstacle::capsule({0, 0, 0}, 0.4, 1.0, Eigen::Quaterniond::Identity());
        const Obstacle b = Obstacle::box({3, 0.3, 0.2}, {1, 1, 1}, Eigen::Quaterniond::Identity());
        Obstacle moved_a = a, moved_b = b;
        moved_a.position = q * a.position + shift;
        moved_a.orientation = q * a.orientation;
        moved_b.position = q * b.position + shift;
        moved_b.orientation = q * b.orientation;
        near(gap(moved_a, moved_b).separator.margin, gap(a, b).separator.margin, 1e-8, "gap under a rigid motion");
    }
    require(gap(Obstacle::sphere({0, 0, 0}, 1.0), Obstacle::sphere({1, 0, 0}, 1.0)).status ==
                xgc2_math::gjk::Result::Status::kOverlap,
            "overlapping spheres");
    require(refuses(
                [&] {
                    occupiedSet(Obstacle::sphere({0, 0, 0}, -1.0));
                },
                "sphere radius"),
            "an invalid obstacle");
}

// ---- cluttered environments
// ---------------------------------------------------------------------------------------------

ClutteredObstacle primitive(ClutteredObstacle::Kind kind, const std::string& type, const Eigen::Vector3d& scale) {
    ClutteredObstacle o;
    o.kind = kind;
    (void)type;
    o.position = Eigen::Vector3d(1.0, -2.0, 0.5);
    o.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 1, 0).normalized()));
    o.scale = scale;
    o.velocity = Eigen::Vector3d(0.1, 0.2, 0.3);
    return o;
}

// Rows (h0 h1 h2 h3) of half-spaces h0 x + h1 y + h2 z + h3 <= 0.
Eigen::MatrixX4d halfSpaces(const std::vector<std::array<double, 4>>& rows) {
    Eigen::MatrixX4d m(static_cast<Eigen::Index>(rows.size()), 4);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        m.row(static_cast<Eigen::Index>(i)) = Eigen::RowVector4d(rows[i][0], rows[i][1], rows[i][2], rows[i][3]);
    }
    return m;
}

ClutteredObstacle cubePolytope(double half) {
    ClutteredObstacle o = primitive(ClutteredObstacle::kHPolytope, "h_polytope:cube", Eigen::Vector3d::Ones());
    o.h_matrix = halfSpaces({{1, 0, 0, -half},
                             {-1, 0, 0, -half},
                             {0, 1, 0, -half},
                             {0, -1, 0, -half},
                             {0, 0, 1, -half},
                             {0, 0, -1, -half}});
    return o;
}

ClutteredObstacle pyramid() {
    ClutteredObstacle o =
        primitive(ClutteredObstacle::kHPolytope, "h_polytope:pyramid_2m", Eigen::Vector3d(1.0, 2.0, 0.5));
    o.h_matrix = halfSpaces({{0, 0, -1, 0}, {1, 0, 0.5, -1}, {-1, 0, 0.5, -1}, {0, 1, 0.5, -1}, {0, -1, 0.5, -1}});
    return o;
}

ClutteredObstacle rock(double s) {
    ClutteredObstacle o =
        primitive(ClutteredObstacle::kVPolytope, "v_polytope:irregular_rock_1", Eigen::Vector3d(s, s, 0.5 * s));
    o.vertices = {{-0.45, -0.40, -0.50}, {0.50, -0.35, -0.50}, {0.45, 0.48, -0.50}, {-0.38, 0.42, -0.50},
                  {-0.25, -0.15, 0.80},  {0.30, -0.20, 0.75},  {0.28, 0.25, 0.85},  {-0.22, 0.30, 0.70}};
    return o;
}

double meshVolume(const ConvexMesh& mesh) {
    return signedVolume(mesh.vertices, mesh.triangles);
}

void clutteredPolytopes() {
    // H-polytope: the vertices of a 2 m cube and of a 2 m pyramid, found from the half-spaces.
    const std::vector<Eigen::Vector3d> corners = polytopeVertices(cubePolytope(1.0));
    require(corners.size() == 8, "a cube has eight vertices");
    for (const Eigen::Vector3d& v : corners) {
        near(v.cwiseAbs().minCoeff(), 1.0, 1e-9, "every cube vertex has |x| = |y| = |z| = 1");
    }
    const std::vector<Eigen::Vector3d> apexes = polytopeVertices(pyramid());
    require(apexes.size() == 5, "a pyramid has five vertices");
    require(std::any_of(apexes.begin(), apexes.end(),
                        [](const Eigen::Vector3d& v) {
                            return (v - Eigen::Vector3d(0, 0, 2)).norm() < 1e-9;
                        }),
            "the apex of the pyramid");
    // Hull meshes are closed convex surfaces, scaled in the local frame.
    const ConvexMesh cube_mesh = polytopeMesh(cubePolytope(1.0));
    require(cube_mesh.vertices.size() == 8 && cube_mesh.triangles.size() == 36,
            "a cube has 8 vertices and 12 triangles");
    near(meshVolume(cube_mesh), 8.0, 1e-9, "cube volume");
    const ConvexMesh pyramid_mesh = polytopeMesh(pyramid());
    near(meshVolume(pyramid_mesh), 4.0 * 2.0 / 3.0 * (1.0 * 2.0 * 0.5), 1e-9, "pyramid volume with scale (1, 2, 0.5)");
    for (const ConvexMesh* mesh : {&cube_mesh, &pyramid_mesh}) {
        const std::vector<std::int64_t> indices(mesh->triangles.begin(), mesh->triangles.end());
        validateConvexMesh(mesh->vertices, indices);
    }
    near(meshVolume(polytopeMesh(rock(1.0))), meshVolume(polytopeMesh(rock(2.0))) / 8.0 * 1.0, 1e-9,
         "scale scales volume by its product");
    // vertexForm gives the same polytope without enumerating again.
    const ClutteredObstacle form = vertexForm(pyramid());
    require(form.kind == ClutteredObstacle::kVPolytope && form.vertices.size() == 5, "vertex form");
    near(meshVolume(polytopeMesh(form)), meshVolume(pyramid_mesh), 1e-12, "same volume in vertex form");
    require(vertexForm(rock(1.0)).vertices == rock(1.0).vertices, "a vertex polytope is unchanged");
    require(vertexForm(primitive(ClutteredObstacle::kSphere, "sphere", Eigen::Vector3d::Ones())).kind ==
                ClutteredObstacle::kSphere,
            "a primitive is unchanged");

    // Refusals.
    require(refuses(
                [&] {
                    polytopeVertices(primitive(ClutteredObstacle::kCube, "cube", Eigen::Vector3d::Ones()));
                },
                "not a polytope"),
            "a cube is not a polytope");
    ClutteredObstacle few = cubePolytope(1.0);
    few.h_matrix.conservativeResize(3, 4);
    require(refuses(
                [&] {
                    polytopeVertices(few);
                },
                "need at least 4 valid H-rows"),
            "three half-spaces");
    ClutteredObstacle empty = cubePolytope(1.0);
    empty.h_matrix.row(1) = Eigen::RowVector4d(-1, 0, 0, 5); // x >= 5 and x <= 1
    require(refuses(
                [&] {
                    polytopeVertices(empty);
                },
                "Failed to enumerate vertices"),
            "an empty polytope");
    ClutteredObstacle triangle = rock(1.0);
    triangle.vertices.resize(3);
    require(refuses(
                [&] {
                    polytopeVertices(triangle);
                },
                "need at least 4 vertices"),
            "three vertices");
}

void clutteredBodies() {
    Cases cases;
    // Primitives: the support function of the occupied set matches the analytic one.
    const ClutteredObstacle sphere = primitive(ClutteredObstacle::kSphere, "sphere", Eigen::Vector3d(0.7, 0.7, 0.7));
    const ClutteredObstacle cylinder =
        primitive(ClutteredObstacle::kCylinder, "cylinder", Eigen::Vector3d(0.5, 0.5, 2.0));
    const ClutteredObstacle capsule = primitive(ClutteredObstacle::kCapsule, "capsule", Eigen::Vector3d(0.4, 0.4, 2.0));
    const ClutteredObstacle cube = primitive(ClutteredObstacle::kCube, "cube", Eigen::Vector3d(1.0, 2.0, 3.0));
    for (int i = 0; i < 50; ++i) {
        const Eigen::Vector3d d = cases.vector3().normalized();
        const Eigen::Vector3d local = sphere.orientation.conjugate() * d;
        near(occupiedSet(toObstacle(sphere))->support(d).support_value, sphere.position.dot(d) + 0.7, 1e-12,
             "sphere support");
        const double radial = std::hypot(local.x(), local.y());
        near(occupiedSet(toObstacle(cylinder))->support(d).support_value,
             cylinder.position.dot(d) + 0.5 * radial + 1.0 * std::abs(local.z()), 1e-12, "cylinder support");
        // a capsule of length 2 and radius 0.4 has caps at +-0.6
        near(occupiedSet(toObstacle(capsule))->support(d).support_value,
             capsule.position.dot(d) + 0.4 + 0.6 * std::abs(local.z()), 1e-12, "capsule support");
        near(occupiedSet(toObstacle(cube))->support(d).support_value,
             cube.position.dot(d) + 0.5 * std::abs(local.x()) + 1.0 * std::abs(local.y()) + 1.5 * std::abs(local.z()),
             1e-12, "cube support");
    }
    vectorNear(occupiedSet(toObstacle(cube))->center(), cube.position, 1e-12, "numeric occupied set center");

    // The same obstacle as a solid for ray casting has the same support function, so the two views agree.
    for (const ClutteredObstacle& o : {sphere, cylinder, capsule, cube, cubePolytope(0.8), pyramid(), rock(1.2)}) {
        const ClutteredObstacle fixed = vertexForm(o);
        const auto set = occupiedSet(toObstacle(fixed));
        const Obstacle solid = toObstacle(fixed);
        // support function of the solid: a lidar scene's samples are on its surface, so the maximum over a dense sample
        // set approaches the support value from below
        const std::vector<Eigen::Vector3d> samples = sampleSurface(solid, 0.05);
        for (int i = 0; i < 12; ++i) {
            const Eigen::Vector3d d = cases.vector3().normalized();
            double best = -kInf;
            for (const Eigen::Vector3d& p : samples) {
                best = std::max(best, p.dot(d));
            }
            const double support = set->support(d).support_value;
            require(best <= support + 1e-9 && best >= support - 0.06,
                    "support of numerical kind " + std::to_string(o.kind) + ": samples reach " + std::to_string(best) +
                        " of " + std::to_string(support));
        }
    }
    // Capsule: the obstacle's straight height is the whole length less its caps.
    const Obstacle pill = toObstacle(capsule);
    require(pill.type == Obstacle::kCapsule && pill.radius == 0.4 && std::abs(pill.height - 1.2) < 1e-15,
            "capsule obstacle");
    const Obstacle stubby =
        toObstacle(primitive(ClutteredObstacle::kCapsule, "capsule", Eigen::Vector3d(1.0, 1.0, 1.0)));
    require(stubby.height == 0.0, "a capsule shorter than its caps is a sphere");
    // A moving obstacle: the body follows the pose.
    ClutteredObstacle moving = rock(1.0);
    moving.position += Eigen::Vector3d(5, 0, 0);
    const auto before = occupiedSet(toObstacle(rock(1.0))), after = occupiedSet(toObstacle(moving));
    near(after->support(Eigen::Vector3d::UnitX()).support_value -
             before->support(Eigen::Vector3d::UnitX()).support_value,
         5.0, 1e-12, "the body moves with the pose");
}

void clutteredTemplates() {
    // Support points of a geometry library.
    const auto cube = templateSupportPoints(primitive(ClutteredObstacle::kCube, "cube", Eigen::Vector3d::Ones()), 50);
    require(cube.size() == 8, "unit cube corners");
    for (const auto& v : cube) {
        near(v.cwiseAbs().minCoeff(), 0.5, 0.0, "cube corner");
        near(v.cwiseAbs().maxCoeff(), 0.5, 0.0, "cube corner");
    }
    const auto ring =
        templateSupportPoints(primitive(ClutteredObstacle::kCylinder, "cylinder", Eigen::Vector3d::Ones()), 8);
    require(ring.size() == 16, "two points per angle");
    for (std::size_t i = 0; i < ring.size(); i += 2) {
        near(std::hypot(ring[i].x(), ring[i].y()), 1.0, 1e-15, "unit radius");
        require(ring[i].z() == -0.5 && ring[i + 1].z() == 0.5 && ring[i].head<2>() == ring[i + 1].head<2>(),
                "bottom and top ring");
    }
    near(std::atan2(ring[2].y(), ring[2].x()), 2.0 * kPi / 8, 1e-15, "angles follow the resolution");
    require(
        templateSupportPoints(primitive(ClutteredObstacle::kCylinder, "cylinder", Eigen::Vector3d::Ones()), 0).empty(),
        "resolution zero");
    require(
        templateSupportPoints(primitive(ClutteredObstacle::kCylinder, "cylinder", Eigen::Vector3d::Ones()), -4).empty(),
        "negative resolution");
    require(templateSupportPoints(primitive(ClutteredObstacle::kSphere, "sphere", Eigen::Vector3d::Ones()), 50).empty(),
            "a sphere is analytic");
    require(
        templateSupportPoints(primitive(ClutteredObstacle::kCapsule, "capsule", Eigen::Vector3d::Ones()), 50).empty(),
        "a capsule is analytic");
    require(templateSupportPoints(rock(1.0), 50) == rock(1.0).vertices, "polytope vertices");
    require(templateSupportPoints(cubePolytope(1.0), 50).size() == 8, "enumerated polytope vertices");
}

void clutteredCubeFrame() {
    CubeFrame frame;
    frame.center = Eigen::Vector3d(5, 6, 1);
    frame.yaw = kPi / 2;
    frame.width = 2.0;
    frame.height = 3.0;
    frame.rod_thickness = 0.2;
    const std::vector<ClutteredObstacle> cubes = expandCubeFrame(frame);
    require(cubes.size() == 4, "four cubes");
    for (std::size_t i = 0; i < 4; ++i) {
        require(cubes[i].kind == ClutteredObstacle::kCube, "numerical cube kind");
        near((cubes[i].orientation.coeffs() - Eigen::Quaterniond(std::cos(kPi / 4), 0, 0, std::sin(kPi / 4)).coeffs())
                 .norm(),
             0.0, 1e-15, "frame orientation");
    }
    // The posts stand a half width either side along the frame's lateral axis, the edges a half height above and below.
    vectorNear(cubes[0].position, Eigen::Vector3d(5 - 1.0 * std::cos(kPi / 2), 6 - 1.0 * std::sin(kPi / 2), 1), 1e-15,
               "left post");
    vectorNear(cubes[1].position, Eigen::Vector3d(5 + 1.0 * std::cos(kPi / 2), 6 + 1.0 * std::sin(kPi / 2), 1), 1e-15,
               "right post");
    vectorNear(cubes[2].position, Eigen::Vector3d(5, 6, 2.5), 1e-15, "top edge");
    vectorNear(cubes[3].position, Eigen::Vector3d(5, 6, -0.5), 1e-15, "bottom edge");
    vectorNear(cubes[0].scale, Eigen::Vector3d(0.2, 0.2, 3.0), 1e-15, "post size");
    vectorNear(cubes[2].scale, Eigen::Vector3d(2.2, 0.2, 0.2), 1e-15, "edge size");
    // The frame is open in the middle: a ray through it passes, rays at the posts and edges hit.
    std::vector<Obstacle> solids;
    std::transform(cubes.begin(), cubes.end(), std::back_inserter(solids), [](const ClutteredObstacle& c) {
        return toObstacle(c);
    });
    WorldLidar lidar(SensorConfig{});
    lidar.setScene(solids);
    // The frame stands in the plane x = 5 (yaw 90 degrees): posts at y = 5 and 7, edges at z = 2.5 and -0.5.
    require(std::isinf(lidar.castRay({0, 6, 1}, Eigen::Vector3d::UnitX(), 20)), "through the frame");
    near(lidar.castRay({0, 6, 2.5}, Eigen::Vector3d::UnitX(), 20), 4.9, 1e-12, "the top edge");
    near(lidar.castRay({0, 7, 1}, Eigen::Vector3d::UnitX(), 20), 4.9, 1e-12, "the right post");
    near(lidar.castRay({0, 5, 1}, Eigen::Vector3d::UnitX(), 20), 4.9, 1e-12, "the left post");
    CubeFrame bad;

    bad.width = 0.0;
    require(refuses(
                [&] {
                    expandCubeFrame(bad);
                },
                "must be positive"),
            "no width");
    bad.width = 1.0;
    bad.rod_thickness = -0.1;
    require(refuses(
                [&] {
                    expandCubeFrame(bad);
                },
                "must be positive"),
            "negative thickness");
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 13> tests{{
        {"document_conversion", documentConversion},
        {"gazebo_conversion", gazeboConversion},
        {"same_scans_from_both_sources", sameScansFromBothSources},
        {"observation_contract", observationContract},
        {"sensor_metadata", sensorMetadata},
        {"placement", placement},
        {"motion", motion},
        {"convex_document", convexDocument},
        {"occupied_sets", occupiedSets},
        {"cluttered_polytopes", clutteredPolytopes},
        {"cluttered_bodies", clutteredBodies},
        {"cluttered_templates", clutteredTemplates},
        {"cluttered_cube_frame", clutteredCubeFrame},
    }};
    if (argc > 2) {
        std::cerr << "Usage: scene_contract_test [case_name|--list]\n";
        return 2;
    }
    const std::string requested = argc == 2 ? argv[1] : "";
    if (requested == "--list") {
        for (const auto& test : tests) {
            std::cout << test.name << '\n';
        }
        return 0;
    }
    int failures = 0;
    int executed = 0;
    for (const auto& test : tests) {
        if (!requested.empty() && requested != test.name) {
            continue;
        }
        ++executed;
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << " (seed=" << kSeed << "): " << error.what() << '\n';
        }
    }
    if (executed == 0) {
        std::cerr << "Unknown case: " << requested << " (use --list)\n";
        return 2;
    }
    // Explicit checks remain active in Release/RelWithDebInfo, including NDEBUG.
    return failures == 0 ? 0 : 1;
}
