// Reference comparison of xgc2_math::scene against the world_lidar core it replaces.
//
// Built only when XGC2_MATH_SCENE_REFERENCE_SOURCE points at a checkout of the scene repository
// (XGC-Team/xgc2-scene-generation, commit 2940a99): the original core sources (world_lidar.cpp, scene_conversion.cpp,
// convex_body_conversion.cpp) are compiled into this executable. Both implementations get the same obstacles, sensor
// configurations, poses and bodies, and their outputs are compared:
//
//   * counts, ordering, hit flags and vehicle ids must be identical;
//   * points, ranges and residuals must agree within kTolerance metres.
//
// The fixtures are those of the original test suite and of its observation regression dump: mixed random scenes of
// every obstacle type, the toy document and Gazebo-truth scenes, and a chain of scene revisions with moving, resized,
// added, removed and reordered obstacles.

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgc2_math/scene.hpp"
#include "xgc2_world_lidar/convex_body_conversion.h"
#include "xgc2_world_lidar/scene_conversion.h"
#include "xgc2_world_lidar/world_lidar.h"

namespace ref = xgc2_world_lidar;
namespace scn = xgc2_math::scene;

namespace {

// Largest accepted difference of a point coordinate, range or residual, in metres. The two cores evaluate the same
// expressions; the tolerance only allows for differently ordered floating-point sums.
constexpr double kTolerance = 1.0e-9;

struct Statistics {
    double worst = 0.0;
    std::size_t compared = 0;
    std::size_t exact = 0;
};

std::string describe(const std::string& what, const Statistics& s) {
    std::ostringstream out;
    out << std::setprecision(3) << std::scientific << what << ": " << s.compared << " values, " << s.exact
        << " bit-identical, worst deviation " << s.worst;
    return out.str();
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare(Statistics* s, double a, double b, const std::string& context) {
    ++s->compared;
    if (std::memcmp(&a, &b, sizeof a) == 0) {
        ++s->exact;
        return;
    }
    const double difference = (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) ? 0.0 : std::abs(a - b);
    if (!(difference <= kTolerance)) {
        std::ostringstream out;
        out << std::setprecision(17) << context << ": " << a << " vs " << b;
        throw std::runtime_error(out.str());
    }
    s->worst = std::max(s->worst, difference);
}

void compare(Statistics* s, const Eigen::Vector3d& a, const Eigen::Vector3d& b, const std::string& context) {
    for (int k = 0; k < 3; ++k) {
        compare(s, a[k], b[k], context);
    }
}

void report(const std::string& line) {
    std::cout << "  " << line << '\n';
}

// ---- fixtures of the original test suite ----------------------------------------------------------------------------

Eigen::Quaterniond randomQuat(std::mt19937& rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    return Eigen::Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized();
}

// A mixed random scene: every primitive type, rotated.
std::vector<ref::Obstacle> randomScene(std::mt19937& rng, int count, double half_extent) {
    std::uniform_real_distribution<double> pos(-half_extent, half_extent);
    std::uniform_real_distribution<double> z(-2.0, 2.0);
    std::uniform_real_distribution<double> sz(0.3, 2.0);
    std::vector<ref::Obstacle> out;
    for (int i = 0; i < count; ++i) {
        const Eigen::Vector3d c(pos(rng), pos(rng), z(rng));
        switch (i % 5) {
        case 0:
            out.push_back(ref::Obstacle::box(c, Eigen::Vector3d(sz(rng), sz(rng), sz(rng)), randomQuat(rng)));
            break;
        case 1:
            out.push_back(ref::Obstacle::sphere(c, 0.5 * sz(rng)));
            break;
        case 2:
            out.push_back(ref::Obstacle::cylinder(c, 0.5 * sz(rng), 2.0 * sz(rng), randomQuat(rng)));
            break;
        case 3:
            out.push_back(ref::Obstacle::capsule(c, 0.4 * sz(rng), sz(rng), randomQuat(rng)));
            break;
        case 4: {
            // irregular_rock_1 from cluttered_environment/config/v_polytope_library.yaml
            std::vector<Eigen::Vector3d> v{{-0.45, -0.40, -0.50}, {0.50, -0.35, -0.50}, {0.45, 0.48, -0.50},
                                           {-0.38, 0.42, -0.50},  {-0.25, -0.15, 0.80}, {0.30, -0.20, 0.75},
                                           {0.28, 0.25, 0.85},    {-0.22, 0.30, 0.70}};
            const double s = sz(rng);
            std::transform(v.begin(), v.end(), v.begin(), [s](const Eigen::Vector3d& p) {
                return Eigen::Vector3d(s * p);
            });
            out.push_back(ref::Obstacle::convexHull(v, c, randomQuat(rng)));
            break;
        }
        }
    }
    return out;
}

// More convex solids than the random scene has: the polytopes of the cluttered-environment libraries, a thin slab,
// a cloud with interior points and a polytope far from the origin.
std::vector<ref::Obstacle> convexFixtures(std::mt19937& rng) {
    const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
    std::vector<ref::Obstacle> out;
    out.push_back(ref::Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {1, 2, 0}, identity));
    out.push_back(ref::Obstacle::convexHull(
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0.5, 0.5, 0.5}},
        {-2, 1, 0}, randomQuat(rng)));
    out.push_back(ref::Obstacle::convexHull({{0, 0, 0},
                                             {2, 0, 0},
                                             {2, 1, 0},
                                             {0, 1, 0},
                                             {0.5, 0.5, 0.6},
                                             {1.5, 0.5, 0.6},
                                             {1.5, 0.6, 0.6},
                                             {0.5, 0.6, 0.6}},
                                            {0, 3, 0}, randomQuat(rng)));
    // pentagonal base, irregular top (debris_angular_1)
    out.push_back(ref::Obstacle::convexHull({{0.00, -0.55, -0.60},
                                             {0.52, -0.17, -0.60},
                                             {0.32, 0.48, -0.60},
                                             {-0.32, 0.48, -0.60},
                                             {-0.52, -0.17, -0.60},
                                             {0.00, -0.25, 1.00},
                                             {0.30, 0.00, 0.85},
                                             {0.10, 0.28, 0.80},
                                             {-0.20, 0.22, 0.82},
                                             {-0.28, -0.10, 0.88}},
                                            {3, -2, 0.5}, randomQuat(rng)));
    // far from the origin
    out.push_back(ref::Obstacle::convexHull({{1000.0, 2000.0, 300.0},
                                             {1001.0, 2000.0, 300.0},
                                             {1000.0, 2001.0, 300.0},
                                             {1000.0, 2000.0, 301.0},
                                             {1001.0, 2001.0, 301.0}},
                                            {0, 0, 0}, identity));
    // a random cloud, tens of vertices
    std::normal_distribution<double> n(0.0, 1.0);
    std::vector<Eigen::Vector3d> cloud;
    for (int i = 0; i < 40; ++i) {
        cloud.push_back(Eigen::Vector3d(n(rng), n(rng), n(rng)).normalized() * (0.6 + 0.2 * n(rng)));
    }
    out.push_back(ref::Obstacle::convexHull(cloud, {5, 5, 1}, randomQuat(rng)));
    return out;
}

scn::Obstacle toNew(const ref::Obstacle& o) {
    scn::Obstacle n;
    n.type = static_cast<scn::Obstacle::Type>(static_cast<int>(o.type));
    n.position = o.position;
    n.orientation = o.orientation;
    n.size = o.size;
    n.radius = o.radius;
    n.height = o.height;
    n.vertices = o.vertices;
    return n;
}

std::vector<scn::Obstacle> toNew(const std::vector<ref::Obstacle>& obstacles) {
    std::vector<scn::Obstacle> out;
    std::transform(obstacles.begin(), obstacles.end(), std::back_inserter(out), [](const ref::Obstacle& o) {
        return toNew(o);
    });
    return out;
}

scn::SensorConfig toNew(const ref::SensorConfig& c) {
    scn::SensorConfig n;
    n.mode = static_cast<scn::SensorConfig::Mode>(static_cast<int>(c.mode));
    n.range = c.range;
    n.min_range = c.min_range;
    n.h_fov_deg = c.h_fov_deg;
    n.v_fov_deg = c.v_fov_deg;
    n.h_res = c.h_res;
    n.v_res = c.v_res;
    n.surface_spacing = c.surface_spacing;
    n.noise_std = c.noise_std;
    n.seed = c.seed;
    n.penetrating_heading_crop = c.penetrating_heading_crop;
    n.heading_cos_min = c.heading_cos_min;
    n.vertical_slab_tan = c.vertical_slab_tan;
    n.penetrating_keep_buried = c.penetrating_keep_buried;
    n.width = c.width;
    n.height = c.height;
    n.fx = c.fx;
    n.fy = c.fy;
    n.cx = c.cx;
    n.cy = c.cy;
    return n;
}

std::vector<scn::VehicleBody> toNew(const std::vector<ref::VehicleBody>& bodies) {
    std::vector<scn::VehicleBody> out;
    std::transform(bodies.begin(), bodies.end(), std::back_inserter(out), [](const ref::VehicleBody& b) {
        return scn::VehicleBody{b.id, b.position, b.radius};
    });
    return out;
}

// ---- comparisons -------------------------------------------------------------------------------------------------

void comparePoints(Statistics* s, const std::vector<Eigen::Vector3d>& a, const std::vector<Eigen::Vector3d>& b,
                   const std::string& context) {
    require(a.size() == b.size(),
            context + ": " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + " points");
    for (std::size_t i = 0; i < a.size(); ++i) {
        compare(s, a[i], b[i], context + " point " + std::to_string(i));
    }
}

void compareTagged(Statistics* s, const std::vector<ref::TaggedPoint>& a, const std::vector<scn::TaggedPoint>& b,
                   const std::string& context) {
    require(a.size() == b.size(),
            context + ": " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + " points");
    for (std::size_t i = 0; i < a.size(); ++i) {
        require(a[i].vehicle_id == b[i].vehicle_id, context + ": vehicle id of point " + std::to_string(i));
        compare(s, a[i].point, b[i].point, context + " point " + std::to_string(i));
    }
}

void compareBeams(Statistics* s, const std::vector<ref::Beam>& a, const scn::BeamScan& b, const std::string& context) {
    require(a.size() == b.size(), context + ": beam count");
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::string where = context + " beam " + std::to_string(i);
        require(a[i].hit == (b.hit[i] != 0), where + ": hit flag");
        require(a[i].vehicle_id == b.vehicle_id[i], where + ": vehicle id");
        compare(s, a[i].origin, b.origin, where + " origin");
        compare(s, a[i].direction, b.direction[i], where + " direction");
        compare(s, a[i].range, b.range[i], where + " range");
    }
}

// Sampled surfaces of single obstacles, compiled by each core on its own.
void sampledSurfaces() {
    std::mt19937 rng(41);
    std::vector<ref::Obstacle> obstacles = randomScene(rng, 50, 8.0);
    const auto convex = convexFixtures(rng);
    obstacles.insert(obstacles.end(), convex.begin(), convex.end());
    Statistics s;
    std::size_t count = 0;
    for (const double spacing : {0.05, 0.1, 0.25, 1.0, 1.0e9}) {
        for (const auto& o : obstacles) {
            const auto a = ref::sampleSurface(o, spacing);
            const auto b = scn::sampleSurface(toNew(o), spacing);
            comparePoints(&s, a, b, "obstacle type " + std::to_string(o.type) + " spacing " + std::to_string(spacing));
            ++count;
        }
    }
    report(describe("sampled surfaces of " + std::to_string(count) + " obstacle/spacing pairs", s));
}

// Scene maps and geometric queries on shared scenes.
void sceneQueries() {
    std::mt19937 rng(5);
    std::vector<ref::Obstacle> obstacles = randomScene(rng, 100, 15.0);
    const auto convex = convexFixtures(rng);
    obstacles.insert(obstacles.end(), convex.begin(), convex.end());
    const std::vector<scn::Obstacle> converted = toNew(obstacles);
    Statistics rays, residuals, maps;
    for (const bool keep : {false, true}) {
        for (const double spacing : {0.1, 0.25}) {
            ref::SensorConfig rc;
            rc.mode = ref::SensorConfig::kPenetrating;
            rc.surface_spacing = spacing;
            rc.penetrating_keep_buried = keep;
            ref::WorldLidar original(rc);
            original.setScene(obstacles);
            scn::WorldLidar port(toNew(rc));
            port.setScene(converted);
            require(original.scene()->sampleCount() == port.scene()->sampleCount(), "sample count of the scene");
            comparePoints(&maps, original.globalMap(spacing), port.globalMap(spacing), "scene map");
            comparePoints(&maps, original.globalMap(0.37), port.globalMap(0.37), "resampled scene map");
            std::uniform_real_distribution<double> u(-15.0, 15.0);
            std::normal_distribution<double> n(0.0, 1.0);
            for (int i = 0; i < 3000; ++i) {
                const Eigen::Vector3d o(u(rng), u(rng), 0.2 * u(rng));
                const Eigen::Vector3d d = Eigen::Vector3d(n(rng), n(rng), n(rng)).normalized();
                compare(&rays, original.castRay(o, d, 25.0), port.castRay(o, d, 25.0), "ray " + std::to_string(i));
                compare(&residuals, original.surfaceResidual(o), port.surfaceResidual(o), "residual");
            }
        }
    }
    report(describe("scene maps", maps));
    report(describe("ray casts", rays));
    report(describe("surface residuals", residuals));
}

// Scans in every mode, with and without buried samples, noise, heading crop and vehicle bodies, over a chain of scene
// revisions (the regression dump of the original suite, which also compares incremental rebuilds).
void scanChain() {
    const Eigen::Quaterniond rotation(Eigen::AngleAxisd(0.43, Eigen::Vector3d(1, 2, 3).normalized()));
    std::vector<ref::Obstacle> initial{
        ref::Obstacle::box({3, 0, 1}, {1, 4, 2}, rotation), ref::Obstacle::sphere({3.2, 0, 1}, 0.8),
        ref::Obstacle::cylinder({-2, 3, 1}, 0.6, 2.5, rotation), ref::Obstacle::capsule({0, -3, 1}, 0.3, 1.5, rotation),
        ref::Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {1, 2, 0}, rotation)};
    std::mt19937 rng(936);
    std::uniform_real_distribution<double> position(-6.0, 6.0);
    for (int i = 0; i < 25; ++i) {
        initial.push_back(ref::Obstacle::sphere({position(rng), position(rng), position(rng)}, 0.25));
    }
    Statistics points, beams, maps;
    std::size_t scans = 0, revisions = 0;
    for (int mode = 0; mode < 3; ++mode) {
        for (const bool keep : {false, true}) {
            for (const bool noisy : {false, true}) {
                for (const bool crop : {false, true}) {
                    ref::SensorConfig config;
                    config.mode = static_cast<ref::SensorConfig::Mode>(mode);
                    config.range = 7;
                    config.min_range = 0.2;
                    config.h_fov_deg = mode == 2 ? 100 : 270;
                    config.v_fov_deg = 90;
                    config.h_res = 48;
                    config.v_res = 9;
                    config.width = 20;
                    config.height = 12;
                    config.surface_spacing = 0.25;
                    config.penetrating_keep_buried = keep;
                    config.penetrating_heading_crop = crop;
                    config.heading_cos_min = 0.3;
                    config.noise_std = noisy ? 0.015 : 0;
                    config.seed = 77;
                    ref::WorldLidar original(config);
                    scn::WorldLidar port(toNew(config));
                    auto obstacles = initial;
                    for (int revision = 0; revision < 10; ++revision) {
                        if (revision == 1 || revision == 2) {
                            obstacles[0].position.y() += 0.25;
                        }
                        if (revision == 3) {
                            obstacles[1].radius *= 1.3;
                        }
                        if (revision == 4) {
                            obstacles.push_back(ref::Obstacle::box({-1, -1, 1}, {1, 2, 1}));
                        }
                        if (revision == 5) {
                            obstacles.erase(obstacles.begin() + 2);
                        }
                        if (revision == 6) {
                            std::reverse(obstacles.begin(), obstacles.end());
                        }
                        if (revision == 7) {
                            obstacles.clear();
                        }
                        if (revision == 8) {
                            obstacles = initial;
                        }
                        // Both cores rebuild from the scene they hold, the way a simulator revises a scene.
                        original.setScene(std::make_shared<ref::LidarScene>(obstacles, 0.25, keep, *original.scene()));
                        port.setScene(std::make_shared<scn::LidarScene>(toNew(obstacles), 0.25, keep, *port.scene()));
                        ++revisions;
                        comparePoints(&maps, original.globalMap(0.25), port.globalMap(0.25), "scene map");
                        for (int scan = 0; scan < 3; ++scan) {
                            const Eigen::Vector3d origin(scan - 1.0, 0.5 * revision, 0.2);
                            const std::vector<ref::VehicleBody> others{{4, {2, 1, 1}, 0.4}, {9, {100, 100, 0}, 0.5}};
                            const std::string where = "mode " + std::to_string(mode) + " keep " + std::to_string(keep) +
                                                      " noisy " + std::to_string(noisy) + " crop " +
                                                      std::to_string(crop) + " revision " + std::to_string(revision) +
                                                      " scan " + std::to_string(scan);
                            compareTagged(&points, original.scanTagged(origin, rotation, others),
                                          port.scanTagged(origin, rotation, toNew(others)), where);
                            if (mode != 1) {
                                compareBeams(&beams, original.scanWithBeams(origin, rotation, others),
                                             port.scanWithBeams(origin, rotation, toNew(others)), where);
                            }
                            ++scans;
                        }
                    }
                }
            }
        }
    }
    report(std::to_string(scans) + " scans over " + std::to_string(revisions) + " scene revisions");
    report(describe("scene maps along the chain", maps));
    report(describe("scan points", points));
    report(describe("beam records", beams));
}

// Packed PointCloud2 records and the scan index: a sensor that scans repeatedly must follow the original's noise
// stream.
void packedClouds() {
    std::mt19937 rng(77);
    const auto obstacles = randomScene(rng, 40, 10.0);
    const auto converted = toNew(obstacles);
    std::size_t clouds = 0, bytes = 0;
    for (int mode = 0; mode < 3; ++mode) {
        for (const bool with_id : {false, true}) {
            ref::SensorConfig c;
            c.mode = static_cast<ref::SensorConfig::Mode>(mode);
            c.range = 8.0;
            c.noise_std = 0.02;
            c.seed = 5;
            c.h_fov_deg = mode == 2 ? 90.0 : 360.0;
            c.v_fov_deg = mode == 1 ? 180.0 : 30.0;
            ref::WorldLidar original(c);
            original.setScene(obstacles);
            scn::WorldLidar port(toNew(c));
            port.setScene(converted);
            std::vector<uint8_t> a(3, 0xab), b(5, 0xcd);
            for (int k = 0; k < 6; ++k) {
                const Eigen::Vector3d p(1.0 * k - 2.0, 0.5, 0.3);
                const std::vector<ref::VehicleBody> others{{4, p + Eigen::Vector3d(1.2, 0, 0), 0.3}};
                const std::size_t na = original.scanInto(p, Eigen::Quaterniond::Identity(), others, with_id, &a);
                const std::size_t nb = port.scanInto(p, Eigen::Quaterniond::Identity(), toNew(others), with_id, &b);
                require(na == nb && a.size() == b.size() &&
                            (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0),
                        "packed cloud of mode " + std::to_string(mode) + " scan " + std::to_string(k) + " differs");
                ++clouds;
                bytes += a.size();
            }
        }
    }
    report(std::to_string(clouds) + " packed clouds, " + std::to_string(bytes) + " bytes, byte-identical");
}

// Conversion of scene documents and of the Gazebo obstacle truth.
void conversions() {
    ref::SceneObstacleDescription o;
    o.id = "wall";
    o.pose.position = Eigen::Vector3d(0, 0, 1);
    o.pose.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
    ref::ScenePartDescription part;
    part.type = "box";
    part.pose.position = Eigen::Vector3d(4, 0, 0);
    part.size = Eigen::Vector3d(1, 2, 2);
    o.parts.push_back(part);
    ref::ScenePartDescription rock;
    rock.type = "convex";
    rock.pose.position = Eigen::Vector3d(0, 0, 5);
    rock.pose.orientation = Eigen::Quaterniond(2.0, 0.1, 0.2, 0.3);
    rock.vertices = {{-0.45, -0.40, -0.50}, {0.50, -0.35, -0.50}, {0.45, 0.48, -0.50},
                     {-0.38, 0.42, -0.50},  {-0.25, -0.15, 0.80}, {0.30, -0.20, 0.75},
                     {0.28, 0.25, 0.85},    {-0.22, 0.30, 0.70},  {0.0, 0.0, 0.0}};
    o.parts.push_back(rock);
    ref::ScenePartDescription pillar;
    pillar.type = "cylinder";
    pillar.radius = 0.4;
    pillar.height = 3.0;
    o.parts.push_back(pillar);
    ref::ScenePartDescription ball;
    ball.type = "sphere";
    ball.radius = 0.7;
    o.parts.push_back(ball);
    ref::ScenePartDescription pill;
    pill.type = "capsule";
    pill.radius = 0.2;
    pill.height = 1.0;
    pill.pose.position = Eigen::Vector3d(1, 1, 0);
    o.parts.push_back(pill);
    const auto a = ref::toObstacles({o});

    scn::SceneObstacleDescription n;
    n.pose.position = o.pose.position;
    n.pose.orientation = o.pose.orientation;
    for (const auto& p : o.parts) {
        scn::ScenePartDescription q;
        xgc2_math::geometry::Geometry g;
        g.type = xgc2_math::geometry::geometryTypeFromName(p.type);
        q.pose.position = p.pose.position;
        q.pose.orientation = p.pose.orientation;
        g.size = {p.size.x(), p.size.y(), p.size.z()};
        g.radius = p.radius;
        g.height = p.height;
        for (const auto& v : p.vertices)
            g.vertices.push_back({v.x(), v.y(), v.z()});
        q.geometry = xgc2_math::geometry::encodeGeometry(g);
        n.parts.push_back(q);
    }
    const auto b = scn::toObstacles({n});
    require(a.size() == b.size(), "converted obstacle count");
    Statistics s;
    for (std::size_t i = 0; i < a.size(); ++i) {
        require(static_cast<int>(a[i].type) == static_cast<int>(b[i].type), "converted obstacle type");
        compare(&s, a[i].position, b[i].position, "converted position");
        compare(&s, a[i].orientation.coeffs().x(), b[i].orientation.coeffs().x(), "converted orientation");
        compare(&s, a[i].orientation.coeffs().y(), b[i].orientation.coeffs().y(), "converted orientation");
        compare(&s, a[i].orientation.coeffs().z(), b[i].orientation.coeffs().z(), "converted orientation");
        compare(&s, a[i].orientation.coeffs().w(), b[i].orientation.coeffs().w(), "converted orientation");
        compare(&s, a[i].size, b[i].size, "converted size");
        compare(&s, a[i].radius, b[i].radius, "converted radius");
        compare(&s, a[i].height, b[i].height, "converted height");
        require(a[i].vertices.size() == b[i].vertices.size(), "converted vertex count");
    }
    report(describe("document conversion", s));

    // The same refusals, with the same words.
    const auto refusal = [](auto&& convert) {
        try {
            convert();
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const auto flat_ref = [&] {
        ref::SceneObstacleDescription bad = o;
        bad.parts = {rock};
        bad.parts[0].vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
        ref::toObstacles({bad});
    };
    const auto flat_new = [&] {
        scn::SceneObstacleDescription bad = n;
        bad.parts = {n.parts[1]};
        bad.parts[0].geometry = xgc2_math::geometry::encodeGeometry(
            xgc2_math::geometry::Geometry::convex({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}));
        scn::toObstacles({bad});
    };
    const std::string message = refusal(flat_ref);
    require(!message.empty() && refusal(flat_new).find("flat") != std::string::npos,
            "refusal text of a flat part: '" + message + "' vs '" + refusal(flat_new) + "'");
    report("refusals carry the original text: " + message);
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 5> tests{{
        {"sampled_surfaces", sampledSurfaces},
        {"scene_queries", sceneQueries},
        {"scan_chain", scanChain},
        {"packed_clouds", packedClouds},
        {"conversions", conversions},
    }};
    if (argc > 2) {
        std::cerr << "Usage: scene_reference_test [case_name|--list]\n";
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
            std::cout << "RUN " << test.name << '\n';
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    if (executed == 0) {
        std::cerr << "Unknown case: " << requested << " (use --list)\n";
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
