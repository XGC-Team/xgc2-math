// The world-frame LiDAR / depth camera model of xgc2_math::scene: raycast, penetrating and depth-frustum scans, field
// of view and heading crop, vehicle bodies, range noise, shared scenes and beam patterns, scan requests and
// configuration validation. The checks are the invariants of the model (no return behind an occluder, every point on a
// surface and within range, the indexed penetrating scan equal to the plain loop it specifies) and run with fixed
// seeds.

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xgc2_math/scene.hpp"

namespace {

using namespace xgc2_math::scene;

constexpr std::uint64_t kSeed = 0x584743324c4944ULL;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.141592653589793238462643383279502884;
const Eigen::Quaterniond kIdentity = Eigen::Quaterniond::Identity();

struct Cases {
    std::mt19937_64 engine{kSeed};
    double unit() { return static_cast<double>(engine() >> 11) / 9007199254740992.0; }
    double scalar() { return 2.0 * unit() - 1.0; }
    double between(double lo, double hi) { return lo + (hi - lo) * unit(); }
    double gaussian() {
        const double u1 = 1.0 - unit(), u2 = unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
    Eigen::Quaterniond rotation() {
        const double w = gaussian(), x = gaussian(), y = gaussian(), z = gaussian();
        return Eigen::Quaterniond(w, x, y, z).normalized();
    }
    std::vector<Eigen::Vector3d> rock(double s) {
        std::vector<Eigen::Vector3d> v{{-0.45, -0.40, -0.50}, {0.50, -0.35, -0.50}, {0.45, 0.48, -0.50},
                                       {-0.38, 0.42, -0.50},  {-0.25, -0.15, 0.80}, {0.30, -0.20, 0.75},
                                       {0.28, 0.25, 0.85},    {-0.22, 0.30, 0.70}};
        std::transform(v.begin(), v.end(), v.begin(), [s](const Eigen::Vector3d& p) {
            return Eigen::Vector3d(s * p);
        });
        return v;
    }
    // A mixed scene: every obstacle type, rotated.
    std::vector<Obstacle> scene(int count, double half_extent) {
        std::vector<Obstacle> out;
        for (int i = 0; i < count; ++i) {
            const double x = between(-half_extent, half_extent), y = between(-half_extent, half_extent);
            const Eigen::Vector3d c(x, y, between(-2.0, 2.0));
            const auto size = [this] {
                return between(0.3, 2.0);
            };
            switch (i % 5) {
            case 0: {
                const double a = size(), b = size(), d = size();
                out.push_back(Obstacle::box(c, Eigen::Vector3d(a, b, d), rotation()));
                break;
            }
            case 1:
                out.push_back(Obstacle::sphere(c, 0.5 * size()));
                break;
            case 2: {
                const double r = 0.5 * size(), h = 2.0 * size();
                out.push_back(Obstacle::cylinder(c, r, h, rotation()));
                break;
            }
            case 3: {
                const double r = 0.4 * size(), h = size();
                out.push_back(Obstacle::capsule(c, r, h, rotation()));
                break;
            }
            default:
                out.push_back(Obstacle::convexHull(rock(size()), c, rotation()));
                break;
            }
        }
        return out;
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

template <class F> bool refuses(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

SensorConfig lidar32(SensorConfig::Mode mode) {
    SensorConfig c;
    c.mode = mode;
    c.range = 20.0;
    c.h_fov_deg = 360.0;
    c.v_fov_deg = 30.0;
    c.h_res = 360;
    c.v_res = 32;
    c.surface_spacing = 0.1;
    return c;
}

// True when the open segment (a, b) passes through the interior of the scene.
bool segmentCrossesInterior(const WorldLidar& lidar, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const int n = 400;
    for (int k = 1; k < n; ++k) {
        const Eigen::Vector3d p = a + (b - a) * (static_cast<double>(k) / n);
        if ((p - b).norm() < 1e-4) {
            continue;
        }
        if (lidar.surfaceResidual(p) < -1e-6) {
            return true;
        }
    }
    return false;
}

// ---- scan modes --------------------------------------------------------------------------------------------------

void raycastOcclusion() {
    const Obstacle front = Obstacle::box({5, 0, 0}, {1, 4, 4});
    const Obstacle back = Obstacle::box({10, 0, 0}, {1, 12, 8});
    WorldLidar lidar(lidar32(SensorConfig::kRaycast));
    lidar.setScene({front, back});
    WorldLidar front_only(lidar32(SensorConfig::kRaycast));
    front_only.setScene({front});
    const Eigen::Vector3d o = Eigen::Vector3d::Zero();
    const auto points = lidar.scan(o, kIdentity);
    int on_back = 0, bad = 0;
    for (const auto& p : points) {
        if (p.x() > 9.0) {
            ++on_back;
            // The beam to a back-wall point must not meet the front box first.
            if (front_only.castRay(o, p - o, 20.0) < (p - o).norm() - 1e-9) {
                ++bad;
            }
        }
        require(p.x() <= 5.5 + 1e-9 || p.x() >= 9.5 - 1e-9, "a return between the boxes");
        require(p.x() <= 4.5 + 1e-6 || p.x() >= 9.5 - 1e-6, "a return on the far face of the front box");
    }
    require(on_back > 0, "the back wall is seen around the front box");
    require(bad == 0, "no occluded return");

    // Random clutter: the segment sensor -> return never crosses a solid.
    Cases cases;
    WorldLidar cluttered(lidar32(SensorConfig::kRaycast));
    cluttered.setScene(cases.scene(100, 12.0));
    int crossings = 0;
    std::size_t total = 0;
    for (int trial = 0; trial < 3; ++trial) {
        const Eigen::Vector3d s(3.0 * trial - 3.0, 0.5, 4.0); // above most clutter (z in [-2, 2])
        const auto cloud = cluttered.scan(s, cases.rotation());
        total += cloud.size();
        for (std::size_t i = 0; i < cloud.size(); i += 7) {
            crossings += segmentCrossesInterior(cluttered, s, cloud[i]) ? 1 : 0;
        }
    }
    require(total > 0, "clutter is seen");
    require(crossings == 0, "no segment crosses a solid");
}

void penetratingBackFaces() {
    const Obstacle front = Obstacle::box({5, 0, 0}, {2, 2, 2});
    const Obstacle hidden = Obstacle::sphere({9, 0, 0}, 0.5); // fully in the box's shadow
    WorldLidar penetrating(lidar32(SensorConfig::kPenetrating));
    penetrating.setScene({front, hidden});
    WorldLidar raycast(lidar32(SensorConfig::kRaycast));
    raycast.setScene({front, hidden});
    int back_face = 0, hidden_points = 0, ray_back = 0, ray_hidden = 0;
    for (const auto& q : penetrating.scan(Eigen::Vector3d::Zero(), kIdentity)) {
        back_face += std::abs(q.x() - 6.0) < 1e-9 ? 1 : 0;
        hidden_points += q.x() > 8.0 ? 1 : 0;
    }
    for (const auto& q : raycast.scan(Eigen::Vector3d::Zero(), kIdentity)) {
        ray_back += std::abs(q.x() - 6.0) < 1e-9 ? 1 : 0;
        ray_hidden += q.x() > 8.0 ? 1 : 0;
    }
    require(back_face > 0, "the penetrating scan returns the back face");
    require(hidden_points > 0, "the penetrating scan returns the hidden sphere");
    require(ray_back == 0 && ray_hidden == 0, "the raycast sees neither");
}

void onSurfaceAndInRange() {
    for (int mode = 0; mode < 2; ++mode) {
        for (const double noise : {0.0, 0.02}) {
            SensorConfig c = lidar32(mode == 0 ? SensorConfig::kRaycast : SensorConfig::kPenetrating);
            c.range = 8.0;
            c.noise_std = noise;
            c.seed = 3;
            Cases cases;
            WorldLidar lidar(c);
            lidar.setScene(cases.scene(100, 10.0));
            const double tolerance = noise == 0.0 ? 1e-6 : 6.0 * noise;
            double worst = 0.0, farthest = 0.0;
            std::size_t n = 0;
            for (int trial = 0; trial < 4; ++trial) {
                const Eigen::Vector3d s(-6.0 + 4.0 * trial, 1.0, 3.5);
                for (const auto& q : lidar.scan(s, cases.rotation())) {
                    worst = std::max(worst, std::abs(lidar.surfaceResidual(q)));
                    farthest = std::max(farthest, (q - s).norm());
                    ++n;
                }
            }
            const std::string where =
                std::string(mode == 0 ? "raycast" : "penetrating") + " noise " + std::to_string(noise);
            require(n > 0, where + ": no points");
            require(worst <= tolerance, where + ": a point " + std::to_string(worst) + " off the surface");
            require(farthest <= c.range + 1e-9, where + ": a point beyond the range");
        }
    }
}

void fieldOfView() {
    const Eigen::Quaterniond yaw90(Eigen::AngleAxisd(kPi / 2, Eigen::Vector3d::UnitZ()));
    for (int mode = 0; mode < 2; ++mode) {
        SensorConfig c = lidar32(mode == 0 ? SensorConfig::kRaycast : SensorConfig::kPenetrating);
        c.h_fov_deg = 90.0;
        c.h_res = 90;
        WorldLidar lidar(c);
        // A ring of pillars all around the sensor.
        std::vector<Obstacle> ring;
        for (int k = 0; k < 16; ++k) {
            const double a = 2 * kPi * k / 16;
            ring.push_back(Obstacle::cylinder({6 * std::cos(a), 6 * std::sin(a), 0}, 0.8, 6.0));
        }
        lidar.setScene(ring);
        const auto points = lidar.scan(Eigen::Vector3d::Zero(), yaw90); // looking along +y
        int outside = 0;
        for (const auto& p : points) {
            const double az = std::atan2(p.y(), p.x()) - kPi / 2;
            outside += std::abs(az) > kPi / 4 + 1e-9 ? 1 : 0;
            const double el = std::atan2(p.z(), std::hypot(p.x(), p.y()));
            outside += std::abs(el) > 15.0 * kPi / 180 + 1e-9 ? 1 : 0;
        }
        require(!points.empty(), "the sector sees a pillar");
        require(outside == 0, "no return outside the sector");
        // The predicate itself: azimuth and elevation limits with a 1e-9 rad margin; full circle and sphere do not
        // limit.
        const FieldOfView fov(90.0, 30.0);
        require(fov.limited() && !FieldOfView(360.0, 180.0).limited(), "limits");
        require(fov.contains({1, 0, 0}) && fov.contains({1, 1, 0}) && !fov.contains({1, 1.01, 0}), "azimuth edge");
        require(fov.contains({1, 0, std::tan(15.0 * kPi / 180)}) && !fov.contains({1, 0, 0.28}), "elevation edge");
        require(!fov.contains({-1, 0, 0}) && FieldOfView(360.0, 30.0).contains({-1, 0, 0}), "behind the sensor");
    }
}

void noiseAndGlobalMap() {
    // Seeded noise is reproducible; the global map lies on the surfaces.
    SensorConfig c = lidar32(SensorConfig::kRaycast);
    c.noise_std = 0.05;
    c.seed = 42;
    Cases first, second;
    WorldLidar a(c), b(c);
    a.setScene(first.scene(30, 8.0));
    b.setScene(second.scene(30, 8.0));
    for (int k = 0; k < 3; ++k) {
        const auto pa = a.scan({0, 0, 3}, kIdentity);
        const auto pb = b.scan({0, 0, 3}, kIdentity);
        require(pa.size() == pb.size(), "reproducible count");
        for (std::size_t i = 0; i < pa.size(); ++i) {
            require(pa[i] == pb[i], "reproducible points");
        }
    }
    const auto map = a.globalMap(0.2);
    double worst = 0.0;
    for (const auto& p : map) {
        worst = std::max(worst, std::abs(a.surfaceResidual(p)));
    }
    require(map.size() > 1000, "the global map is dense");
    require(worst < 1e-9, "the global map lies on the surfaces");
    require(a.globalMap(0.1).size() == a.scene()->sampleCount(),
            "the map at the scene's spacing is the scene's samples");

    // The error of a return along its beam is a zero-mean Gaussian of the configured standard deviation.
    SensorConfig wall = lidar32(SensorConfig::kRaycast);
    wall.range = 30.0;
    wall.v_fov_deg = 60.0;
    WorldLidar clean(wall);
    wall.noise_std = 0.1;
    wall.seed = 9;
    WorldLidar noisy(wall);
    const std::vector<Obstacle> far_wall{Obstacle::box({10, 0, 0}, {1, 100, 100})};
    clean.setScene(far_wall);
    noisy.setScene(far_wall);
    std::vector<double> errors;
    for (int scan = 0; scan < 6; ++scan) {
        const BeamScan reference = clean.scanWithBeams({0, 0, 0}, kIdentity);
        const BeamScan measured = noisy.scanWithBeams({0, 0, 0}, kIdentity);
        for (std::size_t i = 0; i < reference.size(); ++i) {
            if (reference.hit[i] && measured.hit[i]) {
                errors.push_back(measured.range[i] - reference.range[i]);
            }
        }
    }
    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
    const double square = std::accumulate(errors.begin(), errors.end(), 0.0, [mean](double sum, double e) {
        return sum + (e - mean) * (e - mean);
    });
    const double deviation = std::sqrt(square / static_cast<double>(errors.size()));
    require(errors.size() > 4000, "enough noisy returns");
    near(mean, 0.0, 5.0 * 0.1 / std::sqrt(static_cast<double>(errors.size())), "noise mean");
    near(deviation, 0.1, 0.005, "noise standard deviation");
    // Another seed or another scan index gives another stream.
    wall.seed = 10;
    WorldLidar other(wall);
    other.setScene(far_wall);
    const BeamScan x = other.scanWithBeams({0, 0, 0}, kIdentity);
    const BeamScan y = noisy.scanWithBeams({0, 0, 0}, kIdentity);
    require(x.range != y.range, "another seed, another noise");
}

void noiseStream() {
    // The scan seed is the std::seed_seq of (seed, index, index >> 32) without its allocation; engines seeded either
    // way are the same engine.
    for (const unsigned seed : {0u, 1u, 7u, 42u, 0xdeadbeefu, 0xffffffffu}) {
        for (const std::uint64_t index :
             {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{1000003}, std::uint64_t{0xffffffffu},
              std::uint64_t{0x100000000ull}, std::uint64_t{0x123456789abcdefull}}) {
            std::seed_seq reference{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(index),
                                    static_cast<std::uint32_t>(index >> 32)};
            std::mt19937_64 expected(reference);
            detail::ScanSeedSequence sequence(static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(index),
                                              static_cast<std::uint32_t>(index >> 32));
            std::mt19937_64 actual(sequence);
            for (int i = 0; i < 700; ++i) {
                require(expected() == actual(), "engine stream differs for seed " + std::to_string(seed) + " index " +
                                                    std::to_string(index) + " at draw " + std::to_string(i));
            }
            // Other sizes of the generated block (the 32-bit engine takes 624 words).
            std::mt19937 expected32(reference);
            std::mt19937 actual32(sequence);
            for (int i = 0; i < 700; ++i) {
                require(expected32() == actual32(), "32-bit engine stream differs");
            }
        }
    }
    // The noise of a scan: reproducible, reseeding restarts it, another index is another stream.
    RangeNoise a, b;
    a.reseed(5, 3, 0.1);
    b.reseed(5, 3, 0.1);
    std::vector<double> first;
    for (int i = 0; i < 50; ++i) {
        first.push_back(a());
        require(first.back() == b(), "equal noise streams");
    }
    a.reseed(5, 3, 0.1);
    for (int i = 0; i < 50; ++i) {
        require(a() == first[static_cast<std::size_t>(i)], "reseeding restarts the stream");
    }
    a.reseed(5, 4, 0.1);
    require(a() != first[0], "another scan index, another stream");
    detail::ScanSeedSequence sequence(5, 3, 0);
    std::mt19937_64 engine(sequence);
    std::normal_distribution<double> distribution(0.0, 0.1);
    require(distribution(engine) == first[0], "the stream is the standard normal distribution of the seeded engine");
}

void beamRecords() {
    // Beam records equal the points of scan(); free segments are free; no-return beams are empty.
    for (const double noise : {0.0, 0.02}) {
        SensorConfig c = lidar32(SensorConfig::kRaycast);
        c.range = 10.0;
        c.noise_std = noise;
        c.seed = 17;
        Cases first, second;
        WorldLidar a(c), b(c);
        a.setScene(first.scene(80, 10.0));
        b.setScene(second.scene(80, 10.0));
        const Eigen::Vector3d o(0.5, -0.5, 3.0);
        const Eigen::Quaterniond q(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized()));
        const auto points = a.scan(o, q);
        const BeamScan beams = b.scanWithBeams(o, q);
        require(beams.size() == b.beamCount(), "one record per beam");
        require(beams.direction.size() == beams.size() && beams.vehicle_id.size() == beams.size() &&
                    beams.hit.size() == beams.size(),
                "parallel arrays");
        require(beams.origin == o, "the shared origin");
        std::vector<Eigen::Vector3d> from_beams;
        int crossing = 0, bad_miss = 0, bad_record = 0;
        for (std::size_t i = 0; i < beams.size(); ++i) {
            bad_record += std::abs(beams.direction[i].norm() - 1.0) > 1e-12 || beams.vehicle_id[i] != -1 ? 1 : 0;
            if (beams.hit[i]) {
                from_beams.push_back(beams.point(i));
                bad_record += !(beams.range[i] > 0.0 && beams.range[i] <= c.range) ? 1 : 0;
                // [origin, return) shrunk by the noise bound is free.
                const double shrink = noise == 0.0 ? 1e-6 : 6.0 * noise;
                if (i % 5 == 0 && beams.range[i] > shrink) {
                    crossing +=
                        segmentCrossesInterior(b, o, o + (beams.range[i] - shrink) * beams.direction[i]) ? 1 : 0;
                }
            } else {
                bad_record += beams.range[i] != c.range ? 1 : 0;
                if (noise == 0.0) {
                    bad_miss += std::isfinite(b.castRay(o, beams.direction[i], c.range)) ? 1 : 0;
                }
            }
        }
        require(from_beams == points, "the hits are the points of scan()");
        require(crossing == 0, "free segments cross no solid");
        require(bad_miss == 0, "no-return beams have no surface in range");
        require(bad_record == 0, "well-formed records");
        // A second scan overwrites the records.
        BeamScan reused = beams;
        b.scanBeamsInto(o, q, {}, &reused);
        require(reused.size() == beams.size() && reused.range.capacity() >= beams.size(), "records are overwritten");
    }
    WorldLidar penetrating(lidar32(SensorConfig::kPenetrating));
    bool threw = false;
    try {
        penetrating.scanWithBeams(Eigen::Vector3d::Zero(), kIdentity);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "a penetrating sensor has no beams");
    require(penetrating.beamCount() == 0, "no beams");
}

void depthFrustum() {
    SensorConfig c;
    c.mode = SensorConfig::kDepthFrustum;
    c.width = 160;
    c.height = 120;
    c.h_fov_deg = 90.0;
    c.range = 20.0;
    c.min_range = 0.3;
    WorldLidar lidar(c);
    require(lidar.beamCount() == 160u * 120u, "one beam per pixel");
    near(lidar.config().fx, 80.0, 1e-9, "derived focal length");
    near(lidar.config().fy, 80.0, 1e-9, "square pixels");
    near(lidar.config().cx, 80.0, 0.0, "centred principal point");
    near(lidar.config().cy, 60.0, 0.0, "centred principal point");
    const Eigen::Quaterniond yaw90(Eigen::AngleAxisd(kPi / 2, Eigen::Vector3d::UnitZ())); // looks along +y
    const Obstacle front = Obstacle::box({0, 5, 0}, {4, 1, 4});
    const Obstacle back = Obstacle::box({0, 10, 0}, {30, 1, 30});
    const Obstacle behind = Obstacle::box({0, -5, 0}, {4, 1, 4});
    lidar.setScene({front, back, behind});
    WorldLidar front_only(c);
    front_only.setScene({front});
    const Eigen::Vector3d o = Eigen::Vector3d::Zero();
    const auto points = lidar.scan(o, yaw90);
    int occluded = 0, off_surface = 0, out_of_frustum = 0, far_face = 0, on_back = 0, on_behind = 0;
    const double tan_h = 1.0; // 90 degrees
    const double tan_v = tan_h * c.height / c.width;
    for (const auto& p : points) {
        on_back += p.y() > 9.0 ? 1 : 0;
        on_behind += p.y() < 0.0 ? 1 : 0;
        far_face += std::abs(p.y() - 5.5) < 1e-6 ? 1 : 0;
        if (p.y() > 9.0 && front_only.castRay(o, p - o, 20.0) < p.norm() - 1e-9) {
            ++occluded;
        }
        off_surface += std::abs(lidar.surfaceResidual(p)) > 1e-6 ? 1 : 0;
        const Eigen::Vector3d s = yaw90.inverse() * p; // sensor frame
        out_of_frustum +=
            !(s.x() > 0 && std::abs(s.y() / s.x()) <= tan_h + 1e-9 && std::abs(s.z() / s.x()) <= tan_v + 1e-9) ? 1 : 0;
        out_of_frustum += p.norm() > c.range + 1e-9 || p.norm() < c.min_range - 1e-9 ? 1 : 0;
    }
    require(!points.empty() && on_back > 0, "the wall behind the front box is seen around it");
    require(occluded == 0 && far_face == 0 && on_behind == 0, "occlusion, front faces only, one direction");
    require(off_surface == 0 && out_of_frustum == 0, "points on surfaces inside the frustum and the range");
    // Camera axes: pixel u grows to sensor -y, v to sensor -z.
    const BeamScan all = lidar.scanWithBeams(o, kIdentity);
    require(all.direction[60 * 160 + 80].x() > 0.99, "the principal pixel looks along +x");
    require(all.direction[60 * 160 + 10].y() > all.direction[60 * 160 + 150].y(), "u grows to -y");
    require(all.direction[10 * 160 + 80].z() > all.direction[100 * 160 + 80].z(), "v grows to -z");
    // Clutter: no return behind an occluder, exactly as raycast.
    Cases cases;
    WorldLidar cluttered(c);
    cluttered.setScene(cases.scene(100, 12.0));
    int crossings = 0;
    std::size_t total = 0;
    for (int k = 0; k < 3; ++k) {
        const Eigen::Vector3d s(-3.0 + 3.0 * k, 0.0, 3.5);
        const Eigen::Quaterniond q(Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitY()) *
                                   Eigen::AngleAxisd(2.0 * k, Eigen::Vector3d::UnitZ()));
        const auto cloud = cluttered.scan(s, q);
        total += cloud.size();
        for (std::size_t i = 0; i < cloud.size(); i += 5) {
            crossings += segmentCrossesInterior(cluttered, s, cloud[i]) ? 1 : 0;
        }
    }
    require(total > 0 && crossings == 0, "clutter seen without segments through solids");
    // min_range: a wall at 0.1 m (<= 0.16 m along the corner pixels) blocks the view and gives no return (unknown).
    lidar.setScene({Obstacle::box({0.6, 0, 0}, {1.0, 10, 10})});
    const BeamScan near_wall = lidar.scanWithBeams(o, kIdentity);
    require(std::count(near_wall.hit.begin(), near_wall.hit.end(), std::uint8_t{1}) == 0,
            "returns closer than min_range");
}

void headingCrop() {
    std::vector<Obstacle> ring;
    for (int k = 0; k < 12; ++k) {
        const double a = 2 * kPi * k / 12;
        ring.push_back(Obstacle::cylinder({3 * std::cos(a), 3 * std::sin(a), 0}, 0.4, 10.0));
    }
    const Eigen::Quaterniond attitude(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()) *
                                      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
    const Eigen::Vector3d x_body = attitude * Eigen::Vector3d::UnitX();
    for (const double cos_min : {0.0, 0.5}) {
        SensorConfig c = lidar32(SensorConfig::kPenetrating);
        c.range = 5.0;
        c.v_fov_deg = 180.0;
        c.penetrating_heading_crop = true;
        c.heading_cos_min = cos_min;
        WorldLidar cropped(c);
        cropped.setScene(ring);
        c.penetrating_heading_crop = false;
        WorldLidar full(c);
        full.setScene(ring);
        const auto a = cropped.scan(Eigen::Vector3d::Zero(), attitude);
        const auto b = full.scan(Eigen::Vector3d::Zero(), attitude);
        const auto violations = std::count_if(a.begin(), a.end(), [&](const Eigen::Vector3d& p) {
            return p.normalized().dot(x_body) < cos_min || std::abs(p.z()) > std::tan(kPi / 6) * 5.0 + 1e-12;
        });
        const auto expected = std::count_if(b.begin(), b.end(), [&](const Eigen::Vector3d& p) {
            return p.normalized().dot(x_body) >= cos_min && std::abs(p.z()) <= std::tan(kPi / 6) * 5.0;
        });
        require(!a.empty(), "the crop keeps points");
        require(violations == 0, "no point violates the crop");
        require(static_cast<std::ptrdiff_t>(a.size()) == expected, "the crop keeps exactly the points that satisfy it");
        require(a.size() < b.size(), "the crop removes points");
    }
    const HeadingCrop crop(true, 0.5, 0.5, 4.0);
    require(crop.enabled() && crop.contains({10, 0, 1.9}, {1, 0, 0}) && !crop.contains({10, 0, 2.1}, {1, 0, 0}),
            "slab half-height is tan * range");
    require(!crop.contains({0, 1, 0}, {1, 0, 0}) && crop.contains({1, 1, 0}, {1, 0, 0}) &&
                !crop.contains({1, 2, 0}, {1, 0, 0}),
            "heading cone of 60 degrees");
    require(HeadingCrop().contains({-1, 0, 100}, {1, 0, 0}), "a disabled crop keeps everything");
}

void vehicleBodies() {
    // Neighbours occlude, are tagged and are off by default.
    const Obstacle wall = Obstacle::box({8, 0, 0}, {1, 20, 20});
    const std::vector<VehicleBody> others{{3, {4, 0, 0}, 0.5}, {7, {-4, 0, 0}, 0.3}};
    for (int mode = 0; mode < 3; ++mode) {
        SensorConfig c = lidar32(mode == 0 ? SensorConfig::kRaycast
                                           : (mode == 1 ? SensorConfig::kPenetrating : SensorConfig::kDepthFrustum));
        c.v_fov_deg = 60.0;
        c.h_fov_deg = mode == 2 ? 90.0 : 360.0;
        WorldLidar lidar(c);
        lidar.setScene({wall});
        const auto tagged = lidar.scanTagged(Eigen::Vector3d::Zero(), kIdentity, others);
        const auto plain = lidar.scanTagged(Eigen::Vector3d::Zero(), kIdentity);
        int on3 = 0, on7 = 0, bad_tag = 0, static_points = 0;
        for (const auto& t : tagged) {
            if (t.vehicle_id == 3) {
                ++on3;
                bad_tag += std::abs((t.point - others[0].position).norm() - 0.5) > 1e-9 ? 1 : 0;
            } else if (t.vehicle_id == 7) {
                ++on7;
                bad_tag += std::abs((t.point - others[1].position).norm() - 0.3) > 1e-9 ? 1 : 0;
            } else {
                ++static_points;
                bad_tag += t.vehicle_id != -1 || std::abs(lidar.surfaceResidual(t.point)) > 1e-6 ? 1 : 0;
            }
        }
        const auto plain_tagged = std::count_if(plain.begin(), plain.end(), [](const TaggedPoint& t) {
            return t.vehicle_id != -1;
        });
        const std::string where = "mode " + std::to_string(mode);
        require(on3 > 0, where + ": the near neighbour is seen");
        require(mode == 2 ? on7 == 0 : on7 > 0, where + ": the camera looks along +x only");
        require(bad_tag == 0, where + ": tags and positions");
        require(plain_tagged == 0, where + ": bodies are off unless passed");
        if (mode != 1) {
            // Non-penetrating: the neighbour at x = 4 hides part of the wall.
            require(static_points < static_cast<int>(plain.size()), where + ": the neighbour hides the wall");
            const BeamScan beams = lidar.scanWithBeams(Eigen::Vector3d::Zero(), kIdentity, others);
            int tagged_beams = 0, bad_beam = 0;
            for (std::size_t i = 0; i < beams.size(); ++i) {
                if (beams.vehicle_id[i] < 0) {
                    continue;
                }
                ++tagged_beams;
                const Eigen::Vector3d p = beams.point(i);
                const bool on_body = std::abs((p - others[0].position).norm() - 0.5) <= 1e-9 ||
                                     std::abs((p - others[1].position).norm() - 0.3) <= 1e-9;
                bad_beam += !beams.hit[i] || !on_body ? 1 : 0;
            }
            require(tagged_beams == on3 + on7, where + ": beams and points agree on the tagged returns");
            require(bad_beam == 0, where + ": tagged beams hit their body");
        } else {
            require(static_points == static_cast<int>(plain.size()), where + ": see-through, nothing hidden");
        }
    }
    // A body out of range is not sampled, and an invalid body is refused.
    SensorConfig c = lidar32(SensorConfig::kPenetrating);
    c.range = 5.0;
    WorldLidar lidar(c);
    lidar.setScene(std::vector<Obstacle>{});
    require(lidar.scanTagged({0, 0, 0}, kIdentity, {{1, {100, 0, 0}, 0.5}}).empty(), "a far body is not sampled");
    require(refuses(
                [&] {
                    lidar.scanTagged({0, 0, 0}, kIdentity, {{1, {0, 0, 0}, 0.0}});
                },
                "invalid vehicle body"),
            "zero radius body");
    require(refuses(
                [&] {
                    lidar.scanTagged({0, 0, 0}, kIdentity, {{1, {kInf, 0, 0}, 0.3}});
                },
                "invalid vehicle body"),
            "infinite body position");
}

// ---- shared scenes, brute-force model, packed clouds
// -----------------------------------------------------------------

void sharedSceneIndex() {
    SensorConfig c;
    c.mode = SensorConfig::kPenetrating;
    c.range = 3.0;
    c.h_fov_deg = 90.0;
    c.v_fov_deg = 180.0;
    const std::vector<Obstacle> obstacles{Obstacle::box({2, 0, 0}, {1, 1, 1}), Obstacle::box({-2, 0, 0}, {1, 1, 1}),
                                          Obstacle::box({80, 0, 0}, {1, 1, 1})};
    auto scene = std::make_shared<LidarScene>(obstacles);
    WorldLidar a(c), b(c);
    a.setScene(scene);
    b.setScene(scene);
    require(a.scene().get() == b.scene().get(), "one scene for both sensors");
    const auto all = a.globalMap(c.surface_spacing);
    require(all.size() == scene->sampleCount(), "the global map is the scene's samples");
    const auto local = a.scan(Eigen::Vector3d::Zero(), kIdentity);
    const std::size_t expected =
        static_cast<std::size_t>(std::count_if(all.begin(), all.end(), [](const Eigen::Vector3d& p) {
            return p.squaredNorm() <= 9.0 && p.x() > 0 && std::abs(p.y()) <= p.x();
        }));
    require(local.size() == expected, "the scan keeps the samples in range and in the field of view");
    for (const auto& p : local) {
        require(p.x() > 0 && p.squaredNorm() <= 9.0, "a kept sample");
    }
    a.setScene(std::make_shared<LidarScene>(std::vector<Obstacle>{}));
    require(a.scan(Eigen::Vector3d::Zero(), kIdentity).empty(), "an empty scene gives no points");
    require(b.scan(Eigen::Vector3d::Zero(), kIdentity).size() == expected, "the other sensor keeps its scene");
    require(refuses(
                [&] {
                    b.setScene(std::make_shared<LidarScene>(obstacles, 0.2));
                },
                "sampling policy"),
            "a penetrating sensor refuses another sampling spacing");
    require(refuses(
                [&] {
                    b.setScene(std::make_shared<LidarScene>(obstacles, 0.1, true));
                },
                "sampling policy"),
            "and another burial policy");
    require(b.scene().get() == scene.get(), "the refused scene is not installed");
    // A raycast sensor takes any scene.
    WorldLidar ray(lidar32(SensorConfig::kRaycast));
    ray.setScene(std::make_shared<LidarScene>(obstacles, 0.7, true));
    require(refuses(
                [&] {
                    ray.setScene(std::shared_ptr<const LidarScene>());
                },
                "null lidar scene"),
            "null scene");
    // An invalid obstacle leaves the previous scene installed.
    require(refuses(
                [&] {
                    ray.setScene({Obstacle::sphere({0, 0, 0}, -1.0)});
                },
                "sphere radius"),
            "invalid obstacle");
    require(ray.obstacleCount() == 3, "the previous scene stays");
}

// The penetrating model, written as the plain loop it specifies: every scene sample in scene order (then every
// vehicle-body sample), kept when it lies in (0, range], at or beyond min_range, inside the field of view and the
// heading crop, with seeded range noise drawn per kept sample in that order. The indexed scan must return exactly
// these points in exactly this order.
std::vector<TaggedPoint> bruteForcePenetrating(const WorldLidar& lidar, const Eigen::Vector3d& position,
                                               const Eigen::Quaterniond& attitude,
                                               const std::vector<VehicleBody>& others, std::uint64_t scan_index) {
    const SensorConfig& c = lidar.config();
    const Eigen::Matrix3d rotation = Eigen::Quaterniond(attitude.coeffs() / attitude.norm()).toRotationMatrix();
    const Eigen::Matrix3d to_sensor = rotation.transpose();
    const double r2 = c.range * c.range, rmin2 = c.min_range * c.min_range;
    const bool noisy = c.noise_std > 0.0;
    std::seed_seq seq{static_cast<std::uint32_t>(c.seed), static_cast<std::uint32_t>(scan_index),
                      static_cast<std::uint32_t>(scan_index >> 32)};
    std::mt19937_64 rng(seq);
    std::normal_distribution<double> noise(0.0, noisy ? c.noise_std : 1.0);
    std::vector<TaggedPoint> out;
    const auto consider = [&](const Eigen::Vector3d& p, int vehicle) {
        const Eigen::Vector3d v = p - position;
        const double d2 = v.squaredNorm();
        if (d2 > r2 || d2 == 0.0 || d2 < rmin2) {
            return;
        }
        const Eigen::Vector3d s = to_sensor * v;
        if (c.h_fov_deg < 360.0 && std::abs(std::atan2(s.y(), s.x())) > 0.5 * c.h_fov_deg * kPi / 180.0 + 1e-9) {
            return;
        }
        if (c.v_fov_deg < 180.0 &&
            std::abs(std::atan2(s.z(), std::hypot(s.x(), s.y()))) > 0.5 * c.v_fov_deg * kPi / 180.0 + 1e-9) {
            return;
        }
        if (c.penetrating_heading_crop && (v.normalized().dot(rotation.col(0)) < c.heading_cos_min ||
                                           std::abs(v.z()) > c.vertical_slab_tan * c.range)) {
            return;
        }
        if (!noisy) {
            out.push_back({p, vehicle});
            return;
        }
        const double dist = std::sqrt(d2);
        const double t = dist + noise(rng);
        if (t <= 0.0 || t > c.range || t < c.min_range) {
            return;
        }
        out.push_back({position + v * (t / dist), vehicle});
    };
    for (const auto& p : lidar.globalMap(c.surface_spacing)) {
        consider(p, -1);
    }
    for (const auto& b : others) {
        if ((b.position - position).norm() - b.radius > c.range) {
            continue;
        }
        const int n = std::max(
            12, static_cast<int>(std::ceil(4.0 * kPi * b.radius * b.radius / (c.surface_spacing * c.surface_spacing))));
        const double golden = kPi * (3.0 - std::sqrt(5.0));
        for (int i = 0; i < n; ++i) {
            const double z = 1.0 - 2.0 * (i + 0.5) / n;
            const double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = golden * i;
            consider(b.position + b.radius * Eigen::Vector3d(rho * std::cos(phi), rho * std::sin(phi), z), b.id);
        }
    }
    return out;
}

void penetratingBruteForce() {
    Cases cases;
    const std::vector<Obstacle> scene = cases.scene(60, 12.0);
    int scans = 0, mismatches = 0;
    std::size_t points = 0;
    for (int variant = 0; variant < 8; ++variant) {
        SensorConfig c;
        c.mode = SensorConfig::kPenetrating;
        c.range = (variant % 2) != 0 ? 5.0 : 9.0;
        c.surface_spacing = variant < 4 ? 0.1 : 0.25;
        c.h_fov_deg = variant == 2 ? 120.0 : 360.0;
        c.v_fov_deg = variant == 2 ? 40.0 : 180.0;
        c.min_range = variant == 3 ? 1.5 : 0.0;
        c.penetrating_keep_buried = variant % 3 != 0;
        c.penetrating_heading_crop = variant == 5 || variant == 6;
        c.heading_cos_min = variant == 5 ? 0.5 : 0.0;
        c.noise_std = variant == 7 ? 0.05 : 0.0;
        c.seed = 11;
        WorldLidar lidar(c);
        lidar.setScene(scene);
        for (std::uint64_t k = 0; k < 6; ++k) {
            const Eigen::Vector3d p(cases.between(-12, 12), cases.between(-12, 12), cases.between(-2, 2));
            const Eigen::Quaterniond q = cases.rotation();
            std::vector<VehicleBody> others;
            if (k % 2) {
                others = {{3, p + Eigen::Vector3d(1.0, 0.5, 0.0), 0.3}, {9, p + Eigen::Vector3d(-2.0, 0.0, 0.4), 0.25}};
            }
            const auto expected = bruteForcePenetrating(lidar, p, q, others, k);
            const auto got = lidar.scanTagged(p, q, others);
            bool same = got.size() == expected.size();
            for (std::size_t i = 0; same && i < got.size(); ++i) {
                same = got[i].point == expected[i].point && got[i].vehicle_id == expected[i].vehicle_id;
            }
            mismatches += same ? 0 : 1;
            points += got.size();
            ++scans;
        }
    }
    require(mismatches == 0,
            std::to_string(mismatches) + " of " + std::to_string(scans) + " scans differ from the plain loop");
    require(points > 10000, "enough points compared");
}

void packedCloud() {
    // scanInto() writes the PointCloud2 records of exactly the points that scanTagged() returns for the same scan
    // index, in every mode.
    Cases cases;
    const std::vector<Obstacle> scene = cases.scene(40, 10.0);
    for (int mode = 0; mode < 3; ++mode) {
        for (const bool with_id : {false, true}) {
            const SensorConfig::Mode modes[] = {SensorConfig::kRaycast, SensorConfig::kPenetrating,
                                                SensorConfig::kDepthFrustum};
            SensorConfig c = lidar32(modes[mode]);
            c.range = 8.0;
            c.noise_std = 0.02;
            c.seed = 5;
            if (mode == 2) {
                c.h_fov_deg = 90.0;
            }
            WorldLidar tagged(c), packed(c);
            tagged.setScene(scene);
            packed.setScene(tagged.scene());
            std::vector<std::uint8_t> data(3, 0xab); // stale contents are replaced
            for (int k = 0; k < 4; ++k) {
                const Eigen::Vector3d p(1.0 * k - 2.0, 0.5, 0.3);
                const std::vector<VehicleBody> others{{4, p + Eigen::Vector3d(1.2, 0, 0), 0.3}};
                const auto expected = tagged.scanTagged(p, kIdentity, others);
                const std::size_t n = packed.scanInto(p, kIdentity, others, with_id, &data);
                const std::size_t stride = with_id ? 16 : 12;
                require(n == expected.size() && data.size() == n * stride,
                        "record count of mode " + std::to_string(mode));
                for (std::size_t i = 0; i < n; ++i) {
                    float xyz[3];
                    std::int32_t id = -1;
                    std::memcpy(xyz, data.data() + i * stride, sizeof xyz);
                    if (with_id) {
                        std::memcpy(&id, data.data() + i * stride + 12, sizeof id);
                    }
                    require(xyz[0] == static_cast<float>(expected[i].point.x()) &&
                                xyz[1] == static_cast<float>(expected[i].point.y()) &&
                                xyz[2] == static_cast<float>(expected[i].point.z()) &&
                                (!with_id || id == expected[i].vehicle_id),
                            "record " + std::to_string(i) + " of mode " + std::to_string(mode));
                }
            }
        }
    }
    // A reused buffer keeps its capacity and a custom sink receives the same points.
    WorldLidar lidar(lidar32(SensorConfig::kPenetrating));
    lidar.setScene(scene);
    std::vector<std::uint8_t> data;
    lidar.scanInto({0, 0, 0}, kIdentity, {}, false, &data);
    const std::size_t capacity = data.capacity();
    lidar.scanInto({0.1, 0, 0}, kIdentity, {}, false, &data);
    require(data.capacity() == capacity || data.capacity() > capacity, "capacity is kept");
    struct CountingSink {
        std::size_t bound = 0, pushed = 0;
        void reserve(std::size_t n) { bound = n; }
        void push(const Eigen::Vector3d&, int) { ++pushed; }
    } sink;
    lidar.scanTo({0, 0, 0}, kIdentity, {}, sink);
    require(sink.pushed > 0 && sink.pushed <= sink.bound, "a custom sink gets an upper bound and the points");
    require(refuses(
                [&] {
                    lidar.scanInto({0, 0, 0}, kIdentity, {}, false, nullptr);
                },
                "null cloud buffer"),
            "null buffer");
}

void fleetThreads() {
    // A fleet of sensors on one shared scene scanned by several threads publishes exactly the clouds of a serial pass:
    // every robot has its own sensor and scan counter, the scene is shared and never written.
    Cases cases;
    const auto scene = std::make_shared<const LidarScene>(cases.scene(50, 10.0), 0.1, true);
    SensorConfig c;
    c.mode = SensorConfig::kPenetrating;
    c.range = 8.0;
    c.h_fov_deg = 360.0;
    c.v_fov_deg = 180.0;
    c.penetrating_keep_buried = true;
    c.noise_std = 0.01;
    c.seed = 3;
    const int robots = 12;
    std::vector<std::unique_ptr<WorldLidar>> serial, pooled;
    for (int i = 0; i < robots; ++i) {
        serial.push_back(std::make_unique<WorldLidar>(c));
        pooled.push_back(std::make_unique<WorldLidar>(c));
        serial.back()->setScene(scene);
        pooled.back()->setScene(scene);
    }
    std::vector<std::vector<std::uint8_t>> a(robots), b(robots);
    int mismatches = 0;
    for (int tick = 0; tick < 5; ++tick) {
        const auto at = [&](int i) {
            return Eigen::Vector3d(-6.0 + i, 0.3 * tick, 0.5);
        };
        for (int i = 0; i < robots; ++i) {
            serial[static_cast<std::size_t>(i)]->scanInto(at(i), kIdentity, {}, false, &a[static_cast<std::size_t>(i)]);
        }
        std::vector<std::thread> threads;
        threads.reserve(4);
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&, t] {
                for (int i = t; i < robots; i += 4) {
                    pooled[static_cast<std::size_t>(i)]->scanInto(at(i), kIdentity, {}, false,
                                                                  &b[static_cast<std::size_t>(i)]);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (int i = 0; i < robots; ++i) {
            mismatches += a[static_cast<std::size_t>(i)] == b[static_cast<std::size_t>(i)] ? 0 : 1;
        }
    }
    require(mismatches == 0, std::to_string(mismatches) + " clouds differ from the serial pass");
}

// ---- configuration, requests, patterns ----------------------------------------------------------------------------

void configValidation() {
    const auto bad = [](auto&& edit, const std::string& needle) {
        SensorConfig c;
        edit(c);
        return refuses(
            [&] {
                WorldLidar lidar(c);
            },
            needle);
    };
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    require(bad(
                [](SensorConfig& c) {
                    c.range = 0.0;
                },
                "range must be > 0"),
            "zero range");
    require(bad(
                [](SensorConfig& c) {
                    c.range = nan;
                },
                "range must be > 0"),
            "NaN range");
    require(bad(
                [](SensorConfig& c) {
                    c.min_range = -1.0;
                },
                "min_range"),
            "negative min_range");
    require(bad(
                [](SensorConfig& c) {
                    c.min_range = c.range;
                },
                "min_range"),
            "min_range at range");
    require(bad(
                [](SensorConfig& c) {
                    c.h_fov_deg = 0.0;
                },
                "h_fov_deg"),
            "zero horizontal fov");
    require(bad(
                [](SensorConfig& c) {
                    c.h_fov_deg = 361.0;
                },
                "h_fov_deg"),
            "horizontal fov over 360");
    require(bad(
                [](SensorConfig& c) {
                    c.v_fov_deg = -1.0;
                },
                "v_fov_deg"),
            "negative vertical fov");
    require(bad(
                [](SensorConfig& c) {
                    c.v_fov_deg = 181.0;
                },
                "v_fov_deg"),
            "vertical fov over 180");
    require(bad(
                [](SensorConfig& c) {
                    c.noise_std = -0.1;
                },
                "noise_std"),
            "negative noise");
    require(bad(
                [](SensorConfig& c) {
                    c.noise_std = kInf;
                },
                "noise_std"),
            "infinite noise");
    require(bad(
                [](SensorConfig& c) {
                    c.h_res = 0;
                },
                "h_res and v_res"),
            "no columns");
    require(bad(
                [](SensorConfig& c) {
                    c.v_res = 0;
                },
                "h_res and v_res"),
            "no rows");
    require(bad(
                [](SensorConfig& c) {
                    c.surface_spacing = 0.0;
                },
                "spacing"),
            "zero spacing");
    require(bad(
                [](SensorConfig& c) {
                    c.surface_spacing = nan;
                },
                "spacing"),
            "NaN spacing, in any mode");
    require(bad(
                [](SensorConfig& c) {
                    c.heading_cos_min = 1.5;
                },
                "heading_cos_min"),
            "heading cosine");
    require(bad(
                [](SensorConfig& c) {
                    c.vertical_slab_tan = -1.0;
                },
                "vertical_slab_tan"),
            "slab");
    require(bad(
                [](SensorConfig& c) {
                    c.mode = SensorConfig::kDepthFrustum;
                    c.width = 0;
                },
                "width and height"),
            "no pixels");
    require(bad(
                [](SensorConfig& c) {
                    c.mode = SensorConfig::kDepthFrustum;
                    c.h_fov_deg = 180.0;
                },
                "h_fov_deg < 180"),
            "camera with a half-space field of view");
    require(bad(
                [](SensorConfig& c) {
                    c.mode = SensorConfig::kDepthFrustum;
                    c.fx = -5.0;
                    c.fy = 5.0;
                },
                "pinhole"),
            "negative focal length");
    // Depth cameras do not need h_res and v_res; raycast and penetrating sensors do not need width and height.
    SensorConfig camera;
    camera.mode = SensorConfig::kDepthFrustum;
    camera.h_fov_deg = 60.0;
    camera.h_res = 0;
    camera.v_res = 0;
    const WorldLidar camera_without_grid(camera);
    SensorConfig ray;
    ray.width = 0;
    const WorldLidar ray_without_pixels(ray);
    // Explicit intrinsics are kept.
    camera.fx = 100.0;
    camera.fy = 120.0;
    camera.cx = 77.0;
    camera.cy = 55.0;
    const WorldLidar explicit_camera(camera);
    require(explicit_camera.config().fx == 100.0 && explicit_camera.config().fy == 120.0 &&
                explicit_camera.config().cx == 77.0 && explicit_camera.config().cy == 55.0,
            "explicit intrinsics are kept");
    camera.fx = 0.0;
    camera.fy = 0.0;
    camera.cx = 0.0;
    camera.cy = 0.0;
    near(WorldLidar(camera).config().fx, 0.5 * camera.width / std::tan(30.0 * kPi / 180.0), 1e-12, "derived fx");
}

void scanRequests() {
    // A refused request does not consume a scan index: the noise of the next scan is that of a fresh sensor's first.
    SensorConfig c = lidar32(SensorConfig::kRaycast);
    c.noise_std = 0.05;
    c.seed = 8;
    Cases cases;
    const std::vector<Obstacle> scene = cases.scene(30, 6.0);
    WorldLidar fresh(c), tried(c);
    fresh.setScene(scene);
    tried.setScene(scene);
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    require(refuses(
                [&] {
                    tried.scan({nan, 0, 0}, kIdentity);
                },
                "non-finite sensor position"),
            "NaN position");
    require(refuses(
                [&] {
                    tried.scan({0, 0, 0}, Eigen::Quaterniond(0, 0, 0, 0));
                },
                "orientation"),
            "zero attitude");
    require(refuses(
                [&] {
                    tried.scanTagged({0, 0, 0}, kIdentity, {{1, {0, 0, nan}, 0.3}});
                },
                "invalid vehicle body"),
            "NaN body");
    require(tried.scan({0, 0, 3}, kIdentity) == fresh.scan({0, 0, 3}, kIdentity),
            "refused requests consume no scan index");
    require(tried.scan({0, 0, 3}, kIdentity) != tried.scan({0, 0, 3}, kIdentity), "every scan has its own noise");
    // A non-unit attitude quaternion is normalised.
    WorldLidar a(lidar32(SensorConfig::kRaycast)), b(lidar32(SensorConfig::kRaycast));
    a.setScene(scene);
    b.setScene(scene);
    const Eigen::Quaterniond q = cases.rotation();
    const Eigen::Quaterniond scaled(q.coeffs() * 3.0);
    const auto pa = a.scan({0, 0, 3}, q);
    const auto pb = b.scan({0, 0, 3}, scaled);
    require(pa.size() == pb.size(), "a scaled attitude gives the same scan");
    for (std::size_t i = 0; i < pa.size(); ++i) {
        require((pa[i] - pb[i]).norm() < 1e-9, "a scaled attitude gives the same points");
    }
    // Buffers must exist.
    require(refuses(
                [&] {
                    a.scanTaggedInto({0, 0, 0}, kIdentity, {}, nullptr);
                },
                "null point buffer"),
            "null points");
    require(refuses(
                [&] {
                    a.scanBeamsInto({0, 0, 0}, kIdentity, {}, nullptr);
                },
                "null beam buffer"),
            "null beams");
    // castRay normalises its direction and refuses nonsense quietly.
    a.setScene({Obstacle::sphere({5, 0, 0}, 1.0)});
    near(a.castRay({0, 0, 0}, {7, 0, 0}, 20.0), 4.0, 1e-12, "castRay normalises");
    require(std::isinf(a.castRay({0, 0, 0}, {0, 0, 0}, 20.0)), "zero direction");
    require(std::isinf(a.castRay({nan, 0, 0}, {1, 0, 0}, 20.0)), "NaN origin");
}

void beamPatterns() {
    SensorConfig c = lidar32(SensorConfig::kRaycast);
    c.h_res = 8;
    c.v_res = 3;
    c.h_fov_deg = 360.0;
    c.v_fov_deg = 20.0;
    resolveSensorConfig(&c);
    const auto full = BeamPattern::make(c);
    require(full->size() == 24, "h_res x v_res beams");
    // Rows run from the lowest elevation; a full circle starts at -pi and has no seam beam.
    near(std::asin(full->directions()[0].z()), -10.0 * kPi / 180, 1e-12, "lowest row");
    near(std::asin(full->directions()[8].z()), 0.0, 1e-12, "middle row");
    near(std::asin(full->directions()[16].z()), 10.0 * kPi / 180, 1e-12, "top row");
    near(std::atan2(full->directions()[0].y(), full->directions()[0].x()), -kPi, 1e-12, "first azimuth");
    near(std::atan2(full->directions()[7].y(), full->directions()[7].x()), kPi * 3.0 / 4.0, 1e-12, "last azimuth");
    for (const auto& d : full->directions()) {
        near(d.norm(), 1.0, 1e-12, "unit beams");
    }
    // A sector spans its field of view edge to edge; one row is level, one column looks ahead.
    c.h_fov_deg = 90.0;
    c.h_res = 5;
    c.v_res = 1;
    const auto sector = BeamPattern::make(c);
    near(std::atan2(sector->directions().front().y(), sector->directions().front().x()), -kPi / 4, 1e-12,
         "sector start");
    near(std::atan2(sector->directions().back().y(), sector->directions().back().x()), kPi / 4, 1e-12, "sector end");
    near(sector->directions()[2].x(), 1.0, 1e-12, "middle beam looks ahead");
    c.h_res = 1;
    const auto one = BeamPattern::make(c);
    require(one->size() == 1 && one->directions()[0].isApprox(Eigen::Vector3d::UnitX(), 1e-12), "one beam ahead");
    // Sensors with equal parameters share one pattern; another configuration is refused.
    SensorConfig config = lidar32(SensorConfig::kRaycast);
    resolveSensorConfig(&config);
    const auto shared = BeamPattern::make(config);
    WorldLidar a(config, shared), b(config, shared);
    require(a.beamPattern() == b.beamPattern() && a.beamPattern() == shared, "the pattern is shared");
    require(shared->matches(config), "the pattern matches its configuration");
    SensorConfig other = config;
    other.h_res = 180;
    require(!shared->matches(other), "another resolution is another pattern");
    require(refuses(
                [&] {
                    WorldLidar(other, shared);
                },
                "beam pattern differs"),
            "a mismatching pattern is refused");
    SensorConfig deeper = config;
    deeper.range = 50.0;
    require(shared->matches(deeper), "range does not change the pattern");
    WorldLidar c2(deeper, shared);
    require(c2.config().range == 50.0, "the sensor keeps its own range");
    // Scans through a shared pattern equal scans with an own one.
    Cases cases;
    const std::vector<Obstacle> scene = cases.scene(30, 6.0);
    WorldLidar own(config);
    a.setScene(scene);
    own.setScene(scene);
    require(a.scan({0, 0, 2}, kIdentity) == own.scan({0, 0, 2}, kIdentity), "shared and own patterns scan alike");
    // A pinhole camera: principal ray ahead, u to -y, v to -z, and matching parameters.
    SensorConfig camera;
    camera.mode = SensorConfig::kDepthFrustum;
    camera.width = 4;
    camera.height = 2;
    camera.fx = camera.fy = 2.0;
    resolveSensorConfig(&camera);
    const auto pinhole = BeamPattern::make(camera);
    require(pinhole->size() == 8, "one beam per pixel");
    const Eigen::Vector3d expected = Eigen::Vector3d(1.0, -(0 + 0.5 - 2.0) / 2.0, -(0 + 0.5 - 1.0) / 2.0).normalized();
    require((pinhole->directions()[0] - expected).norm() < 1e-15, "first pixel");
    require(pinhole->matches(camera) && !pinhole->matches(c), "pinhole parameters identify the pattern");
    require(BeamPattern::make(lidar32(SensorConfig::kPenetrating))->size() == 0, "a penetrating sensor has no beams");
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 17> tests{{
        {"raycast_occlusion", raycastOcclusion},
        {"penetrating_back_faces", penetratingBackFaces},
        {"on_surface_and_in_range", onSurfaceAndInRange},
        {"field_of_view", fieldOfView},
        {"noise_and_global_map", noiseAndGlobalMap},
        {"noise_stream", noiseStream},
        {"beams", beamRecords},
        {"depth_frustum", depthFrustum},
        {"heading_crop", headingCrop},
        {"vehicle_bodies", vehicleBodies},
        {"shared_scene_index", sharedSceneIndex},
        {"penetrating_brute_force", penetratingBruteForce},
        {"packed_cloud", packedCloud},
        {"fleet_threads", fleetThreads},
        {"config_validation", configValidation},
        {"scan_requests", scanRequests},
        {"beam_patterns", beamPatterns},
    }};
    if (argc > 2) {
        std::cerr << "Usage: scene_lidar_test [case_name|--list]\n";
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
