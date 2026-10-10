#pragma once

// World-frame LiDAR / point-cloud sensor model, without ROS.
//
// One scene, one sensor model and one output frame for every simulated robot that enables it, on every simulator.
// The scene is a set of closed convex solids; a scan returns WORLD-frame points.
//
//   kRaycast      first return per beam (non-penetrating). Beam pattern: h_res x v_res beams over h_fov_deg x
//                 v_fov_deg, sensor frame x forward, y left, z up, elevation centred on the x-y plane. A beam that
//                 hits nothing inside `range` yields no point.
//   kPenetrating  every surface sample of every obstacle (spacing `surface_spacing`) that lies within `range` and
//                 inside the angular field of view; no occlusion, so back faces and hidden obstacles are returned
//                 (depth-map / global-map style). Samples strictly inside another obstacle are dropped, so the cloud
//                 lies on the boundary of the union of the solids, unless `penetrating_keep_buried`. Optional
//                 `penetrating_heading_crop` reproduces the CPU ZJU local_sensing crop (pointcloud_render_node.cpp:
//                 127-136): keep dot(normalized(p - x), body x axis) >= heading_cos_min and
//                 |p_z - x_z| <= vertical_slab_tan * range.
//   kDepthFrustum first return per pixel of a pinhole camera (width x height, intrinsics fx fy cx cy; 0 derives them
//                 from h_fov_deg with square pixels and a centred principal point). The camera looks along sensor +x;
//                 pixel u grows to sensor -y, v to sensor -z. Same ray caster, occlusion and range limits as kRaycast.
//
// Range limits: a return is kept when min_range <= distance <= range, the distance measured along the beam
// (Euclidean, also for kDepthFrustum).
//
// Noise: zero-mean Gaussian of standard deviation `noise_std` along the sensor-to-point direction (range noise).
// Scans draw from a generator seeded by `seed` and the scan index, so a run is reproducible.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "xgc2_math/scene/surface_sampling.hpp"

namespace xgc2_math::scene {

struct SensorConfig {
    enum Mode { kRaycast, kPenetrating, kDepthFrustum };
    Mode mode = kRaycast;
    double range = 20.0;          // metres, max range
    double min_range = 0.0;       // metres, returns closer than this are dropped
    double h_fov_deg = 360.0;     // (0, 360]
    double v_fov_deg = 30.0;      // [0, 180]
    int h_res = 360;              // beams per row
    int v_res = 32;               // rows
    double surface_spacing = 0.1; // metres, kPenetrating sample spacing
    double noise_std = 0.0;       // metres, range noise
    unsigned seed = 0;
    // kPenetrating: ZJU local_sensing heading half-space / cone and z slab.
    bool penetrating_heading_crop = false;
    double heading_cos_min = 0.0;                   // 0.0 half-space, 0.5 +-60 deg cone
    double vertical_slab_tan = 0.57735026918962576; // tan(pi/6)
    // kPenetrating: keep samples that lie strictly inside another obstacle (every part is sampled on its own).
    bool penetrating_keep_buried = false;
    // kDepthFrustum pinhole camera (ZJU camera.yaml is 640 x 480).
    int width = 160;
    int height = 120;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0; // pixels; 0 = derive
};

// Checks a configuration and fills the derived kDepthFrustum intrinsics (fx, fy, cx, cy that are 0). Throws
// std::invalid_argument naming the offending field.
inline void resolveSensorConfig(SensorConfig* config) {
    SensorConfig& c = *config;
    if (!std::isfinite(c.range) || c.range <= 0.0) {
        throw std::invalid_argument("range must be > 0");
    }
    if (!std::isfinite(c.min_range) || c.min_range < 0.0 || c.min_range >= c.range) {
        throw std::invalid_argument("min_range must be in [0, range)");
    }
    if (!(c.h_fov_deg > 0.0 && c.h_fov_deg <= 360.0)) {
        throw std::invalid_argument("h_fov_deg must be in (0, 360]");
    }
    if (!(c.v_fov_deg >= 0.0 && c.v_fov_deg <= 180.0)) {
        throw std::invalid_argument("v_fov_deg must be in [0, 180]");
    }
    if (!std::isfinite(c.noise_std) || c.noise_std < 0.0) {
        throw std::invalid_argument("noise_std must be >= 0");
    }
    if (c.mode == SensorConfig::kPenetrating && !(std::isfinite(c.surface_spacing) && c.surface_spacing > 0.0)) {
        throw std::invalid_argument("surface_spacing must be > 0");
    }
    if (!(c.heading_cos_min >= -1.0 && c.heading_cos_min <= 1.0)) {
        throw std::invalid_argument("heading_cos_min must be in [-1, 1]");
    }
    if (!(std::isfinite(c.vertical_slab_tan) && c.vertical_slab_tan >= 0.0)) {
        throw std::invalid_argument("vertical_slab_tan must be >= 0");
    }
    if (c.mode == SensorConfig::kDepthFrustum) {
        if (c.width < 1 || c.height < 1) {
            throw std::invalid_argument("width and height must be >= 1");
        }
        if (c.fx == 0.0 || c.fy == 0.0) {
            if (!(c.h_fov_deg < 180.0)) {
                throw std::invalid_argument("kDepthFrustum needs h_fov_deg < 180");
            }
            const double f = 0.5 * c.width / std::tan(0.5 * c.h_fov_deg * detail::kPi / 180.0);
            if (c.fx == 0.0) {
                c.fx = f;
            }
            if (c.fy == 0.0) {
                c.fy = c.fx;
            }
        }
        if (c.cx == 0.0) {
            c.cx = 0.5 * c.width;
        }
        if (c.cy == 0.0) {
            c.cy = 0.5 * c.height;
        }
        if (!(c.fx > 0.0 && c.fy > 0.0 && std::isfinite(c.cx) && std::isfinite(c.cy))) {
            throw std::invalid_argument("invalid pinhole intrinsics");
        }
    } else if (c.h_res < 1 || c.v_res < 1) {
        throw std::invalid_argument("h_res and v_res must be >= 1");
    }
}

// A returned point with its source (-1 static geometry, else a vehicle id).
struct TaggedPoint {
    Eigen::Vector3d point;
    int vehicle_id = -1;
};

// Another vehicle's body, a sphere, seen by the sensor (off unless passed).
struct VehicleBody {
    int id = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double radius = 0.3;
};

// Per-beam records of one kRaycast / kDepthFrustum scan, in beam order, as parallel arrays. The free segment of a beam
// is [origin, origin + range * direction) only when `hit`. A scan overwrites the arrays and keeps their capacity.
struct BeamScan {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero(); // world, shared by every beam of the scan
    std::vector<Eigen::Vector3d> direction;           // world, unit
    std::vector<double> range;                        // return distance (with noise) if hit, else SensorConfig::range
    std::vector<std::int32_t> vehicle_id;             // >= 0: the return is on that vehicle's body, -1: static or none
    std::vector<std::uint8_t> hit;

    std::size_t size() const { return range.size(); }
    Eigen::Vector3d point(std::size_t i) const { return origin + range[i] * direction[i]; }
    void clear() {
        direction.clear();
        range.clear();
        vehicle_id.clear();
        hit.clear();
    }
};

// Angular field of view of a sensor in its own frame (x forward, y left, z up): azimuth about +z and elevation off the
// x-y plane, both centred on the x axis. A full circle (360 deg) or full sphere (180 deg) does not limit.
class FieldOfView {
  public:
    FieldOfView(double h_fov_deg, double v_fov_deg)
        : h_limited_(h_fov_deg < 360.0), v_limited_(v_fov_deg < 180.0), h_half_(0.5 * h_fov_deg * detail::kPi / 180.0),
          v_half_(0.5 * v_fov_deg * detail::kPi / 180.0) {}

    bool limited() const { return h_limited_ || v_limited_; }

    // Whether a vector in the sensor frame points into the field of view (within 1e-9 rad of its edge).
    bool contains(const Eigen::Vector3d& sensor_frame_vector) const {
        constexpr double tol = 1e-9;
        if (h_limited_) {
            const double az = std::atan2(sensor_frame_vector.y(), sensor_frame_vector.x());
            if (std::abs(az) > h_half_ + tol) {
                return false;
            }
        }
        if (v_limited_) {
            const double el =
                std::atan2(sensor_frame_vector.z(), std::hypot(sensor_frame_vector.x(), sensor_frame_vector.y()));
            if (std::abs(el) > v_half_ + tol) {
                return false;
            }
        }
        return true;
    }

  private:
    bool h_limited_, v_limited_;
    double h_half_, v_half_;
};

// ZJU local_sensing CPU crop: a heading test on the full body x axis and a z slab of half-height
// vertical_slab_tan * range (not a per-point elevation).
class HeadingCrop {
  public:
    HeadingCrop() = default;
    HeadingCrop(bool enabled, double heading_cos_min, double vertical_slab_tan, double range)
        : enabled_(enabled), cos_min_(heading_cos_min), slab_half_height_(vertical_slab_tan * range) {}

    bool enabled() const { return enabled_; }

    // `world_offset` is the point minus the sensor position, `body_x` the sensor x axis in the world frame.
    bool contains(const Eigen::Vector3d& world_offset, const Eigen::Vector3d& body_x) const {
        if (!enabled_) {
            return true;
        }
        if (world_offset.normalized().dot(body_x) < cos_min_) {
            return false;
        }
        return std::abs(world_offset.z()) <= slab_half_height_;
    }

  private:
    bool enabled_ = false;
    double cos_min_ = 0.0;
    double slab_half_height_ = 0.0;
};

namespace detail {

// The three 32-bit words of a scan's seed as a std::seed_seq, without its heap allocation. generate() is the
// algorithm of the C++ standard as libstdc++ implements it, so the engines it seeds match those seeded with
// std::seed_seq (the test suite compares them). It is also cheaper: the index arithmetic of libstdc++ divides for
// every access.
class ScanSeedSequence {
  public:
    using result_type = std::uint32_t;

    ScanSeedSequence(std::uint32_t a, std::uint32_t b, std::uint32_t c) : v_{{a, b, c}} {}

    std::size_t size() const { return v_.size(); }

    template <class OutputIt> void param(OutputIt out) const { std::copy(v_.begin(), v_.end(), out); }

    template <class RandomIt> void generate(RandomIt begin, RandomIt end) const {
        using Word = std::uint32_t;
        if (begin == end) {
            return;
        }
        std::fill(begin, end, Word{0x8b8b8b8bu});
        const std::size_t n = static_cast<std::size_t>(end - begin);
        const std::size_t s = v_.size();
        const std::size_t t = (n >= 623) ? 11 : (n >= 68) ? 7 : (n >= 39) ? 5 : (n >= 7) ? 3 : (n - 1) / 2;
        const std::size_t p = (n - t) / 2;
        const std::size_t q = p + t;
        const std::size_t m = std::max(s + 1, n);
        // Indices k % n, (k + p) % n, (k + q) % n and (k - 1) % n, advanced without division. The last one wraps
        // from k = 0 through the unsigned range like the library's, and then follows k.
        std::size_t k0 = 0, kp = p % n, kq = q % n, km = (static_cast<std::size_t>(0) - 1) % n;
        const auto advance = [n](std::size_t* i) {
            if (++*i == n) {
                *i = 0;
            }
        };
        for (std::size_t k = 0; k < m; ++k) {
            Word arg = begin[static_cast<std::ptrdiff_t>(k0)] ^ begin[static_cast<std::ptrdiff_t>(kp)] ^
                       begin[static_cast<std::ptrdiff_t>(km)];
            const Word r1 = Word{1664525u} * (arg ^ (arg >> 27));
            Word r2 = r1;
            if (k == 0) {
                r2 += static_cast<Word>(s);
            } else if (k <= s) {
                r2 += static_cast<Word>(k0) + v_[k - 1];
            } else {
                r2 += static_cast<Word>(k0);
            }
            begin[static_cast<std::ptrdiff_t>(kp)] += r1;
            begin[static_cast<std::ptrdiff_t>(kq)] += r2;
            begin[static_cast<std::ptrdiff_t>(k0)] = r2;
            km = k0;
            advance(&k0);
            advance(&kp);
            advance(&kq);
        }
        for (std::size_t k = m; k < m + n; ++k) {
            const Word arg = begin[static_cast<std::ptrdiff_t>(k0)] + begin[static_cast<std::ptrdiff_t>(kp)] +
                             begin[static_cast<std::ptrdiff_t>(km)];
            const Word r3 = Word{1566083941u} * (arg ^ (arg >> 27));
            const Word r4 = r3 - static_cast<Word>(k0);
            begin[static_cast<std::ptrdiff_t>(kp)] ^= r3;
            begin[static_cast<std::ptrdiff_t>(kq)] ^= r4;
            begin[static_cast<std::ptrdiff_t>(k0)] = r4;
            km = k0;
            advance(&k0);
            advance(&kp);
            advance(&kq);
        }
    }

  private:
    std::array<std::uint32_t, 3> v_;
};

} // namespace detail

// Zero-mean Gaussian range noise of one scan: the stream is a function of the sensor seed and the scan index, so a
// run is reproducible. Drawing and reseeding allocate nothing.
class RangeNoise {
  public:
    // The generator is seeded on the first reseed().
    RangeNoise() = default;

    // Starts the stream of scan `scan_index` of a sensor with the given seed and standard deviation (metres).
    void reseed(unsigned seed, std::uint64_t scan_index, double std_dev) {
        detail::ScanSeedSequence seq(static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(scan_index),
                                     static_cast<std::uint32_t>(scan_index >> 32));
        engine_.seed(seq);
        distribution_ = std::normal_distribution<double>(0.0, std_dev);
    }

    // The next range error.
    double operator()() { return distribution_(engine_); }

  private:
    std::mt19937_64 engine_;
    std::normal_distribution<double> distribution_;
};

} // namespace xgc2_math::scene
