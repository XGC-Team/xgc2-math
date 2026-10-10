#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "xgc2_math/scene/beam_pattern.hpp"
#include "xgc2_math/scene/lidar_scene.hpp"
#include "xgc2_math/scene/sensor_model.hpp"

namespace xgc2_math::scene {

namespace detail {

// PointCloud2 records written in place: float32 x y z (+ int32 vehicle id) in host byte order. The buffer only grows
// to the announced bound, so a buffer reused across scans does not reallocate.
class PackedCloudSink {
  public:
    PackedCloudSink(std::vector<std::uint8_t>* data, bool with_id)
        : data_(data), stride_(with_id ? 16 : 12), with_id_(with_id) {}

    void reserve(std::size_t n) {
        if (data_->size() < n * stride_) {
            data_->resize(n * stride_);
        }
    }

    void push(const Eigen::Vector3d& p, int vehicle) {
        if (data_->size() < (count_ + 1) * stride_) {
            data_->resize(std::max<std::size_t>(2 * data_->size(), (count_ + 1) * stride_));
        }
        std::uint8_t* out = data_->data() + count_ * stride_;
        const float xyz[3] = {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
        std::memcpy(out, xyz, sizeof xyz);
        if (with_id_) {
            const std::int32_t id = vehicle;
            std::memcpy(out + sizeof xyz, &id, sizeof id);
        }
        ++count_;
    }

    // Cuts the buffer to the records written and returns their number.
    std::size_t finish() {
        data_->resize(count_ * stride_);
        return count_;
    }

  private:
    std::vector<std::uint8_t>* data_;
    std::size_t stride_;
    bool with_id_;
    std::size_t count_ = 0;
};

class TaggedPointSink {
  public:
    explicit TaggedPointSink(std::vector<TaggedPoint>* points) : points_(points) { points_->clear(); }
    void reserve(std::size_t n) { points_->reserve(n); }
    void push(const Eigen::Vector3d& p, int vehicle) { points_->push_back({p, vehicle}); }

  private:
    std::vector<TaggedPoint>* points_;
};

} // namespace detail

// A sensor in the world frame: a configuration, its beams and the scene it observes. A scan returns world-frame
// points, optionally tagged with the vehicle body they lie on.
//
// Sensors with equal parameters can share a BeamPattern, and any number of sensors share one immutable LidarScene.
// The sensor itself is not thread safe: a scan advances the scan index and works in the sensor's own scratch buffers,
// so give every concurrent scanner its own WorldLidar. After the first scan with a given scene and number of bodies a
// scan allocates nothing (range noise included) and writes into buffers the caller reuses.
class WorldLidar {
  public:
    // Throws std::invalid_argument on an invalid configuration.
    explicit WorldLidar(const SensorConfig& config, std::shared_ptr<const BeamPattern> pattern = nullptr)
        // An invalid sample spacing is an error in every mode, as a sensor may be handed scenes of its own policy.
        : config_(config), scene_(std::make_shared<LidarScene>(std::vector<Obstacle>{}, config.surface_spacing,
                                                               config.penetrating_keep_buried)) {
        resolveSensorConfig(&config_);
        if (pattern) {
            if (!pattern->matches(config_)) {
                throw std::invalid_argument("beam pattern differs from the sensor configuration");
            }
            pattern_ = std::move(pattern);
        } else {
            pattern_ = BeamPattern::make(config_);
        }
        field_of_view_ = FieldOfView(config_.h_fov_deg, config_.v_fov_deg);
        crop_ = HeadingCrop(config_.penetrating_heading_crop, config_.heading_cos_min, config_.vertical_slab_tan,
                            config_.range);
    }

    WorldLidar(const WorldLidar&) = delete;
    WorldLidar& operator=(const WorldLidar&) = delete;

    // Replaces the scene (rebuilds the hierarchy, shared surface samples and index). Throws std::invalid_argument on a
    // degenerate obstacle and then leaves the previous scene installed.
    void setScene(const std::vector<Obstacle>& obstacles) {
        setScene(std::make_shared<LidarScene>(obstacles, config_.surface_spacing, config_.penetrating_keep_buried));
    }

    // Installs an already compiled fleet scene. Penetrating scans require its sampling spacing and buried-surface
    // policy to match their configuration.
    void setScene(std::shared_ptr<const LidarScene> scene) {
        if (!scene) {
            throw std::invalid_argument("null lidar scene");
        }
        if (config_.mode == SensorConfig::kPenetrating &&
            (scene->spacing() != config_.surface_spacing || scene->keepBuried() != config_.penetrating_keep_buried)) {
            throw std::invalid_argument("shared scene sampling policy differs from sensor");
        }
        scene_ = std::move(scene);
        bits_.reserve((scene_->sampleCount() + 63) / 64);
    }

    const std::shared_ptr<const LidarScene>& scene() const { return scene_; }
    // The configuration with its derived pinhole intrinsics.
    const SensorConfig& config() const { return config_; }
    const std::shared_ptr<const BeamPattern>& beamPattern() const { return pattern_; }
    std::size_t obstacleCount() const { return scene_->obstacleCount(); }
    // Beams of one scan; kPenetrating has none.
    std::size_t beamCount() const { return pattern_->size(); }

    // ---- scans ------------------------------------------------------------------------------------------------
    // A scan is made from a sensor at `position` with attitude `attitude` (sensor -> world); `others` are the bodies of
    // other vehicles the sensor can see (kPenetrating samples them as spheres at surface_spacing). Throws
    // std::invalid_argument for a non-finite position, an invalid body or a zero attitude quaternion; such a call does
    // not consume a scan index.

    // The points of one scan, delivered to `sink`: sink.reserve(n) announces an upper bound of the points to come once,
    // then sink.push(point, vehicle_id) appends one (vehicle_id is -1 for static geometry).
    template <class Sink>
    void scanTo(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                const std::vector<VehicleBody>& others, Sink& sink) {
        const Eigen::Matrix3d rotation = beginScan(position, attitude, others);
        const std::uint64_t index = scan_index_++;
        if (config_.mode == SensorConfig::kPenetrating) {
            samplePenetrating(position, rotation, index, others, sink);
            return;
        }
        sink.reserve(pattern_->size());
        traceBeams(position, rotation, index, others,
                   [&](const Eigen::Vector3d& direction, bool hit, double range, int vehicle) {
                       if (hit) {
                           sink.push(position + range * direction, vehicle);
                       }
                   });
    }

    // The points of one scan written straight into `data` as the records of a sensor_msgs/PointCloud2: float32 x y z
    // in host byte order, then int32 vehicle_id (-1 static geometry) when `with_id` (point_step 12 or 16). `data` is
    // resized to exactly the records; its capacity is kept, so a buffer reused across scans does not reallocate.
    // Returns the number of points.
    std::size_t scanInto(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                         const std::vector<VehicleBody>& others, bool with_id, std::vector<std::uint8_t>* data) {
        if (data == nullptr) {
            throw std::invalid_argument("null cloud buffer");
        }
        detail::PackedCloudSink sink(data, with_id);
        scanTo(position, attitude, others, sink);
        return sink.finish();
    }

    // The points of one scan with their sources, replacing the contents of `points` (its capacity is kept).
    void scanTaggedInto(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                        const std::vector<VehicleBody>& others, std::vector<TaggedPoint>* points) {
        if (points == nullptr) {
            throw std::invalid_argument("null point buffer");
        }
        detail::TaggedPointSink sink(points);
        scanTo(position, attitude, others, sink);
    }

    // Per-beam records of one kRaycast / kDepthFrustum scan, replacing the contents of `out`. The hits of this call
    // are exactly the points scanTo() delivers for the same scan index. Throws std::logic_error in kPenetrating, which
    // has no beams.
    void scanBeamsInto(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                       const std::vector<VehicleBody>& others, BeamScan* out) {
        if (out == nullptr) {
            throw std::invalid_argument("null beam buffer");
        }
        if (config_.mode == SensorConfig::kPenetrating) {
            throw std::logic_error("scanBeamsInto: kPenetrating has no beams");
        }
        const Eigen::Matrix3d rotation = beginScan(position, attitude, others);
        const std::uint64_t index = scan_index_++;
        out->clear();
        out->origin = position;
        out->direction.reserve(pattern_->size());
        out->range.reserve(pattern_->size());
        out->vehicle_id.reserve(pattern_->size());
        out->hit.reserve(pattern_->size());
        traceBeams(position, rotation, index, others,
                   [&](const Eigen::Vector3d& direction, bool hit, double range, int vehicle) {
                       out->direction.push_back(direction);
                       out->range.push_back(range);
                       out->vehicle_id.push_back(vehicle);
                       out->hit.push_back(hit ? 1 : 0);
                   });
    }

    // Allocating conveniences of the scans above.
    std::vector<Eigen::Vector3d> scan(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude) {
        std::vector<TaggedPoint> tagged;
        scanTaggedInto(position, attitude, {}, &tagged);
        std::vector<Eigen::Vector3d> points;
        points.reserve(tagged.size());
        std::transform(tagged.begin(), tagged.end(), std::back_inserter(points), [](const TaggedPoint& p) {
            return p.point;
        });
        return points;
    }

    std::vector<TaggedPoint> scanTagged(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                                        const std::vector<VehicleBody>& others = {}) {
        std::vector<TaggedPoint> points;
        scanTaggedInto(position, attitude, others, &points);
        return points;
    }

    BeamScan scanWithBeams(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                           const std::vector<VehicleBody>& others = {}) {
        BeamScan beams;
        scanBeamsInto(position, attitude, others, &beams);
        return beams;
    }

    // ---- scene queries ----------------------------------------------------------------------------------------
    // Distance along `direction` to the first surface within max_range, or +infinity. A ray starting inside a solid
    // returns its exit surface.
    double castRay(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction, double max_range) const {
        const double n = direction.norm();
        if (!(n > 0.0) || !origin.allFinite()) {
            return detail::kInf;
        }
        return scene_->closest(origin, direction / n, max_range);
    }

    // Surface samples of the whole scene at `spacing`, without samples buried inside another obstacle unless
    // penetrating_keep_buried (the latched global map). No range/FOV filter.
    std::vector<Eigen::Vector3d> globalMap(double spacing) const {
        if (spacing == scene_->spacing() && config_.penetrating_keep_buried == scene_->keepBuried()) {
            return scene_->samples();
        }
        return scene_->resampled(spacing, config_.penetrating_keep_buried);
    }

    // Scene residual: min over obstacles of a per-obstacle function that is 0 on the surface, < 0 inside and > 0
    // outside (|value| <= distance to that surface). Also a cheap occupancy query.
    double surfaceResidual(const Eigen::Vector3d& point) const { return scene_->surfaceResidual(point); }

  private:
    static int bodySampleCount(double radius, double spacing) { return detail::sphereSampleCount(radius, spacing); }

    // Validates a scan request and returns the unit rotation matrix of the attitude.
    static Eigen::Matrix3d beginScan(const Eigen::Vector3d& position, const Eigen::Quaterniond& attitude,
                                     const std::vector<VehicleBody>& others) {
        if (!position.allFinite()) {
            throw std::invalid_argument("non-finite sensor position");
        }
        for (const auto& b : others) {
            if (!b.position.allFinite() || !std::isfinite(b.radius) || b.radius <= 0.0) {
                throw std::invalid_argument("invalid vehicle body");
            }
        }
        return detail::rotationOf(attitude);
    }

    // Casts every beam of the pattern. sink(direction, hit, range, vehicle) receives them in beam order.
    template <class BeamSink>
    void traceBeams(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation, std::uint64_t scan_index,
                    const std::vector<VehicleBody>& others, BeamSink&& sink) {
        const double range = config_.range;
        const bool noisy = config_.noise_std > 0.0;
        if (noisy) {
            noise_.reseed(config_.seed, scan_index, config_.noise_std);
        }
        near_.clear(); // bodies that can be reached within range
        for (const auto& b : others) {
            if ((b.position - position).norm() - b.radius <= range) {
                near_.push_back(&b);
            }
        }
        const LidarScene& map = *scene_;
        for (const Eigen::Vector3d& local : pattern_->directions()) {
            const Eigen::Vector3d d = rotation * local;
            double t = map.closest(position, d, range);
            int vehicle = -1;
            for (const VehicleBody* b : near_) {
                double t0, t1;
                if (!detail::sphereInterval(position, d, b->position, b->radius, &t0, &t1)) {
                    continue;
                }
                const double tb = t0 >= 0.0 ? t0 : t1;
                if (tb >= 0.0 && tb <= range && tb < t) {
                    t = tb;
                    vehicle = b->id;
                }
            }
            bool hit = t != detail::kInf;
            if (hit && noisy) {
                t += noise_();
            }
            if (hit && ((noisy && t <= 0.0) || t > range || t < config_.min_range)) {
                hit = false;
            }
            if (!hit) {
                vehicle = -1;
            }
            sink(d, hit, hit ? t : range, vehicle);
        }
    }

    // kPenetrating: every sample of the scene in scene order, then every vehicle-body sample, kept when it lies in
    // (0, range], at or beyond min_range, inside the field of view and the heading crop, with seeded range noise drawn
    // per kept sample in that order.
    template <class Sink>
    void samplePenetrating(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation, std::uint64_t scan_index,
                           const std::vector<VehicleBody>& others, Sink& sink) {
        const double range = config_.range;
        const double r2 = range * range;
        const double rmin2 = config_.min_range * config_.min_range;
        const bool noisy = config_.noise_std > 0.0;
        if (noisy) {
            noise_.reseed(config_.seed, scan_index, config_.noise_std);
        }
        const Eigen::Matrix3d to_sensor = rotation.transpose();
        const Eigen::Vector3d body_x = rotation.col(0);
        const bool limited = field_of_view_.limited();
        const auto consider = [&](const Eigen::Vector3d& p, int vehicle) {
            const Eigen::Vector3d v = p - position;
            const double d2 = v.squaredNorm();
            if (d2 > r2 || d2 == 0.0 || d2 < rmin2) {
                return;
            }
            if (limited && !field_of_view_.contains(to_sensor * v)) {
                return;
            }
            if (!crop_.contains(v, body_x)) {
                return;
            }
            if (!noisy) {
                sink.push(p, vehicle);
                return;
            }
            const double dist = std::sqrt(d2);
            const double t = dist + noise_();
            if (t <= 0.0 || t > range || t < config_.min_range) {
                return;
            }
            sink.push(Eigen::Vector3d(position + v * (t / dist)), vehicle);
        };
        const LidarScene& map = *scene_;
        bits_.assign((map.sampleCount() + 63) / 64, 0);
        std::size_t bound = map.mark(position, r2, bits_.data());
        body_samples_.assign(others.size(), 0);
        for (std::size_t i = 0; i < others.size(); ++i) {
            const VehicleBody& b = others[i];
            if ((b.position - position).norm() - b.radius > range) {
                continue;
            }
            body_samples_[i] = bodySampleCount(b.radius, config_.surface_spacing);
            bound += static_cast<std::size_t>(body_samples_[i]);
        }
        sink.reserve(bound);
        // Ascending scene index: the order of the scene samples.
        const std::vector<Eigen::Vector3d>& samples = map.samples();
        for (std::size_t w = 0; w < bits_.size(); ++w) {
            for (std::uint64_t word = bits_[w]; word != 0; word &= word - 1) {
                consider(samples[64 * w + static_cast<std::size_t>(__builtin_ctzll(word))], -1);
            }
        }
        for (std::size_t i = 0; i < others.size(); ++i) {
            const VehicleBody& b = others[i];
            const int n = body_samples_[i];
            for (int k = 0; k < n; ++k) {
                consider(b.position + b.radius * detail::fibonacciPoint(k, n), b.id);
            }
        }
    }

    SensorConfig config_;
    std::shared_ptr<const BeamPattern> pattern_;
    std::shared_ptr<const LidarScene> scene_;
    FieldOfView field_of_view_{360.0, 180.0};
    HeadingCrop crop_;
    std::uint64_t scan_index_ = 0;
    // Scratch space of a scan, reused by the next one.
    RangeNoise noise_;
    std::vector<std::uint64_t> bits_;
    std::vector<int> body_samples_;
    std::vector<const VehicleBody*> near_;
};

} // namespace xgc2_math::scene
