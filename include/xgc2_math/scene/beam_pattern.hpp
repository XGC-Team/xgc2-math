#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "xgc2_math/scene/sensor_model.hpp"
#include "xgc2_math/scene/surface_sampling.hpp"

namespace xgc2_math::scene {

// The unit beam directions of a sensor in its own frame (x forward, y left, z up), in scan order. The pattern is a
// function of the sensor model parameters only, so sensors with equal parameters share one immutable pattern.
class BeamPattern {
  public:
    // The beams of a kRaycast sensor (h_res x v_res beams over h_fov_deg x v_fov_deg, rows from the lowest elevation
    // up, a full circle without a duplicate seam beam) or of a kDepthFrustum sensor (one beam per pixel through its
    // centre, row by row from the top-left). `config` must have passed resolveSensorConfig(); kPenetrating has no
    // beams, so its pattern is empty.
    static std::shared_ptr<const BeamPattern> make(const SensorConfig& config) {
        auto pattern = std::make_shared<BeamPattern>();
        pattern->mode_ = config.mode;
        pattern->h_fov_deg_ = config.h_fov_deg;
        pattern->v_fov_deg_ = config.v_fov_deg;
        pattern->h_res_ = config.h_res;
        pattern->v_res_ = config.v_res;
        pattern->width_ = config.width;
        pattern->height_ = config.height;
        pattern->fx_ = config.fx;
        pattern->fy_ = config.fy;
        pattern->cx_ = config.cx;
        pattern->cy_ = config.cy;
        if (config.mode == SensorConfig::kDepthFrustum) {
            pattern->directions_.reserve(static_cast<std::size_t>(config.width) *
                                         static_cast<std::size_t>(config.height));
            for (int v = 0; v < config.height; ++v) {
                for (int u = 0; u < config.width; ++u) {
                    pattern->directions_.push_back(
                        Eigen::Vector3d(1.0, -(u + 0.5 - config.cx) / config.fx, -(v + 0.5 - config.cy) / config.fy)
                            .normalized());
                }
            }
        } else if (config.mode == SensorConfig::kRaycast) {
            const double h = config.h_fov_deg * detail::kPi / 180.0;
            const double v = config.v_fov_deg * detail::kPi / 180.0;
            const bool full_circle = config.h_fov_deg >= 360.0;
            pattern->directions_.reserve(static_cast<std::size_t>(config.h_res) *
                                         static_cast<std::size_t>(config.v_res));
            for (int j = 0; j < config.v_res; ++j) {
                const double el = config.v_res == 1 ? 0.0 : -0.5 * v + v * j / (config.v_res - 1);
                for (int i = 0; i < config.h_res; ++i) {
                    double az;
                    if (full_circle) {
                        az = -detail::kPi + 2.0 * detail::kPi * i / config.h_res;
                    } else {
                        az = config.h_res == 1 ? 0.0 : -0.5 * h + h * i / (config.h_res - 1);
                    }
                    pattern->directions_.emplace_back(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az),
                                                      std::sin(el));
                }
            }
        }
        return pattern;
    }

    std::size_t size() const { return directions_.size(); }
    const std::vector<Eigen::Vector3d>& directions() const { return directions_; }

    // Whether this pattern is the one `config` (resolved) describes.
    bool matches(const SensorConfig& config) const {
        if (config.mode != mode_) {
            return false;
        }
        switch (mode_) {
        case SensorConfig::kRaycast:
            return config.h_fov_deg == h_fov_deg_ && config.v_fov_deg == v_fov_deg_ && config.h_res == h_res_ &&
                   config.v_res == v_res_;
        case SensorConfig::kDepthFrustum:
            return config.width == width_ && config.height == height_ && config.fx == fx_ && config.fy == fy_ &&
                   config.cx == cx_ && config.cy == cy_;
        case SensorConfig::kPenetrating:
            return true;
        }
        return false;
    }

  private:
    SensorConfig::Mode mode_ = SensorConfig::kRaycast;
    double h_fov_deg_ = 0.0, v_fov_deg_ = 0.0;
    int h_res_ = 0, v_res_ = 0;
    int width_ = 0, height_ = 0;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
    std::vector<Eigen::Vector3d> directions_;
};

} // namespace xgc2_math::scene
