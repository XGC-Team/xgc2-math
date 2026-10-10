#pragma once

#include "xgc2_math/trajectory/analytic/3d/torus_knot_3d.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>

namespace xgc2_math::trajectory {

struct TorusKnotEntryCurveParameters3 {
    uint32_t flags{kFlagNone};
    double duration{35.0};
    Eigen::Vector3d start{Eigen::Vector3d::Zero()};
    double origin_yaw{0.0};
    double entry_duration{5.0};
    TorusKnotCurveParameters3 torus{};
};

inline TorusKnotCurveParameters3 torusKnot3dParametersWithYaw(const TorusKnotCurveParameters3& torus, double yaw) {
    TorusKnotCurveParameters3 params = torus;
    params.yaw = yaw;
    return params;
}

class TorusKnotEntryCurveEvaluator3 final : public TrajectoryEvaluator3 {
  public:
    explicit TorusKnotEntryCurveEvaluator3(const TorusKnotEntryCurveParameters3& params = {})
        : params_(params), torus_(torusKnot3dParametersWithYaw(params_.torus, params_.origin_yaw)) {
        // Match circle-entry's yaw-hold contract across the entire trajectory.
        params_.torus.yaw = params_.origin_yaw;
        params_.entry_duration = std::max(0.0, params_.entry_duration);
        if (!analytic_detail::finiteScalar(params_.duration) || params_.duration <= 0.0) {
            params_.duration = torus_.duration() + params_.entry_duration;
        }
    }

    bool evaluate(double t, FlatOutput3& output) const override {
        if (!analytic_detail::finiteScalar(t)) {
            output.flags |= kFlagInvalidInput;
            return false;
        }
        t = analytic_detail::clamp(t, 0.0, params_.duration);
        const double entry = std::max(analytic_detail::kMinDuration, params_.entry_duration);
        output = FlatOutput3{};
        if (t >= entry) {
            const bool ok = torus_.evaluate(t - entry, output);
            output.flags |= params_.flags;
            return ok && TrajectoryValidator3::finite(output);
        }

        FlatOutput3 end;
        torus_.evaluate(0.0, end);
        const auto cx = analytic_detail::septicBoundary(params_.start.x(), 0.0, 0.0, 0.0, end.position.x(),
                                                        end.velocity.x(), end.acceleration.x(), end.jerk.x(), entry);
        const auto cy = analytic_detail::septicBoundary(params_.start.y(), 0.0, 0.0, 0.0, end.position.y(),
                                                        end.velocity.y(), end.acceleration.y(), end.jerk.y(), entry);
        const auto cz = analytic_detail::septicBoundary(params_.start.z(), 0.0, 0.0, 0.0, end.position.z(),
                                                        end.velocity.z(), end.acceleration.z(), end.jerk.z(), entry);
        analytic_detail::evalSeptic(cx, t, output.position.x(), output.velocity.x(), output.acceleration.x(),
                                    output.jerk.x(), output.snap.x());
        analytic_detail::evalSeptic(cy, t, output.position.y(), output.velocity.y(), output.acceleration.y(),
                                    output.jerk.y(), output.snap.y());
        analytic_detail::evalSeptic(cz, t, output.position.z(), output.velocity.z(), output.acceleration.z(),
                                    output.jerk.z(), output.snap.z());
        analytic_detail::fillYawHold(output, params_.origin_yaw);
        output.flags |= params_.flags;
        return TrajectoryValidator3::finite(output);
    }

    double duration() const override { return params_.duration; }
    TrajectoryModelType type() const override { return TrajectoryModelType::kAnalytic; }
    uint32_t flags() const override { return params_.flags | torus_.flags(); }
    const TorusKnotEntryCurveParameters3& params() const { return params_; }

  private:
    TorusKnotEntryCurveParameters3 params_;
    TorusKnotCurveEvaluator3 torus_;
};

} // namespace xgc2_math::trajectory
