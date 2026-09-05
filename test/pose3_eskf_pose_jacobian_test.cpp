#include <cmath>
#include <cstdio>

#include <xgc2_math/estimation/pose3_inertial_eskf.hpp>

namespace xgc2_math {

struct Pose3InertialEskfTestAccess {
    static Pose3 predictedMarkerPose(const RigidBodyState& state) {
        return Pose3InertialEskf::predictedMarkerPose(state);
    }

    static Pose3InertialEskf::MeasurementVector measurementResidual(const Pose3& predicted_marker,
                                                                    const Pose3& measured_marker) {
        return Pose3InertialEskf::measurementResidual(predicted_marker, measured_marker);
    }

    static Pose3InertialEskf::MeasurementMatrix
    measurementJacobian(const Pose3InertialEskf& estimator, const Pose3& predicted_marker,
                        const Pose3InertialEskf::MeasurementVector& innovation) {
        return estimator.measurementJacobian(predicted_marker, innovation);
    }

    static void injectError(const Pose3InertialEskf& estimator, const Pose3InertialEskf::ErrorVector& delta,
                            RigidBodyState& state) {
        estimator.injectError(delta, state);
    }
};

} // namespace xgc2_math

int main() {
    using xgc2_math::Pose3;
    using xgc2_math::Pose3InertialEskf;
    using xgc2_math::Pose3InertialEskfConfig;
    using xgc2_math::Pose3InertialEskfTestAccess;
    using xgc2_math::RigidBodyState;

    Pose3InertialEskfConfig config;
    config.body_to_marker.position = Eigen::Vector3d(0.17, -0.08, 0.11);
    config.body_to_marker.orientation =
        xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.31, -0.22, 0.27));

    Pose3InertialEskf estimator;
    estimator.setConfig(config);

    RigidBodyState base;
    base.initialized = true;
    base.position = Eigen::Vector3d(1.2, -0.7, 0.9);
    base.orientation = xgc2_math::rpyToQuaternion(Eigen::Vector3d(-0.35, 0.28, 0.63));
    base.body_to_marker = config.body_to_marker;

    const Pose3 predicted = Pose3InertialEskfTestAccess::predictedMarkerPose(base);
    Pose3 delta_measurement;
    delta_measurement.position = Eigen::Vector3d(0.12, -0.09, 0.07);
    delta_measurement.orientation = xgc2_math::expMap(Eigen::Vector3d(0.45, -0.35, 0.25));
    const Pose3 measured = xgc2_math::compose(predicted, delta_measurement);

    const Pose3InertialEskf::MeasurementVector innovation =
        Pose3InertialEskfTestAccess::measurementResidual(predicted, measured);
    const Pose3InertialEskf::MeasurementMatrix analytic =
        Pose3InertialEskfTestAccess::measurementJacobian(estimator, predicted, innovation);

    Eigen::Matrix3d finite_difference = Eigen::Matrix3d::Zero();
    constexpr double epsilon = 1.0e-7;
    for (int axis = 0; axis < 3; ++axis) {
        Pose3InertialEskf::ErrorVector plus_delta = Pose3InertialEskf::ErrorVector::Zero();
        Pose3InertialEskf::ErrorVector minus_delta = Pose3InertialEskf::ErrorVector::Zero();
        plus_delta(6 + axis) = epsilon;
        minus_delta(6 + axis) = -epsilon;

        RigidBodyState plus = base;
        RigidBodyState minus = base;
        Pose3InertialEskfTestAccess::injectError(estimator, plus_delta, plus);
        Pose3InertialEskfTestAccess::injectError(estimator, minus_delta, minus);

        const Eigen::Vector3d plus_residual =
            Pose3InertialEskfTestAccess::measurementResidual(
                Pose3InertialEskfTestAccess::predictedMarkerPose(plus), measured)
                .tail<3>();
        const Eigen::Vector3d minus_residual =
            Pose3InertialEskfTestAccess::measurementResidual(
                Pose3InertialEskfTestAccess::predictedMarkerPose(minus), measured)
                .tail<3>();
        finite_difference.col(axis) = (plus_residual - minus_residual) / (2.0 * epsilon);
    }

    const Eigen::Matrix3d analytic_orientation = analytic.block<3, 3>(3, 6);
    const double max_error = (analytic_orientation - finite_difference).cwiseAbs().maxCoeff();

    // Ensure this fixture actually distinguishes the full SO(3) derivative from
    // the old small-residual approximation -R_bm^T.
    const Eigen::Matrix3d small_angle_approximation =
        -xgc2_math::normalizedQuaternion(config.body_to_marker.orientation).toRotationMatrix().transpose();
    const double approximation_gap = (small_angle_approximation - finite_difference).norm();
    if (!(approximation_gap > 1.0e-2)) {
        std::fprintf(stderr, "pose Jacobian fixture is not discriminating: approximation_gap=%.12g\n",
                     approximation_gap);
        return 2;
    }

    if (!(max_error < 2.0e-6)) {
        std::fprintf(stderr,
                     "pose orientation Jacobian mismatch: max_error=%.12g approximation_gap=%.12g\n",
                     max_error, approximation_gap);
        std::fprintf(stderr, "analytic:\n%s\n", "see matrix values in debugger/CI artifact");
        return 1;
    }

    return 0;
}
