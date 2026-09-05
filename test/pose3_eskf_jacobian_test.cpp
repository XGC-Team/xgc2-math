#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include "xgc2_math/estimation/pose3_inertial_eskf.hpp"

// Separate executable: do not link this access shim with math_header_test.cpp.
namespace xgc2_math {
struct Pose3InertialEskfTestAccess {
    static void setState(Pose3InertialEskf& filter, const RigidBodyState& state) { filter.state_ = state; }
    static Pose3 predicted(const RigidBodyState& state) { return Pose3InertialEskf::predictedMarkerPose(state); }
    static Pose3InertialEskf::MeasurementVector residual(const RigidBodyState& state, const Pose3& measured) {
        return Pose3InertialEskf::measurementResidual(predicted(state), measured);
    }
    static Pose3InertialEskf::MeasurementMatrix jacobian(const Pose3InertialEskf& filter, const Pose3& measured) {
        const auto prediction = predicted(filter.state_);
        return filter.measurementJacobian(prediction, Pose3InertialEskf::measurementResidual(prediction, measured));
    }
    static RigidBodyState inject(const Pose3InertialEskf& filter, const Pose3InertialEskf::ErrorVector& error) {
        RigidBodyState result = filter.state_;
        filter.injectError(error, result);
        return result;
    }
};
} // namespace xgc2_math

namespace {
using Filter = xgc2_math::Pose3InertialEskf;
using Access = xgc2_math::Pose3InertialEskfTestAccess;
using Vector = Eigen::Vector3d;
constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Vector randomVector(std::mt19937& generator) {
    // Specify draw order and integer conversion so the fixtures are reproducible.
    const double x = 2.0 * static_cast<double>(generator()) / 4294967295.0 - 1.0;
    const double y = 2.0 * static_cast<double>(generator()) / 4294967295.0 - 1.0;
    const double z = 2.0 * static_cast<double>(generator()) / 4294967295.0 - 1.0;
    return Vector(x, y, z);
}

bool testFullResidualJacobian() {
    std::mt19937 generator(20260905u);
    bool passed = true;
    std::size_t comparisons = 0;
    double worst = 0.0;
    for (const double angle : {0.0, 1.0e-9, 1.0e-6, 0.000999, 0.001001, 0.05, 0.3, 0.8, 1.5, 2.6, kPi - 1.0e-4}) {
        double angle_worst = 0.0;
        for (int trial = 0; trial < 32; ++trial) {
            Filter filter;
            xgc2_math::RigidBodyState state;
            state.initialized = true;
            state.position = randomVector(generator);
            state.orientation = xgc2_math::expMap(2.0 * randomVector(generator));
            state.body_to_marker.position = 0.2 * randomVector(generator);
            state.body_to_marker.orientation = xgc2_math::expMap(randomVector(generator));
            Access::setState(filter, state);
            auto measured = Access::predicted(state);
            measured.position += 0.3 * randomVector(generator);
            measured.orientation = xgc2_math::normalizedQuaternion(
                measured.orientation * xgc2_math::expMap(angle * randomVector(generator).normalized()));
            const auto analytical = Access::jacobian(filter, measured);
            require(analytical.allFinite(), "non-finite analytical Jacobian");
            for (const double step : {1.0e-5, 1.0e-6, 1.0e-7}) {
                Filter::MeasurementMatrix numerical;
                for (int column = 0; column < Filter::kErrorStateDim; ++column) {
                    Filter::ErrorVector delta = Filter::ErrorVector::Zero();
                    delta(column) = step;
                    const auto plus = Access::residual(Access::inject(filter, delta), measured);
                    const auto minus = Access::residual(Access::inject(filter, -delta), measured);
                    numerical.col(column) = (plus - minus) / (2.0 * step);
                }
                require(numerical.allFinite(), "non-finite numerical Jacobian");
                const double error = (analytical - numerical).cwiseAbs().maxCoeff();
                angle_worst = std::max(angle_worst, error);
                passed = passed && error < 1.0e-6;
                ++comparisons;
            }
            // A quaternion sign change is not a new rotation or a new residual.
            auto negated_measurement = measured;
            negated_measurement.orientation.coeffs() *= -1.0;
            require((Access::residual(state, measured) - Access::residual(state, negated_measurement)).norm() < 1.0e-12,
                    "measurement quaternion sign changed residual");
            require((analytical - Access::jacobian(filter, negated_measurement)).norm() < 1.0e-12,
                    "measurement quaternion sign changed Jacobian");
            state.orientation.coeffs() *= -1.0;
            Access::setState(filter, state);
            require((analytical - Access::jacobian(filter, measured)).norm() < 1.0e-12,
                    "state quaternion sign changed Jacobian");
        }
        worst = std::max(worst, angle_worst);
        std::cout << "angle_rad=" << angle << " max_abs_error=" << angle_worst << '\n';
    }
    std::cout << "full_6x15_comparisons=" << comparisons << " worst=" << worst << '\n';
    return passed;
}

xgc2_math::PoseMeasurement poseSample(double stamp, const xgc2_math::Pose3& pose) {
    xgc2_math::PoseMeasurement sample;
    sample.received = true;
    sample.valid = true;
    sample.stamp_sec = stamp;
    sample.pose = pose;
    return sample;
}

void requireCovariance(const Filter& filter) {
    const auto& covariance = filter.covariance();
    require(covariance.allFinite(), "non-finite covariance");
    require((covariance - covariance.transpose()).norm() < 1.0e-10, "asymmetric covariance");
    Eigen::SelfAdjointEigenSolver<Filter::ErrorCovariance> solver(covariance);
    require(solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -1.0e-10,
            "non-positive-semidefinite covariance");
}

void testPublicUpdateAndReplay() {
    for (const int iterations : {1, 3, 8}) {
        for (const bool floor : {false, true}) {
            xgc2_math::Pose3InertialEskfConfig config;
            config.pose_update_iterations = iterations;
            config.apply_pose_covariance_floor = floor;
            config.body_to_marker.position = Vector(0.12, -0.08, 0.05);
            config.body_to_marker.orientation = xgc2_math::expMap(Vector(0.15, -0.2, 0.1));
            Filter filter;
            filter.setConfig(config);
            const xgc2_math::Pose3 initial = config.body_to_marker;
            filter.initializeFromPose(poseSample(1.0, initial));
            auto measured = initial;
            measured.position += Vector(0.08, -0.03, 0.02);
            measured.orientation =
                xgc2_math::normalizedQuaternion(measured.orientation * xgc2_math::expMap(Vector(0.3, -0.2, 0.1)));
            const double before = Access::residual(filter.state(), measured).norm();
            require(filter.updatePose(poseSample(1.01, measured)).accepted, "moderate pose update rejected");
            require(Access::residual(filter.state(), measured).norm() < before, "pose update did not reduce residual");
            requireCovariance(filter);
            require(std::abs(filter.state().orientation.norm() - 1.0) < 1.0e-12, "non-unit quaternion");
            auto jumped = measured;
            jumped.position.x() += 10.0;
            const auto rejection = filter.updatePose(poseSample(1.02, jumped));
            require(!rejection.accepted && rejection.innovation_rejected, "raw jump gate changed");
        }
    }
    Filter filter;
    xgc2_math::InertialSample imu;
    imu.received = true;
    imu.valid = true;
    imu.stamp_sec = 1.0;
    imu.linear_acceleration = Vector(0.0, 0.0, 9.8066);
    filter.initializeFromPose(poseSample(1.0, xgc2_math::Pose3{}), &imu);
    for (int i = 1; i <= 20; ++i) {
        imu.stamp_sec = 1.0 + 0.01 * i;
        filter.propagateInertial(imu);
    }
    xgc2_math::Pose3 measured;
    measured.position = Vector(0.02, 0.0, 0.0);
    measured.orientation = xgc2_math::expMap(Vector(0.0, 0.0, 0.2));
    require(filter.updatePose(poseSample(1.1, measured)).accepted, "delayed pose rejected");
    require(std::abs(filter.state().last_inertial_stamp_sec - 1.2) < 1.0e-12, "replay lost IMU endpoint");
    requireCovariance(filter);
    measured.position.x() = std::numeric_limits<double>::quiet_NaN();
    require(!filter.updatePose(poseSample(1.21, measured)).accepted, "invalid pose accepted");
    std::cout << "public_update_floor_gate_and_replay=PASS\n";
}
} // namespace

int main() {
    std::cout << std::setprecision(12);
    try {
        const bool jacobian_passed = testFullResidualJacobian();
        testPublicUpdateAndReplay();
        require(jacobian_passed, "full pose Jacobian differs from central finite differences");
        std::cout << "pose3_eskf_jacobian_test=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pose3_eskf_jacobian_test=FAIL: " << error.what() << '\n';
        return 1;
    }
}
