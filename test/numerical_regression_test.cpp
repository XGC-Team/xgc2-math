#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "xgc2_math/algebra/angle.hpp"
#include "xgc2_math/control/se3_nmpc_problem.hpp"
#include "xgc2_math/estimation/recursive_least_squares.hpp"
#include "xgc2_math/geometry/se3.hpp"
#include "xgc2_math/trajectory/analytic/detail.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, double tolerance, const char* message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

void rlsInformativeSample() {
    xgc2_math::ScalarRecursiveLeastSquares estimator;
    estimator.reset(0.0);
    const auto sample = estimator.update(2.0e8, 1.0e8);
    require(sample.measurement_accepted, "finite informative sample must be accepted");
    requireNear(sample.parameter, 2.0, 1.0e-14, "incorrect RLS parameter");
    requireNear(sample.covariance, estimator.options().min_covariance, 1.0e-20,
                "informative sample must not reset covariance to its initial value");
}

void rlsCovarianceBounds() {
    xgc2_math::ScalarRecursiveLeastSquaresOptions options;
    options.initial_covariance = 1.0e12;
    options.max_covariance = 1.0e9;
    xgc2_math::ScalarRecursiveLeastSquares estimator(options);
    requireNear(estimator.covariance(), options.max_covariance, 0.0, "constructor must respect covariance bounds");
    estimator.reset(0.0, std::numeric_limits<double>::quiet_NaN());
    requireNear(estimator.covariance(), options.max_covariance, 0.0, "reset fallback must respect covariance bounds");
    options.initial_covariance = 1.0;
    options.min_covariance = 10.0;
    estimator.setOptions(options);
    estimator.reset();
    requireNear(estimator.covariance(), options.min_covariance, 0.0, "initial covariance must respect lower bound");
}

void rlsUnderflowAndInvalidInput() {
    xgc2_math::ScalarRecursiveLeastSquaresOptions options;
    options.initial_covariance = 1.0e-200;
    options.min_covariance = 1.0e-250;
    xgc2_math::ScalarRecursiveLeastSquares estimator(options);
    estimator.reset(0.0);
    const auto sample = estimator.update(1.0e200, 1.0e200);
    require(sample.measurement_accepted, "representable parameter update must be accepted");
    requireNear(sample.parameter, 1.0, 1.0e-14, "incorrect parameter for underflow case");
    require(sample.covariance == options.min_covariance, "covariance underflow must saturate at lower bound");
    const auto rejected = estimator.update(std::numeric_limits<double>::infinity(), 1.0);
    require(!rejected.measurement_accepted, "nonfinite measurement must be rejected");
    require(rejected.parameter == sample.parameter && rejected.covariance == sample.covariance,
            "rejected measurement must not mutate the estimate");
}

void rlsInformationFormReference() {
    xgc2_math::ScalarRecursiveLeastSquaresOptions options;
    options.forgetting_factor = 0.997;
    xgc2_math::ScalarRecursiveLeastSquares estimator(options);
    estimator.reset(0.0);
    long double precision = 1.0L / static_cast<long double>(options.initial_covariance);
    long double information = 0.0L;
    for (int i = 0; i < 400; ++i) {
        const double phi = 0.2 + std::sin(static_cast<double>(i) * 0.17);
        const double measurement = 3.5 * phi + 0.03 * std::cos(static_cast<double>(i) * 0.37);
        precision = static_cast<long double>(options.forgetting_factor) * precision +
                    static_cast<long double>(phi) * phi;
        information = static_cast<long double>(options.forgetting_factor) * information +
                      static_cast<long double>(phi) * measurement;
        const auto sample = estimator.update(measurement, phi);
        require(sample.measurement_accepted, "ordinary RLS sample was rejected");
        requireNear(sample.parameter, static_cast<double>(information / precision), 2.0e-12,
                    "parameter disagrees with independent information-form recurrence");
        requireNear(sample.covariance, static_cast<double>(1.0L / precision), 2.0e-12,
                    "covariance disagrees with independent information-form recurrence");
    }
}

void scaledQuaternionNormalization() {
    const Eigen::Quaterniond reference(Eigen::AngleAxisd(1.3, Eigen::Vector3d(1.0, -2.0, 3.0).normalized()));
    for (double scale : {1.0e-10, 1.0, 1.0e100, 1.0e200, 1.0e300, -1.0e300}) {
        Eigen::Quaterniond input = reference;
        input.coeffs() *= scale;
        const Eigen::Quaterniond normalized = xgc2_math::normalizedQuaternion(input);
        requireNear(normalized.norm(), 1.0, 5.0e-15, "normalized quaternion must have unit norm");
        require((normalized.toRotationMatrix() - reference.toRotationMatrix()).norm() < 1.0e-14,
                "finite rescaling must preserve the represented rotation");
        require(normalized.w() >= 0.0, "quaternion sign convention changed");
    }
    Eigen::Quaterniond largest(1.0, 1.0, 1.0, 1.0);
    largest.coeffs() *= std::numeric_limits<double>::max();
    requireNear(xgc2_math::normalizedQuaternion(largest).norm(), 1.0, 5.0e-15,
                "normalization must work even when the unscaled norm is unrepresentable");
}

void quaternionFallbackConvention() {
    for (const Eigen::Quaterniond& input :
         {Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0), Eigen::Quaterniond(1.0e-13, 0.0, 0.0, 0.0),
          Eigen::Quaterniond(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0)}) {
        require(xgc2_math::normalizedQuaternion(input).coeffs() == Eigen::Quaterniond::Identity().coeffs(),
                "invalid and near-zero quaternions must still fall back to identity");
    }
}

void rotationFrameSingularities() {
    for (double yaw : {0.0, 0.3, xgc2_math::kPi / 2.0, xgc2_math::kPi, -xgc2_math::kPi / 2.0}) {
        for (double sign : {-1.0, 1.0}) {
            const Eigen::Vector3d body_z = sign * Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0.0);
            const Eigen::Matrix3d rotation = xgc2_math::control::rotationFromBodyZAndYaw(body_z, yaw);
            require((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1.0e-14,
                    "singular heading fallback must return an orthonormal frame");
            requireNear(rotation.determinant(), 1.0, 1.0e-14, "heading fallback must be right handed");
            require((rotation.col(2) - body_z).norm() < 1.0e-14, "heading fallback must preserve body Z");
        }
    }
}

void septicNormalizedBoundaryConditions() {
    for (double duration : {1.0e-6, 1.0e-4, 0.01, 0.1, 1.0, 30.0, 1000.0, 1.0e6}) {
        const double t2 = duration * duration;
        const double t3 = t2 * duration;
        const auto coefficients = xgc2_math::trajectory::trajectory2_detail::septicBoundary(
            0.2, 0.3 / duration, -0.15 / t2, 0.27 / t3, 1.2, -0.6 / duration, 0.4 / t2, -0.2 / t3, duration);
        const std::array<double, 4> start{{0.2, 0.3, -0.15, 0.27}};
        const std::array<double, 4> end{{1.2, -0.6, 0.4, -0.2}};
        double time_scale = 1.0;
        for (int derivative = 0; derivative < 4; ++derivative) {
            const auto index = static_cast<std::size_t>(derivative);
            requireNear(xgc2_math::trajectory::trajectory2_detail::polyValue(coefficients, 0.0, derivative) *
                            time_scale,
                        start[index], 1.0e-9, "septic start boundary condition violated");
            requireNear(xgc2_math::trajectory::trajectory2_detail::polyValue(coefficients, duration, derivative) *
                            time_scale,
                        end[index], 1.0e-9, "septic end boundary condition violated");
            time_scale *= duration;
        }
    }
}

void analyticSepticBoundaryConditions() {
    for (double duration : {1.0e-6, 1.0e-4, 0.1, 1.0, 1000.0, 1.0e6}) {
        const auto coefficients = xgc2_math::trajectory::analytic_detail::septicBoundary(
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, duration);
        requireNear(xgc2_math::trajectory::analytic_detail::polyValue(coefficients, duration, 0), 1.0, 1.0e-11,
                    "analytic entry polynomial did not reach its endpoint");
    }
}

xgc2_math::trajectory::FlatOutput3 movingThrust(double t, bool singular) {
    xgc2_math::trajectory::FlatOutput3 flat;
    const Eigen::Vector3d acceleration =
        singular ? Eigen::Vector3d(0.0, 1.0, -9.8066) : Eigen::Vector3d(0.7, -0.4, -0.5);
    const Eigen::Vector3d jerk = singular ? Eigen::Vector3d::UnitZ().eval() : Eigen::Vector3d(0.3, 0.8, -0.2);
    const Eigen::Vector3d snap = singular ? Eigen::Vector3d(0.0, 0.0, 0.3) : Eigen::Vector3d(0.6, 0.2, 0.1);
    flat.acceleration = acceleration + jerk * t + 0.5 * snap * t * t;
    flat.jerk = jerk + snap * t;
    flat.snap = snap;
    flat.yaw = singular ? xgc2_math::kPi / 2.0 : 0.25 + 0.4 * t - 0.1 * t * t;
    flat.yaw_rate = singular ? 0.0 : 0.4 - 0.2 * t;
    flat.yaw_accel = singular ? 0.0 : -0.2;
    return flat;
}

void checkFlatnessDerivatives(bool singular) {
    const xgc2_math::trajectory::FlatnessMapper3 mapper;
    const double step = singular ? 2.0e-8 : 1.0e-5;
    const double tolerance = singular ? 1.0e-6 : 1.0e-8;
    const auto center = mapper.map(movingThrust(0.0, singular));
    const auto plus = mapper.map(movingThrust(step, singular));
    const auto minus = mapper.map(movingThrust(-step, singular));
    const Eigen::Matrix3d rotation = center.attitude.toRotationMatrix();
    const Eigen::Matrix3d rotation_dot =
        (plus.attitude.toRotationMatrix() - minus.attitude.toRotationMatrix()) / (2.0 * step);
    const Eigen::Matrix3d omega_hat = rotation.transpose() * rotation_dot;
    const Eigen::Vector3d numerical_rate(omega_hat(2, 1), omega_hat(0, 2), omega_hat(1, 0));
    require((center.body_rate - numerical_rate).norm() < tolerance,
            "body rate must differentiate the actual returned attitude");
    const Eigen::Vector3d numerical_acceleration = (plus.body_rate - minus.body_rate) / (2.0 * step);
    require((center.angular_acceleration - numerical_acceleration).norm() < tolerance,
            "angular acceleration must differentiate the returned body rate");
    if (singular) {
        require((center.flags & xgc2_math::trajectory::kFlagYawSingularity) != 0U, "singularity must remain flagged");
        requireNear(center.body_rate.x(), 1.0, 1.0e-14, "fallback body rate must not be halved");
        requireNear(center.angular_acceleration.x(), 0.3, 1.0e-14, "fallback angular acceleration is incorrect");
    }
}

void dynamicPenaltyGradient() {
    using namespace xgc2_math::trajectory;
    WaypointProblem3 problem;
    problem.dynamic_penalty_weight = 3.0;
    problem.limits.max_velocity = 1.0;
    Eigen::VectorXd times = Eigen::VectorXd::Constant(1, 1.3);
    Eigen::MatrixX3d coefficients = Eigen::MatrixX3d::Zero(6, 3);
    coefficients.row(1) << 2.0, 0.2, 0.1;
    coefficients.row(2) << 0.2, -0.1, 0.03;
    coefficients.row(3) << 0.03, 0.02, -0.01;
    const auto penalty = [&](const Eigen::VectorXd& t, const Eigen::MatrixX3d& c) {
        double value = 0.0;
        Eigen::VectorXd time_gradient = Eigen::VectorXd::Zero(t.size());
        Eigen::MatrixX3d coefficient_gradient = Eigen::MatrixX3d::Zero(c.rows(), 3);
        trajectory3_detail::dynamicPenalty(t, c, problem, value, time_gradient, coefficient_gradient);
        return value;
    };
    double cost = 0.0;
    Eigen::VectorXd time_gradient = Eigen::VectorXd::Zero(1);
    Eigen::MatrixX3d coefficient_gradient = Eigen::MatrixX3d::Zero(6, 3);
    trajectory3_detail::dynamicPenalty(times, coefficients, problem, cost, time_gradient, coefficient_gradient);
    const double step = 1.0e-6;
    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 3; ++col) {
            Eigen::MatrixX3d plus = coefficients;
            Eigen::MatrixX3d minus = coefficients;
            plus(row, col) += step;
            minus(row, col) -= step;
            const double numerical = (penalty(times, plus) - penalty(times, minus)) / (2.0 * step);
            requireNear(coefficient_gradient(row, col), numerical, 2.0e-7,
                        "dynamic penalty coefficient gradient disagrees with its cost");
        }
    }
    Eigen::VectorXd plus = times;
    Eigen::VectorXd minus = times;
    plus(0) += step;
    minus(0) -= step;
    const double numerical = (penalty(plus, coefficients) - penalty(minus, coefficients)) / (2.0 * step);
    requireNear(time_gradient(0), numerical, 2.0e-7, "dynamic penalty time gradient disagrees with its cost");
}

void flatnessRegularDerivatives() {
    checkFlatnessDerivatives(false);
}

void flatnessFallbackDerivatives() {
    checkFlatnessDerivatives(true);
}

} // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*run)();
    };
    const std::array<TestCase, 12> tests{{
        {"RLS informative sample", rlsInformativeSample},
        {"RLS covariance bounds", rlsCovarianceBounds},
        {"RLS underflow and invalid input", rlsUnderflowAndInvalidInput},
        {"RLS independent information-form reference", rlsInformationFormReference},
        {"scaled quaternion normalization", scaledQuaternionNormalization},
        {"quaternion fallback convention", quaternionFallbackConvention},
        {"rotation frame singularities", rotationFrameSingularities},
        {"septic normalized boundary conditions", septicNormalizedBoundaryConditions},
        {"analytic septic boundary conditions", analyticSepticBoundaryConditions},
        {"dynamic penalty gradient", dynamicPenaltyGradient},
        {"flatness regular derivatives", flatnessRegularDerivatives},
        {"flatness fallback derivatives", flatnessFallbackDerivatives},
    }};
    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    // Do not use assert: these checks must also run in Release/RelWithDebInfo.
    return failures == 0 ? 0 : 1;
}
