#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#include <xgc2_math/math.hpp>

enum class TestPipelineStatus {
    kReady,
    kBlocked,
};

namespace xgc2_math {

template <> struct StatusRegistry<::TestPipelineStatus> {
    static constexpr std::array<StatusDescriptor<::TestPipelineStatus>, 2> statuses{{
        {::TestPipelineStatus::kReady, "ready"},
        {::TestPipelineStatus::kBlocked, "blocked"},
    }};
};

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

    static bool hasLastInertial(const Pose3InertialEskf& estimator) { return estimator.has_last_inertial_; }

    static double lastInertialStamp(const Pose3InertialEskf& estimator) { return estimator.last_inertial_.stamp_sec; }
};

} // namespace xgc2_math

namespace {

void expect(bool condition) {
    if (!condition) {
        std::abort();
    }
}

void testButterworth() {
    xgc2_math::SecondOrderButterworthLowPass filter(5.0, 0.0);
    double y = 0.0;
    for (int i = 0; i < 200; ++i) {
        y = filter.filter(1.0, 0.01);
        expect(std::isfinite(y));
    }
    expect(y > 0.95);

    const double held = filter.filter(std::numeric_limits<double>::quiet_NaN(), 0.01);
    expect(std::isfinite(held));
    expect(held == filter.value());

    const double before_bad_dt = filter.value();
    expect(filter.filter(10.0, 0.0) == before_bad_dt);
    expect(filter.filter(10.0, -0.01) == before_bad_dt);
    expect(filter.filter(10.0, std::numeric_limits<double>::quiet_NaN()) == before_bad_dt);
    expect(filter.filter(10.0, std::numeric_limits<double>::infinity()) == before_bad_dt);

    xgc2_math::SecondOrderButterworthLowPass invalid_cutoff(std::numeric_limits<double>::quiet_NaN(), 2.0);
    expect(invalid_cutoff.filter(4.0, 0.01) == 4.0);
    expect(invalid_cutoff.value() == 4.0);
    invalid_cutoff.setCutoffFrequencyHz(std::numeric_limits<double>::infinity());
    expect(invalid_cutoff.filter(6.0, 0.01) == 6.0);

    filter.resetState(std::numeric_limits<double>::quiet_NaN());
    expect(filter.initialized());
    expect(filter.value() == 0.0);
    expect(std::isfinite(filter.filter(1.0, 0.01)));

    xgc2_math::SecondOrderButterworthLowPass retuned_filter(1.0, 0.0);
    retuned_filter.filter(1.0, 0.01);
    const double before_retune = retuned_filter.value();
    retuned_filter.setCutoffFrequencyHz(10.0);
    expect(retuned_filter.cutoffFrequencyHz() == 10.0);
    expect(retuned_filter.value() == before_retune);
    expect(std::isfinite(retuned_filter.filter(2.0, 0.01)));

    xgc2_math::SecondOrderButterworthLowPass overflow_filter(5.0, 0.0);
    overflow_filter.resetState(std::numeric_limits<double>::max());
    expect(overflow_filter.filter(1.0, 0.01) == 1.0);
    expect(overflow_filter.value() == 1.0);
}

void testExponentialLowPass() {
    xgc2_math::ExponentialLowPass filter(2.0, 0.0);
    double y = 0.0;
    for (int i = 0; i < 100; ++i) {
        y = filter.filter(1.0, 0.02);
        expect(std::isfinite(y));
    }
    expect(y > 0.9);

    const double held = filter.filter(std::numeric_limits<double>::quiet_NaN(), 0.02);
    expect(held == filter.value());

    const double before_bad_dt = filter.value();
    expect(filter.filter(10.0, 0.0) == before_bad_dt);
    expect(filter.filter(10.0, -0.01) == before_bad_dt);
    expect(filter.filter(10.0, std::numeric_limits<double>::quiet_NaN()) == before_bad_dt);
    expect(filter.filter(10.0, std::numeric_limits<double>::infinity()) == before_bad_dt);

    xgc2_math::ExponentialLowPass invalid_initial(2.0, std::numeric_limits<double>::quiet_NaN());
    expect(!invalid_initial.initialized());
    expect(invalid_initial.value() == 0.0);
    expect(invalid_initial.filter(3.0, 0.02) == 3.0);
    expect(invalid_initial.initialized());

    xgc2_math::ExponentialLowPass bypass(0.0, 1.0);
    expect(bypass.filter(5.0, 0.02) == 5.0);
    expect(bypass.value() == 5.0);
    bypass.setCutoffFrequencyHz(-1.0);
    expect(bypass.cutoffFrequencyHz() == 0.0);
    expect(bypass.filter(7.0, std::numeric_limits<double>::quiet_NaN()) == 7.0);

    xgc2_math::ExponentialLowPass step_filter(1.0, 0.0);
    const double first = step_filter.filter(1.0, 0.001);
    const double second = step_filter.filter(1.0, 0.001);
    expect(first > 0.0);
    expect(second > first);
    expect(second < 1.0);
}

void testSlewRateLimiter() {
    xgc2_math::SlewRateLimiter limiter(0.5, 0.0);

    expect(std::fabs(limiter.filter(10.0, 0.2) - 0.1) < 1.0e-12);
    const double before_bad_dt = limiter.value();
    expect(limiter.filter(10.0, 0.0) == before_bad_dt);
    expect(limiter.filter(10.0, -0.01) == before_bad_dt);
    expect(limiter.filter(10.0, std::numeric_limits<double>::quiet_NaN()) == before_bad_dt);
    expect(limiter.filter(10.0, std::numeric_limits<double>::infinity()) == before_bad_dt);
    expect(limiter.filter(std::numeric_limits<double>::quiet_NaN(), 0.2) == before_bad_dt);

    expect(std::fabs(limiter.filter(-10.0, 0.2) - 0.0) < 1.0e-12);

    xgc2_math::SlewRateLimiter invalid_initial(0.5, std::numeric_limits<double>::quiet_NaN());
    expect(!invalid_initial.initialized());
    expect(invalid_initial.value() == 0.0);
    expect(invalid_initial.filter(3.0, 0.2) == 3.0);
    expect(invalid_initial.initialized());

    xgc2_math::SlewRateLimiter frozen(0.0, 1.0);
    expect(frozen.filter(2.0, 0.2) == 1.0);
    frozen.setMaxRatePerSecond(-1.0);
    expect(frozen.maxRatePerSecond() == 0.0);
    expect(frozen.filter(3.0, 0.2) == 1.0);
    frozen.setMaxRatePerSecond(std::numeric_limits<double>::quiet_NaN());
    expect(frozen.maxRatePerSecond() == 0.0);
    expect(frozen.filter(4.0, 0.2) == 1.0);
    frozen.setMaxRatePerSecond(std::numeric_limits<double>::infinity());
    expect(frozen.maxRatePerSecond() == 0.0);
    expect(frozen.filter(5.0, 0.2) == 1.0);

    xgc2_math::SlewRateLimiter overflow_filter(std::numeric_limits<double>::max(), 2.0);
    expect(overflow_filter.filter(3.0, std::numeric_limits<double>::max()) == 2.0);
}

void testStatusHelpers() {
    const auto& time_delta_statuses = xgc2_math::registeredStatuses<xgc2_math::TimeDeltaStatus>();
    expect(time_delta_statuses.size() == 5);
    expect(time_delta_statuses[0].status == xgc2_math::TimeDeltaStatus::kInitialized);
    expect(std::string(xgc2_math::toString(xgc2_math::TimeDeltaStatus::kHeldInvalidDt)) == "held_invalid_dt");

    const auto* invalid_dt = xgc2_math::statusDescriptor(xgc2_math::TimeDeltaStatus::kHeldInvalidDt);
    expect(invalid_dt != nullptr);
    if (invalid_dt != nullptr) {
        expect(std::string(invalid_dt->name) == "held_invalid_dt");
    }

    const auto& test_statuses = xgc2_math::registeredStatuses<TestPipelineStatus>();
    expect(test_statuses.size() == 2);
    expect(std::string(xgc2_math::toString(TestPipelineStatus::kReady)) == "ready");
    expect(std::string(xgc2_math::toString(TestPipelineStatus::kBlocked)) == "blocked");
}

void testTimeDeltaGuard() {
    xgc2_math::TimeDeltaGuardOptions options;
    options.min_dt_s = 0.01;
    options.max_dt_s = 0.1;

    xgc2_math::TimeDeltaGuard guard(options);
    auto sample = guard.update(10.0);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kInitialized);
    expect(!sample.accepted);
    expect(sample.dt_s == 0.0);
    expect(std::fabs(guard.lastTimestampS() - 10.0) < 1.0e-12);

    sample = guard.update(10.02);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kAccepted);
    expect(sample.accepted);
    expect(std::fabs(sample.dt_s - 0.02) < 1.0e-12);

    sample = guard.update(10.0205);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kHeldInvalidDt);
    expect(!sample.accepted);
    expect(std::fabs(guard.lastTimestampS() - 10.02) < 1.0e-12);

    sample = guard.update(10.05);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kAccepted);
    expect(sample.accepted);
    expect(std::fabs(sample.dt_s - 0.03) < 1.0e-12);

    sample = guard.update(std::numeric_limits<double>::quiet_NaN());
    expect(sample.status == xgc2_math::TimeDeltaStatus::kHeldInvalidInput);
    expect(!sample.accepted);
    expect(std::fabs(guard.lastTimestampS() - 10.05) < 1.0e-12);

    sample = guard.update(10.08);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kAccepted);
    expect(sample.accepted);
    expect(std::fabs(sample.dt_s - 0.03) < 1.0e-12);

    sample = guard.update(10.5);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kHeldInvalidDt);
    expect(!sample.accepted);
    expect(std::fabs(guard.lastTimestampS() - 10.5) < 1.0e-12);

    sample = guard.update(10.54);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kAccepted);
    expect(sample.accepted);
    expect(std::fabs(sample.dt_s - 0.04) < 1.0e-12);

    sample = guard.update(10.4);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kHeldTimeWentBack);
    expect(!sample.accepted);
    expect(std::fabs(guard.lastTimestampS() - 10.4) < 1.0e-12);

    sample = guard.update(10.45);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kAccepted);
    expect(sample.accepted);
    expect(std::fabs(sample.dt_s - 0.05) < 1.0e-12);

    options.reset_on_large_dt = false;
    guard.setOptions(options);
    guard.reset(20.0);
    sample = guard.update(20.5);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kHeldInvalidDt);
    expect(!sample.accepted);
    expect(std::fabs(guard.lastTimestampS() - 20.0) < 1.0e-12);

    sample = guard.update(20.08);
    expect(sample.status == xgc2_math::TimeDeltaStatus::kAccepted);
    expect(sample.accepted);
    expect(std::fabs(sample.dt_s - 0.08) < 1.0e-12);
}

void testSampleTimingPredicates() {
    expect(!xgc2_math::sampleStale(10.0, 9.9, 0.2));
    expect(xgc2_math::sampleStale(10.0, 9.7, 0.2));
    expect(xgc2_math::sampleStale(std::numeric_limits<double>::quiet_NaN(), 9.7, 0.2));
    expect(xgc2_math::sampleStale(10.0, 9.7, std::numeric_limits<double>::quiet_NaN()));

    expect(!xgc2_math::sampleRateLow(false, 1.0, 5.0));
    expect(!xgc2_math::sampleRateLow(true, 0.0, 5.0));
    expect(!xgc2_math::sampleRateLow(true, 1.0, 0.0));
    expect(!xgc2_math::sampleRateLow(true, 0.1, 5.0));
    expect(xgc2_math::sampleRateLow(true, 0.3, 5.0));

    expect(!xgc2_math::sampleTimeJumped(10.0, 10.0, 0.01));
    expect(xgc2_math::sampleTimeJumped(10.0, 10.2, 0.01));
    expect(xgc2_math::sampleTimeJumped(10.0, 10.0, -0.1));
    expect(!xgc2_math::sampleTimeJumped(10.0, 10.0, -0.01));
    expect(xgc2_math::sampleTimeJumped(std::numeric_limits<double>::quiet_NaN(), 10.0, 0.01));
}

void testOptionNormalization() {
    xgc2_math::DifferentiatorOptions bad_diff_options;
    bad_diff_options.min_dt_s = -1.0;
    bad_diff_options.max_dt_s = -2.0;
    bad_diff_options.max_input_step = -std::numeric_limits<double>::infinity();
    bad_diff_options.max_derivative = -std::numeric_limits<double>::infinity();
    bad_diff_options.derivative_cutoff_hz = -3.0;
    const auto diff_options = xgc2_math::normalized(bad_diff_options);
    expect(xgc2_math::isValid(diff_options));
    expect(diff_options.max_input_step == std::numeric_limits<double>::infinity());
    expect(diff_options.max_derivative == std::numeric_limits<double>::infinity());

    xgc2_math::PositionVelocityObserverOptions bad_observer_options;
    bad_observer_options.position_gain = -1.0;
    bad_observer_options.max_velocity = std::numeric_limits<double>::quiet_NaN();
    const auto observer_options = xgc2_math::normalized(bad_observer_options);
    expect(xgc2_math::isValid(observer_options));
}

void testDifferentiator() {
    xgc2_math::DifferentiatorOptions options;
    options.min_dt_s = 0.001;
    options.max_dt_s = 0.1;
    options.max_input_step = 1.0;
    options.max_derivative = 20.0;
    options.derivative_cutoff_hz = 10.0;

    xgc2_math::Differentiator default_differentiator;
    expect(xgc2_math::isValid(default_differentiator.options()));

    xgc2_math::Differentiator invalid_reset(options);
    invalid_reset.reset(0.2, 1.0);
    expect(invalid_reset.initialized());
    invalid_reset.reset(std::numeric_limits<double>::quiet_NaN(), 0.0);
    expect(!invalid_reset.initialized());
    expect(invalid_reset.value() == 0.0);
    expect(invalid_reset.derivative() == 0.0);
    invalid_reset.reset(0.0, std::numeric_limits<double>::infinity());
    expect(!invalid_reset.initialized());
    expect(invalid_reset.value() == 0.0);
    expect(invalid_reset.derivative() == 0.0);

    xgc2_math::Differentiator differentiator(options);
    auto sample = differentiator.update(0.0, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kInitialized);
    expect(sample.measurement_accepted);
    expect(sample.derivative == 0.0);

    sample = differentiator.update(std::numeric_limits<double>::quiet_NaN(), 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidInput);
    expect(!sample.measurement_accepted);
    expect(differentiator.value() == 0.0);
    expect(differentiator.derivative() == 0.0);

    sample = differentiator.update(0.1, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kAccepted);
    expect(sample.measurement_accepted);

    const double derivative = sample.derivative;
    const double accepted_value = differentiator.value();
    sample = differentiator.update(10.0, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldOutlier);
    expect(!sample.measurement_accepted);
    expect(differentiator.value() == accepted_value);
    expect(std::fabs(sample.derivative - derivative) < 1.0e-12);

    sample = differentiator.update(0.2, 0.0);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    expect(!sample.measurement_accepted);
    sample = differentiator.update(0.2, -0.01);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    sample = differentiator.update(0.2, std::numeric_limits<double>::quiet_NaN());
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    sample = differentiator.update(0.2, std::numeric_limits<double>::infinity());
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    expect(differentiator.value() == accepted_value);
    expect(differentiator.derivative() == derivative);

    xgc2_math::DifferentiatorOptions unbounded_options;
    unbounded_options.min_dt_s = 0.001;
    unbounded_options.max_dt_s = 0.1;
    xgc2_math::Differentiator delta_overflow(unbounded_options);
    delta_overflow.reset(-std::numeric_limits<double>::max(), 0.0);
    sample = delta_overflow.update(std::numeric_limits<double>::max(), 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldOutlier);
    expect(!sample.measurement_accepted);
    expect(delta_overflow.value() == -std::numeric_limits<double>::max());
    expect(delta_overflow.derivative() == 0.0);

    xgc2_math::Differentiator derivative_overflow(unbounded_options);
    derivative_overflow.reset(0.0, 0.0);
    sample = derivative_overflow.update(std::numeric_limits<double>::max(), 0.001);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldOutlier);
    expect(!sample.measurement_accepted);
    expect(derivative_overflow.value() == 0.0);
    expect(derivative_overflow.derivative() == 0.0);

    xgc2_math::DifferentiatorOptions reset_options = options;
    reset_options.reset_on_large_dt = true;
    xgc2_math::Differentiator reset_on_large_dt(reset_options);
    sample = reset_on_large_dt.update(0.0, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kInitialized);
    sample = reset_on_large_dt.update(2.0, 0.2);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kInitialized);
    expect(sample.measurement_accepted);
    expect(reset_on_large_dt.value() == 2.0);
    expect(reset_on_large_dt.derivative() == 0.0);
}

void testAngleUtilitiesAndDifferentiator() {
    expect(std::fabs(xgc2_math::normalizeAngle(3.0 * xgc2_math::kPi) - xgc2_math::kPi) < 1.0e-12);

    const double from = xgc2_math::kPi - 0.01;
    const double to = -xgc2_math::kPi + 0.01;
    const double delta = xgc2_math::shortestAngularDistance(from, to);
    expect(std::fabs(delta - 0.02) < 1.0e-12);

    xgc2_math::DifferentiatorOptions options;
    options.min_dt_s = 0.001;
    options.max_dt_s = 0.1;
    options.max_input_step = 0.1;
    options.max_derivative = 10.0;

    xgc2_math::AngleDifferentiator default_differentiator;
    expect(xgc2_math::isValid(default_differentiator.options()));

    xgc2_math::AngleDifferentiator invalid_reset(options);
    invalid_reset.reset(0.2, 1.0);
    expect(invalid_reset.initialized());
    invalid_reset.reset(std::numeric_limits<double>::quiet_NaN(), 0.0);
    expect(!invalid_reset.initialized());
    expect(invalid_reset.value() == 0.0);
    expect(invalid_reset.derivative() == 0.0);
    invalid_reset.reset(0.0, std::numeric_limits<double>::infinity());
    expect(!invalid_reset.initialized());
    expect(invalid_reset.value() == 0.0);
    expect(invalid_reset.derivative() == 0.0);

    xgc2_math::AngleDifferentiator differentiator(options);
    auto sample = differentiator.update(from, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kInitialized);
    expect(sample.measurement_accepted);
    expect(sample.derivative == 0.0);

    const double initialized_angle = differentiator.value();
    sample = differentiator.update(std::numeric_limits<double>::quiet_NaN(), 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidInput);
    expect(!sample.measurement_accepted);
    expect(differentiator.value() == initialized_angle);
    expect(differentiator.derivative() == 0.0);

    sample = differentiator.update(to, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kAccepted);
    expect(sample.measurement_accepted);
    expect(std::fabs(sample.derivative - 1.0) < 1.0e-12);

    const double accepted_angle = differentiator.value();
    const double accepted_derivative = differentiator.derivative();
    sample = differentiator.update(accepted_angle + 0.01, 0.0);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    expect(!sample.measurement_accepted);
    expect(differentiator.value() == accepted_angle);
    expect(differentiator.derivative() == accepted_derivative);
    sample = differentiator.update(accepted_angle + 0.01, -0.01);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    sample = differentiator.update(accepted_angle + 0.01, std::numeric_limits<double>::quiet_NaN());
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    sample = differentiator.update(accepted_angle + 0.01, std::numeric_limits<double>::infinity());
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldInvalidDt);
    expect(differentiator.value() == accepted_angle);
    expect(differentiator.derivative() == accepted_derivative);

    sample = differentiator.update(accepted_angle + 0.2, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldOutlier);
    expect(!sample.measurement_accepted);
    expect(differentiator.value() == accepted_angle);
    expect(differentiator.derivative() == accepted_derivative);

    sample = differentiator.update(accepted_angle + 0.08, 0.001);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kHeldOutlier);
    expect(!sample.measurement_accepted);
    expect(differentiator.value() == accepted_angle);
    expect(differentiator.derivative() == accepted_derivative);

    auto reset_options = options;
    reset_options.reset_on_large_dt = true;
    xgc2_math::AngleDifferentiator reset_on_large_dt(reset_options);
    sample = reset_on_large_dt.update(from, 0.02);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kInitialized);
    sample = reset_on_large_dt.update(to, 0.2);
    expect(sample.status == xgc2_math::DifferentiatorStatus::kInitialized);
    expect(sample.measurement_accepted);
    expect(std::fabs(reset_on_large_dt.value() - to) < 1.0e-12);
    expect(reset_on_large_dt.derivative() == 0.0);
}

void testPositionVelocityObserver() {
    xgc2_math::PositionVelocityObserverOptions options;
    options.position_gain = 0.45;
    options.velocity_gain = 0.12;
    options.max_position_residual = 2.0;
    options.max_velocity = 5.0;

    xgc2_math::PositionVelocityLuenbergerObserver observer(options);
    auto estimate = observer.update(0.0, 0.02);
    expect(estimate.status == xgc2_math::PositionVelocityObserverStatus::kInitialized);

    for (int i = 1; i <= 200; ++i) {
        estimate = observer.update(0.5 * i * 0.02, 0.02);
        expect(estimate.status == xgc2_math::PositionVelocityObserverStatus::kAccepted);
        expect(std::isfinite(estimate.position));
        expect(std::isfinite(estimate.velocity));
    }

    expect(observer.velocity() > 0.2);
    expect(observer.velocity() < 1.0);

    estimate = observer.update(100.0, 0.02);
    expect(estimate.status == xgc2_math::PositionVelocityObserverStatus::kHeldOutlier);
    expect(!estimate.measurement_accepted);
}

void testAngularPositionVelocityObserver() {
    xgc2_math::PositionVelocityObserverOptions options;
    options.position_gain = 0.4;
    options.velocity_gain = 0.1;
    options.max_position_residual = 0.5;

    xgc2_math::AngularPositionVelocityLuenbergerObserver observer(options);
    auto estimate = observer.update(xgc2_math::kPi - 0.01, 0.02);
    expect(estimate.status == xgc2_math::PositionVelocityObserverStatus::kInitialized);

    estimate = observer.update(-xgc2_math::kPi + 0.01, 0.02);
    expect(estimate.status == xgc2_math::PositionVelocityObserverStatus::kAccepted);
    expect(std::fabs(estimate.residual - 0.02) < 1.0e-12);
}

void testArrayWrappers() {
    xgc2_math::ArrayDifferentiator<3> differentiator;
    std::array<double, 3> p0{{0.0, 1.0, 2.0}};
    std::array<double, 3> p1{{0.1, 1.2, 2.3}};
    auto samples = differentiator.update(p0, 0.02);
    expect(samples[0].status == xgc2_math::DifferentiatorStatus::kInitialized);
    samples = differentiator.update(p1, 0.1);
    expect(samples[0].status == xgc2_math::DifferentiatorStatus::kAccepted);
    expect(std::fabs(samples[2].derivative - 3.0) < 1.0e-12);

    std::array<double, 3> reset_values{{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0}};
    differentiator.reset(reset_values);
    expect(differentiator.axis(0).initialized());
    expect(differentiator.axis(0).value() == 1.0);
    expect(!differentiator.axis(1).initialized());
    expect(differentiator.axis(1).value() == 0.0);
    expect(differentiator.axis(1).derivative() == 0.0);
    expect(differentiator.axis(2).initialized());
    expect(differentiator.axis(2).value() == 3.0);
    samples = differentiator.update(p1, 0.1);
    expect(samples[0].status == xgc2_math::DifferentiatorStatus::kAccepted);
    expect(samples[1].status == xgc2_math::DifferentiatorStatus::kInitialized);
    expect(samples[2].status == xgc2_math::DifferentiatorStatus::kAccepted);

    xgc2_math::ArrayPositionVelocityLuenbergerObserver<3> observer;
    auto estimates = observer.update(p0, 0.02);
    expect(estimates[1].status == xgc2_math::PositionVelocityObserverStatus::kInitialized);
    estimates = observer.update(p1, 0.02);
    expect(estimates[1].status == xgc2_math::PositionVelocityObserverStatus::kAccepted);
}

void testScalarRecursiveLeastSquares() {
    xgc2_math::ScalarRecursiveLeastSquaresOptions bad_options;
    bad_options.forgetting_factor = 2.0;
    bad_options.initial_covariance = std::numeric_limits<double>::infinity();
    bad_options.min_abs_regressor = -1.0;
    const auto options = xgc2_math::normalized(bad_options);
    expect(xgc2_math::isValid(options));

    xgc2_math::ScalarRecursiveLeastSquares estimator(options);
    auto sample = estimator.update(4.0, 2.0);
    expect(sample.status == xgc2_math::ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput);

    estimator.reset(1.0, 10.0);
    sample = estimator.update(6.0, 2.0);
    expect(sample.status == xgc2_math::ScalarRecursiveLeastSquaresStatus::kAccepted);
    expect(sample.measurement_accepted);
    expect(sample.parameter > 2.0);
    expect(sample.parameter < 3.0);
    expect(sample.covariance > 0.0);

    const double held_parameter = sample.parameter;
    sample = estimator.update(std::numeric_limits<double>::quiet_NaN(), 2.0);
    expect(sample.status == xgc2_math::ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput);
    expect(std::fabs(sample.parameter - held_parameter) < 1.0e-12);

    sample = estimator.update(6.0, 0.0);
    expect(sample.status == xgc2_math::ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput);

    sample = estimator.update(6.0, std::numeric_limits<double>::quiet_NaN());
    expect(sample.status == xgc2_math::ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput);
    expect(!sample.measurement_accepted);
}

struct InertialPoseTestSamples {
    static xgc2_math::InertialSample inertial(double stamp_sec, const Eigen::Vector3d& gyro = Eigen::Vector3d::Zero(),
                                              const Eigen::Vector3d& accel = Eigen::Vector3d(0.0, 0.0, 9.8066)) {
        xgc2_math::InertialSample sample;
        sample.received = true;
        sample.valid = true;
        sample.stamp_sec = stamp_sec;
        sample.angular_velocity = gyro;
        sample.linear_acceleration = accel;
        return sample;
    }

    static xgc2_math::PoseMeasurement pose(double stamp_sec, const Eigen::Vector3d& position,
                                           const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
        xgc2_math::PoseMeasurement sample;
        sample.received = true;
        sample.valid = true;
        sample.stamp_sec = stamp_sec;
        sample.pose.position = position;
        sample.pose.orientation = xgc2_math::normalizedQuaternion(orientation);
        return sample;
    }
};

void testSe3Utilities() {
    const Eigen::Quaterniond q = xgc2_math::normalizedQuaternion(Eigen::Quaterniond(-2.0, 0.1, -0.2, 0.3));
    expect(std::fabs(q.norm() - 1.0) < 1.0e-12);
    expect(q.w() >= 0.0);

    const Eigen::Quaterniond invalid =
        xgc2_math::normalizedQuaternion(Eigen::Quaterniond(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0));
    expect(std::fabs(invalid.w() - 1.0) < 1.0e-12);
    expect(invalid.vec().norm() < 1.0e-12);

    xgc2_math::Pose3 transform;
    transform.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    transform.orientation = xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.1, -0.2, 0.3));
    const xgc2_math::Pose3 identity = xgc2_math::compose(transform, xgc2_math::inverse(transform));
    expect(identity.position.norm() < 1.0e-12);
    expect(std::fabs(identity.orientation.w() - 1.0) < 1.0e-12);

    xgc2_math::Pose3 delta;
    delta.position = Eigen::Vector3d(0.2, -0.1, 0.05);
    delta.orientation = xgc2_math::expMap(Eigen::Vector3d(0.03, -0.02, 0.01));
    const xgc2_math::Pose3 measured = xgc2_math::compose(transform, delta);

    const Eigen::Matrix<double, 6, 1> error = xgc2_math::se3Error(transform, measured);
    expect((error.head<3>() - delta.position).norm() < 1.0e-12);
    expect((error.tail<3>() - xgc2_math::logMap(delta.orientation)).norm() < 1.0e-12);
}

void testSe2Utilities() {
    xgc2_math::Pose2 transform;
    transform.position = Eigen::Vector2d(1.0, -0.5);
    transform.yaw = 0.4;

    const xgc2_math::Pose2 identity = xgc2_math::compose(transform, xgc2_math::inverse(transform));
    expect(identity.position.norm() < 1.0e-12);
    expect(std::fabs(identity.yaw) < 1.0e-12);

    xgc2_math::Pose2 delta;
    delta.position = Eigen::Vector2d(0.2, -0.1);
    delta.yaw = 0.03;
    const xgc2_math::Pose2 measured = xgc2_math::compose(transform, delta);
    const Eigen::Vector3d error = xgc2_math::se2Error(transform, measured);
    expect((error.head<2>() - delta.position).norm() < 1.0e-12);
    expect(std::fabs(error.z() - delta.yaw) < 1.0e-12);
}

void testObservationHealthTrackerFixedWindow() {
    xgc2_math::ObservationHealthConfig config;
    config.window_size = 3;
    config.window_chi_square_gate = 1000.0;

    xgc2_math::ObservationHealthTracker tracker;
    tracker.setConfig(config);
    tracker.recordAccepted(1.0);
    tracker.recordAccepted(2.0);
    tracker.recordAccepted(3.0);
    expect(std::fabs(tracker.chiSquareWindowSum() - 6.0) < 1.0e-12);
    tracker.recordAccepted(4.0);
    expect(std::fabs(tracker.chiSquareWindowSum() - 9.0) < 1.0e-12);

    config.window_size = xgc2_math::ObservationHealthTracker::kMaxWindowSize + 10u;
    tracker.setConfig(config);
    tracker.reset();
    tracker.setConfig(config);
    for (std::size_t i = 0; i < xgc2_math::ObservationHealthTracker::kMaxWindowSize + 5u; ++i) {
        tracker.recordAccepted(1.0);
    }
    expect(std::fabs(tracker.chiSquareWindowSum() -
                     static_cast<double>(xgc2_math::ObservationHealthTracker::kMaxWindowSize)) < 1.0e-12);
}

void testPose2InertialEskf() {
    xgc2_math::Pose2InertialEskfConfig config;
    config.measurement_frame_to_world.position = Eigen::Vector2d(1.0, 2.0);
    config.measurement_frame_to_world.yaw = 0.0;
    config.body_to_marker.position = Eigen::Vector2d(0.2, 0.0);
    config.pose_position_noise_std = 0.01;
    config.pose_yaw_noise_std = 0.01;
    config.accel_noise_std = 0.3;
    config.gyro_noise_std = 0.03;
    config.gyro_bias_random_walk_std = 1.0e-4;
    config.accel_bias_random_walk_std = 1.0e-3;
    config.innovation_position_gate_m = 0.8;
    config.innovation_yaw_gate_rad = 0.5;

    xgc2_math::Pose2InertialEskf estimator;
    estimator.setConfig(config);

    xgc2_math::PlanarPoseMeasurement first_pose;
    first_pose.received = true;
    first_pose.valid = true;
    first_pose.stamp_sec = 1.0;
    first_pose.pose.position = Eigen::Vector2d(2.0, 0.0);
    first_pose.pose.yaw = 0.1;
    xgc2_math::PlanarInertialSample first_imu;
    first_imu.received = true;
    first_imu.valid = true;
    first_imu.stamp_sec = 1.0;
    first_imu.angular_velocity_z = 0.0;
    first_imu.linear_acceleration = Eigen::Vector2d::Zero();
    xgc2_math::Pose2InertialEskf uninitialized_estimator;
    const auto first_imu_result = uninitialized_estimator.propagateInertial(first_imu);
    expect(!first_imu_result.accepted);
    expect(first_imu_result.initialized_stamp);
    expect(!uninitialized_estimator.initialized());
    expect(std::fabs(uninitialized_estimator.state().last_inertial_stamp_sec - first_imu.stamp_sec) < 1.0e-12);

    estimator.initializeFromPose(first_pose, &first_imu);

    expect(estimator.initialized());
    const xgc2_math::Pose2 expected_marker_world =
        xgc2_math::compose(config.measurement_frame_to_world, first_pose.pose);
    const xgc2_math::Pose2 expected_body_world =
        xgc2_math::compose(expected_marker_world, xgc2_math::inverse(config.body_to_marker));
    expect((estimator.state().position - expected_body_world.position).norm() < 1.0e-12);
    expect(std::fabs(estimator.state().yaw - 0.1) < 1.0e-12);
    expect(estimator.hasCorrectedBodyPose());
    expect(estimator.hasRawProjectedBodyPose());
    expect((estimator.correctedBodyPose().position - estimator.state().position).norm() < 1.0e-12);
    expect((estimator.rawProjectedBodyPose().position - expected_body_world.position).norm() < 1.0e-12);

    xgc2_math::PlanarInertialSample imu = first_imu;
    imu.stamp_sec = 1.02;
    imu.angular_velocity_z = 0.02;
    imu.linear_acceleration = Eigen::Vector2d(0.1, 0.0);
    const auto propagated = estimator.propagateInertial(imu);
    expect(propagated.accepted);
    expect(estimator.state().position.x() > expected_body_world.position.x());
    expect(estimator.state().yaw > 0.1);
    const auto held_after_propagation = estimator.state();
    auto out_of_order_imu = imu;
    out_of_order_imu.stamp_sec = 1.01;
    const auto out_of_order_result = estimator.propagateInertial(out_of_order_imu);
    expect(!out_of_order_result.accepted);
    expect(std::fabs(estimator.state().last_inertial_stamp_sec - held_after_propagation.last_inertial_stamp_sec) <
           1.0e-12);
    expect((estimator.state().position - held_after_propagation.position).norm() < 1.0e-12);

    xgc2_math::PlanarPoseMeasurement second_pose = first_pose;
    second_pose.stamp_sec = 1.1;
    second_pose.pose.position = Eigen::Vector2d(2.2, 0.0);
    second_pose.pose.yaw = 0.12;
    const auto accepted = estimator.updatePose(second_pose);
    expect(accepted.accepted);
    expect(!accepted.innovation_rejected);
    expect(estimator.state().position.x() > 2.8);
    expect(estimator.state().velocity.x() > 0.0);
    expect(estimator.state().yaw_rate > 0.0);
    expect(estimator.covariance().trace() > 0.0);
    const xgc2_math::Pose2 expected_raw_body =
        xgc2_math::compose(xgc2_math::compose(config.measurement_frame_to_world, second_pose.pose),
                           xgc2_math::inverse(config.body_to_marker));
    expect(estimator.hasRawProjectedBodyPose());
    expect((estimator.rawProjectedBodyPose().position - expected_raw_body.position).norm() < 1.0e-12);
    expect((estimator.correctedBodyPose().position - estimator.state().position).norm() < 1.0e-12);

    const Eigen::Vector2d held_position = estimator.state().position;
    xgc2_math::PlanarPoseMeasurement jump_pose = second_pose;
    jump_pose.stamp_sec = 1.2;
    jump_pose.pose.position = Eigen::Vector2d(10.0, 0.0);
    const auto rejected = estimator.updatePose(jump_pose);
    expect(!rejected.accepted);
    expect(rejected.innovation_rejected);
    expect((estimator.state().position - held_position).norm() < 1.0e-12);
}

void testPose2InertialEskfOutOfOrderPoseDrop() {
    xgc2_math::Pose2InertialEskfConfig config;
    config.innovation_position_gate_m = 1.0;

    xgc2_math::Pose2InertialEskf estimator;
    estimator.setConfig(config);

    xgc2_math::PlanarPoseMeasurement pose0;
    pose0.received = true;
    pose0.valid = true;
    pose0.stamp_sec = 1.0;
    pose0.pose.position = Eigen::Vector2d::Zero();

    xgc2_math::PlanarInertialSample imu0;
    imu0.received = true;
    imu0.valid = true;
    imu0.stamp_sec = 1.0;
    estimator.initializeFromPose(pose0, &imu0);

    auto imu = imu0;
    imu.linear_acceleration = Eigen::Vector2d(1.0, 0.0);
    imu.stamp_sec = 1.02;
    estimator.propagateInertial(imu);
    imu.stamp_sec = 1.04;
    estimator.propagateInertial(imu);
    const auto held_state = estimator.state();
    const auto held_corrected = estimator.correctedBodyPose();
    const double held_last_fused_stamp = estimator.lastFusedPoseStampS();

    auto out_of_order_pose = pose0;
    out_of_order_pose.stamp_sec = 1.02;
    out_of_order_pose.pose.position = Eigen::Vector2d(0.0002, 0.0);
    const auto out_of_order_result = estimator.updatePose(out_of_order_pose);
    expect(!out_of_order_result.accepted);
    expect(out_of_order_result.time_alignment_rejected);
    expect(out_of_order_result.reject_reason == xgc2_math::PoseFusionRejectReason::kTimeAlignment);
    expect(std::fabs(estimator.state().last_inertial_stamp_sec - 1.04) < 1.0e-12);
    expect(std::fabs(estimator.lastFusedPoseStampS() - held_last_fused_stamp) < 1.0e-12);
    expect((estimator.state().position - held_state.position).norm() < 1.0e-12);
    expect((estimator.correctedBodyPose().position - held_corrected.position).norm() < 1.0e-12);
    expect((estimator.correctedBodyPose().position - estimator.state().position).norm() < 1.0e-12);

    auto too_old_pose = pose0;
    too_old_pose.stamp_sec = 0.5;
    const auto rejected = estimator.updatePose(too_old_pose);
    expect(!rejected.accepted);
    expect(rejected.time_alignment_rejected);
    expect(rejected.reject_reason == xgc2_math::PoseFusionRejectReason::kTimeAlignment);
}

void testPose3InertialEskfInitialization() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.measurement_frame_to_world.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    config.body_to_marker.position = Eigen::Vector3d(0.1, 0.0, 0.0);

    xgc2_math::Pose3InertialEskf inertial_clear_eskf;
    inertial_clear_eskf.propagateInertial(InertialPoseTestSamples::inertial(2.0));
    expect(xgc2_math::Pose3InertialEskfTestAccess::hasLastInertial(inertial_clear_eskf));
    inertial_clear_eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), nullptr);
    expect(!xgc2_math::Pose3InertialEskfTestAccess::hasLastInertial(inertial_clear_eskf));
    expect(std::fabs(xgc2_math::Pose3InertialEskfTestAccess::lastInertialStamp(inertial_clear_eskf)) < 1.0e-12);

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);

    const auto imu = InertialPoseTestSamples::inertial(1.0);
    const auto pose = InertialPoseTestSamples::pose(1.0, Eigen::Vector3d(2.0, 0.0, 1.0));
    eskf.initializeFromPose(pose, &imu);

    expect(eskf.initialized());
    expect(std::fabs(eskf.state().position.x() - 2.9) < 1.0e-12);
    expect(std::fabs(eskf.state().position.y() - 2.0) < 1.0e-12);
    expect(std::fabs(eskf.state().position.z() - 4.0) < 1.0e-12);
    expect(eskf.hasCorrectedBodyPose());
    expect(eskf.hasRawProjectedBodyPose());
    expect(std::fabs(eskf.correctedBodyPose().position.x() - 2.9) < 1.0e-12);
    expect(std::fabs(eskf.rawProjectedBodyPose().position.x() - 2.9) < 1.0e-12);
}

void testPose3InertialEskfUpdatePoseInitializesAndHealthUsesCovariance() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.innovation_position_gate_m = 0.1;
    config.covariance_high_threshold = 1.0e-9;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);

    const auto result = eskf.updatePose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d(5.0, 0.0, 0.0)));
    expect(result.accepted);
    expect(eskf.initialized());
    expect(std::fabs(eskf.state().position.x() - 5.0) < 1.0e-12);
    expect(eskf.filterHealth() == xgc2_math::FilterHealth::kDegraded);
}

void testPose3InertialEskfStationaryPropagation() {
    xgc2_math::Pose3InertialEskf eskf;
    const auto imu0 = InertialPoseTestSamples::inertial(1.0, Eigen::Vector3d(0.1, 0.0, 0.0));
    xgc2_math::Pose3InertialEskf uninitialized_eskf;
    uninitialized_eskf.propagateInertial(imu0);
    expect(!uninitialized_eskf.initialized());
    expect(std::fabs(uninitialized_eskf.state().last_inertial_stamp_sec - 1.0) < 1.0e-12);

    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    const auto imu1 = InertialPoseTestSamples::inertial(1.01, Eigen::Vector3d(0.1, 0.0, 0.0));
    eskf.propagateInertial(imu1);
    const auto held_state = eskf.state();
    eskf.propagateInertial(InertialPoseTestSamples::inertial(1.005, Eigen::Vector3d(0.1, 0.0, 0.0)));

    expect(std::fabs(eskf.state().angular_velocity.x() - 0.1) < 1.0e-12);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - held_state.last_inertial_stamp_sec) < 1.0e-12);
    expect((eskf.state().position - held_state.position).norm() < 1.0e-12);
    expect(eskf.state().velocity.norm() < 1.0e-3);
    expect(std::fabs(eskf.state().orientation.norm() - 1.0) < 1.0e-12);
    expect(eskf.state().orientation.w() >= 0.0);
}

void testPose3InertialEskfMidpointIntegrationPublishesCurrentAngularRate() {
    xgc2_math::Pose3InertialEskf eskf;
    const auto imu0 = InertialPoseTestSamples::inertial(1.0, Eigen::Vector3d::Zero());
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    const Eigen::Vector3d current_gyro(1.0, -0.4, 0.2);
    const auto imu1 = InertialPoseTestSamples::inertial(1.02, current_gyro);
    eskf.propagateInertial(imu1);

    expect((eskf.state().angular_velocity - current_gyro).norm() < 1.0e-12);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - 1.02) < 1.0e-12);
}

void testPose3InertialEskfTimeJumpResets() {
    xgc2_math::Pose3InertialEskf eskf;
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);
    expect(eskf.initialized());

    auto jumped = InertialPoseTestSamples::inertial(2.0);
    jumped.time_jump = true;
    eskf.propagateInertial(jumped);
    expect(!eskf.initialized());
    expect(eskf.filterHealth() == xgc2_math::FilterHealth::kLost);
}

void testPose3InertialEskfPoseUpdateAndReject() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.innovation_position_gate_m = 0.5;
    config.pose_nis_gate = 1.0e6;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    auto result = eskf.updatePose(InertialPoseTestSamples::pose(1.02, Eigen::Vector3d(0.4, 0.0, 0.0)));
    expect(result.accepted);
    expect(eskf.state().position.x() > 0.3);
    expect(eskf.state().position.x() < 0.4);

    const double held_position = eskf.state().position.x();
    result = eskf.updatePose(InertialPoseTestSamples::pose(1.04, Eigen::Vector3d(2.0, 0.0, 0.0)));
    expect(!result.accepted);
    expect(result.innovation_rejected);
    expect(eskf.state().position.x() < held_position + 0.1);
}

void testPose3InertialEskfHighNisWithinAbsoluteGateIsSoftHealth() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.innovation_position_gate_m = 1.0;
    config.pose_position_noise_std = 1.0e-3;
    config.pose_orientation_noise_std = 1.0e-3;
    config.pose_nis_gate = 0.1;
    config.vrpn_health.window_chi_square_gate = 0.1;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    const auto result = eskf.updatePose(InertialPoseTestSamples::pose(1.01, Eigen::Vector3d(0.05, 0.0, 0.0)));
    expect(result.accepted);
    expect(!result.innovation_rejected);
    expect(result.reject_reason == xgc2_math::PoseFusionRejectReason::kNone);
    expect(result.mahalanobis_distance > config.pose_nis_gate);
    expect(result.vrpn_observation_state == xgc2_math::VrpnObservationState::kSuspected);
    expect(result.filter_health == xgc2_math::FilterHealth::kDegraded);
    expect(eskf.vrpnConsecutiveRejects() == 0u);
    expect(eskf.lastFusedPoseStampS() == 1.01);
}

void testPose3InertialEskfPredictionErrorDoesNotRejectContinuousVrpn() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.innovation_position_gate_m = 0.5;
    config.pose_position_noise_std = 0.01;
    config.pose_orientation_noise_std = 0.01;
    config.pose_nis_gate = 1.0e6;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    auto result = eskf.updatePose(InertialPoseTestSamples::pose(1.01, Eigen::Vector3d(0.1, 0.0, 0.0)));
    expect(result.accepted);

    const auto bad_imu =
        InertialPoseTestSamples::inertial(1.41, Eigen::Vector3d::Zero(), Eigen::Vector3d(80.0, 0.0, 9.8066));
    eskf.propagateInertial(bad_imu);
    expect(eskf.state().position.x() > 0.7);

    result = eskf.updatePose(InertialPoseTestSamples::pose(1.42, Eigen::Vector3d(0.2, 0.0, 0.0)));
    expect(result.accepted);
    expect(!result.innovation_rejected);
    expect(result.reject_reason == xgc2_math::PoseFusionRejectReason::kNone);
    expect(result.position_innovation_norm > config.innovation_position_gate_m);
    expect(eskf.vrpnConsecutiveRejects() == 0u);
    expect(std::fabs(eskf.rawProjectedBodyPose().position.x() - 0.2) < 1.0e-12);
    expect(std::fabs(eskf.lastFusedPoseStampS() - 1.42) < 1.0e-12);
    expect(std::fabs(eskf.state().position.x() - 0.2) < 0.08);
}

void testPose3InertialEskfInvalidAndLargeDtHoldState() {
    xgc2_math::Pose3InertialEskf eskf;
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    xgc2_math::InertialSample invalid_imu = InertialPoseTestSamples::inertial(1.01);
    invalid_imu.angular_velocity.x() = std::numeric_limits<double>::quiet_NaN();
    eskf.propagateInertial(invalid_imu);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - 1.0) < 1.0e-12);

    const double trace_before_large_dt = eskf.covariance().trace();
    eskf.propagateInertial(InertialPoseTestSamples::inertial(2.0, Eigen::Vector3d(0.0, 0.0, 0.1)));
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - 2.0) < 1.0e-12);
    expect(eskf.state().position.norm() < 1.0e-12);
    expect(eskf.covariance().trace() > trace_before_large_dt);
    expect(xgc2_math::logMap(eskf.state().orientation).norm() > 0.02);

    xgc2_math::PoseMeasurement invalid_pose = InertialPoseTestSamples::pose(2.01, Eigen::Vector3d::Zero());
    invalid_pose.pose.position.z() = std::numeric_limits<double>::quiet_NaN();
    const auto result = eskf.updatePose(invalid_pose);
    expect(!result.accepted);
    expect(!result.innovation_rejected);
    expect(eskf.state().position.norm() < 1.0e-12);
}

void testPose3InertialEskfGyroBiasAndCorrectedPose() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.pose_position_noise_std = 0.01;
    config.pose_orientation_noise_std = 0.01;
    config.gyro_bias_random_walk_std = 1.0e-4;
    config.pose_nis_gate = 1.0e6;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);
    eskf.propagateInertial(InertialPoseTestSamples::inertial(1.1));

    const auto yaw_pose = InertialPoseTestSamples::pose(1.1, Eigen::Vector3d(0.2, 0.0, 0.0),
                                                        xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.0, 0.0, 0.1)));
    const auto result = eskf.updatePose(yaw_pose);
    expect(result.accepted);
    expect(eskf.hasRawProjectedBodyPose());
    expect(std::fabs(eskf.rawProjectedBodyPose().position.x() - 0.2) < 1.0e-12);
    expect((eskf.correctedBodyPose().position - eskf.state().position).norm() < 1.0e-12);
    expect(eskf.state().position.x() > 0.0);
    expect(eskf.covariance().trace() > 0.0);
    expect(eskf.state().orientation.w() >= 0.0);
}

void testPose3InertialEskfPoseUpdateRefreshesDerivedImuState() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.pose_position_noise_std = 0.01;
    config.pose_orientation_noise_std = 0.001;
    config.initial_gyro_bias_variance = 0.5;
    config.pose_nis_gate = 1.0e6;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    const Eigen::Vector3d latest_gyro(0.4, -0.2, 0.1);
    const auto imu1 = InertialPoseTestSamples::inertial(1.1, latest_gyro);
    eskf.propagateInertial(imu1);

    const auto pose = InertialPoseTestSamples::pose(1.1, Eigen::Vector3d::Zero(),
                                                    xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.0, 0.0, 0.2)));
    const auto result = eskf.updatePose(pose);
    expect(result.accepted);
    expect(eskf.state().gyro_bias.norm() > 1.0e-6);
    const Eigen::Vector3d expected_angular_velocity = latest_gyro - eskf.state().gyro_bias;
    expect((eskf.state().angular_velocity - expected_angular_velocity).norm() < 1.0e-12);
}

void testPose3InertialEskfFixedOfflineExtrinsic() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.measurement_frame_to_world.position = Eigen::Vector3d(1.0, -0.5, 0.25);
    config.measurement_frame_to_world.orientation = xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.0, 0.0, 0.2));
    config.body_to_marker.position = Eigen::Vector3d(0.12, -0.03, 0.05);
    config.body_to_marker.orientation = xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.03, -0.02, 0.04));
    config.estimate_extrinsic = false;
    config.pose_nis_gate = 1.0e6;
    config.innovation_position_gate_m = 2.0;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);

    const auto initial_pose = InertialPoseTestSamples::pose(
        1.0, Eigen::Vector3d(0.3, -0.1, 0.2), xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.05, 0.02, -0.03)));
    const xgc2_math::Pose3 expected_marker_world =
        xgc2_math::compose(config.measurement_frame_to_world, initial_pose.pose);
    const xgc2_math::Pose3 expected_body_world =
        xgc2_math::compose(expected_marker_world, xgc2_math::inverse(config.body_to_marker));

    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(initial_pose, &imu0);
    static_assert(xgc2_math::Pose3InertialEskf::kErrorStateDim == 15,
                  "Pose3InertialEskf should not estimate extrinsic states online");
    expect(eskf.covariance().rows() == 15);
    expect(eskf.covariance().cols() == 15);
    expect((eskf.state().position - expected_body_world.position).norm() < 1.0e-12);
    expect(xgc2_math::logMap(eskf.state().orientation.conjugate() * expected_body_world.orientation).norm() < 1.0e-12);
    expect((eskf.state().body_to_marker.position - config.body_to_marker.position).norm() < 1.0e-12);
    expect(xgc2_math::logMap(eskf.state().body_to_marker.orientation.conjugate() * config.body_to_marker.orientation)
               .norm() < 1.0e-12);

    auto update_pose = initial_pose;
    update_pose.stamp_sec = 1.02;
    update_pose.pose.position.x() += 0.02;
    const auto result = eskf.updatePose(update_pose);
    expect(result.accepted);
    expect((eskf.state().body_to_marker.position - config.body_to_marker.position).norm() < 1.0e-12);
    expect(xgc2_math::logMap(eskf.state().body_to_marker.orientation.conjugate() * config.body_to_marker.orientation)
               .norm() < 1.0e-12);
}

void testPose3InertialEskfMultiRateSequentialPoseFusion() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.innovation_position_gate_m = 1.0;
    config.pose_nis_gate = 1.0e6;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    auto imu = InertialPoseTestSamples::inertial(1.017, Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 9.8066));
    eskf.propagateInertial(imu);
    imu.stamp_sec = 1.041;
    eskf.propagateInertial(imu);
    const auto held_state = eskf.state();
    const double held_last_fused_stamp = eskf.lastFusedPoseStampS();

    auto time_jump_pose = InertialPoseTestSamples::pose(1.03, Eigen::Vector3d::Zero());
    time_jump_pose.time_jump = true;
    const auto time_jump_result = eskf.updatePose(time_jump_pose);
    expect(!time_jump_result.accepted);
    expect(time_jump_result.time_alignment_rejected);

    const auto out_of_order_result =
        eskf.updatePose(InertialPoseTestSamples::pose(1.017, Eigen::Vector3d(0.0002, 0.0, 0.0)));
    expect(!out_of_order_result.accepted);
    expect(out_of_order_result.time_alignment_rejected);
    expect(out_of_order_result.reject_reason == xgc2_math::PoseFusionRejectReason::kTimeAlignment);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - held_state.last_inertial_stamp_sec) < 1.0e-12);
    expect(std::fabs(eskf.lastFusedPoseStampS() - held_last_fused_stamp) < 1.0e-12);
    expect((eskf.state().position - held_state.position).norm() < 1.0e-12);

    const auto stale_inertial_pose = InertialPoseTestSamples::pose(1.083, Eigen::Vector3d(0.0035, 0.0, 0.0));
    const auto stale_inertial_result = eskf.updatePose(stale_inertial_pose);
    expect(!stale_inertial_result.accepted);
    expect(stale_inertial_result.time_alignment_rejected);
    expect(stale_inertial_result.reject_reason == xgc2_math::PoseFusionRejectReason::kTimeAlignment);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - held_state.last_inertial_stamp_sec) < 1.0e-12);

    auto imu2 =
        InertialPoseTestSamples::inertial(1.083, Eigen::Vector3d(0.2, -0.1, 0.03), Eigen::Vector3d(1.0, 0.0, 9.8066));
    eskf.propagateInertial(imu2);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - 1.083) < 1.0e-12);
    expect((eskf.state().angular_velocity - imu2.angular_velocity).norm() < 1.0e-12);

    const auto accepted_result = eskf.updatePose(stale_inertial_pose);
    expect(accepted_result.accepted);
    expect(!accepted_result.time_alignment_rejected);
    expect(std::fabs(eskf.state().last_inertial_stamp_sec - 1.083) < 1.0e-12);
    expect(std::fabs(eskf.lastFusedPoseStampS() - 1.083) < 1.0e-12);
    expect((eskf.correctedBodyPose().position - eskf.state().position).norm() < 1.0e-12);

    const auto rejected = eskf.updatePose(InertialPoseTestSamples::pose(0.5, Eigen::Vector3d::Zero()));
    expect(!rejected.accepted);
    expect(rejected.time_alignment_rejected);
    expect(rejected.reject_reason == xgc2_math::PoseFusionRejectReason::kTimeAlignment);
}

void testPose3InertialEskfMeasurementJacobianMatchesFiniteDifference() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.estimate_extrinsic = true;
    config.body_to_marker.position = Eigen::Vector3d(0.17, -0.04, 0.08);
    config.body_to_marker.orientation = xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.07, -0.03, 0.04));

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d(0.4, -0.2, 0.7),
                                                          xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.2, -0.1, 0.3))),
                            nullptr);

    const xgc2_math::RigidBodyState base_state = eskf.state();
    const xgc2_math::Pose3 predicted_marker = xgc2_math::Pose3InertialEskfTestAccess::predictedMarkerPose(base_state);

    xgc2_math::Pose3 residual_pose;
    residual_pose.position = Eigen::Vector3d(0.03, -0.02, 0.01);
    residual_pose.orientation = xgc2_math::expMap(Eigen::Vector3d(1.0e-4, -2.0e-4, 1.5e-4));
    const xgc2_math::Pose3 measured_marker = xgc2_math::compose(predicted_marker, residual_pose);
    const auto innovation =
        xgc2_math::Pose3InertialEskfTestAccess::measurementResidual(predicted_marker, measured_marker);
    const auto analytic =
        xgc2_math::Pose3InertialEskfTestAccess::measurementJacobian(eskf, predicted_marker, innovation);

    constexpr double kEps = 1.0e-6;
    for (int i = 0; i < xgc2_math::Pose3InertialEskf::kErrorStateDim; ++i) {
        xgc2_math::Pose3InertialEskf::ErrorVector delta = xgc2_math::Pose3InertialEskf::ErrorVector::Zero();
        delta(i) = kEps;
        xgc2_math::RigidBodyState perturbed_state = base_state;
        xgc2_math::Pose3InertialEskfTestAccess::injectError(eskf, delta, perturbed_state);
        const xgc2_math::Pose3 perturbed_marker =
            xgc2_math::Pose3InertialEskfTestAccess::predictedMarkerPose(perturbed_state);
        const auto perturbed_innovation =
            xgc2_math::Pose3InertialEskfTestAccess::measurementResidual(perturbed_marker, measured_marker);
        const auto numeric_column = (perturbed_innovation - innovation) / kEps;
        const double error_norm = (numeric_column - analytic.col(i)).norm();
        if (error_norm >= 5.0e-4) {
            std::fprintf(
                stderr,
                "Pose3 H column %d mismatch: error=%g numeric=[%g %g %g %g %g %g] analytic=[%g %g %g %g %g %g]\n", i,
                error_norm, numeric_column(0), numeric_column(1), numeric_column(2), numeric_column(3),
                numeric_column(4), numeric_column(5), analytic(0, i), analytic(1, i), analytic(2, i), analytic(3, i),
                analytic(4, i), analytic(5, i));
        }
        expect(error_norm < 5.0e-4);
    }
}

void testPose3InertialEskfVrpnHealthTransitions() {
    xgc2_math::Pose3InertialEskfConfig config;
    config.innovation_position_gate_m = 0.1;
    config.vrpn_health.fault_after_rejects = 2;
    config.vrpn_health.recovery_after_accepts = 2;

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(config);
    const auto imu0 = InertialPoseTestSamples::inertial(1.0);
    eskf.initializeFromPose(InertialPoseTestSamples::pose(1.0, Eigen::Vector3d::Zero()), &imu0);

    auto result = eskf.updatePose(InertialPoseTestSamples::pose(1.01, Eigen::Vector3d(1.0, 0.0, 0.0)));
    expect(!result.accepted);
    expect(result.innovation_rejected);
    expect(result.vrpn_observation_state == xgc2_math::VrpnObservationState::kSuspected);
    expect(result.filter_health == xgc2_math::FilterHealth::kDegraded);

    result = eskf.updatePose(InertialPoseTestSamples::pose(1.02, Eigen::Vector3d(1.0, 0.0, 0.0)));
    expect(!result.accepted);
    expect(result.vrpn_observation_state == xgc2_math::VrpnObservationState::kFault);
    expect(result.filter_health == xgc2_math::FilterHealth::kImuOnly);
    expect(std::fabs(eskf.vrpnInnovationWindowChiSquare()) < 1.0e-12);

    result = eskf.updatePose(InertialPoseTestSamples::pose(1.03, Eigen::Vector3d::Zero()));
    expect(result.accepted);
    expect(result.reject_reason == xgc2_math::PoseFusionRejectReason::kNone);
    expect(result.vrpn_observation_state != xgc2_math::VrpnObservationState::kFault);
    expect(result.filter_health == xgc2_math::FilterHealth::kDegraded);
    expect(std::fabs(eskf.rawProjectedBodyPose().position.x()) < 1.0e-12);

    result = eskf.updatePose(InertialPoseTestSamples::pose(1.04, Eigen::Vector3d::Zero()));
    expect(result.accepted);
    expect(result.vrpn_observation_state == xgc2_math::VrpnObservationState::kTrusted);
    expect(result.filter_health == xgc2_math::FilterHealth::kNominal);
}

void testTrajectoryAndNmpcProblemContracts() {
    auto angle_error = [](double a, double b) {
        return std::atan2(std::sin(a - b), std::cos(a - b));
    };

    xgc2_math::trajectory::CircleEntryCurveParameters3 circle_entry_params;
    circle_entry_params.duration = 12.0;
    circle_entry_params.origin = Eigen::Vector3d(0.0, 0.0, 3.0);
    circle_entry_params.origin_yaw = 0.4;
    circle_entry_params.entry_duration = 2.0;
    circle_entry_params.circle.radius = 3.0;
    circle_entry_params.circle.line_speed = 2.0;
    circle_entry_params.circle.height = 3.0;
    circle_entry_params.circle.z_amplitude = 0.5;

    xgc2_math::trajectory::CircleEntryCurveEvaluator3 circle_entry(circle_entry_params);
    xgc2_math::trajectory::FlatOutput3 flat;
    expect(circle_entry.evaluate(0.05, flat));
    expect(std::fabs(angle_error(flat.yaw, circle_entry_params.origin_yaw)) < 0.1);
    expect(std::fabs(flat.yaw_rate) < 1.0e-12);
    expect(std::fabs(flat.yaw_accel) < 1.0e-12);
    expect(circle_entry.evaluate(0.5, flat));
    expect(xgc2_math::trajectory::TrajectoryValidator3::finite(flat));
    expect(circle_entry.evaluate(1.8, flat));
    expect(std::fabs(angle_error(flat.yaw, circle_entry_params.origin_yaw)) < 1.0e-12);
    expect(std::fabs(flat.yaw_rate) < 1.0e-12);
    expect(std::fabs(flat.yaw_accel) < 1.0e-12);

    xgc2_math::trajectory::FigureEightCurveEvaluator2 figure_eight;
    xgc2_math::trajectory::PlanarReference2 planar_ref;
    expect(figure_eight.evaluate(0.5, planar_ref));
    expect(xgc2_math::trajectory::TrajectoryValidator2::finite(planar_ref));

    xgc2_math::trajectory::Se2TargetState2 se2_start;
    se2_start.position = Eigen::Vector2d(0.0, 0.0);
    se2_start.yaw = 0.0;
    se2_start.speed = 0.0;
    xgc2_math::trajectory::Se2TargetState2 se2_target;
    se2_target.position = Eigen::Vector2d(2.0, 1.0);
    se2_target.yaw = 0.5;

    xgc2_math::trajectory::Se2TargetTrajectoryOptions2 se2_options;
    se2_options.piece_count = 3;
    se2_options.sample_dt = 0.05;
    se2_options.validation_sample_dt = 0.05;
    se2_options.desired_speed = 1.0;
    se2_options.max_velocity = 5.0;
    se2_options.max_acceleration = 10.0;
    se2_options.max_yaw_rate = 5.0;
    se2_options.max_iterations = 20;
    xgc2_math::trajectory::Se2TargetTrajectoryResult2 se2_result;
    expect(xgc2_math::trajectory::Se2MincoTargetPlanner2().plan(se2_start, se2_target, se2_options, se2_result));
    expect(!se2_result.samples.empty());
    expect(se2_result.samples.front().reference.position.isApprox(se2_start.position, 1.0e-9));
    expect(se2_result.samples.back().reference.position.isApprox(se2_target.position, 1.0e-9));
    expect(std::fabs(angle_error(se2_result.samples.back().reference.yaw, se2_target.yaw)) < 1.0e-9);
    for (const auto& sample : se2_result.samples) {
        expect(xgc2_math::trajectory::TrajectoryValidator2::finite(sample.reference));
        expect(sample.reference.speed <= se2_options.max_velocity + 1.0e-6);
        expect(std::abs(sample.reference.yaw_rate) <= se2_options.max_yaw_rate + 1.0e-6);
        if (sample.reference.speed > 1.0e-3) {
            const double velocity_yaw = std::atan2(sample.reference.velocity.y(), sample.reference.velocity.x());
            expect(std::fabs(angle_error(sample.reference.yaw, velocity_yaw)) < 1.0e-9);
        }
    }

    xgc2_math::trajectory::Se2TargetTrajectoryOptions2 physical_options = se2_options;
    physical_options.max_velocity = 1.5;
    physical_options.max_acceleration = 2.0;
    physical_options.max_yaw_rate = 0.5235;
    physical_options.max_iterations = 80;
    xgc2_math::trajectory::Se2TargetTrajectoryResult2 physical_result;
    expect(
        xgc2_math::trajectory::Se2MincoTargetPlanner2().plan(se2_start, se2_target, physical_options, physical_result));
    expect(!physical_result.samples.empty());
    expect((physical_result.flags &
            (xgc2_math::trajectory::kFlagVelocityLimit | xgc2_math::trajectory::kFlagAccelerationLimit |
             xgc2_math::trajectory::kFlagYawRateLimit)) == 0U);
    for (const auto& sample : physical_result.samples) {
        expect(std::abs(sample.reference.speed) <= physical_options.max_velocity + 1.0e-6);
        expect(std::abs(sample.reference.linear_acceleration) <= physical_options.max_acceleration + 1.0e-6);
        expect(std::abs(sample.reference.yaw_rate) <= physical_options.max_yaw_rate + 1.0e-6);
    }

    xgc2_math::trajectory::Se2TargetState2 yaw_target = se2_start;
    yaw_target.yaw = 1.2;
    xgc2_math::trajectory::Se2TargetTrajectoryResult2 yaw_result;
    expect(xgc2_math::trajectory::Se2MincoTargetPlanner2().plan(se2_start, yaw_target, se2_options, yaw_result));
    expect(!yaw_result.samples.empty());
    expect(!yaw_result.optimized);
    expect(yaw_result.samples.back().reference.position.isApprox(se2_start.position, 1.0e-12));
    expect(std::fabs(angle_error(yaw_result.samples.back().reference.yaw, yaw_target.yaw)) < 1.0e-9);
    expect((yaw_result.samples.back().reference.flags & xgc2_math::trajectory::kFlagLowSpeedSingularity) != 0U);

    xgc2_math::trajectory::LineCurveParameters3 line_params;
    line_params.duration = 2.0;
    line_params.target = Eigen::Vector3d(1.0, 2.0, 3.0);
    xgc2_math::trajectory::LineCurveEvaluator3 line(line_params);
    expect(line.evaluate(2.0, flat));
    expect(flat.position.isApprox(line_params.target, 1.0e-12));
    expect(std::fabs(flat.yaw) < 1.0e-12);
    expect(std::fabs(flat.yaw_rate) < 1.0e-12);
    expect(std::fabs(flat.yaw_accel) < 1.0e-12);

    xgc2_math::trajectory::LemniscateCurveParameters3 lemniscate_params;
    lemniscate_params.radius = 2.0;
    lemniscate_params.omega = 0.5;
    xgc2_math::trajectory::LemniscateCurveEvaluator3 lemniscate(lemniscate_params);
    expect(lemniscate.evaluate(0.0, flat));
    expect(std::fabs(flat.velocity.x() - 1.0) < 1.0e-12);
    expect(std::fabs(flat.velocity.y() - 1.0) < 1.0e-12);
    expect(std::fabs(flat.yaw) < 1.0e-12);
    expect(std::fabs(flat.yaw_rate) < 1.0e-12);

    xgc2_math::trajectory::HelixYzCurveParameters3 helix_yz_params;
    helix_yz_params.radius = 1.0;
    helix_yz_params.omega = 0.8;
    helix_yz_params.linear_scale = 5.0;
    xgc2_math::trajectory::HelixYzCurveEvaluator3 helix_yz(helix_yz_params);
    expect(helix_yz.evaluate(0.0, flat));
    expect(std::fabs(flat.velocity.x() - 0.2) < 1.0e-12);
    expect(std::fabs(flat.yaw) < 1.0e-12);
    xgc2_math::trajectory::HelixXyCurveParameters3 helix_xy_params;
    helix_xy_params.radius = 1.0;
    helix_xy_params.omega = 0.8;
    helix_xy_params.linear_scale = 5.0;
    xgc2_math::trajectory::HelixXyCurveEvaluator3 helix_xy(helix_xy_params);
    expect(helix_xy.evaluate(0.0, flat));
    expect(std::fabs(flat.velocity.z() - 0.2) < 1.0e-12);
    expect(std::fabs(flat.yaw) < 1.0e-12);

    xgc2_math::trajectory::TorusKnotCurveParameters3 torus_params;
    torus_params.omega = 0.6;
    torus_params.scale = 0.4;
    xgc2_math::trajectory::TorusKnotCurveEvaluator3 torus(torus_params);
    expect(torus.evaluate(0.0, flat));
    expect(flat.position.isApprox(Eigen::Vector3d(0.0, -0.4, 1.6), 1.0e-12));
    expect(std::fabs(flat.yaw) < 1.0e-12);
    expect(std::fabs(flat.yaw_rate) < 1.0e-12);

    xgc2_math::trajectory::FigureEightCurveParameters3 figure_eight3_params;
    figure_eight3_params.yaw = -0.2;
    xgc2_math::trajectory::FigureEightCurveEvaluator3 figure_eight3(figure_eight3_params);
    expect(figure_eight3.evaluate(0.5, flat));
    expect(std::fabs(angle_error(flat.yaw, figure_eight3_params.yaw)) < 1.0e-12);
    expect(std::fabs(flat.yaw_rate) < 1.0e-12);

    xgc2_math::control::Se3State se3_state;
    se3_state.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    xgc2_math::control::Se3Control se3_control;
    se3_control.body_z_specific_force = 9.8;
    se3_control.angular_acceleration = Eigen::Vector3d(0.1, 0.2, 0.3);
    const auto se3_x = xgc2_math::control::packState(se3_state);
    const auto se3_u = xgc2_math::control::packControl(se3_control);
    expect(se3_x.size() == 13);
    expect(std::fabs(se3_u(0) - 9.8) < 1.0e-12);
    expect(std::fabs(xgc2_math::control::unpackControl(se3_u).angular_acceleration.z() - 0.3) < 1.0e-12);

    xgc2_math::control::Se2State se2_state;
    se2_state.position = Eigen::Vector2d(1.0, 2.0);
    se2_state.yaw = 4.0;
    se2_state.linear_speed = 1.5;
    xgc2_math::control::Se2Control se2_control;
    se2_control.linear_acceleration = 0.4;
    se2_control.yaw_rate = 0.7;
    const auto se2_x = xgc2_math::control::packState(se2_state);
    const auto se2_u = xgc2_math::control::packControl(se2_control);
    expect(se2_x.size() == 4);
    expect(std::fabs(xgc2_math::control::unpackState(se2_x).yaw - xgc2_math::normalizeAngle(4.0)) < 1.0e-12);
    expect(std::fabs(se2_u(0) - 0.4) < 1.0e-12);
    expect(std::fabs(se2_u(1) - 0.7) < 1.0e-12);
}

} // namespace

int main() {
    testButterworth();
    testExponentialLowPass();
    testSlewRateLimiter();
    testStatusHelpers();
    testTimeDeltaGuard();
    testSampleTimingPredicates();
    testOptionNormalization();
    testDifferentiator();
    testAngleUtilitiesAndDifferentiator();
    testPositionVelocityObserver();
    testAngularPositionVelocityObserver();
    testArrayWrappers();
    testScalarRecursiveLeastSquares();
    testSe3Utilities();
    testSe2Utilities();
    testObservationHealthTrackerFixedWindow();
    testPose2InertialEskf();
    testPose2InertialEskfOutOfOrderPoseDrop();
    testPose3InertialEskfInitialization();
    testPose3InertialEskfUpdatePoseInitializesAndHealthUsesCovariance();
    testPose3InertialEskfStationaryPropagation();
    testPose3InertialEskfMidpointIntegrationPublishesCurrentAngularRate();
    testPose3InertialEskfTimeJumpResets();
    testPose3InertialEskfPoseUpdateAndReject();
    testPose3InertialEskfHighNisWithinAbsoluteGateIsSoftHealth();
    testPose3InertialEskfPredictionErrorDoesNotRejectContinuousVrpn();
    testPose3InertialEskfInvalidAndLargeDtHoldState();
    testPose3InertialEskfGyroBiasAndCorrectedPose();
    testPose3InertialEskfPoseUpdateRefreshesDerivedImuState();
    testPose3InertialEskfFixedOfflineExtrinsic();
    testPose3InertialEskfMultiRateSequentialPoseFusion();
    testPose3InertialEskfMeasurementJacobianMatchesFiniteDifference();
    testPose3InertialEskfVrpnHealthTransitions();
    testTrajectoryAndNmpcProblemContracts();
    return 0;
}
