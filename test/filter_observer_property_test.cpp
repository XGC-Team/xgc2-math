#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgc2_math/estimation/recursive_least_squares.hpp"
#include "xgc2_math/filter/butterworth_filter.hpp"
#include "xgc2_math/filter/exponential_filter.hpp"
#include "xgc2_math/filter/slew_rate_limiter.hpp"
#include "xgc2_math/observer/luenberger_observer.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// Keep numerical checks active in Release builds as well as Debug builds.
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream detail;
        detail.precision(17);
        detail << message << ": actual=" << actual << ", expected=" << expected << ", tolerance=" << tolerance;
        throw std::runtime_error(detail.str());
    }
}

double wrapped(double angle) {
    // Independent circular comparison, without using the library angle helper.
    return std::atan2(std::sin(angle), std::cos(angle));
}

void exponentialPartition() {
    constexpr std::array<double, 6> steps{{1.0e-7, 0.001, 0.007, 0.013, 0.037, 0.09}};
    for (double cutoff : {0.02, 0.7, 8.0, 30.0}) {
        constexpr double initial = -1.7;
        constexpr double target = 2.3;
        xgc2_math::ExponentialLowPass filter(cutoff, initial);
        long double elapsed = 0.0L;
        double previous = initial;
        for (std::size_t i = 0; i < 600; ++i) {
            const double dt = steps[i % steps.size()];
            elapsed += dt;
            const double value = filter.filter(target, dt);
            const double exact =
                target + (initial - target) * std::exp(-2.0 * kPi * cutoff * static_cast<double>(elapsed));
            requireNear(value, exact, 8.0e-13, "variable-step exponential filter disagrees with exact ODE solution");
            require(value >= previous - 1.0e-14 && value <= target + 1.0e-14,
                    "constant-input exponential response must be monotone without overshoot");
            previous = value;
        }
        xgc2_math::ExponentialLowPass unsplit(cutoff, initial);
        requireNear(filter.value(), unsplit.filter(target, static_cast<double>(elapsed)), 8.0e-13,
                    "partitioning the elapsed time changed the constant-input result");

        // A tiny second interval exercises expm1's small-step accuracy.
        for (double first_dt : {1.0e-12, 0.004, 0.17, 10.0}) {
            xgc2_math::ExponentialLowPass whole(cutoff, initial);
            xgc2_math::ExponentialLowPass split(cutoff, initial);
            split.filter(target, first_dt);
            requireNear(split.filter(target, 1.0e-8), whole.filter(target, first_dt + 1.0e-8), 4.0e-15,
                        "exponential semigroup identity failed");
        }
    }
}

void butterworthResponse() {
    constexpr double sample_hz = 200.0;
    constexpr double dt = 1.0 / sample_hz;
    constexpr int settle_samples = 4000;
    constexpr int measured_samples = 4000;
    for (double cutoff : {2.5, 12.0, 80.0}) {
        // The public implementation limits fc to 45 percent of Nyquist.
        const double effective_cutoff = std::min(cutoff, 0.225 * sample_hz);
        for (double frequency : {0.5, 2.5, 12.0, 30.0, 80.0}) {
            xgc2_math::SecondOrderButterworthLowPass filter(cutoff, 0.0);
            long double sine_projection = 0.0L;
            long double cosine_projection = 0.0L;
            for (int n = 0; n < settle_samples + measured_samples; ++n) {
                const double phase = 2.0 * kPi * frequency * static_cast<double>(n) * dt;
                const double value = filter.filter(std::sin(phase), dt);
                require(std::isfinite(value), "Butterworth sweep produced a nonfinite output");
                if (n >= settle_samples) {
                    sine_projection += value * std::sin(phase);
                    cosine_projection += value * std::cos(phase);
                }
            }
            const double sine_gain = static_cast<double>(2.0L * sine_projection / measured_samples);
            const double cosine_gain = static_cast<double>(2.0L * cosine_projection / measured_samples);

            // Independent oracle: analog H(s)=1/(s^2+sqrt(2)*s+1),
            // evaluated at the bilinear/prewarped imaginary frequency.
            // No production filter coefficients are reproduced here.
            const double ratio = std::tan(kPi * frequency / sample_hz) / std::tan(kPi * effective_cutoff / sample_hz);
            const std::complex<double> expected =
                1.0 / std::complex<double>(1.0 - ratio * ratio, std::sqrt(2.0) * ratio);
            requireNear(sine_gain, expected.real(), 2.0e-10, "Butterworth in-phase frequency response mismatch");
            requireNear(cosine_gain, expected.imag(), 2.0e-10, "Butterworth quadrature frequency response mismatch");
            requireNear(std::hypot(sine_gain, cosine_gain), std::abs(expected), 2.0e-10,
                        "Butterworth gain disagrees with analytic transfer function");
        }
    }

    for (double cutoff : {2.5, 12.0}) {
        xgc2_math::SecondOrderButterworthLowPass step_filter(cutoff, 0.0);
        double peak = 0.0;
        for (int n = 0; n < 2000; ++n) {
            const double value = step_filter.filter(1.0, dt);
            require(std::isfinite(value) && value >= -1.0e-12 && value <= 1.06,
                    "Butterworth step response is unstable or has excessive overshoot");
            peak = std::max(peak, value);
        }
        require(peak > 1.03, "second-order Butterworth step must retain its characteristic overshoot");
        requireNear(step_filter.value(), 1.0, 2.0e-12, "Butterworth DC gain is not one");
    }
}

void slewVariableDt() {
    constexpr double rate = 0.75;
    constexpr std::array<double, 5> steps{{1.0e-8, 0.001, 0.007, 0.025, 0.12}};
    xgc2_math::SlewRateLimiter limiter(rate, -0.2);
    double variation = 0.0;
    double elapsed = 0.0;
    for (std::size_t i = 0; i < 1200; ++i) {
        const double dt = steps[i % steps.size()];
        const double target = (i / 71 % 2 == 0 ? 1.5 : -1.2) + 0.2 * std::sin(0.31 * static_cast<double>(i));
        const double previous = limiter.value();
        const double value = limiter.filter(target, dt);
        const double movement = std::abs(value - previous);
        require(std::isfinite(value) && movement <= rate * dt + 3.0e-16,
                "slew limiter violated its physical rate limit at a variable time step");
        require(value >= std::min(previous, target) - 3.0e-16 && value <= std::max(previous, target) + 3.0e-16,
                "slew limiter moved away from or past its target");
        if (std::abs(target - previous) <= rate * dt) {
            requireNear(value, target, 3.0e-16, "reachable target must be attained in one step");
        } else {
            requireNear(movement, rate * dt, 3.0e-16, "distant target must use the entire feasible movement");
        }
        variation += movement;
        elapsed += dt;
    }
    require(variation <= rate * elapsed + 1.0e-11, "slew total variation exceeds rate times elapsed time");

    for (double target : {-8.0, -0.25, 0.4, 0.45, 12.0}) {
        xgc2_math::SlewRateLimiter split(rate, 0.4);
        xgc2_math::SlewRateLimiter whole(rate, 0.4);
        double total_dt = 0.0;
        for (double dt : steps) {
            split.filter(target, dt);
            total_dt += dt;
        }
        requireNear(split.value(), whole.filter(target, total_dt), 3.0e-16,
                    "slew result depends on partitioning constant-target elapsed time");
    }
}

void observerAccelerationGating() {
    xgc2_math::PositionVelocityObserverOptions options;
    options.min_dt_s = 1.0e-6;
    options.max_dt_s = 0.1;
    options.max_position_residual = 0.15;
    xgc2_math::PositionVelocityLuenbergerObserver observer(options);
    constexpr double initial_position = -0.7;
    constexpr double initial_velocity = 0.4;
    constexpr double acceleration = 0.8;
    observer.reset(initial_position, initial_velocity);
    constexpr std::array<double, 4> steps{{0.003, 0.007, 0.013, 0.027}};
    long double elapsed = 0.0L;
    int rejected = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        const double dt = steps[i % steps.size()];
        elapsed += dt;
        const double time = static_cast<double>(elapsed);
        const double truth = initial_position + initial_velocity * time + 0.5 * acceleration * time * time;
        const bool outlier = i % 67 == 31;
        const auto estimate = observer.update(truth + (outlier ? 3.0 : 0.0), dt, acceleration);
        requireNear(estimate.position, truth, 5.0e-12, "accelerated observer disagrees with analytic position");
        requireNear(estimate.velocity, initial_velocity + acceleration * time, 5.0e-12,
                    "accelerated observer disagrees with analytic velocity");
        if (outlier) {
            ++rejected;
            require(!estimate.measurement_accepted &&
                        estimate.status == xgc2_math::PositionVelocityObserverStatus::kHeldOutlier,
                    "outlier must be rejected while the model prediction advances");
            requireNear(estimate.residual, 3.0, 5.0e-12, "outlier residual must be computed after prediction");
        } else {
            require(estimate.measurement_accepted &&
                        estimate.status == xgc2_math::PositionVelocityObserverStatus::kAccepted,
                    "valid sample after a gate event failed to recover");
        }
    }
    require(rejected == 15, "observer gate scenario did not exercise the intended outlier count");
    const double held_position = observer.position();
    const double held_velocity = observer.velocity();
    const auto invalid = observer.update(0.0, 0.2, acceleration);
    require(invalid.status == xgc2_math::PositionVelocityObserverStatus::kHeldInvalidDt,
            "invalid dt must have a distinct status from an outlier");
    requireNear(observer.position(), held_position, 0.0, "invalid dt changed observer position");
    requireNear(observer.velocity(), held_velocity, 0.0, "invalid dt changed observer velocity");

    // Start away from the exact trajectory as well: a predictor with its
    // correction accidentally disabled must not pass the analytic-motion run.
    xgc2_math::PositionVelocityLuenbergerObserver recovering(options);
    recovering.reset(initial_position + 0.1, initial_velocity - 0.2);
    constexpr double recovery_dt = 0.02;
    for (int i = 1; i <= 160; ++i) {
        const double time = i * recovery_dt;
        const double truth = initial_position + initial_velocity * time + 0.5 * acceleration * time * time;
        const auto estimate = recovering.update(truth, recovery_dt, acceleration);
        require(estimate.measurement_accepted, "bounded initial observer error unexpectedly triggered the gate");
        if (i == 160) {
            requireNear(estimate.position, truth, 1.0e-11, "position correction did not remove initial observer error");
            requireNear(estimate.velocity, initial_velocity + acceleration * time, 1.0e-11,
                        "velocity correction did not remove initial observer error");
        }
    }
}

void angularWrap() {
    xgc2_math::PositionVelocityObserverOptions options;
    options.position_gain = 0.35;
    options.velocity_gain = 0.08;
    options.min_dt_s = 1.0e-6;
    options.max_dt_s = 0.1;
    options.max_position_residual = 0.2;
    xgc2_math::AngularPositionVelocityLuenbergerObserver observer(options);
    constexpr double initial_angle = kPi - 0.08;
    constexpr double initial_velocity = 1.7;
    constexpr double acceleration = 0.15;
    constexpr std::array<double, 3> steps{{0.007, 0.012, 0.031}};
    observer.reset(initial_angle, initial_velocity);
    long double elapsed = 0.0L;
    double previous_wrapped = initial_angle;
    int wraps = 0;
    for (std::size_t i = 0; i < 1200; ++i) {
        const double dt = steps[i % steps.size()];
        elapsed += dt;
        const double time = static_cast<double>(elapsed);
        const double truth = initial_angle + initial_velocity * time + 0.5 * acceleration * time * time;
        const double measured_wrapped = wrapped(truth);
        if (std::abs(measured_wrapped - previous_wrapped) > kPi) {
            ++wraps;
        }
        previous_wrapped = measured_wrapped;
        const bool outlier = i % 83 == 17;
        // Equivalent input representatives must not produce innovation spikes.
        const double representative = measured_wrapped + 2.0 * kPi * static_cast<double>(static_cast<int>(i % 5) - 2);
        const auto estimate = observer.update(representative + (outlier ? 0.8 : 0.0), dt, acceleration);
        requireNear(wrapped(estimate.position - truth), 0.0, 3.0e-12,
                    "angular observer lost its physical angle across the branch cut");
        requireNear(estimate.velocity, initial_velocity + acceleration * time, 3.0e-12,
                    "angular wrap caused a velocity spike");
        require(std::abs(estimate.position) <= kPi + 1.0e-14, "observer angle left the principal interval");
        require(estimate.measurement_accepted != outlier, "angular outlier gate accepted the wrong samples");
        require(estimate.status == (outlier ? xgc2_math::PositionVelocityObserverStatus::kHeldOutlier
                                            : xgc2_math::PositionVelocityObserverStatus::kAccepted),
                "angular gate status changed across the branch cut");
        if (outlier) {
            requireNear(estimate.residual, 0.8, 3.0e-12,
                        "angular outlier must report the positive short residual of 0.8 rad");
        }
    }
    require(wraps >= 10, "angular scenario did not exercise repeated full rotations");
    const double held_position = observer.position();
    const double held_velocity = observer.velocity();
    const auto invalid = observer.update(0.0, 0.2, acceleration);
    require(invalid.status == xgc2_math::PositionVelocityObserverStatus::kHeldInvalidDt,
            "invalid angular dt must have a distinct status from an outlier");
    requireNear(observer.position(), held_position, 0.0, "invalid dt changed angular observer position");
    requireNear(observer.velocity(), held_velocity, 0.0, "invalid dt changed angular observer velocity");

    // Nonzero innovation across +pi/-pi. The prediction is pi-0.01,
    // the measurement is -pi+0.03, and the physical residual is +0.04.
    // With alpha=0.35, beta=0.08 and dt=0.05 the posterior must cross
    // the branch cut: angle=-pi+0.004, rate=0.2+0.08*0.04/0.05=0.264.
    // These explicit values also reject a predictor with correction removed.
    xgc2_math::AngularPositionVelocityLuenbergerObserver correcting(options);
    correcting.reset(kPi - 0.02, 0.2);
    const auto corrected = correcting.update(-kPi + 0.03, 0.05);
    require(corrected.measurement_accepted && corrected.status == xgc2_math::PositionVelocityObserverStatus::kAccepted,
            "small physical innovation across the angle branch cut must be accepted");
    requireNear(corrected.residual, 0.04, 2.0e-14, "angular correction used the long or wrong-sign residual");
    requireNear(corrected.position, -kPi + 0.004, 2.0e-14,
                "angular posterior position omitted correction or principal-angle normalization");
    requireNear(corrected.velocity, 0.264, 2.0e-14, "angular posterior velocity omitted or mis-scaled correction");
}

void rlsParameterChange() {
    xgc2_math::ScalarRecursiveLeastSquaresOptions options;
    options.forgetting_factor = 0.985;
    options.initial_covariance = 7.0;
    constexpr double prior = -0.6;
    xgc2_math::ScalarRecursiveLeastSquares adaptive(options);
    adaptive.reset(prior);
    auto no_forgetting_options = options;
    no_forgetting_options.forgetting_factor = 1.0;
    xgc2_math::ScalarRecursiveLeastSquares no_forgetting(no_forgetting_options);
    no_forgetting.reset(prior);
    std::vector<double> regressors;
    std::vector<double> measurements;
    for (std::size_t i = 0; i < 1200; ++i) {
        const double index = static_cast<double>(i);
        const double regressor = 0.8 + 0.3 * std::sin(0.17 * index);
        const double truth = i < 600 ? 1.8 : -0.7;
        const double measurement = truth * regressor + 0.02 * std::sin(0.41 * index) + 0.01 * std::cos(0.13 * index);
        regressors.push_back(regressor);
        measurements.push_back(measurement);
        const auto estimate = adaptive.update(measurement, regressor);
        require(estimate.measurement_accepted, "RLS rejected bounded persistently excited data");
        no_forgetting.update(measurement, regressor);

        if (i < 3 || i % 53 == 0 || i == 599 || i == 1199) {
            // Solve weighted batch LS from the original observations, not from
            // another recursive estimator. The prior precision also decays.
            const long double lambda = options.forgetting_factor;
            const long double prior_weight = std::pow(lambda, static_cast<int>(i + 1));
            long double normal_matrix = prior_weight / options.initial_covariance;
            long double normal_rhs = prior_weight * prior / options.initial_covariance;
            for (std::size_t j = 0; j <= i; ++j) {
                const long double weight = std::pow(lambda, static_cast<int>(i - j));
                normal_matrix += weight * regressors[j] * regressors[j];
                normal_rhs += weight * regressors[j] * measurements[j];
            }
            requireNear(estimate.parameter, static_cast<double>(normal_rhs / normal_matrix), 3.0e-12,
                        "RLS parameter disagrees with the full weighted batch least-squares solution");
            requireNear(estimate.covariance, static_cast<double>(1.0L / normal_matrix), 3.0e-12,
                        "RLS covariance disagrees with the inverse weighted batch normal matrix");
        }
        if (i == 599) {
            requireNear(estimate.parameter, 1.8, 0.02, "RLS did not converge before the parameter change");
        }
    }
    requireNear(adaptive.parameter(), -0.7, 0.02, "forgetting RLS failed to track the changed parameter");
    require(std::abs(no_forgetting.parameter() + 0.7) > 0.7,
            "parameter-change scenario failed to distinguish forgetting from cumulative least squares");
}

struct TestCase {
    const char* name;
    void (*run)();
};

constexpr std::array<TestCase, 6> kCases{{
    {"exponential_partition", exponentialPartition},
    {"butterworth_response", butterworthResponse},
    {"slew_variable_dt", slewVariableDt},
    {"observer_acceleration_gating", observerAccelerationGating},
    {"angular_wrap", angularWrap},
    {"rls_parameter_change", rlsParameterChange},
}};

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: filter_observer_property_test [case_name|--list]\n";
        return 2;
    }
    const std::string selected = argc == 2 ? argv[1] : "";
    if (selected == "--list") {
        for (const auto& test : kCases) {
            std::cout << test.name << '\n';
        }
        return 0;
    }
    bool matched = false;
    int failed = 0;
    for (const auto& test : kCases) {
        if (!selected.empty() && selected != test.name) {
            continue;
        }
        matched = true;
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }
    if (!matched) {
        std::cerr << "unknown case: " << selected << '\n';
        return 2;
    }
    return failed == 0 ? 0 : 1;
}
