// Public-API numerical scenarios. REQUIRE stays active in Release/NDEBUG.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/Eigenvalues>

#include "xgc2_math/estimation/pose3_inertial_eskf.hpp"

namespace {
using Filter = xgc2_math::Pose3InertialEskf;
using Config = xgc2_math::Pose3InertialEskfConfig;
using Vector = Eigen::Vector3d;
using Pose = xgc2_math::Pose3;
constexpr double kGravity = 9.8066;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

xgc2_math::InertialSample imu(double stamp, const Vector& omega = Vector::Zero(),
                              const Vector& force = Vector(0.0, 0.0, kGravity)) {
    xgc2_math::InertialSample sample;
    sample.received = true;
    sample.valid = true;
    sample.stamp_sec = stamp;
    sample.angular_velocity = omega;
    sample.linear_acceleration = force;
    return sample;
}

xgc2_math::PoseMeasurement pose(double stamp, const Pose& value = Pose{}) {
    xgc2_math::PoseMeasurement sample;
    sample.received = true;
    sample.valid = true;
    sample.stamp_sec = stamp;
    sample.pose = value;
    return sample;
}

void requireStateEqual(const xgc2_math::RigidBodyState& a, const xgc2_math::RigidBodyState& b, double tolerance,
                       const std::string& label) {
    require(a.initialized == b.initialized && (a.position - b.position).norm() <= tolerance &&
                (a.velocity - b.velocity).norm() <= tolerance &&
                (a.orientation.coeffs() - b.orientation.coeffs()).norm() <= tolerance &&
                (a.angular_velocity - b.angular_velocity).norm() <= tolerance &&
                (a.linear_acceleration - b.linear_acceleration).norm() <= tolerance &&
                (a.gyro_bias - b.gyro_bias).norm() <= tolerance && (a.accel_bias - b.accel_bias).norm() <= tolerance &&
                std::abs(a.last_inertial_stamp_sec - b.last_inertial_stamp_sec) <= tolerance,
            label + ": state differs");
}

void requireCovariance(const Filter& filter) {
    const auto& p = filter.covariance();
    require(p.allFinite(), "covariance contains non-finite values");
    require((p - p.transpose()).norm() <= 1e-11 * (1.0 + p.norm()), "covariance is asymmetric");
    Eigen::SelfAdjointEigenSolver<Filter::ErrorCovariance> solver(p);
    require(solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -1e-10,
            "covariance is not positive semidefinite");
    require(std::abs(filter.state().covariance_trace - p.trace()) < 1e-10, "reported covariance trace is stale");
    require(std::abs(filter.state().orientation.norm() - 1.0) < 1e-12, "orientation is not a unit quaternion");
}

void testInitializationImuSeed() {
    // updatePose passes its stored IMU to initializeFromPose. Resetting that
    // member before copying the argument silently loses the first reading.
    const auto first = imu(1.0, Vector(0.4, -0.2, 0.1), Vector(0.6, -0.3, kGravity + 0.2));
    Pose initial;
    initial.orientation = xgc2_math::expMap(Vector(0.12, -0.1, 0.3));
    Filter explicit_seed;
    explicit_seed.initializeFromPose(pose(1.0, initial), &first);
    Filter queued_seed;
    queued_seed.propagateInertial(first);
    require(queued_seed.updatePose(pose(1.0, initial)).accepted, "initial pose was rejected");
    requireStateEqual(queued_seed.state(), explicit_seed.state(), 1e-14, "queued IMU initialization");
    const auto second = imu(1.008, Vector(-0.3, 0.6, 0.8), Vector(-0.4, 0.2, kGravity - 0.1));
    explicit_seed.propagateInertial(second);
    queued_seed.propagateInertial(second);
    requireStateEqual(queued_seed.state(), explicit_seed.state(), 1e-13, "first midpoint step");
    require((queued_seed.covariance() - explicit_seed.covariance()).norm() < 1e-13, "IMU seed changed covariance");
    const double dt = second.stamp_sec - first.stamp_sec;
    const Vector omega = 0.5 * (first.angular_velocity + second.angular_velocity);
    const Eigen::Quaterniond expected_q = initial.orientation * xgc2_math::expMap(omega * dt);
    const Eigen::Quaterniond midpoint_q = initial.orientation * xgc2_math::expMap(0.5 * omega * dt);
    const Vector acceleration =
        midpoint_q * (0.5 * (first.linear_acceleration + second.linear_acceleration)) + Vector(0.0, 0.0, -kGravity);
    require(xgc2_math::logMap(expected_q.conjugate() * queued_seed.state().orientation).norm() < 1e-13,
            "first IMU pair did not use mean angular velocity");
    require((queued_seed.state().position - 0.5 * acceleration * dt * dt).norm() < 1e-13,
            "first IMU pair did not use midpoint specific force");
    requireCovariance(queued_seed);
    std::cout << "initialization_queued_vs_explicit_imu=PASS\n";
}

void testFramesAndConfig() {
    Config config;
    config.measurement_frame_to_world.position = Vector(1.0, -2.0, 0.5);
    config.measurement_frame_to_world.orientation = xgc2_math::expMap(Vector(0.1, -0.2, 0.4));
    config.body_to_marker.position = Vector(0.15, -0.04, 0.08);
    config.body_to_marker.orientation = xgc2_math::expMap(Vector(-0.2, 0.05, 0.1));
    Pose expected;
    expected.position = Vector(0.7, 1.1, -0.2);
    expected.orientation = xgc2_math::expMap(Vector(0.3, -0.1, 0.6));
    const Pose measured = xgc2_math::compose(xgc2_math::inverse(config.measurement_frame_to_world),
                                             xgc2_math::compose(expected, config.body_to_marker));
    Filter filter;
    filter.setConfig(config);
    filter.initializeFromPose(pose(1.0, measured));
    require((filter.state().position - expected.position).norm() < 1e-12, "marker/world lever arm transform is wrong");
    require(xgc2_math::logMap(expected.orientation.conjugate() * filter.state().orientation).norm() < 1e-12,
            "marker/world orientation transform is wrong");
    require(filter.hasCorrectedBodyPose() && filter.hasRawProjectedBodyPose(), "initial body poses missing");
    requireCovariance(filter);

    config.gravity_mps2 = std::numeric_limits<double>::quiet_NaN();
    config.accel_noise_std = -1.0;
    config.pose_update_iterations = 99;
    config.inertial_buffer_capacity = 1;
    config.estimate_extrinsic = true;
    const Config normalized = xgc2_math::normalized(config);
    require(normalized.gravity_mps2 == kGravity && normalized.accel_noise_std == 0.35 &&
                normalized.pose_update_iterations == 8 && normalized.inertial_buffer_capacity == 128 &&
                !normalized.estimate_extrinsic && normalized.initial_extrinsic_position_variance == 0.0,
            "config normalization changed its documented fallbacks");
    std::cout << "frames_and_config=PASS\n";
}

void testAnalyticPropagation() {
    const Vector acceleration(0.3, -0.15, 0.08);
    for (const int hz : {50, 100, 200}) {
        for (const bool stationary : {true, false}) {
            Filter filter;
            const Vector a = stationary ? Vector::Zero().eval() : acceleration;
            const Vector force = a + Vector(0.0, 0.0, kGravity);
            const auto seed = imu(1.0, Vector::Zero(), force);
            filter.initializeFromPose(pose(1.0), &seed);
            for (int i = 1; i <= 3 * hz; ++i) {
                const double t = static_cast<double>(i) / hz;
                filter.propagateInertial(imu(1.0 + t, Vector::Zero(), force));
                require((filter.state().position - 0.5 * a * t * t).norm() < 1e-10, "analytic p=at^2/2 mismatch");
                require((filter.state().velocity - a * t).norm() < 1e-10, "analytic v=at mismatch");
            }
            requireCovariance(filter);
        }
    }

    // A large gap is integrated in bounded substeps; it is not silently dropped.
    Filter one_gap, many_steps;
    const auto seed = imu(1.0, Vector::Zero(), acceleration + Vector(0.0, 0.0, kGravity));
    one_gap.initializeFromPose(pose(1.0), &seed);
    many_steps.initializeFromPose(pose(1.0), &seed);
    one_gap.propagateInertial(imu(1.25, seed.angular_velocity, seed.linear_acceleration));
    for (int i = 1; i <= 25; ++i)
        many_steps.propagateInertial(imu(1.0 + 0.01 * i, seed.angular_velocity, seed.linear_acceleration));
    requireStateEqual(one_gap.state(), many_steps.state(), 1e-12, "segmented long IMU interval");
    require((one_gap.covariance() - many_steps.covariance()).norm() < 1e-10, "gap segmentation changed covariance");
    std::cout << "stationary_acceleration_50_100_200hz_and_long_gap=PASS\n";
}

struct Truth {
    Pose body;
    Vector velocity;
};

Truth turnTruth(double time, double a = 0.2, double w = 0.35) {
    Truth truth;
    truth.body.position =
        Vector(a * (1.0 - std::cos(w * time)) / (w * w), a * (time / w - std::sin(w * time) / (w * w)), 0.0);
    truth.body.orientation = xgc2_math::expMap(Vector(0.0, 0.0, w * time));
    truth.velocity = Vector(a * std::sin(w * time) / w, a * (1.0 - std::cos(w * time)) / w, 0.0);
    return truth;
}

void testTurnConvergence() {
    double previous_error = 0.0;
    for (const int hz : {50, 100, 200}) {
        Config config;
        config.max_propagation_dt_s = 0.05;
        Filter filter;
        filter.setConfig(config);
        const auto seed = imu(1.0, Vector(0.0, 0.0, 0.35), Vector(0.2, 0.0, kGravity));
        filter.initializeFromPose(pose(1.0), &seed);
        for (int i = 1; i <= 4 * hz; ++i)
            filter.propagateInertial(
                imu(1.0 + static_cast<double>(i) / hz, seed.angular_velocity, seed.linear_acceleration));
        const Truth truth = turnTruth(4.0);
        const double error = (filter.state().position - truth.body.position).norm();
        require(error < 2e-5, "constant-turn analytic position error too large");
        require((filter.state().velocity - truth.velocity).norm() < 1e-5,
                "constant-turn analytic velocity error too large");
        require(xgc2_math::logMap(truth.body.orientation.conjugate() * filter.state().orientation).norm() < 1e-12,
                "constant-turn attitude is not exact");
        if (previous_error > 0.0)
            require(error < 0.3 * previous_error, "midpoint convergence is not second order");
        previous_error = error;
        requireCovariance(filter);
        std::cout << "turn_imu_hz=" << hz << " final_position_error_m=" << error << '\n';
    }
}

void testNoiseUnitsAcrossRates() {
    // A free-fall, nonrotating fixture decouples orientation from translation.
    // Independent continuous-white and sample-held acceleration formulas give
    // exact p/v covariance blocks; tiny initial variances isolate process noise.
    const double sigma = 0.4;
    const double duration = 2.0;
    for (const bool density : {false, true}) {
        for (const int hz : {50, 100, 200}) {
            Config config;
            config.max_propagation_dt_s = 0.05;
            config.imu_noise_std_is_density = density;
            config.accel_noise_std = sigma;
            config.gyro_noise_std = 1e-20;
            config.gyro_bias_random_walk_std = 0.0;
            config.accel_bias_random_walk_std = 0.0;
            config.initial_position_variance = 1e-20;
            config.initial_velocity_variance = 1e-20;
            config.initial_orientation_variance = 1e-20;
            config.initial_gyro_bias_variance = 1e-20;
            config.initial_accel_bias_variance = 1e-20;
            Filter filter;
            filter.setConfig(config);
            const auto seed = imu(1.0, Vector::Zero(), Vector::Zero());
            filter.initializeFromPose(pose(1.0), &seed);
            for (int i = 1; i <= static_cast<int>(duration * hz); ++i)
                filter.propagateInertial(imu(1.0 + static_cast<double>(i) / hz, Vector::Zero(), Vector::Zero()));
            const double dt = 1.0 / hz;
            const double q = sigma * sigma;
            const double pp = density
                                  ? q * duration * duration * duration / 3.0
                                  : q * (dt * duration * duration * duration / 3.0 - duration * dt * dt * dt / 12.0);
            const double pv = q * duration * duration / 2.0 * (density ? 1.0 : dt);
            const double vv = q * duration * (density ? 1.0 : dt);
            const auto& p = filter.covariance();
            for (int axis = 0; axis < 3; ++axis) {
                require(std::abs(p(axis, axis) - pp) < 1e-11, "acceleration-noise position covariance has wrong units");
                require(std::abs(p(axis, axis + 3) - pv) < 1e-11, "acceleration-noise p/v covariance has wrong units");
                require(std::abs(p(axis + 3, axis + 3) - vv) < 1e-11,
                        "acceleration-noise velocity covariance has wrong units");
            }
            requireCovariance(filter);
        }
    }
    std::cout << "density_vs_sample_noise_covariance_50_100_200hz=PASS\n";
}

// std::mt19937::operator() advances engine state; the reference cannot be const.
// cppcheck-suppress constParameterReference
double normal(std::mt19937& rng) {
    // Box-Muller with explicit mt19937 draws, independent of library-specific
    // std::normal_distribution implementations. Bounds avoid log(0).
    const double u = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
    const double v = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
    return std::sqrt(-2.0 * std::log(u)) * std::cos(6.2831853071795864769 * v);
}

Vector normalVector(std::mt19937& rng) {
    const double x = normal(rng), y = normal(rng), z = normal(rng);
    return Vector(x, y, z);
}

void testNoisyFusionAndDropout() {
    for (const int hz : {100, 200}) {
        for (const bool dropout : {false, true}) {
            Config config;
            config.imu_noise_std_is_density = true;
            Filter filter, inertial_only;
            filter.setConfig(config);
            inertial_only.setConfig(config);
            std::mt19937 rng(20261010u + static_cast<unsigned>(hz) + static_cast<unsigned>(dropout));
            const auto seed = imu(1.0, Vector(0.0, 0.0, 0.35), Vector(0.2, 0.0, kGravity));
            filter.initializeFromPose(pose(1.0), &seed);
            inertial_only.initializeFromPose(pose(1.0), &seed);
            double position_square = 0.0, baseline_square = 0.0, velocity_square = 0.0, orientation_square = 0.0;
            double recovery_square = 0.0;
            int recovery_count = 0, accepted = 0;
            for (int i = 1; i <= 6 * hz; ++i) {
                const double t = static_cast<double>(i) / hz;
                const Truth truth = turnTruth(t);
                const Vector gyro = seed.angular_velocity + Vector(0.001, -0.0008, 0.002) + 0.003 * normalVector(rng);
                const Vector force = seed.linear_acceleration + Vector(0.04, -0.03, 0.02) + 0.06 * normalVector(rng);
                const auto reading = imu(1.0 + t, gyro, force);
                filter.propagateInertial(reading);
                inertial_only.propagateInertial(reading);
                if (i % (hz / 20) == 0 && !(dropout && t >= 2.0 && t <= 3.2)) {
                    Pose measured = truth.body;
                    measured.position += 0.008 * normalVector(rng);
                    measured.orientation = measured.orientation * xgc2_math::expMap(0.004 * normalVector(rng));
                    require(filter.updatePose(pose(1.0 + t, measured)).accepted, "valid noisy pose rejected");
                    ++accepted;
                    requireCovariance(filter);
                }
                const double p2 = (filter.state().position - truth.body.position).squaredNorm();
                position_square += p2;
                baseline_square += (inertial_only.state().position - truth.body.position).squaredNorm();
                velocity_square += (filter.state().velocity - truth.velocity).squaredNorm();
                orientation_square +=
                    xgc2_math::logMap(truth.body.orientation.conjugate() * filter.state().orientation).squaredNorm();
                if (t >= 3.7) {
                    recovery_square += p2;
                    ++recovery_count;
                }
            }
            const double pr = std::sqrt(position_square / (6 * hz));
            const double br = std::sqrt(baseline_square / (6 * hz));
            const double vr = std::sqrt(velocity_square / (6 * hz));
            const double qr = std::sqrt(orientation_square / (6 * hz));
            require(accepted >= 90, "insufficient fused poses in noise/dropout scenario");
            require(pr < 0.12 && vr < 0.25 && qr < 0.04, "noisy ESKF error exceeded physical regression bounds");
            require(pr < 0.5 * br, "pose fusion did not halve biased inertial position RMSE");
            require(std::sqrt(recovery_square / recovery_count) < 0.04, "pose dropout did not recover within 4 cm");
            requireCovariance(filter);
            std::cout << "noise_imu_hz=" << hz << " dropout=" << dropout << " position_rmse_m=" << pr
                      << " baseline_rmse_m=" << br << " velocity_rmse_mps=" << vr << '\n';
        }
    }
}

void testDelayedReplayEquivalence() {
    Config config;
    config.apply_pose_covariance_floor = false;
    config.max_propagation_dt_s = 0.005;
    Filter in_order, delayed;
    in_order.setConfig(config);
    delayed.setConfig(config);
    const auto seed = imu(1.0, Vector(0.0, 0.0, 0.35), Vector(0.2, 0.0, kGravity));
    in_order.initializeFromPose(pose(1.0), &seed);
    delayed.initializeFromPose(pose(1.0), &seed);
    Pose correction = turnTruth(0.1).body;
    correction.position += Vector(0.015, -0.007, 0.003);
    correction.orientation = correction.orientation * xgc2_math::expMap(Vector(0.01, -0.02, 0.03));
    for (int i = 1; i <= 40; ++i) {
        const auto reading = imu(1.0 + 0.005 * i, seed.angular_velocity, seed.linear_acceleration);
        in_order.propagateInertial(reading);
        delayed.propagateInertial(reading);
        if (i == 20)
            require(in_order.updatePose(pose(1.1, correction)).accepted, "in-order pose rejected");
    }
    require(delayed.updatePose(pose(1.1, correction)).accepted, "within-window delayed pose rejected");
    requireStateEqual(delayed.state(), in_order.state(), 1e-10, "rewind/replay vs in-order fusion");
    require((delayed.covariance() - in_order.covariance()).norm() < 1e-10, "rewind/replay changed covariance");
    require(std::abs(delayed.state().last_inertial_stamp_sec - 1.2) < 1e-12, "replay did not restore IMU endpoint");
    requireCovariance(delayed);
    std::cout << "delayed_pose_replay_matches_in_order=PASS\n";
}

void testInvalidAndTimeBoundaries() {
    Filter filter;
    const auto seed = imu(1.0);
    filter.initializeFromPose(pose(1.0), &seed);
    filter.propagateInertial(imu(1.01));
    const auto before = filter.state();
    const auto covariance = filter.covariance();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const double bad_stamp : {1.01, 1.0, 1.010005, nan, inf}) {
        filter.propagateInertial(imu(bad_stamp, Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)));
        requireStateEqual(filter.state(), before, 0.0, "ignored IMU timestamp");
        require((filter.covariance() - covariance).norm() == 0.0, "ignored IMU changed covariance");
    }
    auto invalid = imu(1.02);
    invalid.linear_acceleration.x() = nan;
    filter.propagateInertial(invalid);
    requireStateEqual(filter.state(), before, 0.0, "invalid IMU force");
    for (int kind = 0; kind < 5; ++kind) {
        auto measurement = pose(1.01);
        if (kind == 0)
            measurement.pose.position.z() = nan;
        if (kind == 1)
            measurement.pose.orientation.x() = inf;
        if (kind == 2)
            measurement.stamp_sec = nan;
        if (kind == 3)
            measurement.received = false;
        if (kind == 4)
            measurement.valid = false;
        const auto result = filter.updatePose(measurement);
        require(!result.accepted && result.reject_reason == xgc2_math::PoseFusionRejectReason::kInvalidInput,
                "invalid pose did not report invalid input");
        requireStateEqual(filter.state(), before, 0.0, "invalid pose");
        require((filter.covariance() - covariance).norm() == 0.0, "invalid pose changed covariance");
    }
    for (const double bad_stamp : {0.5, 1.5}) {
        const auto result = filter.updatePose(pose(bad_stamp));
        require(!result.accepted && result.time_alignment_rejected, "outside-window pose accepted");
        requireStateEqual(filter.state(), before, 0.0, "outside-window pose");
    }
    Pose jumped;
    jumped.position.x() = 4.0;
    const auto jump = filter.updatePose(pose(1.01, jumped));
    require(!jump.accepted && jump.innovation_rejected, "raw position jump accepted");
    requireStateEqual(filter.state(), before, 0.0, "raw jump");
    require((filter.covariance() - covariance).norm() == 0.0, "raw jump changed covariance");
    auto time_jump = pose(1.02);
    time_jump.time_jump = true;
    require(filter.updatePose(time_jump).time_alignment_rejected, "pose time jump accepted");
    requireStateEqual(filter.state(), before, 0.0, "pose time jump");
    auto reset_imu = imu(2.0);
    reset_imu.time_jump = true;
    filter.propagateInertial(reset_imu);
    require(!filter.initialized() && !filter.hasCorrectedBodyPose() && !filter.hasRawProjectedBodyPose(),
            "IMU time jump did not reset filter");
    require(filter.updatePose(pose(2.0)).accepted, "filter did not reinitialize after IMU time jump");
    requireCovariance(filter);
    std::cout << "invalid_inputs_timestamps_gates_and_reset=PASS\n";
}

void testObservationWindowEdges() {
    // Binary-exact intervals avoid confusing an equality policy with rounding.
    for (const double offset : {-0.125, 0.125}) {
        for (const bool outside : {false, true}) {
            Config config;
            config.pose_max_late_s = 0.125;
            config.pose_max_early_s = 0.125;
            Filter filter;
            filter.setConfig(config);
            const auto seed = imu(1.0);
            filter.initializeFromPose(pose(1.0), &seed);
            for (int i = 1; i <= 32; ++i)
                filter.propagateInertial(imu(1.0 + i / 128.0));
            double stamp = 1.25 + offset;
            if (outside)
                stamp = std::nextafter(stamp, offset < 0 ? 0.0 : 2.0);
            const auto before = filter.state();
            const auto covariance = filter.covariance();
            const auto update = filter.updatePose(pose(stamp));
            require(update.accepted == !outside && update.time_alignment_rejected == outside,
                    "pose window equality/one-ULP-outside policy changed");
            require(filter.state().last_inertial_stamp_sec == 1.25, "pose window edge moved the IMU clock");
            if (outside) {
                requireStateEqual(filter.state(), before, 0.0, "one-ULP outside pose window");
                require((filter.covariance() - covariance).norm() == 0.0, "outside-window pose changed covariance");
            }
            requireCovariance(filter);
        }
    }
    Config config;
    config.inertial_buffer_capacity = 2;
    Filter filter;
    filter.setConfig(config);
    const auto seed = imu(1.0);
    filter.initializeFromPose(pose(1.0), &seed);
    for (int i = 1; i <= 4; ++i)
        filter.propagateInertial(imu(1.0 + 0.01 * i));
    const auto before = filter.state();
    const auto covariance = filter.covariance();
    const auto missing_history = filter.updatePose(pose(1.01));
    require(!missing_history.accepted && missing_history.time_alignment_rejected,
            "within-window pose with evicted history was accepted");
    requireStateEqual(filter.state(), before, 0.0, "missing rewind history");
    require((filter.covariance() - covariance).norm() == 0.0, "missing history changed covariance");
    std::cout << "observation_window_equalities_and_history_eviction=PASS\n";
}

void testVelocityUpdate() {
    Config config;
    Filter filter;
    filter.setConfig(config);
    const auto seed = imu(1.0);
    filter.initializeFromPose(pose(1.0), &seed);
    xgc2_math::VelocityMeasurement measurement;
    measurement.received = true;
    measurement.valid = true;
    measurement.stamp_sec = 1.0;
    measurement.velocity = Vector(1.0, -0.2, 0.3);
    require(filter.updateVelocity(measurement).accepted, "current-time velocity measurement was rejected");
    const double prior = config.initial_velocity_variance;
    const double noise = config.velocity_noise_std * config.velocity_noise_std;
    const double gain = prior / (prior + noise);
    require((filter.state().velocity - gain * measurement.velocity).norm() < 1e-12,
            "velocity update differs from scalar Gaussian conditioning");
    require(
        (filter.covariance().block<3, 3>(3, 3) - Eigen::Matrix3d::Identity() * prior * noise / (prior + noise)).norm() <
            1e-12,
        "velocity posterior covariance is incorrect");
    requireCovariance(filter);
    const auto before = filter.state();
    const auto covariance = filter.covariance();
    measurement.velocity.x() = std::numeric_limits<double>::quiet_NaN();
    require(filter.updateVelocity(measurement).reject_reason == xgc2_math::PoseFusionRejectReason::kInvalidInput,
            "NaN velocity was not rejected");
    measurement.velocity = Vector(100.0, 0.0, 0.0);
    require(filter.updateVelocity(measurement).innovation_rejected, "velocity outlier gate failed");
    measurement.velocity = Vector::Zero();
    measurement.stamp_sec = 0.0;
    require(filter.updateVelocity(measurement).time_alignment_rejected, "late velocity measurement was accepted");
    requireStateEqual(filter.state(), before, 0.0, "rejected velocity measurement");
    require((filter.covariance() - covariance).norm() == 0.0, "rejected velocity changed covariance");
    std::cout << "velocity_posterior_and_reject_invariance=PASS\n";
}
} // namespace

int main() {
    std::cout << std::setprecision(12);
    try {
        testInitializationImuSeed();
        testFramesAndConfig();
        testAnalyticPropagation();
        testTurnConvergence();
        testNoiseUnitsAcrossRates();
        testNoisyFusionAndDropout();
        testDelayedReplayEquivalence();
        testInvalidAndTimeBoundaries();
        testObservationWindowEdges();
        testVelocityUpdate();
        std::cout << "pose3_eskf_scenarios_test=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pose3_eskf_scenarios_test=FAIL: " << error.what() << '\n';
        return 1;
    }
}
