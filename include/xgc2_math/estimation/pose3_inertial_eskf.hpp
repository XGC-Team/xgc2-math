#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "xgc2_math/estimation/health.hpp"
#include "xgc2_math/geometry/se3.hpp"

namespace xgc2_math {

struct Pose3InertialEskfConfig {
    double gravity_mps2{9.8066};
    Pose3 measurement_frame_to_world{};
    Pose3 body_to_marker{};
    // Retained for config compatibility. Pose3InertialEskf is the fixed-extrinsic
    // flight estimator; online extrinsic calibration belongs in a separate estimator.
    bool estimate_extrinsic{false};

    double accel_noise_std{0.35};
    double gyro_noise_std{0.03};
    double pose_position_noise_std{0.01};
    double pose_orientation_noise_std{0.01};
    double velocity_noise_std{0.05};
    double gyro_bias_random_walk_std{1.0e-4};
    double accel_bias_random_walk_std{1.0e-3};
    double extrinsic_position_random_walk_std{1.0e-5};
    double extrinsic_orientation_random_walk_std{1.0e-5};
    double innovation_position_gate_m{1.5};
    double innovation_orientation_gate_rad{0.8};
    double velocity_innovation_gate_mps{3.0};
    double pose_nis_gate{22.5};
    double covariance_high_threshold{100.0};
    double max_propagation_dt_s{0.01};
    // Observation time window versus the IMU clock. Pose updates never move
    // last_inertial_stamp_sec. VRPN at 30 Hz or 120 Hz uses the same rule:
    // apply at the current IMU state, or reject if the stamp is far stale/future.
    double pose_max_late_s{0.12};
    double pose_max_early_s{0.12};
    double initial_position_variance{0.01};
    double initial_velocity_variance{0.1};
    double initial_orientation_variance{0.01};
    double initial_gyro_bias_variance{0.01};
    double initial_accel_bias_variance{0.1};
    double initial_extrinsic_position_variance{1.0e-8};
    double initial_extrinsic_orientation_variance{1.0e-8};
    std::size_t inertial_buffer_capacity{128};
    ObservationHealthConfig vrpn_health{};
};

struct InertialSample {
    bool received{false};
    bool valid{false};
    bool time_jump{false};
    Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
    double stamp_sec{0.0};
    double last_dt_sec{0.0};
    double estimated_rate_hz{0.0};
};

struct PoseMeasurement {
    bool received{false};
    bool valid{false};
    bool time_jump{false};
    Pose3 pose{};
    double stamp_sec{0.0};
    double last_dt_sec{0.0};
    double estimated_rate_hz{0.0};
};

struct VelocityMeasurement {
    bool received{false};
    bool valid{false};
    bool time_jump{false};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    double stamp_sec{0.0};
    double last_dt_sec{0.0};
    double estimated_rate_hz{0.0};
};

struct RigidBodyState {
    bool initialized{false};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d accel_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gravity{0.0, 0.0, -9.8066};
    Pose3 body_to_marker{};
    double last_inertial_stamp_sec{0.0};
    double last_pose_stamp_sec{0.0};
    double covariance_trace{1.0};
};

namespace pose3_inertial_eskf_detail {

inline double positiveOr(double value, double fallback) {
    return std::isfinite(value) && value > 0.0 ? value : fallback;
}

inline double nonNegativeOr(double value, double fallback) {
    return std::isfinite(value) && value >= 0.0 ? value : fallback;
}

inline bool validInertialSample(const InertialSample& sample) {
    return sample.received && sample.valid && std::isfinite(sample.stamp_sec) && isFinite(sample.angular_velocity) &&
           isFinite(sample.linear_acceleration);
}

inline bool validPoseMeasurement(const PoseMeasurement& sample) {
    return sample.received && sample.valid && std::isfinite(sample.stamp_sec) && isFinite(sample.pose);
}

inline bool validVelocityMeasurement(const VelocityMeasurement& sample) {
    return sample.received && sample.valid && std::isfinite(sample.stamp_sec) && isFinite(sample.velocity);
}

inline Eigen::Matrix3d skewMatrix(const Eigen::Vector3d& value) {
    Eigen::Matrix3d result;
    result(0, 0) = 0.0;
    result(0, 1) = -value.z();
    result(0, 2) = value.y();
    result(1, 0) = value.z();
    result(1, 1) = 0.0;
    result(1, 2) = -value.x();
    result(2, 0) = -value.y();
    result(2, 1) = value.x();
    result(2, 2) = 0.0;
    return result;
}

inline Eigen::Matrix<double, 6, 1> normalizedInnovation(Eigen::Matrix<double, 6, 1> value) {
    value.tail<3>() = logMap(expMap(value.tail<3>()));
    return value;
}

} // namespace pose3_inertial_eskf_detail

inline void normalize(Pose3InertialEskfConfig& config) {
    config.gravity_mps2 = pose3_inertial_eskf_detail::positiveOr(config.gravity_mps2, 9.8066);
    config.measurement_frame_to_world.orientation = normalizedQuaternion(config.measurement_frame_to_world.orientation);
    config.body_to_marker.orientation = normalizedQuaternion(config.body_to_marker.orientation);
    config.accel_noise_std = pose3_inertial_eskf_detail::positiveOr(config.accel_noise_std, 0.35);
    config.gyro_noise_std = pose3_inertial_eskf_detail::positiveOr(config.gyro_noise_std, 0.03);
    config.pose_position_noise_std = pose3_inertial_eskf_detail::positiveOr(config.pose_position_noise_std, 0.01);
    config.pose_orientation_noise_std = pose3_inertial_eskf_detail::positiveOr(config.pose_orientation_noise_std, 0.01);
    config.velocity_noise_std = pose3_inertial_eskf_detail::positiveOr(config.velocity_noise_std, 0.05);
    config.gyro_bias_random_walk_std =
        pose3_inertial_eskf_detail::nonNegativeOr(config.gyro_bias_random_walk_std, 1.0e-4);
    config.accel_bias_random_walk_std =
        pose3_inertial_eskf_detail::nonNegativeOr(config.accel_bias_random_walk_std, 1.0e-3);
    config.estimate_extrinsic = false;
    config.extrinsic_position_random_walk_std = 0.0;
    config.extrinsic_orientation_random_walk_std = 0.0;
    config.innovation_position_gate_m = pose3_inertial_eskf_detail::positiveOr(config.innovation_position_gate_m, 1.5);
    config.innovation_orientation_gate_rad =
        pose3_inertial_eskf_detail::positiveOr(config.innovation_orientation_gate_rad, 0.8);
    config.velocity_innovation_gate_mps =
        pose3_inertial_eskf_detail::positiveOr(config.velocity_innovation_gate_mps, 3.0);
    config.pose_nis_gate = pose3_inertial_eskf_detail::positiveOr(config.pose_nis_gate, 22.5);
    config.covariance_high_threshold = pose3_inertial_eskf_detail::positiveOr(config.covariance_high_threshold, 100.0);
    config.max_propagation_dt_s = pose3_inertial_eskf_detail::positiveOr(config.max_propagation_dt_s, 0.01);
    config.pose_max_late_s = pose3_inertial_eskf_detail::positiveOr(config.pose_max_late_s, 0.12);
    config.pose_max_early_s = pose3_inertial_eskf_detail::positiveOr(config.pose_max_early_s, 0.12);
    config.initial_position_variance = pose3_inertial_eskf_detail::positiveOr(config.initial_position_variance, 0.01);
    config.initial_velocity_variance = pose3_inertial_eskf_detail::positiveOr(config.initial_velocity_variance, 0.1);
    config.initial_orientation_variance =
        pose3_inertial_eskf_detail::positiveOr(config.initial_orientation_variance, 0.01);
    config.initial_gyro_bias_variance = pose3_inertial_eskf_detail::positiveOr(config.initial_gyro_bias_variance, 0.01);
    config.initial_accel_bias_variance =
        pose3_inertial_eskf_detail::positiveOr(config.initial_accel_bias_variance, 0.1);
    config.initial_extrinsic_position_variance = 0.0;
    config.initial_extrinsic_orientation_variance = 0.0;
    // Compatibility-only field. Pose3InertialEskf is a sequential multi-rate estimator
    // and does not keep an internal history buffer.
    config.inertial_buffer_capacity = config.inertial_buffer_capacity == 0u ? 1u : config.inertial_buffer_capacity;
}

inline Pose3InertialEskfConfig normalized(Pose3InertialEskfConfig config) {
    normalize(config);
    return config;
}

struct Pose3InertialEskfTestAccess;

class Pose3InertialEskf {
  public:
    static constexpr int kErrorStateDim = 15;
    using ErrorVector = Eigen::Matrix<double, kErrorStateDim, 1>;
    using ErrorCovariance = Eigen::Matrix<double, kErrorStateDim, kErrorStateDim>;
    using MeasurementVector = Eigen::Matrix<double, 6, 1>;
    using MeasurementMatrix = Eigen::Matrix<double, 6, kErrorStateDim>;
    using MeasurementCovariance = Eigen::Matrix<double, 6, 6>;

    struct PoseUpdateResult {
        bool accepted{false};
        bool innovation_rejected{false};
        bool time_alignment_rejected{false};
        double position_innovation_norm{0.0};
        double orientation_innovation_norm{0.0};
        double mahalanobis_distance{0.0};
        double innovation_window_chi_square{0.0};
        VrpnObservationState vrpn_observation_state{VrpnObservationState::kTrusted};
        FilterHealth filter_health{FilterHealth::kLost};
        PoseFusionRejectReason reject_reason{PoseFusionRejectReason::kNone};
    };

    struct VelocityUpdateResult {
        bool accepted{false};
        bool innovation_rejected{false};
        bool time_alignment_rejected{false};
        double velocity_innovation_norm{0.0};
        PoseFusionRejectReason reject_reason{PoseFusionRejectReason::kNone};
    };

    Pose3InertialEskf() { reset(); }

    void setConfig(const Pose3InertialEskfConfig& config) {
        config_ = normalized(config);
        vrpn_health_.setConfig(config_.vrpn_health);
        reset();
    }

    void reset() {
        state_ = RigidBodyState{};
        state_.gravity = Eigen::Vector3d(0.0, 0.0, -config_.gravity_mps2);
        state_.body_to_marker = config_.body_to_marker;
        resetCovariance();
        corrected_body_pose_ = Pose3{};
        raw_projected_body_pose_ = Pose3{};
        last_raw_body_measurement_pose_ = Pose3{};
        has_corrected_body_pose_ = false;
        has_raw_projected_body_pose_ = false;
        has_last_raw_body_measurement_pose_ = false;
        last_inertial_ = InertialSample{};
        has_last_inertial_ = false;
        vrpn_health_.reset();
        last_fused_pose_stamp_sec_ = 0.0;
        last_raw_pose_measurement_stamp_sec_ = 0.0;
    }

    void initializeFromPose(const PoseMeasurement& pose, const InertialSample* inertial = nullptr) {
        if (!pose3_inertial_eskf_detail::validPoseMeasurement(pose)) {
            return;
        }

        last_inertial_ = InertialSample{};
        has_last_inertial_ = false;
        resetCovariance();
        const Pose3 marker_world = markerPoseInWorld(pose.pose);
        const Pose3 body_world = bodyPoseFromMarkerPose(marker_world, config_.body_to_marker);
        state_.position = body_world.position;
        state_.velocity = Eigen::Vector3d::Zero();
        state_.orientation = normalizedQuaternion(body_world.orientation);
        state_.gravity = Eigen::Vector3d(0.0, 0.0, -config_.gravity_mps2);
        state_.body_to_marker = config_.body_to_marker;
        state_.last_pose_stamp_sec = pose.stamp_sec;
        state_.last_inertial_stamp_sec = pose.stamp_sec;
        if (inertial != nullptr && pose3_inertial_eskf_detail::validInertialSample(*inertial)) {
            last_inertial_ = *inertial;
            last_inertial_.stamp_sec = pose.stamp_sec;
            state_.angular_velocity = last_inertial_.angular_velocity - state_.gyro_bias;
            state_.linear_acceleration =
                state_.orientation.toRotationMatrix() * (last_inertial_.linear_acceleration - state_.accel_bias) +
                state_.gravity;
            has_last_inertial_ = true;
        }
        state_.covariance_trace = covariance_.trace();
        state_.initialized = true;
        corrected_body_pose_ = bodyPoseFromState(state_);
        raw_projected_body_pose_ = body_world;
        rememberRawPoseMeasurement(body_world, pose.stamp_sec);
        has_corrected_body_pose_ = true;
        has_raw_projected_body_pose_ = true;
        vrpn_health_.reset();
        last_fused_pose_stamp_sec_ = pose.stamp_sec;
    }

    void propagateInertial(const InertialSample& inertial) {
        if (!pose3_inertial_eskf_detail::validInertialSample(inertial)) {
            return;
        }
        if (inertial.time_jump) {
            reset();
            return;
        }

        if (!state_.initialized) {
            state_.last_inertial_stamp_sec = inertial.stamp_sec;
            last_inertial_ = inertial;
            has_last_inertial_ = true;
            return;
        }

        const double dt = inertial.stamp_sec - state_.last_inertial_stamp_sec;
        if (!std::isfinite(dt) || dt <= kMinDt) {
            return;
        }

        if (has_last_inertial_) {
            propagateToStampUsingInertialPair(inertial, inertial.stamp_sec);
        } else {
            propagateToStamp(inertial, inertial.stamp_sec);
            last_inertial_ = inertial;
            has_last_inertial_ = true;
        }
    }

    PoseUpdateResult updatePose(const PoseMeasurement& pose) {
        PoseUpdateResult result;
        if (!pose3_inertial_eskf_detail::validPoseMeasurement(pose)) {
            result.reject_reason = PoseFusionRejectReason::kInvalidInput;
            stampResultHealth(result);
            return result;
        }
        if (pose.time_jump) {
            vrpn_health_.reset();
            result.time_alignment_rejected = true;
            result.reject_reason = PoseFusionRejectReason::kTimeAlignment;
            vrpn_health_.recordRejected();
            stampResultHealth(result);
            return result;
        }
        // IMU is the only process clock. A pose is an observation of the current
        // IMU-time state, whether VRPN is slower or faster than the IMU. Do not
        // hold-propagate IMU to the pose stamp (that would drop later IMU samples
        // when VRPN is 120 Hz). Do not estimate link delay or rewrite stamps.
        if (!state_.initialized) {
            initializeFromPose(pose, has_last_inertial_ ? &last_inertial_ : nullptr);
            result.accepted = state_.initialized;
            stampResultHealth(result);
            return result;
        }
        if (!measurementStampWithinImuWindow(pose.stamp_sec)) {
            result.time_alignment_rejected = true;
            result.reject_reason = PoseFusionRejectReason::kTimeAlignment;
            vrpn_health_.recordRejected();
            stampResultHealth(result);
            return result;
        }
        return updatePoseAtCurrentState(pose);
    }

    VelocityUpdateResult updateVelocity(const VelocityMeasurement& velocity) {
        VelocityUpdateResult result;
        if (!pose3_inertial_eskf_detail::validVelocityMeasurement(velocity)) {
            result.reject_reason = PoseFusionRejectReason::kInvalidInput;
            return result;
        }
        if (velocity.time_jump) {
            result.time_alignment_rejected = true;
            result.reject_reason = PoseFusionRejectReason::kTimeAlignment;
            return result;
        }
        if (!state_.initialized) {
            return result;
        }
        if (!measurementStampWithinImuWindow(velocity.stamp_sec)) {
            result.time_alignment_rejected = true;
            result.reject_reason = PoseFusionRejectReason::kTimeAlignment;
            return result;
        }
        return updateVelocityAtCurrentState(velocity);
    }

    const RigidBodyState& state() const { return state_; }
    const ErrorCovariance& covariance() const { return covariance_; }
    bool initialized() const { return state_.initialized; }
    bool hasCorrectedBodyPose() const { return has_corrected_body_pose_; }
    const Pose3& correctedBodyPose() const { return corrected_body_pose_; }
    bool hasRawProjectedBodyPose() const { return has_raw_projected_body_pose_; }
    const Pose3& rawProjectedBodyPose() const { return raw_projected_body_pose_; }
    VrpnObservationState vrpnObservationState() const { return vrpn_health_.state(); }
    FilterHealth filterHealth() const {
        if (!state_.initialized) {
            return FilterHealth::kLost;
        }
        if (!covariance_.allFinite()) {
            return FilterHealth::kLost;
        }
        const double trace = covariance_.trace();
        if (!std::isfinite(trace)) {
            return FilterHealth::kLost;
        }
        if (vrpn_health_.state() == VrpnObservationState::kFault) {
            return FilterHealth::kImuOnly;
        }
        if (trace > config_.covariance_high_threshold) {
            return FilterHealth::kDegraded;
        }
        switch (vrpn_health_.state()) {
        case VrpnObservationState::kTrusted:
            return FilterHealth::kNominal;
        case VrpnObservationState::kSuspected:
        case VrpnObservationState::kRecovery:
            return FilterHealth::kDegraded;
        case VrpnObservationState::kFault:
            return FilterHealth::kImuOnly;
        }
        return FilterHealth::kLost;
    }
    double vrpnInnovationWindowChiSquare() const { return vrpn_health_.chiSquareWindowSum(); }
    double lastFusedPoseStampS() const { return last_fused_pose_stamp_sec_; }
    std::size_t vrpnConsecutiveRejects() const { return vrpn_health_.consecutiveRejects(); }
    std::size_t vrpnConsecutiveAccepts() const { return vrpn_health_.consecutiveAccepts(); }

  private:
    PoseUpdateResult updatePoseAtCurrentState(const PoseMeasurement& pose) {
        PoseUpdateResult result;
        if (!pose3_inertial_eskf_detail::validPoseMeasurement(pose)) {
            result.reject_reason = PoseFusionRejectReason::kInvalidInput;
            stampResultHealth(result);
            return result;
        }
        if (!state_.initialized) {
            initializeFromPose(pose, has_last_inertial_ ? &last_inertial_ : nullptr);
            result.accepted = state_.initialized;
            stampResultHealth(result);
            return result;
        }

        const Pose3 marker_world = markerPoseInWorld(pose.pose);
        const Pose3 measured_body_world = bodyPoseFromMarkerPose(marker_world, state_.body_to_marker);
        if (rawPoseMeasurementJumped(measured_body_world)) {
            result.innovation_rejected = true;
            result.reject_reason = PoseFusionRejectReason::kInnovationGate;
            state_.last_pose_stamp_sec = pose.stamp_sec;
            vrpn_health_.recordRejected();
            stampResultHealth(result);
            return result;
        }

        const Pose3 predicted_marker = predictedMarkerPose(state_);
        const MeasurementVector innovation = measurementResidual(predicted_marker, marker_world);
        result.position_innovation_norm = innovation.head<3>().norm();
        result.orientation_innovation_norm = innovation.tail<3>().norm();
        const bool needs_pose_recapture = result.position_innovation_norm > config_.innovation_position_gate_m ||
                                          result.orientation_innovation_norm > config_.innovation_orientation_gate_rad;
        if (needs_pose_recapture) {
            inflatePoseRecaptureCovariance();
        }

        const MeasurementMatrix H = measurementJacobian(predicted_marker, innovation);
        const MeasurementCovariance R = measurementCovariance();
        const MeasurementCovariance S = H * covariance_ * H.transpose() + R;
        Eigen::LDLT<MeasurementCovariance> ldlt;
        ldlt.compute(S);
        if (ldlt.info() != Eigen::Success || !S.allFinite()) {
            result.reject_reason = PoseFusionRejectReason::kNumericalFailure;
            vrpn_health_.recordRejected();
            stampResultHealth(result);
            return result;
        }

        result.mahalanobis_distance = innovation.transpose() * ldlt.solve(innovation);
        if (!std::isfinite(result.mahalanobis_distance)) {
            result.reject_reason = PoseFusionRejectReason::kNumericalFailure;
            vrpn_health_.recordRejected();
            stampResultHealth(result);
            return result;
        }
        // Treat NIS as a health metric. Hard rejection is reserved for absolute
        // position/orientation gates so model-lag during agile flight cannot
        // permanently switch the estimator to IMU-only.
        vrpn_health_.recordAccepted(result.mahalanobis_distance);

        const Eigen::Matrix<double, kErrorStateDim, 6> K =
            covariance_ * H.transpose() * ldlt.solve(MeasurementCovariance::Identity());
        // H is d residual / d error_state for se3Error(predicted, measured), so the correction
        // moves against the residual gradient.
        const ErrorVector delta = -K * innovation;
        if (!delta.allFinite()) {
            result.reject_reason = PoseFusionRejectReason::kNumericalFailure;
            vrpn_health_.recordRejected();
            stampResultHealth(result);
            return result;
        }
        injectError(delta);
        refreshDerivedInertialState();

        const ErrorCovariance identity = ErrorCovariance::Identity();
        covariance_ = (identity - K * H) * covariance_ * (identity - K * H).transpose() + K * R * K.transpose();
        covariance_ = 0.5 * (covariance_ + covariance_.transpose());
        applyMeasurementCovarianceFloor();
        state_.last_pose_stamp_sec = pose.stamp_sec;
        state_.covariance_trace = covariance_.trace();
        corrected_body_pose_ = bodyPoseFromState(state_);
        raw_projected_body_pose_ = measured_body_world;
        rememberRawPoseMeasurement(measured_body_world, pose.stamp_sec);
        has_corrected_body_pose_ = true;
        has_raw_projected_body_pose_ = true;
        last_fused_pose_stamp_sec_ = pose.stamp_sec;
        result.accepted = true;
        stampResultHealth(result);
        return result;
    }

    VelocityUpdateResult updateVelocityAtCurrentState(const VelocityMeasurement& velocity) {
        VelocityUpdateResult result;
        if (!pose3_inertial_eskf_detail::validVelocityMeasurement(velocity)) {
            result.reject_reason = PoseFusionRejectReason::kInvalidInput;
            return result;
        }
        if (!state_.initialized) {
            return result;
        }

        const Eigen::Vector3d innovation = state_.velocity - velocity.velocity;
        result.velocity_innovation_norm = innovation.norm();
        if (!std::isfinite(result.velocity_innovation_norm)) {
            result.reject_reason = PoseFusionRejectReason::kNumericalFailure;
            return result;
        }
        if (result.velocity_innovation_norm > config_.velocity_innovation_gate_mps) {
            result.innovation_rejected = true;
            result.reject_reason = PoseFusionRejectReason::kInnovationGate;
            return result;
        }

        Eigen::Matrix<double, 3, kErrorStateDim> H;
        H.setZero();
        H.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * config_.velocity_noise_std * config_.velocity_noise_std;
        const Eigen::Matrix3d S = H * covariance_ * H.transpose() + R;
        Eigen::LDLT<Eigen::Matrix3d> ldlt;
        ldlt.compute(S);
        if (ldlt.info() != Eigen::Success || !S.allFinite()) {
            result.reject_reason = PoseFusionRejectReason::kNumericalFailure;
            return result;
        }

        const Eigen::Matrix<double, kErrorStateDim, 3> K =
            covariance_ * H.transpose() * ldlt.solve(Eigen::Matrix3d::Identity());
        const ErrorVector delta = -K * innovation;
        if (!delta.allFinite()) {
            result.reject_reason = PoseFusionRejectReason::kNumericalFailure;
            return result;
        }
        injectError(delta);
        refreshDerivedInertialState();

        const ErrorCovariance identity = ErrorCovariance::Identity();
        covariance_ = (identity - K * H) * covariance_ * (identity - K * H).transpose() + K * R * K.transpose();
        covariance_ = 0.5 * (covariance_ + covariance_.transpose());
        state_.covariance_trace = covariance_.trace();
        corrected_body_pose_ = bodyPoseFromState(state_);
        has_corrected_body_pose_ = true;
        result.accepted = true;
        return result;
    }

    void resetCovariance() {
        covariance_.setZero();
        covariance_.block<3, 3>(0, 0).diagonal().setConstant(config_.initial_position_variance);
        covariance_.block<3, 3>(3, 3).diagonal().setConstant(config_.initial_velocity_variance);
        covariance_.block<3, 3>(6, 6).diagonal().setConstant(config_.initial_orientation_variance);
        covariance_.block<3, 3>(9, 9).diagonal().setConstant(config_.initial_gyro_bias_variance);
        covariance_.block<3, 3>(12, 12).diagonal().setConstant(config_.initial_accel_bias_variance);
        state_.covariance_trace = covariance_.trace();
    }

    void inflatePoseRecaptureCovariance() {
        setMinimumCovarianceDiagonal(0, config_.initial_position_variance);
        setMinimumCovarianceDiagonal(3, config_.initial_velocity_variance);
        setMinimumCovarianceDiagonal(6, config_.initial_orientation_variance);
        state_.covariance_trace = covariance_.trace();
    }

    void setMinimumCovarianceDiagonal(int start_index, double minimum_variance) {
        for (int i = 0; i < 3; ++i) {
            double& value = covariance_(start_index + i, start_index + i);
            if (!std::isfinite(value) || value < minimum_variance) {
                value = minimum_variance;
            }
        }
    }

    void applyMeasurementCovarianceFloor() {
        const double pos_std = config_.pose_position_noise_std;
        const double ori_std = config_.pose_orientation_noise_std;
        setMinimumCovarianceDiagonal(0, 0.25 * pos_std * pos_std);
        setMinimumCovarianceDiagonal(6, 0.25 * ori_std * ori_std);
        state_.covariance_trace = covariance_.trace();
    }

    bool measurementStampWithinImuWindow(double stamp_sec) const {
        return stamp_sec + config_.pose_max_late_s >= state_.last_inertial_stamp_sec &&
               stamp_sec <= state_.last_inertial_stamp_sec + config_.pose_max_early_s;
    }

    void rememberRawPoseMeasurement(const Pose3& body_pose_world, double stamp_sec) {
        last_raw_body_measurement_pose_ = body_pose_world;
        has_last_raw_body_measurement_pose_ = true;
        last_raw_pose_measurement_stamp_sec_ = stamp_sec;
    }

    bool rawPoseMeasurementJumped(const Pose3& body_pose_world) const {
        if (!has_last_raw_body_measurement_pose_) {
            return false;
        }
        const double position_delta = (body_pose_world.position - last_raw_body_measurement_pose_.position).norm();
        const double orientation_delta =
            logMap(last_raw_body_measurement_pose_.orientation.conjugate() * body_pose_world.orientation).norm();
        if (!std::isfinite(position_delta) || !std::isfinite(orientation_delta)) {
            return true;
        }
        return position_delta > config_.innovation_position_gate_m ||
               orientation_delta > config_.innovation_orientation_gate_rad;
    }

    Pose3 markerPoseInWorld(const Pose3& raw_marker_pose) const {
        return compose(config_.measurement_frame_to_world, raw_marker_pose);
    }

    static Pose3 bodyPoseFromMarkerPose(const Pose3& marker_pose_world, const Pose3& body_to_marker) {
        return compose(marker_pose_world, inverse(body_to_marker));
    }

    static Pose3 predictedMarkerPose(const RigidBodyState& state) {
        Pose3 body_pose;
        body_pose.position = state.position;
        body_pose.orientation = state.orientation;
        return compose(body_pose, state.body_to_marker);
    }

    static Pose3 bodyPoseFromState(const RigidBodyState& state) {
        Pose3 body_pose;
        body_pose.position = state.position;
        body_pose.orientation = state.orientation;
        return body_pose;
    }

    static MeasurementVector measurementResidual(const Pose3& predicted_marker, const Pose3& measured_marker) {
        return pose3_inertial_eskf_detail::normalizedInnovation(se3Error(predicted_marker, measured_marker));
    }

    MeasurementCovariance measurementCovariance() const {
        MeasurementCovariance R = MeasurementCovariance::Zero();
        R.block<3, 3>(0, 0).diagonal().setConstant(config_.pose_position_noise_std * config_.pose_position_noise_std);
        R.block<3, 3>(3, 3).diagonal().setConstant(config_.pose_orientation_noise_std *
                                                   config_.pose_orientation_noise_std);
        return R;
    }

    MeasurementMatrix measurementJacobian(const Pose3& predicted_marker, const MeasurementVector& innovation) const {
        MeasurementMatrix H;
        H.setZero();
        // Jacobian of se3Error(predictedMarker(state), measuredMarker) with the same right
        // perturbation convention used by injectError().
        const Eigen::Matrix3d marker_rotation_transpose =
            normalizedQuaternion(predicted_marker.orientation).toRotationMatrix().transpose();
        const Eigen::Matrix3d extrinsic_rotation_transpose =
            normalizedQuaternion(state_.body_to_marker.orientation).toRotationMatrix().transpose();
        const Eigen::Vector3d residual_position = innovation.head<3>();
        const Eigen::Matrix3d residual_position_skew = pose3_inertial_eskf_detail::skewMatrix(residual_position);
        const Eigen::Matrix3d marker_offset_skew =
            pose3_inertial_eskf_detail::skewMatrix(state_.body_to_marker.position);

        H.block<3, 3>(0, 0) = -marker_rotation_transpose;
        H.block<3, 3>(0, 6) =
            residual_position_skew * extrinsic_rotation_transpose + extrinsic_rotation_transpose * marker_offset_skew;
        H.block<3, 3>(3, 6) = -extrinsic_rotation_transpose;

        return H;
    }

    void propagateCovariance(const Eigen::Vector3d& accel_body, const Eigen::Matrix3d& rotation, double dt) {
        ErrorCovariance F = ErrorCovariance::Identity();
        const Eigen::Matrix3d accel_skew = pose3_inertial_eskf_detail::skewMatrix(accel_body);

        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(0, 6) = -0.5 * rotation * accel_skew * dt * dt;
        F.block<3, 3>(0, 12) = -0.5 * rotation * dt * dt;
        F.block<3, 3>(3, 6) = -rotation * accel_skew * dt;
        F.block<3, 3>(3, 12) = -rotation * dt;
        F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;

        ErrorCovariance Q = ErrorCovariance::Zero();
        const double accel_var = config_.accel_noise_std * config_.accel_noise_std;
        const double gyro_var = config_.gyro_noise_std * config_.gyro_noise_std;
        Q.block<3, 3>(0, 0).diagonal().setConstant(0.25 * accel_var * dt * dt * dt * dt);
        Q.block<3, 3>(3, 3).diagonal().setConstant(accel_var * dt * dt);
        Q.block<3, 3>(6, 6).diagonal().setConstant(gyro_var * dt * dt);
        Q.block<3, 3>(9, 9).diagonal().setConstant(config_.gyro_bias_random_walk_std *
                                                   config_.gyro_bias_random_walk_std * dt);
        Q.block<3, 3>(12, 12).diagonal().setConstant(config_.accel_bias_random_walk_std *
                                                     config_.accel_bias_random_walk_std * dt);
        covariance_ = F * covariance_ * F.transpose() + Q;
        covariance_ = 0.5 * (covariance_ + covariance_.transpose());
    }

    void injectError(const ErrorVector& delta) { injectError(delta, state_); }

    void injectError(const ErrorVector& delta, RigidBodyState& state) const {
        state.position += delta.segment<3>(0);
        state.velocity += delta.segment<3>(3);
        state.orientation = normalizedQuaternion(state.orientation * expMap(delta.segment<3>(6)));
        state.gyro_bias += delta.segment<3>(9);
        state.accel_bias += delta.segment<3>(12);
    }

    void integrateInertialStepWithReadings(const Eigen::Vector3d& angular_velocity,
                                           const Eigen::Vector3d& linear_acceleration, double dt) {
        state_.angular_velocity = angular_velocity - state_.gyro_bias;
        const Eigen::Vector3d accel_body = linear_acceleration - state_.accel_bias;
        const Eigen::Quaterniond old_orientation = state_.orientation;
        const Eigen::Quaterniond mid_orientation =
            normalizedQuaternion(old_orientation * expMap(0.5 * state_.angular_velocity * dt));
        const Eigen::Matrix3d rotation = mid_orientation.toRotationMatrix();
        const Eigen::Vector3d acceleration_world = rotation * accel_body + state_.gravity;
        state_.linear_acceleration = acceleration_world;
        state_.position += state_.velocity * dt + 0.5 * acceleration_world * dt * dt;
        state_.velocity += acceleration_world * dt;
        state_.orientation = normalizedQuaternion(old_orientation * expMap(state_.angular_velocity * dt));
        propagateCovariance(accel_body, rotation, dt);
        state_.covariance_trace = covariance_.trace();
    }

    void integrateInertialStep(const InertialSample& inertial, double dt) {
        integrateInertialStepWithReadings(inertial.angular_velocity, inertial.linear_acceleration, dt);
    }

    void refreshDerivedInertialState() {
        if (!has_last_inertial_ || !pose3_inertial_eskf_detail::validInertialSample(last_inertial_)) {
            return;
        }
        state_.angular_velocity = last_inertial_.angular_velocity - state_.gyro_bias;
        state_.linear_acceleration =
            state_.orientation.toRotationMatrix() * (last_inertial_.linear_acceleration - state_.accel_bias) +
            state_.gravity;
    }

    void integrateInertialStepMidpoint(const InertialSample& previous, const InertialSample& current, double dt) {
        integrateInertialStepWithReadings(0.5 * (previous.angular_velocity + current.angular_velocity),
                                          0.5 * (previous.linear_acceleration + current.linear_acceleration), dt);
    }

    bool propagateToStamp(const InertialSample& inertial, double target_stamp_sec) {
        if (!std::isfinite(target_stamp_sec) || target_stamp_sec <= state_.last_inertial_stamp_sec + kMinDt) {
            return false;
        }
        double remaining = target_stamp_sec - state_.last_inertial_stamp_sec;
        while (remaining > kMinDt) {
            const double dt = std::min(remaining, config_.max_propagation_dt_s);
            integrateInertialStep(inertial, dt);
            state_.last_inertial_stamp_sec += dt;
            remaining = target_stamp_sec - state_.last_inertial_stamp_sec;
        }
        // Snap to the requested timestamp after segmented integration to avoid accumulating sub-kMinDt drift.
        state_.last_inertial_stamp_sec = target_stamp_sec;
        refreshDerivedInertialState();
        state_.covariance_trace = covariance_.trace();
        corrected_body_pose_ = bodyPoseFromState(state_);
        has_corrected_body_pose_ = true;
        return true;
    }

    InertialSample interpolatedInertialSample(const InertialSample& start, const InertialSample& end,
                                              double stamp_sec) const {
        InertialSample result = end;
        result.stamp_sec = stamp_sec;
        const double duration = end.stamp_sec - start.stamp_sec;
        if (!std::isfinite(duration) || duration <= kMinDt) {
            return result;
        }
        const double ratio = std::max(0.0, std::min(1.0, (stamp_sec - start.stamp_sec) / duration));
        result.angular_velocity = start.angular_velocity + ratio * (end.angular_velocity - start.angular_velocity);
        result.linear_acceleration =
            start.linear_acceleration + ratio * (end.linear_acceleration - start.linear_acceleration);
        return result;
    }

    bool propagateToStampUsingInertialPair(const InertialSample& next_inertial, double target_stamp_sec) {
        if (!has_last_inertial_) {
            const bool propagated = propagateToStamp(next_inertial, target_stamp_sec);
            if (propagated && std::fabs(target_stamp_sec - next_inertial.stamp_sec) <= kMinDt) {
                last_inertial_ = next_inertial;
                has_last_inertial_ = true;
            }
            return propagated;
        }
        if (!std::isfinite(target_stamp_sec) || target_stamp_sec <= state_.last_inertial_stamp_sec + kMinDt) {
            return false;
        }
        if (next_inertial.stamp_sec <= last_inertial_.stamp_sec + kMinDt) {
            return propagateToStamp(next_inertial, target_stamp_sec);
        }
        while (target_stamp_sec > state_.last_inertial_stamp_sec + kMinDt) {
            const double segment_end =
                std::min(target_stamp_sec, state_.last_inertial_stamp_sec + config_.max_propagation_dt_s);
            InertialSample segment_inertial =
                std::fabs(segment_end - next_inertial.stamp_sec) <= kMinDt
                    ? next_inertial
                    : interpolatedInertialSample(last_inertial_, next_inertial, segment_end);
            const double dt = segment_end - state_.last_inertial_stamp_sec;
            if (!std::isfinite(dt) || dt <= kMinDt) {
                return false;
            }
            integrateInertialStepMidpoint(last_inertial_, segment_inertial, dt);
            state_.last_inertial_stamp_sec = segment_end;
            last_inertial_ = segment_inertial;
        }
        if (std::fabs(target_stamp_sec - next_inertial.stamp_sec) <= kMinDt) {
            last_inertial_ = next_inertial;
        }
        state_.last_inertial_stamp_sec = target_stamp_sec;
        refreshDerivedInertialState();
        state_.covariance_trace = covariance_.trace();
        corrected_body_pose_ = bodyPoseFromState(state_);
        has_corrected_body_pose_ = true;
        return true;
    }

    void stampResultHealth(PoseUpdateResult& result) const {
        result.vrpn_observation_state = vrpn_health_.state();
        result.filter_health = filterHealth();
        result.innovation_window_chi_square = vrpn_health_.chiSquareWindowSum();
    }

    static constexpr double kMinDt = 1.0e-5;

    Pose3InertialEskfConfig config_{};
    RigidBodyState state_{};
    ErrorCovariance covariance_{ErrorCovariance::Identity()};
    Pose3 corrected_body_pose_{};
    Pose3 raw_projected_body_pose_{};
    Pose3 last_raw_body_measurement_pose_{};
    bool has_corrected_body_pose_{false};
    bool has_raw_projected_body_pose_{false};
    bool has_last_raw_body_measurement_pose_{false};
    InertialSample last_inertial_{};
    bool has_last_inertial_{false};
    ObservationHealthTracker vrpn_health_{};
    double last_fused_pose_stamp_sec_{0.0};
    double last_raw_pose_measurement_stamp_sec_{0.0};

    friend struct Pose3InertialEskfTestAccess;
};

} // namespace xgc2_math
