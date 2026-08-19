// Offline Pose3InertialEskf replay. Events CSV, recv order:
// imu,<recv>,<stamp>,gx,gy,gz,ax,ay,az
// vrpn,<recv>,<stamp>,x,y,z,qx,qy,qz,qw
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xgc2_math/estimation/pose3_inertial_eskf.hpp>

namespace {

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cols.push_back(item);
    }
    return cols;
}

double atofCol(const std::vector<std::string>& cols, std::size_t i) {
    return i < cols.size() ? std::atof(cols[i].c_str()) : 0.0;
}

xgc2_math::Pose3InertialEskfConfig bagConfig(double delay_s, double body_to_marker_yaw_rad) {
    xgc2_math::Pose3InertialEskfConfig config;
    config.gravity_mps2 = 9.8066;
    config.body_to_marker.orientation = xgc2_math::rpyToQuaternion(Eigen::Vector3d(0.0, 0.0, body_to_marker_yaw_rad));
    config.accel_noise_std = 0.35;
    config.gyro_noise_std = 0.03;
    config.pose_position_noise_std = 0.01;
    config.pose_orientation_noise_std = 0.01;
    config.gyro_bias_random_walk_std = 1.0e-4;
    config.accel_bias_random_walk_std = 1.0e-3;
    config.innovation_position_gate_m = 1.5;
    config.innovation_orientation_gate_rad = 0.8;
    config.pose_nis_gate = 22.5;
    config.max_propagation_dt_s = 0.01;
    config.pose_max_late_s = 0.12;
    config.pose_max_early_s = 0.12;
    config.pose_observation_delay_s = delay_s;
    config.pose_position_kalman_gain = 0.9;
    config.pose_orientation_kalman_gain = 0.8;
    config.pose_update_iterations = 3;
    config.inertial_buffer_capacity = 128;
    config.initial_position_variance = 0.01;
    config.initial_velocity_variance = 0.1;
    config.initial_orientation_variance = 0.01;
    config.initial_gyro_bias_variance = 0.01;
    config.initial_accel_bias_variance = 0.1;
    return config;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: pose3_eskf_csv_replay events.csv state.csv [delay_s] [body_to_marker_yaw_rad]\n";
        return 2;
    }
    const double delay_s = argc >= 4 ? std::atof(argv[3]) : 0.0;
    const double body_to_marker_yaw_rad = argc >= 5 ? std::atof(argv[4]) : 0.0;

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }
    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "cannot open " << argv[2] << "\n";
        return 1;
    }

    xgc2_math::Pose3InertialEskf eskf;
    eskf.setConfig(bagConfig(delay_s, body_to_marker_yaw_rad));
    int pose_accepted = 0;
    int pose_ta = 0;
    int imu_n = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> cols = splitCsv(line);
        if (cols.size() < 3) {
            continue;
        }
        const std::string& kind = cols[0];
        const double stamp = atofCol(cols, 2);
        if (kind == "imu") {
            if (cols.size() < 9) {
                continue;
            }
            xgc2_math::InertialSample imu;
            imu.received = true;
            imu.valid = true;
            imu.stamp_sec = stamp;
            imu.angular_velocity = Eigen::Vector3d(atofCol(cols, 3), atofCol(cols, 4), atofCol(cols, 5));
            imu.linear_acceleration = Eigen::Vector3d(atofCol(cols, 6), atofCol(cols, 7), atofCol(cols, 8));
            eskf.propagateInertial(imu);
            ++imu_n;
            if (!eskf.initialized()) {
                continue;
            }
            const auto& state = eskf.state();
            out << std::setprecision(17) << state.last_inertial_stamp_sec << ',' << state.position.x() << ','
                << state.position.y() << ',' << state.position.z() << ',' << state.orientation.x() << ','
                << state.orientation.y() << ',' << state.orientation.z() << ',' << state.orientation.w() << ','
                << state.velocity.x() << ',' << state.velocity.y() << ',' << state.velocity.z() << '\n';
        } else if (kind == "vrpn") {
            if (cols.size() < 10) {
                continue;
            }
            xgc2_math::PoseMeasurement pose;
            pose.received = true;
            pose.valid = true;
            pose.stamp_sec = stamp;
            pose.pose.position = Eigen::Vector3d(atofCol(cols, 3), atofCol(cols, 4), atofCol(cols, 5));
            pose.pose.orientation = xgc2_math::normalizedQuaternion(
                Eigen::Quaterniond(atofCol(cols, 9), atofCol(cols, 6), atofCol(cols, 7), atofCol(cols, 8)));
            const auto result = eskf.updatePose(pose);
            if (result.accepted) {
                ++pose_accepted;
            }
            if (result.time_alignment_rejected) {
                ++pose_ta;
            }
        }
    }
    std::cerr << "imu=" << imu_n << " pose_accepted=" << pose_accepted << " pose_ta=" << pose_ta
              << " initialized=" << static_cast<int>(eskf.initialized()) << "\n";
    return 0;
}
