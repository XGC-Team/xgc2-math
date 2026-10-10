// Export actual public C++ algorithms for the MATLAB/Octave parity checks.
// Numeric CSV only, with 17-digit precision; no precomputed golden results.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "xgc2_math/control/delayed_planar_velocity.hpp"
#include "xgc2_math/control/smc_tracking_controller.hpp"
#include "xgc2_math/geometry/kinematics.hpp"
#include "xgc2_math/trajectory/analytic.hpp"

namespace {

void vector(std::ostream& out, const Eigen::Vector3d& value) {
    out << ',' << value.x() << ',' << value.y() << ',' << value.z();
}

void planar(std::ostream& out) {
    // Forward unicycle, mecanum strafe, reverse turn, near-zero turn, pure spin.
    const std::array<std::array<double, 3>, 5> inputs{{
        {{1.2, 0.0, 0.7}},
        {{0.4, 0.8, -0.5}},
        {{-0.9, -0.2, 0.4}},
        {{1.0, 0.3, 1.0e-11}},
        {{0.0, 0.0, 1.1}},
    }};
    for (std::size_t scenario = 0; scenario < inputs.size(); ++scenario) {
        const auto& input = inputs[scenario];
        xgc2_math::Pose2 pose{Eigen::Vector2d(0.4, -0.7), 0.3};
        for (int k = 0; k <= 400; ++k) {
            out << scenario << ',' << 0.01 * k << ',' << pose.position.x() << ',' << pose.position.y() << ','
                << pose.yaw << '\n';
            pose = xgc2_math::stepBodyVelocity(pose, Eigen::Vector2d(input[0], input[1]), input[2], 0.01);
        }
    }
}

void delayed(std::ostream& out) {
    const xgc2_math::DelayedPlanarVelocityParameters config{0.035, 0.12, 0.07};
    xgc2_math::DelayedPlanarVelocity plant(config);
    plant.reset(0.0);
    plant.command(0.0, {1.2, -0.6});
    plant.command(0.63, {-0.4, 0.3});
    for (int k = 0; k <= 200; ++k) {
        const double time = 0.01 * k;
        const auto output = plant.advance(time);
        out << time << ',' << output.linear_m_s << ',' << output.yaw_rad_s << '\n';
    }
}

void tracking(std::ostream& out) {
    const xgc2_math::control::SmcTrackingConfig config;
    const Eigen::Vector3d constant_acceleration(0.2, -0.1, 0.03);
    const Eigen::Vector3d frequency(0.4, 0.3, 0.2);
    const Eigen::Vector3d amplitude(0.8, 0.6, 0.3);
    for (int scenario = 0; scenario < 4; ++scenario) {
        const double dt = scenario % 2 == 0 ? 0.005 : 0.01;
        const int steps = static_cast<int>(30.0 / dt);
        xgc2_math::TranslationalState state{Eigen::Vector3d(1.0, -1.0, 0.4), Eigen::Vector3d(-0.5, 0.5, 0.0)};
        for (int k = 0; k <= steps; ++k) {
            const double time = dt * k;
            Eigen::Vector3d position = 0.5 * time * time * constant_acceleration;
            Eigen::Vector3d velocity = time * constant_acceleration;
            Eigen::Vector3d acceleration = constant_acceleration;
            if (scenario >= 2) {
                for (int axis = 0; axis < 3; ++axis) {
                    const double phase = frequency[axis] * time;
                    position[axis] = amplitude[axis] * std::sin(phase);
                    velocity[axis] = amplitude[axis] * frequency[axis] * std::cos(phase);
                    acceleration[axis] = -amplitude[axis] * frequency[axis] * frequency[axis] * std::sin(phase);
                }
            }
            const auto control = xgc2_math::control::computeSmcTracking(config, state.position, state.velocity,
                                                                        position, velocity, acceleration);
            if (!control.success)
                throw std::runtime_error("SMC returned an invalid command");
            out << scenario << ',' << time;
            vector(out, state.position);
            vector(out, state.velocity);
            vector(out, control.acceleration);
            vector(out, control.feedback);
            out << '\n';
            const double sign = time < 15.0 ? 1.0 : -1.0;
            const Eigen::Vector3d disturbance = sign * Eigen::Vector3d(0.1, -0.05, 0.02);
            state = xgc2_math::stepWorldAcceleration(state, control.acceleration + disturbance, dt);
        }
    }
}

void trajectories(std::ostream& out) {
    using namespace xgc2_math::trajectory;
    const CircleCurveEvaluator3 circle;
    const FigureEightCurveEvaluator3 eight;
    const HelixXyCurveEvaluator3 helix;
    const TorusKnotCurveEvaluator3 torus;
    const std::array<const TrajectoryEvaluator3*, 4> evaluators{{&circle, &eight, &helix, &torus}};
    for (std::size_t scenario = 0; scenario < evaluators.size(); ++scenario) {
        for (int k = 0; k <= 500; ++k) {
            const double time = 0.02 * k;
            FlatOutput3 flat;
            if (!evaluators[scenario]->evaluate(time, flat))
                throw std::runtime_error("analytic trajectory evaluation failed");
            out << scenario << ',' << time;
            vector(out, flat.position);
            vector(out, flat.velocity);
            vector(out, flat.acceleration);
            vector(out, flat.jerk);
            vector(out, flat.snap);
            out << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument(
                "usage: simulation_reference_export planar|delayed|tracking|trajectories output.csv");
        const std::string mode(argv[1]);
        if (mode != "planar" && mode != "delayed" && mode != "tracking" && mode != "trajectories")
            throw std::invalid_argument("unknown simulation mode: " + mode);
        std::ofstream out(argv[2]);
        if (!out)
            throw std::runtime_error("cannot open output CSV");
        out << std::setprecision(17);
        if (mode == "planar")
            planar(out);
        else if (mode == "delayed")
            delayed(out);
        else if (mode == "tracking")
            tracking(out);
        else
            trajectories(out);
        out.flush();
        if (!out)
            throw std::runtime_error("cannot finish output CSV");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
