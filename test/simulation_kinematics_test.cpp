#include "xgc2_math/geometry/kinematics.hpp"

#include <iostream>
#include <stdexcept>

using namespace xgc2_math;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

int main() {
    const double pi = std::acos(-1.0);
    TranslationalState start{Eigen::Vector3d(1, -2, 3), Eigen::Vector3d(2, 1, -1)};
    const Eigen::Vector3d acceleration(-1, 2, 0.5);
    const auto full = stepWorldAcceleration(start, acceleration, 2.0);
    require(full.position.isApprox(Eigen::Vector3d(3, 4, 2)) && full.velocity.isApprox(Eigen::Vector3d(0, 5, 0)),
            "world acceleration plant must integrate motion, not assign a reference");
    auto split = start;
    for (int i = 0; i < 200; ++i)
        split = stepWorldAcceleration(split, acceleration, 0.01);
    require(split.position.isApprox(full.position, 1e-12) && split.velocity.isApprox(full.velocity, 1e-12),
            "constant acceleration result must not depend on caller subdivision");
    const auto hold = stepWorldAcceleration(start, Eigen::Vector3d::Zero(), 1.0);
    require(hold.velocity == start.velocity && hold.position == start.position + start.velocity,
            "zero control is ballistic, not an implicit hover controller");

    const Pose2 origin{};
    const auto quarter = stepBodyVelocity(origin, Eigen::Vector2d(2, 0), 1.0, pi / 2);
    require(quarter.position.isApprox(Eigen::Vector2d(2, 2), 1e-12) && std::abs(quarter.yaw - pi / 2) < 1e-12,
            "constant unicycle input traces the correct circular arc");
    const auto lateral = stepBodyVelocity(origin, Eigen::Vector2d(0, 2), 1.0, pi / 2);
    require(lateral.position.isApprox(Eigen::Vector2d(-2, 2), 1e-12),
            "mecanum body-left velocity rotates throughout the step");
    const auto straight = stepBodyVelocity({Eigen::Vector2d(1, 2), pi / 2}, Eigen::Vector2d(3, -2), 0, 2);
    require(straight.position.isApprox(Eigen::Vector2d(5, 8), 1e-12), "body-frame velocity must rotate to world");
    const auto tiny_turn = stepBodyVelocity(origin, Eigen::Vector2d(1, 0), 1e-12, 1);
    require(std::abs(tiny_turn.position.x() - 1) < 1e-14 && std::abs(tiny_turn.position.y() - 5e-13) < 1e-25,
            "near-zero yaw rate must have a continuous straight-line limit");
    Pose2 circle{};
    for (int i = 0; i < 1000; ++i)
        circle = stepBodyVelocity(circle, Eigen::Vector2d(2, 0.5), 1, 2 * pi / 1000);
    require(circle.position.norm() < 1e-11 && std::abs(circle.yaw) < 1e-11,
            "full revolution closes with no integration drift");
    const auto still = stepBodyVelocity(straight, Eigen::Vector2d(5, 6), 2, 0);
    require(still.position == straight.position && still.yaw == straight.yaw, "zero elapsed time preserves state");
    std::cout << "Acceleration and planar kinematics passed\n";
}
