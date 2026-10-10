#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "xgc2_math/geometry/collision/distance_gjk_query.h"
#include "xgc2_math/geometry/kinematics.hpp"
#include "xgc2_math/geometry/occupied_sets/ball_inflated_set.h"
#include "xgc2_math/geometry/occupied_sets/box_set.h"
#include "xgc2_math/geometry/occupied_sets/capsule_set.h"
#include "xgc2_math/geometry/occupied_sets/ellipsoid_set.h"
#include "xgc2_math/geometry/occupied_sets/sphere_set.h"
#include "xgc2_math/geometry/occupied_sets/translated_set_view.h"
#include "xgc2_math/geometry/se3.hpp"

namespace {

constexpr std::uint64_t kSeed = 0x584743324d415448ULL;
constexpr double kPi = 3.141592653589793238462643383279502884;

// Use the specified engine bits directly so cases do not depend on the standard
// library's choice of uniform_real_distribution implementation.
struct Cases {
    std::mt19937_64 engine{kSeed};
    double scalar() { return 2.0 * static_cast<double>(engine() >> 11) / 9007199254740992.0 - 1.0; }
    Eigen::Vector2d vector2() {
        const double x = scalar(), y = scalar();
        return Eigen::Vector2d(x, y);
    }
    Eigen::Vector3d vector3() {
        const double x = scalar(), y = scalar(), z = scalar();
        return Eigen::Vector3d(x, y, z);
    }
    Eigen::Quaterniond rotation() {
        const Eigen::Vector3d axis = vector3().normalized();
        return Eigen::Quaterniond(Eigen::AngleAxisd(kPi * scalar(), axis));
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, const std::string& context) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << std::setprecision(17) << context << ": actual=" << actual << ", expected=" << expected
            << ", tolerance=" << tolerance;
        throw std::runtime_error(out.str());
    }
}

template <typename A, typename B>
void matrixNear(const Eigen::MatrixBase<A>& actual, const Eigen::MatrixBase<B>& expected, double tolerance,
                const std::string& context) {
    require(actual.allFinite() && expected.allFinite(), context + ": nonfinite matrix");
    near((actual - expected).norm(), 0.0, tolerance, context + " (norm of difference)");
}

std::string sample(int i) {
    return "sample " + std::to_string(i);
}

// Matrix multiplication is an independent representation of the pose API.
Eigen::Matrix3d homogeneous(const xgc2_math::Pose2& pose) {
    Eigen::Matrix3d result = Eigen::Matrix3d::Identity();
    result.topLeftCorner<2, 2>() = Eigen::Rotation2Dd(pose.yaw).toRotationMatrix();
    result.topRightCorner<2, 1>() = pose.position;
    return result;
}

Eigen::Matrix4d homogeneous(const xgc2_math::Pose3& pose) {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.topLeftCorner<3, 3>() = pose.orientation.normalized().toRotationMatrix();
    result.topRightCorner<3, 1>() = pose.position;
    return result;
}

void se2Group() {
    Cases cases;
    for (int i = 0; i < 128; ++i) {
        const xgc2_math::Pose2 a{5.0 * cases.vector2(), 8.0 * kPi * cases.scalar()};
        const xgc2_math::Pose2 b{5.0 * cases.vector2(), 8.0 * kPi * cases.scalar()};
        const xgc2_math::Pose2 c{5.0 * cases.vector2(), 8.0 * kPi * cases.scalar()};
        const auto ab = xgc2_math::compose(a, b);
        matrixNear(homogeneous(ab), homogeneous(a) * homogeneous(b), 5.0e-14, sample(i) + " SE2 composition");
        matrixNear(homogeneous(xgc2_math::inverse(a)), homogeneous(a).inverse(), 5.0e-14, sample(i) + " SE2 inverse");
        matrixNear(homogeneous(xgc2_math::compose(a, xgc2_math::inverse(a))), Eigen::Matrix3d::Identity(), 5.0e-14,
                   sample(i) + " SE2 right inverse");
        matrixNear(homogeneous(xgc2_math::compose(xgc2_math::inverse(a), a)), Eigen::Matrix3d::Identity(), 5.0e-14,
                   sample(i) + " SE2 left inverse");
        matrixNear(homogeneous(xgc2_math::compose(ab, c)), homogeneous(xgc2_math::compose(a, xgc2_math::compose(b, c))),
                   1.0e-13, sample(i) + " SE2 associativity");
        matrixNear(xgc2_math::se2Error(a, b), xgc2_math::se2Error(xgc2_math::compose(c, a), xgc2_math::compose(c, b)),
                   1.0e-13, sample(i) + " reference-frame error must be invariant under a common world transform");
    }
}

void se3Group() {
    Cases cases;
    for (int i = 0; i < 128; ++i) {
        const xgc2_math::Pose3 a{5.0 * cases.vector3(), cases.rotation()};
        const xgc2_math::Pose3 b{5.0 * cases.vector3(), cases.rotation()};
        const xgc2_math::Pose3 c{5.0 * cases.vector3(), cases.rotation()};
        const auto ab = xgc2_math::compose(a, b);
        matrixNear(homogeneous(ab), homogeneous(a) * homogeneous(b), 2.0e-14, sample(i) + " SE3 composition");
        matrixNear(homogeneous(xgc2_math::inverse(a)), homogeneous(a).inverse(), 2.0e-14, sample(i) + " SE3 inverse");
        matrixNear(homogeneous(xgc2_math::compose(a, xgc2_math::inverse(a))), Eigen::Matrix4d::Identity(), 2.0e-14,
                   sample(i) + " SE3 right inverse");
        matrixNear(homogeneous(xgc2_math::compose(xgc2_math::inverse(a), a)), Eigen::Matrix4d::Identity(), 2.0e-14,
                   sample(i) + " SE3 left inverse");
        matrixNear(homogeneous(xgc2_math::compose(ab, c)), homogeneous(xgc2_math::compose(a, xgc2_math::compose(b, c))),
                   3.0e-14, sample(i) + " SE3 associativity");
        matrixNear(xgc2_math::se3Error(a, b), xgc2_math::se3Error(xgc2_math::compose(c, a), xgc2_math::compose(c, b)),
                   4.0e-14, sample(i) + " SE3 error under a common world transform");
    }
}

void rotationMaps() {
    Cases cases;
    for (double angle : {1.0e-8, 1.0e-5, 0.2, 1.7, kPi - 1.0e-7}) {
        for (int i = 0; i < 32; ++i) {
            const Eigen::Vector3d vector = angle * cases.vector3().normalized();
            const auto frame = cases.rotation();
            const auto rotation = xgc2_math::expMap(vector);
            matrixNear(xgc2_math::logMap(rotation), vector, 4.0e-15, "exp/log round trip");
            matrixNear(xgc2_math::expMap(frame * vector).toRotationMatrix(),
                       (frame * rotation * frame.conjugate()).toRotationMatrix(), 3.0e-15,
                       "rotation-vector change of frame");
            Eigen::Quaterniond opposite = rotation;
            opposite.coeffs() *= -1.0;
            matrixNear(xgc2_math::logMap(opposite), xgc2_math::logMap(rotation), 2.0e-15,
                       "quaternion double cover must preserve logarithm");
            const auto matrix = rotation.toRotationMatrix();
            matrixNear(matrix.transpose() * matrix, Eigen::Matrix3d::Identity(), 3.0e-15, "rotation orthogonality");
            near(matrix.determinant(), 1.0, 3.0e-15, "proper rotation determinant");
        }
    }
}

void bodyVelocityFlow() {
    Cases cases;
    constexpr std::array<double, 5> fractions{{0.0, 0.013, 0.217, 0.319, 0.451}};
    for (int i = 0; i < 96; ++i) {
        const xgc2_math::Pose2 pose{3.0 * cases.vector2(), kPi * cases.scalar()};
        const xgc2_math::Pose2 frame{3.0 * cases.vector2(), kPi * cases.scalar()};
        const Eigen::Vector2d velocity = 4.0 * cases.vector2();
        const double yaw_rate = 4.0 * cases.scalar();
        const double dt = 0.05 + 2.0 * std::abs(cases.scalar());
        const auto full = xgc2_math::stepBodyVelocity(pose, velocity, yaw_rate, dt);
        const auto split =
            std::accumulate(fractions.begin(), fractions.end(), pose,
                            [&velocity, yaw_rate, dt](const xgc2_math::Pose2& current, double fraction) {
                                return xgc2_math::stepBodyVelocity(current, velocity, yaw_rate, dt * fraction);
                            });
        matrixNear(homogeneous(split), homogeneous(full), 2.0e-14, sample(i) + " body ZOH subdivision");
        const auto reversed = xgc2_math::stepBodyVelocity(full, -velocity, -yaw_rate, dt);
        matrixNear(homogeneous(reversed), homogeneous(pose), 2.0e-14, sample(i) + " body ZOH command reversal");
        matrixNear(homogeneous(xgc2_math::stepBodyVelocity(xgc2_math::compose(frame, pose), velocity, yaw_rate, dt)),
                   homogeneous(xgc2_math::compose(frame, full)), 2.0e-14,
                   sample(i) + " body ZOH world-frame equivariance");
    }
}

void bodyVelocityQuadrature() {
    // Integrate the world-frame differential equation by long-double composite
    // Simpson quadrature. This does not reuse the implementation's sinc arc.
    const xgc2_math::Pose2 pose{Eigen::Vector2d(1.2, -0.8), 2.9};
    const Eigen::Vector2d velocity(1.7, -0.6);
    constexpr int intervals = 4096;
    constexpr long double dt = 1.7L;
    const long double h = dt / intervals;
    for (double yaw_rate : {-4.0, -0.3, -1.2e-6, -1.17e-6, -1.0e-12, 0.0, 1.0e-12, 1.17e-6, 1.2e-6, 0.3, 4.0}) {
        Eigen::Matrix<long double, 2, 1> integral = Eigen::Matrix<long double, 2, 1>::Zero();
        for (int j = 0; j <= intervals; ++j) {
            const long double theta = static_cast<long double>(pose.yaw) + yaw_rate * h * j;
            const long double c = std::cos(theta);
            const long double s = std::sin(theta);
            const int weight = j == 0 || j == intervals ? 1 : (j % 2 == 0 ? 2 : 4);
            integral.x() += weight * (c * velocity.x() - s * velocity.y());
            integral.y() += weight * (s * velocity.x() + c * velocity.y());
        }
        const Eigen::Vector2d expected = pose.position + (integral * (h / 3.0L)).cast<double>();
        const auto actual = xgc2_math::stepBodyVelocity(pose, velocity, yaw_rate, static_cast<double>(dt));
        matrixNear(actual.position, expected, 2.0e-12, "body ZOH quadrature, yaw_rate=" + std::to_string(yaw_rate));
    }
}

void accelerationFlow() {
    Cases cases;
    constexpr std::array<double, 4> fractions{{0.0, 0.1, 0.3, 0.6}};
    for (int i = 0; i < 96; ++i) {
        const xgc2_math::TranslationalState state{cases.vector3(), 2.0 * cases.vector3()};
        const Eigen::Vector3d acceleration = 2.0 * cases.vector3();
        const double dt = 0.01 + 2.0 * std::abs(cases.scalar());
        const auto full = xgc2_math::stepWorldAcceleration(state, acceleration, dt);
        const auto split =
            std::accumulate(fractions.begin(), fractions.end(), state,
                            [&acceleration, dt](const xgc2_math::TranslationalState& current, double fraction) {
                                return xgc2_math::stepWorldAcceleration(current, acceleration, dt * fraction);
                            });
        matrixNear(split.position, full.position, 2.0e-14, sample(i) + " acceleration subdivision position");
        matrixNear(split.velocity, full.velocity, 2.0e-14, sample(i) + " acceleration subdivision velocity");
        const auto reversed = xgc2_math::stepWorldAcceleration({full.position, -full.velocity}, acceleration, dt);
        matrixNear(reversed.position, state.position, 2.0e-14, sample(i) + " acceleration time reversal position");
        matrixNear(reversed.velocity, -state.velocity, 2.0e-14, sample(i) + " acceleration time reversal velocity");
        near(0.5 * (full.velocity.squaredNorm() - state.velocity.squaredNorm()),
             acceleration.dot(full.position - state.position), 3.0e-14, sample(i) + " unit-mass work-energy identity");
        const auto rotation = cases.rotation();
        const Eigen::Vector3d translation = cases.vector3();
        const auto rotated = xgc2_math::stepWorldAcceleration(
            {translation + rotation * state.position, rotation * state.velocity}, rotation * acceleration, dt);
        matrixNear(rotated.position, translation + rotation * full.position, 2.0e-14,
                   sample(i) + " acceleration world-frame equivariance");
        matrixNear(rotated.velocity, rotation * full.velocity, 2.0e-14, sample(i) + " acceleration rotated velocity");
    }
}

void supportMaps() {
    Cases cases;
    const Eigen::Vector3d center(0.8, -1.3, 2.1);
    const Eigen::Vector3d size(0.8, 1.6, 2.4);
    const Eigen::Vector3d radii(0.4, 0.8, 1.2);
    const auto rotation = cases.rotation();
    auto box = std::make_shared<xgc2_math::BoxSet>(center, size, rotation);
    const xgc2_math::EllipsoidSet ellipsoid(center, radii, rotation);
    const xgc2_math::CapsuleSet capsule(center, 0.4, 2.8, rotation);
    const xgc2_math::BallInflatedSet inflated(box, 0.35);
    const Eigen::Vector3d translation(1.7, 0.4, -0.9);
    const xgc2_math::TranslatedSetView<xgc2_math::BoxSet> translated(*box, translation);
    for (int i = 0; i < 96; ++i) {
        const Eigen::Vector3d direction = (0.1 + std::abs(cases.scalar())) * cases.vector3().normalized();
        const auto support = box->support(direction);
        double vertex_max = -1.0e100;
        for (int corner = 0; corner < 8; ++corner) {
            const Eigen::Vector3d signs((corner & 1) != 0 ? 1.0 : -1.0, (corner & 2) != 0 ? 1.0 : -1.0,
                                        (corner & 4) != 0 ? 1.0 : -1.0);
            const Eigen::Vector3d vertex = center + rotation * (0.5 * size.cwiseProduct(signs));
            vertex_max = std::max(vertex_max, direction.dot(vertex));
        }
        near(support.support_value, vertex_max, 4.0e-15, sample(i) + " box support versus all eight vertices");
        near(direction.dot(support.support_point), support.support_value, 3.0e-15, "support witness projection");
        matrixNear(box->support(7.3 * direction).support_point, support.support_point, 2.0e-15,
                   "positive direction scaling must preserve the support point");
        const auto ellipsoid_support = ellipsoid.support(direction);
        const Eigen::Vector3d local = rotation.conjugate() * (ellipsoid_support.support_point - center);
        near(local.cwiseQuotient(radii).squaredNorm(), 1.0, 3.0e-15, "ellipsoid support lies on its boundary");
        const Eigen::Vector3d normal = rotation * local.cwiseQuotient(radii.cwiseProduct(radii));
        matrixNear(normal.normalized(), direction.normalized(), 3.0e-15,
                   "ellipsoid supporting normal is parallel to query");
        const auto capsule_support = capsule.support(direction);
        const Eigen::Vector3d capsule_local = rotation.conjugate() * (capsule_support.support_point - center);
        const Eigen::Vector3d nearest_axis(0.0, 0.0, std::clamp(capsule_local.z(), -1.0, 1.0));
        near((capsule_local - nearest_axis).norm(), 0.4, 3.0e-15, "capsule support lies on swept-ball boundary");
        near(inflated.support(direction).support_value - support.support_value, 0.35 * direction.norm(), 2.0e-15,
             "Minkowski ball support additivity");
        matrixNear(translated.support(direction).support_point, support.support_point + translation, 2.0e-15,
                   "translated support witness");
        near(translated.support(direction).support_value, support.support_value + translation.dot(direction), 2.0e-15,
             "translated support value");
    }
}

template <typename A, typename B>
xgc2_math::gjk::Result distance(const A& a, const B& b, const xgc2_math::gjk::WarmStart* warm = nullptr) {
    return xgc2_math::gjk::DistanceGjkQuery::query(a, b, warm, 1.0e-9, 256, 1.0e-12);
}

void checkSeparator(const xgc2_math::gjk::Result& result, double expected_distance, double tolerance,
                    const std::string& context) {
    require(result.status == xgc2_math::gjk::Result::Status::kSuccess,
            context + ": query did not converge; status=" + std::to_string(static_cast<int>(result.status)));
    near(result.separator.margin, expected_distance, tolerance, context + " distance");
    near(result.separator.normal.norm(), 1.0, 2.0e-14, context + " unit separator");
    matrixNear(result.separator.point_b - result.separator.point_a, result.separator.margin * result.separator.normal,
               1.0e-12, context + " witness separation");
    require(result.iterations <= 256, context + ": iteration budget exceeded");
}

void sphereDistance() {
    Cases cases;
    for (int i = 0; i < 64; ++i) {
        const Eigen::Vector3d center_a = cases.vector3();
        const Eigen::Vector3d normal = cases.vector3().normalized();
        const double radius_a = 0.1 + std::abs(cases.scalar());
        const double radius_b = 0.1 + std::abs(cases.scalar());
        const double gap = 0.05 + std::abs(cases.scalar());
        const Eigen::Vector3d center_b = center_a + (radius_a + radius_b + gap) * normal;
        const xgc2_math::SphereSet a(center_a, radius_a), b(center_b, radius_b);
        const auto result = distance(a, b);
        checkSeparator(result, gap, 4.0e-14, sample(i) + " sphere distance");
        matrixNear(result.separator.point_a, center_a + radius_a * normal, 4.0e-14, "sphere A closest witness");
        matrixNear(result.separator.point_b, center_b - radius_b * normal, 4.0e-14, "sphere B closest witness");
        const auto reverse = distance(b, a);
        checkSeparator(reverse, gap, 4.0e-14, "swapped sphere distance");
        matrixNear(reverse.separator.normal, -result.separator.normal, 3.0e-14, "swapped sphere normal");
        const xgc2_math::gjk::WarmStart warm{result.separator, true};
        checkSeparator(distance(a, b, &warm), gap, 4.0e-14, "warm sphere query");
        const xgc2_math::SphereSet intersecting(center_a + 0.5 * (radius_a + radius_b) * normal, radius_b);
        require(distance(a, intersecting).status == xgc2_math::gjk::Result::Status::kOverlap,
                "intersecting spheres must report overlap, not a positive-distance certificate");
    }
}

void boxSphereDistance() {
    Cases cases;
    const Eigen::Vector3d half_size(0.4, 0.7, 1.1);
    // Face, edge and corner projections have independent point-to-AABB oracles.
    const std::array<Eigen::Vector3d, 5> local_centers{
        {Eigen::Vector3d(2.0, 0.2, -0.3), Eigen::Vector3d(2.0, 1.8, 0.2), Eigen::Vector3d(2.0, 1.8, 2.1),
         Eigen::Vector3d(-1.9, -1.4, 1.6), Eigen::Vector3d(0.1, -0.2, -2.1)}};
    for (int i = 0; i < 25; ++i) {
        const auto rotation = cases.rotation();
        const Eigen::Vector3d center = 2.0 * cases.vector3();
        const Eigen::Vector3d local_center = local_centers[static_cast<std::size_t>(i) % local_centers.size()];
        const Eigen::Vector3d projection = local_center.cwiseMax(-half_size).cwiseMin(half_size);
        const Eigen::Vector3d normal = (local_center - projection).normalized();
        const double radius = 0.23;
        const double gap = (local_center - projection).norm() - radius;
        const xgc2_math::BoxSet box(center, 2.0 * half_size, rotation);
        const xgc2_math::SphereSet sphere(center + rotation * local_center, radius);
        const auto result = distance(box, sphere);
        checkSeparator(result, gap, 2.0e-8, sample(i) + " oriented box/sphere");
        matrixNear(result.separator.point_a, center + rotation * projection, 3.0e-6, "box/sphere box witness");
        matrixNear(result.separator.point_b, center + rotation * (local_center - radius * normal), 3.0e-6,
                   "box/sphere sphere witness");
        checkSeparator(distance(sphere, box), gap, 2.0e-8, "box/sphere swapped distance");
    }
}

void boxBoxDistance() {
    Cases cases;
    const Eigen::Vector3d half_a(0.4, 0.7, 0.9), half_b(0.6, 0.2, 0.3);
    const std::array<Eigen::Vector3d, 4> offsets{{Eigen::Vector3d(2.0, 0.1, 0.1), Eigen::Vector3d(1.5, -1.6, 0.4),
                                                  Eigen::Vector3d(-1.4, 1.7, 1.9), Eigen::Vector3d(0.1, -0.2, -2.0)}};
    for (int i = 0; i < 32; ++i) {
        const auto rotation = cases.rotation();
        const Eigen::Vector3d origin = 2.0 * cases.vector3();
        const Eigen::Vector3d offset = offsets[static_cast<std::size_t>(i) % offsets.size()];
        const double expected = (offset.cwiseAbs() - half_a - half_b).cwiseMax(0.0).norm();
        const xgc2_math::BoxSet a(origin, 2.0 * half_a, rotation);
        const xgc2_math::BoxSet b(origin + rotation * offset, 2.0 * half_b, rotation);
        const auto result = distance(a, b);
        checkSeparator(result, expected, 2.0e-12, sample(i) + " co-oriented boxes");
        const Eigen::Vector3d local_a = rotation.conjugate() * (result.separator.point_a - origin);
        const Eigen::Vector3d local_b = rotation.conjugate() * (result.separator.point_b - origin) - offset;
        require((local_a.cwiseAbs().array() <= half_a.array() + 1.0e-12).all(), "box A witness is outside box A");
        require((local_b.cwiseAbs().array() <= half_b.array() + 1.0e-12).all(), "box B witness is outside box B");
        checkSeparator(distance(b, a), expected, 2.0e-12, "co-oriented boxes swapped");
    }
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 10> tests{{
        {"se2_group", se2Group},
        {"se3_group", se3Group},
        {"rotation_maps", rotationMaps},
        {"body_velocity_flow", bodyVelocityFlow},
        {"body_velocity_quadrature", bodyVelocityQuadrature},
        {"acceleration_flow", accelerationFlow},
        {"support_maps", supportMaps},
        {"sphere_distance", sphereDistance},
        {"box_sphere_distance", boxSphereDistance},
        {"box_box_distance", boxBoxDistance},
    }};
    if (argc > 2) {
        std::cerr << "Usage: geometry_property_test [case_name|--list]\n";
        return 2;
    }
    const std::string requested = argc == 2 ? argv[1] : "";
    if (requested == "--list") {
        for (const auto& test : tests) {
            std::cout << test.name << '\n';
        }
        return 0;
    }
    int failures = 0;
    int executed = 0;
    for (const auto& test : tests) {
        if (!requested.empty() && requested != test.name) {
            continue;
        }
        ++executed;
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << " (seed=" << kSeed << "): " << error.what() << '\n';
        }
    }
    if (executed == 0) {
        std::cerr << "Unknown case: " << requested << " (use --list)\n";
        return 2;
    }
    // Explicit checks remain active in Release/RelWithDebInfo, including NDEBUG.
    return failures == 0 ? 0 : 1;
}
