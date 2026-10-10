#!/usr/bin/env bash

set -euo pipefail

dpkg -s libxgc2-math-dev >/dev/null
dpkg -s libxgc2-math-algebra-dev >/dev/null
dpkg -s libxgc2-math-utils-dev >/dev/null
dpkg -s libxgc2-math-geometry-dev >/dev/null
dpkg -s libxgc2-math-filter-dev >/dev/null
dpkg -s libxgc2-math-observer-dev >/dev/null
dpkg -s libxgc2-math-estimation-dev >/dev/null
dpkg -s libxgc2-math-optimization-dev >/dev/null
dpkg -s libxgc2-math-trajectory-dev >/dev/null
dpkg -s libxgc2-math-control-dev >/dev/null
dpkg -s libxgc2-math-scene-dev >/dev/null

test -f /usr/include/xgc2_math/math.hpp
test -f /usr/include/xgc2_math/types.hpp
test -f /usr/include/xgc2_math/algebra.hpp
test -f /usr/include/xgc2_math/algebra/angle.hpp
test -f /usr/include/xgc2_math/utils/sample_timing.hpp
test -f /usr/include/xgc2_math/geometry/se2.hpp
test -f /usr/include/xgc2_math/geometry/se3.hpp
test -f /usr/include/xgc2_math/geometry/occupied_sets/sphere_set.h
test -f /usr/include/xgc2_math/filter/exponential_filter.hpp
test -f /usr/include/xgc2_math/observer/differentiator.hpp
test -f /usr/include/xgc2_math/estimation/pose3_inertial_eskf.hpp
test ! -f /usr/include/xgc2_math/estimation/pose2_inertial_eskf.hpp
test -f /usr/include/xgc2_math/optimization/minco.hpp
test -f /usr/include/xgc2_math/optimization/lbfgs_lite/lbfgs.hpp
test -f /usr/include/xgc2_math/optimization/lbfgs_lite/LICENSE
test -f /usr/include/xgc2_math/optimization/lbfgs_lite/source.json
test -f /usr/include/xgc2_math/trajectory/trajectory3.hpp
test -f /usr/include/xgc2_math/trajectory/analytic/2d/circle_entry_2d.hpp
test -f /usr/include/xgc2_math/trajectory/analytic/3d/circle_entry_3d.hpp
test -f /usr/include/xgc2_math/trajectory/analytic/3d/helix_xy_3d.hpp
test -f /usr/include/xgc2_math/trajectory/analytic/3d/helix_yz_3d.hpp
test -f /usr/include/xgc2_math/trajectory/analytic/3d/torus_knot_3d.hpp
test -f /usr/include/xgc2_math/control/se2_nmpc_problem.hpp
test -f /usr/include/xgc2_math/control/se3_nmpc_problem.hpp
test -f /usr/include/xgc2_math/control/dfbc_geometric_controller.hpp
test -f /usr/include/xgc2_math/scene.hpp
test -f /usr/include/xgc2_math/scene/world_lidar.hpp
test -f /usr/include/xgc2_math/scene/lidar_scene.hpp
test -f /usr/include/xgc2_math/scene/README.md
test -f /usr/lib/cmake/xgc2_math/xgc2_mathConfig.cmake

probe_dir="${XGC2_MATH_SMOKE_DIR:-$(mktemp -d -t xgc2-math-smoke-XXXXXX)}"
mkdir -p "${probe_dir}"

cat > "${probe_dir}/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.10)
project(xgc2_math_probe LANGUAGES CXX)

find_package(xgc2_math REQUIRED CONFIG)
xgc2_math_require()

add_executable(link_probe link_probe.cpp)
target_compile_features(link_probe PRIVATE cxx_std_17)
target_link_libraries(link_probe PRIVATE xgc2_math::math)

# The scene component alone: its target must bring everything its headers include.
add_executable(scene_probe scene_probe.cpp)
target_compile_features(scene_probe PRIVATE cxx_std_17)
target_link_libraries(scene_probe PRIVATE xgc2_math::scene)
CMAKE

cat > "${probe_dir}/link_probe.cpp" <<'CPP'
#include <xgc2_math/math.hpp>
#include <xgc2_math/control/dfbc_geometric_controller.hpp>

int main()
{
  xgc2_math::SecondOrderButterworthLowPass filter(5.0, 0.0);
  xgc2_math::ExponentialLowPass smoother(2.0, 0.0);
  xgc2_math::TimeDeltaGuard dt_guard;
  xgc2_math::Differentiator diff;
  xgc2_math::AngleDifferentiator yaw_diff;
  xgc2_math::PositionVelocityLuenbergerObserver observer;
  xgc2_math::AngularPositionVelocityLuenbergerObserver yaw_observer;
  xgc2_math::ArrayDifferentiator<3> array_diff;
  xgc2_math::ScalarRecursiveLeastSquares rls;
  xgc2_math::Pose3InertialEskf pose3_inertial_eskf;
  xgc2_math::SphereSet sphere(Eigen::Vector3d::Zero(), 1.0);
  xgc2_math::Pose2 planar_pose;
  xgc2_math::trajectory::CircleEntryCurveEvaluator3 entry_curve;
  xgc2_math::trajectory::FlatOutput3 flat;
  xgc2_math::control::Se3State se3_state;
  xgc2_math::control::Se2Control se2_control;
  xgc2_math::control::DfbcGeometricController dfbc;

  const auto dt0 = dt_guard.update(1.0);
  const auto dt1 = dt_guard.update(1.01);
  const bool stale = xgc2_math::sampleStale(2.0, 1.0, 0.5);
  const bool low_rate = xgc2_math::sampleRateLow(true, 1.0, 5.0);
  const bool jumped = xgc2_math::sampleTimeJumped(1.0, 1.2, 0.01);
  const double filtered = filter.filter(1.0, 0.01);
  const double smoothed = smoother.filter(filtered, 0.01);
  const auto derivative = diff.update(smoothed, dt1.dt_s);
  const auto yaw_derivative = yaw_diff.update(xgc2_math::normalizeAngle(3.14), 0.01);
  const auto estimate = observer.update(smoothed, 0.01);
  const auto yaw_estimate = yaw_observer.update(3.14, 0.01);
  const auto array_samples = array_diff.update({{0.0, 1.0, 2.0}}, 0.01);
  rls.reset(2.0);
  const auto rls_sample = rls.update(4.0, 2.0);
  xgc2_math::PoseMeasurement pose_sample;
  pose_sample.received = true;
  pose_sample.valid = true;
  pose_sample.stamp_sec = 1.0;
  pose3_inertial_eskf.initializeFromPose(pose_sample, nullptr);
  planar_pose = xgc2_math::compose(planar_pose, xgc2_math::Pose2{Eigen::Vector2d::UnitX(), 0.1});
  const bool curve_ok = entry_curve.evaluate(0.1, flat);
  const auto se3_vec = xgc2_math::control::packState(se3_state);
  se2_control.yaw_rate = 0.2;
  const auto se2_u = xgc2_math::control::packControl(se2_control);
  const auto dfbc_output = dfbc.compute(xgc2_math::control::DfbcGeometricInput{});
  xgc2_math::scene::WorldLidar lidar{xgc2_math::scene::SensorConfig{}};
  lidar.setScene({xgc2_math::scene::Obstacle::box({5.0, 0.0, 0.0}, {1.0, 4.0, 4.0})});
  const double wall_distance = lidar.castRay(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(), 20.0);

  return !dt0.accepted && dt1.accepted &&
         stale &&
         low_rate &&
         jumped &&
         derivative.measurement_accepted &&
         yaw_derivative.measurement_accepted &&
         estimate.measurement_accepted &&
         yaw_estimate.measurement_accepted &&
         array_samples[0].measurement_accepted &&
         rls_sample.measurement_accepted &&
         pose3_inertial_eskf.initialized() &&
         curve_ok &&
         se3_vec.size() == 13 &&
         se2_u(1) > 0.1 &&
         dfbc_output.success &&
         std::abs(wall_distance - 4.5) < 1e-9 &&
         std::abs(dfbc_output.specific_thrust - dfbc.config().gravity) < 1e-9 &&
         planar_pose.position.x() > 0.9 &&
         sphere.support(Eigen::Vector3d::UnitX()).support_point.x() > 0.9 ? 0 : 1;
}
CPP

cat > "${probe_dir}/scene_probe.cpp" <<'CPP'
#include <xgc2_math/scene.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

int main()
{
  using namespace xgc2_math::scene;

  // A world with a box and a rock, scanned by a raycast and a penetrating sensor.
  std::vector<Obstacle> obstacles;
  obstacles.push_back(Obstacle::box({5.0, 0.0, 0.0}, {1.0, 4.0, 4.0}));
  obstacles.push_back(Obstacle::convexHull(
      {{-0.5, -0.4, -0.5}, {0.5, -0.4, -0.5}, {0.0, 0.5, -0.5}, {0.0, 0.0, 0.8}}, {0.0, 6.0, 0.0},
      Eigen::Quaterniond::Identity()));
  const auto scene = std::make_shared<const LidarScene>(obstacles, 0.1, true);

  SensorConfig ray;
  WorldLidar raycast(ray);
  raycast.setScene(scene);
  std::vector<std::uint8_t> cloud;
  const std::size_t returns = raycast.scanInto(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), {}, false, &cloud);

  SensorConfig sphere;
  sphere.mode = SensorConfig::kPenetrating;
  sphere.v_fov_deg = 180.0;
  sphere.penetrating_keep_buried = true;
  WorldLidar penetrating(sphere);
  penetrating.setScene(scene);
  const auto points = penetrating.scan(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());

  // Pose algebra and the closed-surface check of scene documents.
  const Motion hold;
  const ElementPlacement element{10.0, 20.0, 1.5707963267948966};
  const auto moved = placeMotion(element, hold);
  const auto triangles = validateConvexMesh(
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3});

  return returns > 0 && cloud.size() == 12 * returns && !points.empty() && moved.type == Motion::kHold &&
         triangles.size() == 12 && std::abs(element.pose().position.y() - 20.0) < 1e-12 ? 0 : 1;
}
CPP

mkdir -p "${probe_dir}/build"
(cd "${probe_dir}/build" && cmake "${probe_dir}")
cmake --build "${probe_dir}/build" -- -j2
"${probe_dir}/build/link_probe"
"${probe_dir}/build/scene_probe"

echo "libxgc2-math-dev installed smoke test passed."
