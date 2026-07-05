#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

bash -n .xgc2/scripts/*.sh

nested_git="$(
  find . \
    -path ./.git -prune -o \
    -path ./.ci -prune -o \
    -path ./build -prune -o \
    -name .git -print
)"
if [[ -n "${nested_git}" ]]; then
  echo "Nested .git directory found." >&2
  echo "${nested_git}" >&2
  exit 1
fi

if git ls-files 2>/dev/null | grep -E '(^|/)(build|devel|install|\.catkin_tools|\.ci)(/|$)' >/dev/null; then
  echo "Generated build artifacts are tracked." >&2
  git ls-files | grep -E '(^|/)(build|devel|install|\.catkin_tools|\.ci)(/|$)' >&2
  exit 1
fi

required_files=(
  .clang-format
  .clang-tidy
  README.md
  CMakeLists.txt
  cmake/xgc2_mathConfig.cmake.in
  include/xgc2_math/math.hpp
  include/xgc2_math/types.hpp
  include/xgc2_math/utils.hpp
  include/xgc2_math/geometry.hpp
  include/xgc2_math/filter.hpp
  include/xgc2_math/observer.hpp
  include/xgc2_math/estimation.hpp
  include/xgc2_math/optimization.hpp
  include/xgc2_math/trajectory.hpp
  include/xgc2_math/control.hpp
  include/xgc2_math/algebra.hpp
  include/xgc2_math/algebra/angle.hpp
  include/xgc2_math/utils/status.hpp
  include/xgc2_math/utils/sample_timing.hpp
  include/xgc2_math/utils/time_delta.hpp
  include/xgc2_math/geometry/se3.hpp
  include/xgc2_math/geometry/occupied_sets/sphere_set.h
  include/xgc2_math/filter/exponential_filter.hpp
  include/xgc2_math/observer/differentiator.hpp
  include/xgc2_math/estimation/pose3_inertial_eskf.hpp
  include/xgc2_math/optimization/minco.hpp
  include/xgc2_math/trajectory/trajectory3.hpp
  include/xgc2_math/trajectory/analytic/2d/circle_entry_2d.hpp
  include/xgc2_math/trajectory/analytic/3d/circle_entry_3d.hpp
  include/xgc2_math/trajectory/analytic/3d/helix_xy_3d.hpp
  include/xgc2_math/trajectory/analytic/3d/helix_yz_3d.hpp
  include/xgc2_math/trajectory/analytic/3d/torus_knot_3d.hpp
  include/xgc2_math/control/se3_nmpc_problem.hpp
  test/math_header_test.cpp
  .github/workflows/ci.yml
  .xgc2/product.yml
  .xgc2/scripts/build_deb.sh
  .xgc2/scripts/check_cpp_quality.sh
  .xgc2/scripts/check_package_compliance.sh
  .xgc2/scripts/publish_self_hosted_apt.sh
  .xgc2/scripts/smoke_test_installed.sh
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required file: ${file}" >&2
    exit 1
  fi
done

for removed_file in package.xml .github/workflows/build-debs.yml include/xgc2_observer cmake/xgc2_observerConfig.cmake.in; do
  if [[ -e "${removed_file}" ]]; then
    echo "xgc2-math package must not keep obsolete file: ${removed_file}" >&2
    exit 1
  fi
done

if grep -R "find_package(catkin\\|catkin_package\\|catkin_add_gtest" CMakeLists.txt cmake 2>/dev/null; then
  echo "system package must not use catkin." >&2
  exit 1
fi

if grep -R "xgc2_observer\\|xgc2_geometry" CMakeLists.txt cmake include test README.md .github .xgc2/product.yml 2>/dev/null; then
  echo "obsolete xgc2_observer/xgc2_geometry identifiers remain." >&2
  exit 1
fi

echo "libxgc2-math-dev package compliance checks passed."
