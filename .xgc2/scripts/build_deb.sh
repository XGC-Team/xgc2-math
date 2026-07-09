#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

product_version() {
  awk -F': *' '/^version:[[:space:]]*/ {print $2; exit}' "${repo_root}/.xgc2/product.yml"
}

package_base_version="${PACKAGE_BASE_VERSION:-$(product_version)}"
package_distribution="${PACKAGE_DISTRIBUTION:-${APT_REPO_DISTRIBUTION:-}}"
if [[ -z "${package_base_version}" ]]; then
  echo "package version is missing; set PACKAGE_BASE_VERSION or .xgc2/product.yml version" >&2
  exit 1
fi

if [[ -z "${package_distribution}" && -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  package_distribution="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
fi

if [[ -n "${PACKAGE_VERSION:-}" ]]; then
  version="${PACKAGE_VERSION}"
else
  if [[ -z "${package_distribution}" ]]; then
    echo "PACKAGE_DISTRIBUTION or VERSION_CODENAME is required for Debian package versioning" >&2
    exit 1
  fi
  version="${package_base_version}~${package_distribution}"
fi

if [[ -n "${package_distribution}" && "${ALLOW_UNSCOPED_DEB_VERSION:-0}" != "1" ]]; then
  case "${version}" in
    *"~${package_distribution}"*|*"+"${package_distribution}*) ;;
    *)
      echo "Debian package version '${version}' must include distribution suffix '${package_distribution}'" >&2
      exit 1
      ;;
  esac
fi

build_dir="${XGC2_MATH_BUILD_DIR:-${repo_root}/.ci/build}"
stage_dir="${XGC2_MATH_STAGE_DIR:-${repo_root}/.ci/stage}"
output_dir="${XGC2_MATH_DEB_OUTPUT_DIR:-${repo_root}/.ci/debs}"
pkg_dir="${repo_root}/.ci/pkg"
arch="all"

rm -rf "${build_dir}" "${stage_dir}" "${output_dir}" "${pkg_dir}"
mkdir -p "${output_dir}" "${pkg_dir}"

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -- -j"$(nproc)"
(cd "${build_dir}" && ctest --output-on-failure)
DESTDIR="${stage_dir}" cmake --install "${build_dir}" --prefix /usr

copy_path() {
  local src="$1"
  local dst_root="$2"
  if [[ -e "${src}" ]]; then
    mkdir -p "${dst_root}$(dirname "${src#${stage_dir}}")"
    cp -a "${src}" "${dst_root}${src#${stage_dir}}"
  fi
}

write_control() {
  local package_name="$1"
  local depends="$2"
  local description="$3"
  local pkg_root="${pkg_dir}/${package_name}"

  mkdir -p "${pkg_root}/DEBIAN" "${pkg_root}/usr/share/doc/${package_name}"
  cat > "${pkg_root}/DEBIAN/control" <<EOF
Package: ${package_name}
Version: ${version}
Section: libdevel
Priority: optional
Architecture: ${arch}
Maintainer: XGC2 <apt@example.com>
Depends: ${depends}
Description: ${description}
 Header-only XGC2 C++ math development package.
EOF
  cp -a "${repo_root}/README.md" "${pkg_root}/usr/share/doc/${package_name}/"
  chmod 0755 "${pkg_root}/DEBIAN"
}

build_package() {
  local package_name="$1"
  local pkg_root="${pkg_dir}/${package_name}"
  find "${pkg_root}" -type d -exec chmod 0755 {} +
  find "${pkg_root}" -type f -exec chmod 0644 {} +
  chmod 0755 "${pkg_root}/DEBIAN"
  fakeroot dpkg-deb --build "${pkg_root}" "${output_dir}/${package_name}_${version}_${arch}.deb" >/dev/null
  dpkg-deb -I "${output_dir}/${package_name}_${version}_${arch}.deb"
}

gcc_runtime_dep="libgcc-s1"
if [[ "${package_distribution}" == "bionic" ]]; then
  gcc_runtime_dep="libgcc1"
fi
base_depends="libeigen3-dev, libc6, ${gcc_runtime_dep}, libstdc++6"

algebra_pkg="libxgc2-math-algebra-dev"
write_control "${algebra_pkg}" "${base_depends}" "XGC2 math algebra headers"
copy_path "${stage_dir}/usr/include/xgc2_math/algebra" "${pkg_dir}/${algebra_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/algebra.hpp" "${pkg_dir}/${algebra_pkg}"

utils_pkg="libxgc2-math-utils-dev"
write_control "${utils_pkg}" "${base_depends}" "XGC2 math utils headers"
copy_path "${stage_dir}/usr/include/xgc2_math/utils" "${pkg_dir}/${utils_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/utils.hpp" "${pkg_dir}/${utils_pkg}"

geometry_pkg="libxgc2-math-geometry-dev"
write_control "${geometry_pkg}" "libxgc2-math-algebra-dev (= ${version}), libxgc2-math-utils-dev (= ${version}), ${base_depends}" \
  "XGC2 math geometry headers"
copy_path "${stage_dir}/usr/include/xgc2_math/geometry" "${pkg_dir}/${geometry_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/geometry.hpp" "${pkg_dir}/${geometry_pkg}"

filter_pkg="libxgc2-math-filter-dev"
write_control "${filter_pkg}" "libxgc2-math-algebra-dev (= ${version}), libxgc2-math-utils-dev (= ${version}), ${base_depends}" \
  "XGC2 math filter headers"
copy_path "${stage_dir}/usr/include/xgc2_math/filter" "${pkg_dir}/${filter_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/filter.hpp" "${pkg_dir}/${filter_pkg}"

observer_pkg="libxgc2-math-observer-dev"
write_control "${observer_pkg}" \
  "libxgc2-math-algebra-dev (= ${version}), libxgc2-math-utils-dev (= ${version}), libxgc2-math-filter-dev (= ${version}), ${base_depends}" \
  "XGC2 math observer headers"
copy_path "${stage_dir}/usr/include/xgc2_math/observer" "${pkg_dir}/${observer_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/observer.hpp" "${pkg_dir}/${observer_pkg}"

estimation_pkg="libxgc2-math-estimation-dev"
write_control "${estimation_pkg}" \
  "libxgc2-math-algebra-dev (= ${version}), libxgc2-math-utils-dev (= ${version}), libxgc2-math-geometry-dev (= ${version}), libxgc2-math-filter-dev (= ${version}), libxgc2-math-observer-dev (= ${version}), ${base_depends}" \
  "XGC2 math estimation headers"
copy_path "${stage_dir}/usr/include/xgc2_math/estimation" "${pkg_dir}/${estimation_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/estimation.hpp" "${pkg_dir}/${estimation_pkg}"

optimization_pkg="libxgc2-math-optimization-dev"
write_control "${optimization_pkg}" "libxgc2-math-utils-dev (= ${version}), ${base_depends}" \
  "XGC2 math optimization headers"
copy_path "${stage_dir}/usr/include/xgc2_math/optimization" "${pkg_dir}/${optimization_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/optimization.hpp" "${pkg_dir}/${optimization_pkg}"

trajectory_pkg="libxgc2-math-trajectory-dev"
write_control "${trajectory_pkg}" \
  "libxgc2-math-utils-dev (= ${version}), libxgc2-math-geometry-dev (= ${version}), libxgc2-math-optimization-dev (= ${version}), ${base_depends}" \
  "XGC2 math trajectory headers"
copy_path "${stage_dir}/usr/include/xgc2_math/trajectory" "${pkg_dir}/${trajectory_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/trajectory.hpp" "${pkg_dir}/${trajectory_pkg}"

control_pkg="libxgc2-math-control-dev"
write_control "${control_pkg}" "libxgc2-math-utils-dev (= ${version}), ${base_depends}" \
  "XGC2 math control headers"
copy_path "${stage_dir}/usr/include/xgc2_math/control" "${pkg_dir}/${control_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/control.hpp" "${pkg_dir}/${control_pkg}"

meta_pkg="libxgc2-math-dev"
write_control "${meta_pkg}" \
  "libxgc2-math-algebra-dev (= ${version}), libxgc2-math-utils-dev (= ${version}), libxgc2-math-geometry-dev (= ${version}), libxgc2-math-filter-dev (= ${version}), libxgc2-math-observer-dev (= ${version}), libxgc2-math-estimation-dev (= ${version}), libxgc2-math-optimization-dev (= ${version}), libxgc2-math-trajectory-dev (= ${version}), libxgc2-math-control-dev (= ${version}), ${base_depends}" \
  "XGC2 math meta development package"
copy_path "${stage_dir}/usr/include/xgc2_math/math.hpp" "${pkg_dir}/${meta_pkg}"
copy_path "${stage_dir}/usr/include/xgc2_math/types.hpp" "${pkg_dir}/${meta_pkg}"
copy_path "${stage_dir}/usr/lib/cmake/xgc2_math" "${pkg_dir}/${meta_pkg}"
copy_path "${stage_dir}/usr/share/doc/libxgc2-math-dev/matlab" "${pkg_dir}/${meta_pkg}"

test -f "${pkg_dir}/${algebra_pkg}/usr/include/xgc2_math/algebra.hpp"
test -f "${pkg_dir}/${algebra_pkg}/usr/include/xgc2_math/algebra/angle.hpp"
test -f "${pkg_dir}/${utils_pkg}/usr/include/xgc2_math/utils/status.hpp"
test -f "${pkg_dir}/${utils_pkg}/usr/include/xgc2_math/utils/sample_timing.hpp"
test -f "${pkg_dir}/${geometry_pkg}/usr/include/xgc2_math/geometry/se2.hpp"
test -f "${pkg_dir}/${geometry_pkg}/usr/include/xgc2_math/geometry/occupied_sets/sphere_set.h"
test -f "${pkg_dir}/${filter_pkg}/usr/include/xgc2_math/filter/exponential_filter.hpp"
test -f "${pkg_dir}/${observer_pkg}/usr/include/xgc2_math/observer/differentiator.hpp"
test -f "${pkg_dir}/${estimation_pkg}/usr/include/xgc2_math/estimation/pose3_inertial_eskf.hpp"
test -f "${pkg_dir}/${estimation_pkg}/usr/include/xgc2_math/estimation/pose2_inertial_eskf.hpp"
test -f "${pkg_dir}/${optimization_pkg}/usr/include/xgc2_math/optimization/minco.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/trajectory3.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/2d/circle_2d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/2d/circle_entry_2d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/2d/figure_eight_2d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/2d/hold_2d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/circle_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/circle_entry_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/figure_eight_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/line_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/lemniscate_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/helix_yz_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/helix_xy_3d.hpp"
test -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/3d/torus_knot_3d.hpp"
test ! -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/circle_entry.hpp"
test ! -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/line.hpp"
test ! -f "${pkg_dir}/${trajectory_pkg}/usr/include/xgc2_math/trajectory/analytic/koopman.hpp"
test -f "${pkg_dir}/${control_pkg}/usr/include/xgc2_math/control.hpp"
test -f "${pkg_dir}/${control_pkg}/usr/include/xgc2_math/control/se3_nmpc_problem.hpp"
test -f "${pkg_dir}/${meta_pkg}/usr/include/xgc2_math/math.hpp"
test -f "${pkg_dir}/${meta_pkg}/usr/lib/cmake/xgc2_math/xgc2_mathConfig.cmake"

for package_name in \
  "${algebra_pkg}" \
  "${utils_pkg}" \
  "${geometry_pkg}" \
  "${filter_pkg}" \
  "${observer_pkg}" \
  "${estimation_pkg}" \
  "${optimization_pkg}" \
  "${trajectory_pkg}" \
  "${control_pkg}" \
  "${meta_pkg}"; do
  build_package "${package_name}"
done

echo "Debian artifacts written to ${output_dir}"
