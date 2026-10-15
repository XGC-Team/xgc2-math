#!/usr/bin/env bash

set -euo pipefail

# Run with --isolated after installing only geometry-dev and its algebra/utils
# dependencies. The default also works as part of the full installed SDK smoke.
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--isolated" ) ]]; then
  echo "usage: $0 [--isolated]" >&2
  exit 2
fi
for package in libxgc2-math-algebra-dev libxgc2-math-utils-dev libxgc2-math-geometry-dev; do
  test "$(dpkg-query -W -f='${Status}' "${package}")" = 'install ok installed'
done
if [[ "${1:-}" == "--isolated" ]]; then
  for package in libxgc2-math-scene-dev libxgc2-math-dev; do
    if dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q '^install ok installed$'; then
      echo "isolated geometry smoke must not have ${package} installed" >&2
      exit 1
    fi
  done
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while IFS= read -r relative_header; do
  case "${relative_header}" in ''|'#'*) continue ;; esac
  header="/usr/include/xgc2_math/${relative_header}"
  test -f "${header}"
  test "$(dpkg-query -S "${header}")" = "libxgc2-math-geometry-dev: ${header}"
done < "${script_dir}/../geometry-scene-headers.txt"
for installed_file in \
  /usr/include/xgc2_math/geometry/geometry_payload.hpp \
  /usr/include/xgc2_math/geometry/geometry_json.hpp \
  /usr/include/xgc2_math/geometry/geometry_registry.def \
  /usr/include/xgc2_math/geometry/prepared_geometry.hpp \
  /usr/lib/python3/dist-packages/xgc2_math/__init__.py \
  /usr/lib/python3/dist-packages/xgc2_math/geometry.py \
  /usr/share/xgc2_math/typescript/geometry.ts \
  /usr/share/xgc2_math/typescript/geometry_registry.ts; do
  test -f "${installed_file}"
  test "$(dpkg-query -S "${installed_file}")" = "libxgc2-math-geometry-dev: ${installed_file}"
done

probe_dir="${XGC2_MATH_GEOMETRY_SMOKE_DIR:-$(mktemp -d -t xgc2-math-geometry-smoke-XXXXXX)}"
mkdir -p "${probe_dir}"
cat > "${probe_dir}/prepared_probe.cpp" <<'CPP'
#include <xgc2_math/geometry/prepared_geometry.hpp>
#include <xgc2_math/geometry/collision/distance_gjk_query.h>
#include <xgc2_math/geometry/occupied_sets/sphere_set.h>

int main() {
    using namespace xgc2_math::geometry;
    const auto payload = encodeGeometry(Geometry::box({1, 2, 3}));
    const auto box = prepareGeometry(payload);
    const auto transform = prepareConvexPieceTransform(box.compiledParts.front(), ConvexInstanceTransform{});
    const xgc2_math::SphereSet sphere({5, 0, 0}, .25);
    const bool separated = visitConvexPiece(transform, [&](const auto& set) {
        const auto result = xgc2_math::gjk::DistanceGjkQuery::query(set, sphere, nullptr, .01, 128, 1e-10);
        return result.status == xgc2_math::gjk::Result::Status::kSuccess && result.separator.margin > 4;
    });
    const auto ring = prepareGeometry(encodeGeometry(Geometry::ring({3, 2}, .25)));
    const auto h = prepareGeometry(encodeGeometry(Geometry::halfspaces(
        {{1, 0, 0, -1}, {-1, 0, 0, -1}, {0, 1, 0, -1}, {0, -1, 0, -1}, {0, 0, 1, -1}, {0, 0, -1, -1}})));
    return separated && box.original == payload && ring.parts.size() == 128 &&
                   ringPointResidual(ring, {0, 0, 0}) > 0 && h.parts.size() == 1
               ? 0
               : 1;
}
CPP
"${CXX:-c++}" -std=c++17 -O2 -I/usr/include/eigen3 "${probe_dir}/prepared_probe.cpp" -o "${probe_dir}/prepared_probe"
"${probe_dir}/prepared_probe"

# -I ignores source directories and PYTHONPATH: this must load the installed package.
python3 -I - <<'PY'
from pathlib import Path
from xgc2_math import geometry
assert Path(geometry.__file__).resolve() == Path('/usr/lib/python3/dist-packages/xgc2_math/geometry.py')
assert len(geometry.REGISTRY) == 9
shape = {'type': 'ring', 'radii': [3, 2], 'tubeRadius': 0.25}
kind, parameters = geometry.encode_geometry(shape)
assert kind == 6 and len(parameters) == 24
assert geometry.decode_geometry(kind, parameters) == shape
halfspace = {'type': 'halfspaces', 'halfspaces': [[1, 0, 0, -1]]}
kind, parameters = geometry.encode_geometry(halfspace)
assert kind == 7 and geometry.decode_geometry(kind, parameters) == halfspace
registry = Path('/usr/share/xgc2_math/typescript/geometry_registry.ts').read_text()
assert 'export const geometryRegistry' in registry and 'export type NamedGeometry' in registry
assert 'Uint8Array' not in registry and 'encodeGeometry' not in registry
print('installed geometry Python codec and TS registry passed')
PY

echo "geometry-dev installed prepared/header/binding smoke passed${1:+ (isolated)}."
