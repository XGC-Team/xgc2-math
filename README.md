# XGC2 Math

Header-only C++17 math library for geometry, filtering, estimation, trajectory
evaluation, optimization and control.

```bash
sudo apt update
sudo apt install libxgc2-math-dev
```

```cmake
find_package(xgc2_math REQUIRED CONFIG)
xgc2_math_require()
target_link_libraries(my_target PRIVATE xgc2_math::math)
```

```cpp
#include <xgc2_math/math.hpp>
```

Component headers and targets are also available, such as
`<xgc2_math/control.hpp>` and `xgc2_math::control`. The scene component
(`<xgc2_math/scene.hpp>`, `xgc2_math::scene`: scene modeling, ray casting and
LiDAR / depth-camera models) is described in
[include/xgc2_math/scene/README.md](include/xgc2_math/scene/README.md).

Geometry payloads use `xgc2_math/geometry/geometry_payload.hpp`: `encodeGeometry(Geometry::box({1,2,3}))`
returns the numeric `{type, parameters}` transport value, and `decodeGeometry` returns local numerical parameters.
The registry is `geometry/geometry_registry.def`; `tools/generate_geometry_bindings.py` produces the minimal
Python and TypeScript codecs. The optional templated `geometry_json.hpp` bridge preserves readable named JSON,
including flat triangle indices. It introduces no JSON or ROS runtime dependency.
Python installs under `${prefix}/lib/python3/dist-packages/xgc2_math`; TypeScript installs under
`${prefix}/share/xgc2_math/typescript`. Override `XGC2_MATH_PYTHON_INSTALL_DIR` for another Python layout.

`geometry/prepared_geometry.hpp` prepares inline numerical geometry once, including arbitrary hull vertices and
bounded halfspaces. `prepareConvexParts` returns numerical shapes and local rigid poses; it does not resolve IDs,
assets, sessions or templates. Closed consistently oriented polyhedral meshes use a bounded face-plane arrangement
and winding classification, preserving cavities. This is intended for non-self-intersecting solid meshes; numerical
plane/hull tolerances still apply. Plane/cell resource limits reject excessive inputs explicitly.
Elliptic rings retain their original parameters and exact point/ray routines. The algorithm-side ring representation
uses 128 chord capsules by default. Each chord radius is `tubeRadius + delta`, with
`delta = max(a,b)*(2*pi/N)^2/8`; their union covers the tube and is within `2*delta` of the original solid.
This bound is reported in the numerical preparation result. Very eccentric rings refine the segment count to preserve the hole, within the numerical resource limit;
`ringSegments` sets the minimum numerical resolution without changing the author geometry or wire format.

For repeated GJK queries, construct `ConvexInstanceTransform` when the pose changes, then
`prepareConvexPieceTransform(piece, transform)` once for each affected part. Reuse that result with
`visitConvexPiece(transformed, callback)`; the callback receives a concrete borrowed support set for the existing
`gjk::DistanceGjkQuery::query`. The hot overload only reads cached rotation/translation. Shape storage remains
immutable and must outlive its borrowed views. `PreparedGeometryCache` is a bounded, owner-local numerical content
cache; the caller owns identity, version and state invalidation. `forEachConvexPieceNear` reuses the existing scene
BVH in local coordinates to prune parts without rebuilding it during rigid motion.

`bench_prepared_geometry` compares existing occupied sets and prepared kernels using the same GJK implementation,
inputs, build flags, tolerances and budgets. Its CSV records support and full-query quantiles, cold/warm query state,
static/dynamic poses, pose update cost, preparation allocations/retained memory, support calls and iterations.
The Linux benchmark intercepts both C++ allocations and header-side `malloc/free`; it changes no host performance
policy. Cold query means no directional warm start, not a flushed CPU cache. It is a numerical library benchmark;
algorithm solve-cycle and end-to-end validation belong to the consumers.
