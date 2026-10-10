# xgc2_math::scene

Scene modeling and sensing geometry: closed convex solids, ray casting, surface samples, LiDAR and depth-camera
models, scene-document geometry checks and placement. Header-only C++17 on Eigen and the `geometry` component; no ROS,
no PCL, no OpenGL and no GPL code.

```cmake
find_package(xgc2_math REQUIRED CONFIG)
target_link_libraries(my_target PRIVATE xgc2_math::scene)   # or xgc2_math::math
```

```cpp
#include <xgc2_math/scene.hpp>
using namespace xgc2_math::scene;

std::vector<Obstacle> obstacles{Obstacle::box({5, 0, 0}, {1, 4, 4}), Obstacle::sphere({0, 6, 1}, 0.5)};
auto scene = std::make_shared<const LidarScene>(obstacles, /*spacing*/ 0.1, /*keep_buried*/ false);
WorldLidar lidar(SensorConfig{});           // 360 x 32 raycast, 20 m
lidar.setScene(scene);                      // one immutable scene, any number of sensors
std::vector<std::uint8_t> cloud;            // reused: float32 x y z records
lidar.scanInto(position, attitude, bodies, /*with_id*/ false, &cloud);

// A revision of a moving scene reuses everything that did not change.
scene = std::make_shared<const LidarScene>(obstacles, 0.1, false, *scene);
lidar.setScene(scene);
```

A penetrating sensor takes only scenes built with its own `surface_spacing` and `penetrating_keep_buried`.

## Contents

| Header | Provides |
|---|---|
| `obstacle.hpp` | `Obstacle`: box, sphere, cylinder, capsule, convex (local frame, placed by position and orientation) |
| `polytope.hpp` | `buildPolytope()`: planes and face polygons of a convex hull (Math's quickhull) |
| `occupied_set.hpp` | `occupiedSet()`: an obstacle as one of Math's support-mapped sets, for GJK distance and separation queries |
| `lidar_scene.hpp` | `LidarScene`: compiled solids, hierarchy, surface samples; incremental rebuild from the previous revision |
| `sensor_model.hpp` | `SensorConfig`, field-of-view and heading-crop predicates, `RangeNoise`, `BeamScan`, `TaggedPoint`, `VehicleBody` |
| `beam_pattern.hpp` | spherical and pinhole beam patterns, shared between sensors |
| `world_lidar.hpp` | `WorldLidar`: raycast, penetrating and depth-frustum scans, `castRay`, `globalMap` |
| `scene_conversion.hpp` | scene-document parts and Gazebo obstacle truth to `Obstacle` |
| `observation_contract.hpp` | pose validity, freshness, mounting composition |
| `sensor_metadata.hpp` | parameter-domain validation (CPU crop-through, original GPU projection) |
| `placement.hpp` | quaternion algebra of scene assembly, placement of elements and motion intents, motion evaluation |
| `convex_mesh.hpp` | closed-convex-surface check of document convex parts, hull meshes |
| `cluttered_environment.hpp` | primitives and polytopes of cluttered scenarios, occupied sets, cube-frame groups |

The OpenGL spherical projection of the old `world_lidar` package is GPL-2.0 and stays with the simulator that links it;
scheduling, thread pools and publication stay with the schedulers (xsim, the scene adapter). Only the parameter-domain
check of that projection, `validateGpuSensorMetadata()`, lives here. The PCL voxel-and-radius crop of an arbitrary
input cloud (`shared_cloud_cpu`) is not part of the component either; `validateSensorMetadata()` keeps its parameter
domain, and `HeadingCrop` applies the same kind of heading cone and vertical slab to the penetrating mode.

## Data layout and threads

- **Scene.** Per shape one 136-byte record holds everything a ray test reads (rotation, position, three parameters,
  plane range); convex planes, bounds, and the polygon vertices that only sampling needs are separate flat arrays.
  Hierarchy nodes keep the boxes of both children next to their codes, one axis at a time with the two children
  side by side, so a step tests both children with two-lane vector operations from one record. The build is a binned
  surface-area split; `refit()` updates the boxes of a moved scene in one sweep and says when the tree has degraded.
- **Samples.** The sampled map is one array of points in obstacle order. Consecutive samples of an obstacle form
  blocks of up to 64 that keep their box; a range query walks the hierarchy to the obstacles near the sensor, marks
  blocks wholly inside as a run of bits, skips blocks wholly outside and tests the rest point by point. A revision of
  the scene reuses the shapes, samples and blocks of every obstacle that is bitwise unchanged.
- **Scans.** The scene is immutable and shared. A `WorldLidar` is not thread safe (a scan advances its scan index and
  works in its own scratch buffers): one sensor per concurrent scanner. After the first scan with a given scene and
  number of bodies a scan allocates nothing, noise and vehicle bodies included, and writes into buffers the caller
  reuses (`scanInto`, `scanTaggedInto`, `scanBeamsInto`, or any sink with `reserve()` and `push()` through `scanTo`).
  `scene_allocation_test` counts allocations to prove it.

## Numerical contract

The component continues the `xgc2_world_lidar` core of the scene repository (sensors/world_lidar, commit 2940a99):
shapes, sampling, ray casting, scan order, range noise (same engine, same seed sequence) and the scene rebuild produce
the same numbers. `scene_reference_test` builds the original core from a checkout of that repository and feeds both
cores the same fixtures; configure with

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXGC2_MATH_SCENE_REFERENCE_SOURCE=/path/to/xgc2-scene-generation
cmake --build build -j3 --target scene_reference_test bench_scene_lidar bench_world_lidar_reference
cd build && ctest -R scene_reference
```

Counts, order, hit flags and vehicle ids must be identical and values must agree within 1e-9 m. On the fixtures
(mixed random scenes of every obstacle type, the polytopes of the cluttered-environment libraries, 240 scene revisions
scanned in all three modes with and without buried samples, noise, heading crop and vehicle bodies) every value is
bit-identical: 4.4 million compared values, 36 packed clouds byte for byte. The placement, motion and convex-surface
math was compared with the Python scene runtime it replaces on 1000 random placements and motions (bit-identical)
and 300 mutated convex meshes (same decision, same message, same triangles).

Deliberate differences from the old API:

| Old | Here |
|---|---|
| hull by plane enumeration, O(n^4) | quickhull; same planes in the same order; a face lists the hull vertices on it (a non-extreme point in a face plane is no longer an outline vertex); a set within 1e-7 of its extent of a plane is flat |
| `Pose` | `xgc2_math::Pose3` |
| `scanWithBeams()` returns `std::vector<Beam>` | `BeamScan`, parallel arrays and one shared origin; `scanBeamsInto()` reuses it |
| `scan*()` const, sensor callable from several threads | non-const, one sensor per thread |
| `beamCount()` of a penetrating sensor is `h_res * v_res` | 0, such a sensor has no beams |
| a geometry type `capsule` of Gazebo obstacle truth is refused | radius `scale.x`, full length `scale.z` |

## Benchmark

`bench_scene_lidar` and `bench_world_lidar_reference` run the fleet scan benchmark of the original core (7 robots, 40
ticks, one thread) on the same synthetic Swarm-Formation forest: 60 pillars and 20 rings of 126 thin capsules, 2580
parts, 118,010 samples with buried samples kept and 65,359 without, a 360 x 32 raycast and a full-sphere 8 m
penetrating sensor. Both are built from the same CMake configuration (g++ 9.4, `-O3 -DNDEBUG`), run alternately nine
times on a shared 8-core Xeon, and the table gives the median CPU time. The output checksums of the two cores are
equal, so the clouds are identical point for point.

| | original | `xgc2_math::scene` | |
|---|---|---|---|
| scene build, buried samples kept | 15.1 ms | 4.4 ms | 3.4x |
| scene build, buried samples dropped | 18.0 ms | 5.8 ms | 3.1x |
| revision moving 5 obstacles, kept (from the previous scene) | 1.02 ms | 0.41 ms | 2.5x |
| revision moving 5 obstacles, dropped (from the previous scene) | 6.24 ms | 0.60 ms | 10x |
| raycast 360 x 32 scan, 11,520 beams | 1.57 ms | 0.80 ms | 2.0x |
| penetrating 8 m scan, 36,969 points (`scanInto`) | 0.298 ms | 0.156 ms | 1.9x |
| penetrating 8 m scan through the allocating tagged path | 0.290 ms | 0.175 ms | 1.7x |

Where it comes from: the hierarchy built by surface area visits 8.8 instead of 12.8 nodes per ray and tests 0.58
instead of 1.51 shapes, with two-lane child tests per node; the block index replaces a k-d tree over all samples (about
54 bytes per sample, now about 25) and survives revisions that change an obstacle's sample count; the scan sets up
nothing per call. No sample age, latency or thread-scaling figures are measured here: those need the schedulers.
