# Validation record: numerical tests and MATLAB simulations

Validated on 2026-10-10 against changes based on upstream
`00c79cef172393889b163acc087f140a0065e4c8`.

## Environment and results

Local environment: Linux x86-64, GCC 13.3.0, Eigen 3.4.0, CMake/CTest 3.28.3,
GNU Octave 8.4.0, clang-format 18.1.3 and cppcheck 2.13.0. The CI-compatible
Focal cppcheck 1.90 was also run locally. Plotting used gnuplot and FreeFont.

| Check | Result |
|---|---|
| Original default CTest suite before changes | 7/7 passed |
| Debug, simulations enabled | 57/57 passed |
| Release with strict warnings as errors, simulations enabled | 57/57 passed |
| Native test registration with simulations disabled | 55 tests |
| Formatting of all 100 non-vendored C++ header/source files | Passed |
| cppcheck on the seven new/changed test translation units | Passed, no diagnostics |
| Whole-tree cppcheck 1.90 with the CI script's exact flags | Passed, 16/16 translation units, no diagnostics |
| `git diff --check` | Passed |
| `.xgc2/scripts/check_package_compliance.sh` | Passed |
| Staged CMake installation and installed consumer probe | Passed |
| Headless reference and native simulation execution | Passed |
| Optional PNG rendering and visual inspection | Passed |

The installed consumer uses the existing probe from
`.xgc2/scripts/smoke_test_installed.sh` with a staged CMake prefix. This is not
a claim that the Debian package/distribution matrix was run locally.
The full `.xgc2/scripts/check_cpp_quality.sh` is a separate CI gate; strict
compilation and formatting alone do not substitute for its static analyzers.
The seven test translation units were checked with the script's cppcheck
flags. A whole-tree scan with local cppcheck 2.13.0 still reports legacy
diagnostics in unchanged `include/xgc2_math/optimization/waypoint3.hpp` and
`test/pose3_eskf_jacobian_test.cpp`; this record does not claim a clean local
whole-tree cppcheck run or a local clang-tidy run.

The first remote run passed all 55 native tests with GCC 9.4.0, then exposed
ten diagnostics specific to the CI image's cppcheck 1.90. These were
reproduced locally against that exact failed source snapshot. The follow-up
changes make calculation precedence explicit, use ordered accumulation for
partitioned motion, initialize Eigen fixture data without comma expressions,
and construct independent measurement objects for each negative test. The
same whole-tree cppcheck 1.90 command then passed without diagnostics. All
original inputs, seeds, tolerances and assertions were retained.

These counts are CTest entries, not line or branch coverage percentages.
The checks use explicit failures and remain active with `NDEBUG` defined.

## Reproduce the two complete runs

```bash
cmake -S . -B build-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DXGC2_MATH_TEST_SIMULATIONS=ON \
  -DXGC2_MATH_SIMULATION_ENGINE=Octave
cmake --build build-debug --parallel 2
ctest --test-dir build-debug --output-on-failure -j 2

cmake -S . -B build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXGC2_MATH_TEST_SIMULATIONS=ON \
  -DXGC2_MATH_SIMULATION_ENGINE=Octave \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Werror"
cmake --build build-release --parallel 2
ctest --test-dir build-release --output-on-failure -j 2
```

Ninja is optional; omit `-G Ninja` to use the default generator. The scripts
and CSV outputs are documented in [the simulation guide](../matlab/README.md).
The CTest runs use seed `20261010`, `plot=false`, and distinct output
directories for reference-only and native validation.

## Observed native simulation errors

The motion checks compare independent MATLAB/Octave reference samples with
the actual C++ exporter, including array shape and finiteness before tolerance
checks. Maximum absolute differences across all numeric entries:

| Group | Maximum absolute difference | Required bound |
|---|---:|---:|
| Planar motion | 1.29e-13 | 2e-9 |
| Delayed actuator response | 4.17e-16 | 2e-9 |
| Sampled SMC tracking | 5.96e-16 | 2e-9 |
| Analytic trajectory derivatives through snap | 4.86e-14 | 2e-9 |

The following are position RMSE values from the actual 15-state C++ ESKF,
computed against analytic truth. The IMU-only column is the independent
strapdown baseline, not a second implementation of the ESKF.

| Scenario | IMU-only baseline (m) | C++ ESKF (m) |
|---|---:|---:|
| Stationary | 0 | 0 |
| Constant acceleration | 1.57e-15 | 8.70e-17 |
| Constant yaw turn | 3.17e-7 | 5.51e-10 |
| Noisy turn, 100 Hz IMU | 0.38657 | 0.013803 |
| Noisy turn, 200 Hz IMU | 0.42635 | 0.013351 |
| Pose dropout, 1.2 seconds | 0.42399 | 0.022224 |
| Pose arrival delayed by 40 ms | 0.41321 | 0.014669 |

The dropout recovery position RMSE was 0.012948 m after the documented
0.5-second recovery allowance. Noise realizations can differ between MATLAB
and Octave even with the same seed; assertions use physical error bounds,
not exact saved stochastic outputs. Detailed rates, noise, units and gates
are in [the inertial simulation guide](../matlab/inertial_pose_eskf/README.md).

## Regression sensitivity and failure handling

The queued-IMU initialization test failed on the old ESKF initialization and
passed after preserving the sample before resetting internal state. The
torus-entry test similarly exposed the non-default-yaw discontinuity and
passes after applying the existing yaw-hold contract to the whole path.

Additional development checks exercised failure paths beyond successful
numeric runs:

- Independent angular-observer mutations removing correction, short-arc
  residual handling, or posterior wrapping were each caught in both C++
  and Octave.
- Pre-existing native CSV files plus a successful executable that writes
  nothing were rejected. Each invocation removes its own stale native
  outputs before validating newly generated data.
- Equivalent input/output paths, hard links and symbolic links were rejected
  by the replay tool while preserving input bytes.
- Real native execution succeeded with spaces and apostrophes in executable
  and output paths on Linux/Octave.
- RNG/search-path cleanup and execution without open figures were checked.

## Validation boundaries

No native MATLAB installation or license was available in this environment;
MATLAB compatibility was exercised through Octave, not claimed as an actual
MATLAB run. The MATLAB CTest engine is available for installations supporting
`-batch`. Filter/observer MATLAB simulations are independent reference
implementations; companion C++ property tests check the library classes.

The finite synthetic scenarios verify the stated numerical models and
contracts. They do not establish vehicle-level performance or statistical
consistency across arbitrary sensor/plant distributions. Generated results
are reproducible build artifacts and are not checked in as golden outputs.

## Scene component (`xgc2_math::scene`)

Validated on 2026-10-10 on the branch `issue-215-scene-geometry`, based on `4a9686f`. The environment is the CI
image's: Ubuntu 20.04 x86-64 with GCC 9.4.0, clang++ 10.0.0, Eigen 3.3.7, CMake 3.16.3, clang-format 10, clang-tidy 10
and cppcheck 1.90. The oldest supported toolchain was exercised with Ubuntu 18.04's g++ 7.5.0 and Eigen 3.3.4.

| Check | Result |
|---|---|
| Default CTest suite, GCC 9.4, RelWithDebInfo, strict warnings as errors | 99/99 passed, 44 of them scene tests |
| Scene tests, clang++ 10, the same warnings as errors | 44/44 passed |
| Scene tests, Ubuntu 18.04 g++ 7.5 with Eigen 3.3.4, the same warnings as errors | All 44 cases passed |
| Scene probe of the installed layout against the four packages it depends on (scene, geometry, algebra, utils), GCC 9.4, clang++ 10 and g++ 7.5 | Passed |
| Reference comparison with the original `world_lidar` core of commit `2940a99` (opt-in) | 5/5 passed, every compared value bit-identical |
| AddressSanitizer and UBSan with leak detection, all four scene tests | 44 cases passed, no reports |
| ThreadSanitizer (clang++ 10): the threaded fleet case, and four threads on one shared scene and beam pattern while revisions are built from that scene | Passed, no reports |
| Formatting of all 119 non-vendored C++ files with clang-format 10 | Passed |
| Whole-tree cppcheck 1.90 with the CI script's exact flags | Passed, no diagnostics |
| clang-tidy with the CI configuration on `test/math_header_test.cpp`, which includes the umbrella and so every scene header | Passed, no diagnostics |
| `git diff --check` and `.xgc2/scripts/check_package_compliance.sh` | Passed |
| `.xgc2/scripts/build_deb.sh` for focal, then the installed smoke test on the extracted packages | Passed; 11 packages, all `Architecture: all` |

The reference comparison builds the original core from a checkout of the scene repository, so it is opt-in
(`-DXGC2_MATH_SCENE_REFERENCE_SOURCE=<checkout>`) and the standard suite and the Debian build do not depend on it. The
fixtures are mixed random scenes of every obstacle type, the polytopes of the cluttered-environment libraries and a chain
of 240 scene revisions scanned in all three modes: 4.4 million values and 36 packed clouds are identical to the last
bit, against a documented tolerance of 1e-9 m. The placement, motion and convex-surface math was compared with the
Python scene runtime it replaces on 1000 random placements and motions (bit-identical) and 300 mutated convex meshes
(same decision, message and triangles); that comparison used throwaway scripts and is not part of the suite.

The before and after table of `include/xgc2_math/scene/README.md` comes from `bench_world_lidar_reference` (the original
benchmark built from the same sources) and `bench_scene_lidar`, built with the same compiler and flags and run
alternately nine times on a shared 8-core host; the median CPU time is reported and the output checksums of both are
equal. A repeat of three runs at the final commit gave the same checksums and ratios within about ten percent.

Boundaries of this record:

- The Jammy and Noble toolchains (g++ 11 and 13 with Eigen 3.4) and the Debian build images were not available; the
  headers are plain C++17 and were compiled with three compilers on Eigen 3.3.4 and 3.3.7 only.
- Building the whole tree with clang++ 10 fails on the base commit and still does: `-Wsign-conversion` in
  `optimization/minco.hpp` and `optimization/root_finder.hpp`, and `geometry_property_test.support_maps` misses its
  tolerance by 3.3e-15 against 3e-15. None of these files changed; only the scene targets were built with clang++.
- ThreadSanitizer was run with clang++ 10 because the GCC 9.4 installation lacks `libtsan_preinit.o`. The allocation
  test replaces the global allocation functions; it replaces the nothrow forms as well so that it stays consistent
  under sanitizers that intercept them.
- Performance was measured on the component's own benchmark only. Sample age, latency and thread scaling of a real
  simulator need the schedulers (xsim, the scene adapter) and are not claimed here.

```bash
cmake -S . -B build-scene -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Werror"
cmake --build build-scene -j3 && (cd build-scene && ctest -L scene --output-on-failure)

cmake -S . -B build-reference -DCMAKE_BUILD_TYPE=Release \
  -DXGC2_MATH_SCENE_REFERENCE_SOURCE=/path/to/xgc2-scene-generation
cmake --build build-reference -j3 --target scene_reference_test bench_scene_lidar bench_world_lidar_reference
(cd build-reference && ctest -R scene_reference --output-on-failure)
```
