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
