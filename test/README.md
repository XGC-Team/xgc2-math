# Numerical tests and simulation replay

The [validation record](VALIDATION.md) lists the tested environment, commands,
observed numerical errors and validation boundaries for this expansion.

## Build and run

Source tests require a C++17 compiler, CMake and Eigen3. They use ordinary
executables with explicit failure checks; no external unit-test framework is
required. All checks remain active in Release and RelWithDebInfo builds.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

For CMake 3.10, configure from a pre-created build directory using `cmake ..`
and invoke `ctest --output-on-failure` from that directory. Building with
`XGC2_MATH_BUILD_TESTING=OFF` continues to install just the library and its
documentation. Tests, exporters and external runtime test fixtures are not
installed as public library interfaces.

Enable the MATLAB/Octave reference and actual C++ parity checks using
`-DXGC2_MATH_TEST_SIMULATIONS=ON`. The engine defaults to Octave; use
`-DXGC2_MATH_SIMULATION_ENGINE=MATLAB` for MATLAB with `-batch` support and a
license. Instructions and CSV conventions are in
[matlab/README.md](../matlab/README.md).

## What runs

| Executable / CTest group | Checks |
|---|---|
| `math_header_test` | Existing public-header/filter/observer/ESKF/trajectory/control and geometry contracts |
| `lbfgs_standard_test` | Existing optimizer regression |
| `pose3_eskf_jacobian_test` | Existing analytic residual-Jacobian and replay regressions |
| `numerical_regression_test` | Existing extreme-scale arithmetic, flatness derivatives, penalties and allocation invariants |
| `delayed_planar_velocity_test`, `simulation_kinematics_test`, `smc_tracking_test` | Existing delay, exact motion and sampled-control regressions |
| `implicit_wheel_contact_test` | Existing friction/torque bounds, power identities, passivity, symmetry, stiff systems and solver failures, now included in the default CTest suite |
| `filter_observer_property_test.*` (6 named tests) | Exponential partition invariance, independent Butterworth complex response, slew feasibility, observer prediction through gated measurements, angular wrapping, RLS against batch weighted least squares |
| `geometry_property_test.*` (10 named tests) | SE2/SE3 group identities, rotation maps, motion flow and quadrature, support maps and independently known sphere/box distances/witnesses |
| `trajectory_property_test.*` (15 named tests) | Analytic derivatives/time domains/periodicity, boundary and entry continuity, sampled interpolation, flatness, validators, independent dense-KKT MINCO optima/energy and finite-difference total gradients |
| `pose3_eskf_scenarios_test` | Queued IMU initialization, analytic motion at several rates, covariance/noise conventions, noisy multirate fusion/dropout, delay equivalence, observation-window/history/invalid-input boundaries and velocity conditioning |
| `replay_cli.*` (15 named tests) | Valid/CRLF recording, exact state layout, malformed numbers/columns/event types, missing initialization, nonfinite quaternion passthrough, bad CLI parameters and identical/alias path rejection |
| `simulation.reference_only`, `simulation.cpp_parity` (optional) | MATLAB/Octave analytic simulations and actual native C++ comparison/replay |

CTest lists 55 tests by default and 57 with simulations enabled. This is a
count of independently scheduled test entries, not a line/branch coverage
percentage; several legacy executables contain multiple assertions or cases.
The three new property executables accept a named case or `--list`; with no
argument they run all cases. Unknown case names return a nonzero exit status.

Examples:

```bash
ctest --test-dir build -R 'trajectory_property_test.minco' --output-on-failure
ctest --test-dir build -L geometry --output-on-failure
./build/filter_observer_property_test butterworth_response
./build/geometry_property_test --list
```

## Native ESKF CSV tool

```bash
./build/pose3_eskf_csv_replay events.csv state.csv \
  0 0 0.35 1 1 3
```

Optional values are observation delay in seconds, body-to-marker yaw in
radians, acceleration noise standard deviation, density mode (`0`/`1`),
pose covariance-floor mode (`0`/`1`) and pose-update iterations (integer 1–8).
The input follows **receive order**, not sensor-stamp order:

```text
imu,receive_time,acquisition_time,gx,gy,gz,ax,ay,az
vrpn,receive_time,acquisition_time,x,y,z,qx,qy,qz,qw
```

Blank lines and comment lines starting with `#` are allowed. These example
labels describe the schema; the actual CSV is numeric and has no header.
Extra/missing columns, unknown event kinds, malformed numeric tokens and
invalid CLI parameters fail with a nonzero exit status. The tool also fails
if replay yields no initialized IMU output rows. Receive timestamps must be
finite. Explicit sensor `nan`/`inf` values are passed to the estimator so its
invalid-sample behavior can be tested, rather than being coerced to zero or
normalizing an invalid quaternion into identity.

Each initialized IMU event produces an 11-column state row:
`imu_stamp, px, py, pz, qx, qy, qz, qw, vx, vy, vz`.
The tool reports IMU count, accepted pose count and alignment rejection count
on stderr. MATLAB generates fixtures and asserts their row counts, time
alignment, finiteness, unit quaternions and accuracy against independent truth.

The separate `test/dfbc/` fixture requires Runtime/robotics wire packages and
remains an integration project. It is intentionally not pulled into the
standalone Eigen-only unit-test build.

## CI and reproducibility

The existing `.xgc2/scripts/check_cpp_quality.sh` builds the CMake suite and
invokes CTest, so the new default tests run through the existing CI gate.
The separate wheel workflow remains available as a sanitizer check. MATLAB
and Octave simulations are opt-in: CI build images must provide the chosen
runtime before enabling them; product CI does not bootstrap new OS/toolchain
dependencies. No unexecuted simulation is reported as a passing native test.

Use Debug for development and Release/RelWithDebInfo to check optimized
behavior. Synthetic scenarios use fixed formulas or recorded seeds. Tolerances
are based on analytic answers, derivative convergence, physical error bounds,
or independent optimization references rather than saved output from the
implementation under test. Numerical/CSV outputs belong in build or `.ci`
directories and are not golden files checked into source control.
