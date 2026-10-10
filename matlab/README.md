# Numerical simulations

Run repeatable numerical checks and optionally generate diagnostic figures
using base MATLAB or GNU Octave. The scripts use SI units, fixed scenarios,
explicit tolerances and assertions; a failed check stops execution with an
error. They do not require Simulink, the Control System Toolbox, the Signal
Processing Toolbox, or the Robotics System Toolbox.

## Run all groups

From the repository root in MATLAB:

```matlab
addpath('matlab');
results = run_all_simulations(struct( ...
    'output_dir', fullfile(pwd, '.ci', 'simulations'), ...
    'plot', true));
```

The default is `plot=false`, allowing execution without a display or graphics
backend. Without an output directory, the unified runner uses a new temporary
directory and returns its locations in the group results. Specifying a
directory overwrites generated files of the same names on subsequent runs.
It does not delete other files. The caller's MATLAB search path and random
number generator state are restored, including when a check fails.

To exercise the **actual C++ implementations**, first build the test targets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel 2
```

Then run:

```matlab
addpath('matlab');
options = struct( ...
    'output_dir', fullfile(pwd, '.ci', 'simulations'), ...
    'plot', true, ...
    'seed', 20261010, ...
    'replay_executable', fullfile(pwd, 'build', 'pose3_eskf_csv_replay'), ...
    'reference_executable', fullfile(pwd, 'build', 'simulation_reference_export'), ...
    'require_cpp', true);
results = run_all_simulations(options);
```

Append `.exe` to executable names on Windows. Explicitly requested missing
executables and nonzero native exit codes are errors. `require_cpp=true`
requires both executables; a missing executable cannot silently pass a native
validation job. Without native paths the MATLAB references still run, while
the results and summary explicitly report that native validation was skipped.

## Coverage and evidence

| Group | Scenarios | Independent check / native connection |
|---|---|---|
| Filters and observers | Exponential response at unequal steps; Butterworth sweep and step response; slew limits; acceleration observer with gated outliers; angular branch crossings; changing-parameter RLS | Exact continuous exponential, frequency-domain transfer function and weighted batch least squares. The companion C++ property suite checks the public classes; this MATLAB group is not itself a C++ replay. |
| Planar motion | Unicycle, mecanum strafe, reverse turn, near-zero yaw rate, pure rotation | Absolute-time SE(2) matrix exponential versus repeated C++ `stepBodyVelocity` updates |
| Actuator dynamics | Two command steps, unequal time constants and a 35 ms delay on a 10 ms sample grid | Superposition of analytic delayed step responses versus C++ `DelayedPlanarVelocity` |
| SMC tracking | Accelerating and sinusoidal references, 100/200 Hz control, disturbance reversal at 15 s | MATLAB feedback drives an ideal held-acceleration plant; compare every state/control sample to actual C++ control and plant output; check final errors and feedback limits |
| Analytic trajectories | Circle, figure eight, helix and torus knot | Generic differentiation of harmonic coefficients versus C++ position, velocity, acceleration, jerk and snap |
| Inertial pose | Stationary, constant acceleration, constant yaw turn, noise at two IMU rates, 1.2 s pose dropout, 40 ms late arrival | MATLAB creates receive-ordered IMU/pose CSV and analytic truth, then assesses the actual 15-state C++ ESKF replay; an IMU-only strapdown baseline provides a comparison |

The trajectory C++ property suite additionally covers the remaining 2D/3D
analytic evaluators, entry continuity, sampled trajectories, flatness and
MINCO energy/gradients. The ESKF C++ suite checks covariance, invalid inputs,
clock/gate boundaries and replay invariants that the MATLAB baseline does
not implement. See [test/README.md](../test/README.md).

## Automated execution

Enable simulations explicitly in CTest; this does not add a MATLAB/Octave
runtime dependency to the header-only library or its installed CMake targets.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DXGC2_MATH_TEST_SIMULATIONS=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

This adds two tests: `simulation.reference_only` and
`simulation.cpp_parity`. The second requires and executes both native tools.
The default engine is Octave. For a licensed MATLAB installation, configure
with `-DXGC2_MATH_SIMULATION_ENGINE=MATLAB`; MATLAB must support `-batch`
(R2019a or later). `XGC2_MATH_MATLAB_EXECUTABLE` or
`XGC2_MATH_OCTAVE_EXECUTABLE` can specify an explicit runtime path. A missing
requested runtime is a configure error, not a skipped passing test.

Run only the simulations with `ctest --test-dir build -L simulation
--output-on-failure`. On older CMake versions without `--test-dir`, run
`ctest` from the build directory. Headless Octave figures additionally need a
graphics backend and fonts; numerical assertions do not need plotting.
Octave test success does not constitute a native MATLAB test run.

## Outputs and conventions

`results.mat` contains all metrics and scenario results. `summary.txt` records
the engine version, seed, group outcomes and whether native code was exercised.
Per-group CSV files preserve the exact inputs, reference data and C++ outputs;
`plot=true` adds PNG diagnostics. Keep generated output under `.ci/`, a build
directory or a chosen results directory, outside the tracked source tree.

Motion/control numeric CSV files have no header:

| File stem | Columns |
|---|---|
| `planar_*` | scenario, time, world x, world y, yaw |
| `delayed_*` | time, forward velocity, yaw rate |
| `tracking_*` | scenario, time, position xyz, velocity xyz, acceleration command xyz, feedback xyz |
| `trajectories_*` | scenario, time, position xyz, velocity xyz, acceleration xyz, jerk xyz, snap xyz |

Scenario IDs are zero-based. Native/reference comparisons require identical
array shape and finite samples before checking numerical error. Yaw errors
are wrapped using `atan2(sin(error),cos(error))`. SMC native rows describe the
state and command at the start of each held-control interval, with the last
row at exactly 30 seconds. The feedback bound is 0.5 m/s² per axis. Motion
parity tolerance is `2e-9` absolute per numeric entry.

For the inertial sensor rates, noise levels, receive/acquisition-time ordering,
quaternion layout and RMSE gates, see
[inertial_pose_eskf/README.md](inertial_pose_eskf/README.md). See
[filter_observer/README.md](filter_observer/README.md) for filter/observer
parameters and CSV headers. These finite synthetic regressions are numerical
checks of the stated models; they do not certify a vehicle's physical tracking,
collision avoidance or general statistical consistency.
