# Inertial pose: analytic simulations and native ESKF replay

This directory supplies reproducible IMU/pose fixtures, independent analytic
truth, a MATLAB/Octave midpoint strapdown baseline, and a bridge to the **real
`xgc2_math::Pose3InertialEskf` C++ implementation**. It needs no MATLAB toolboxes
and opens no figures unless requested.

The previous script had invalid MATLAB syntax, a function/file name mismatch,
and a 21-state filter that no longer matched the 15-state fixed-extrinsic C++
estimator. The new MATLAB baseline propagates only position, velocity, and
orientation from IMU readings. It does **not** model pose fusion, bias
estimation, covariance, health gates, or rewind/replay. Native replay supplies
those behaviors; a skipped replay is reported explicitly and is not a native
ESKF pass.

## Run

From the repository root, in MATLAB or Octave:

```matlab
addpath('matlab/inertial_pose_eskf');
results = run_inertial_pose_eskf_simulation();
assert(results.pass);
```

`run_pose3_inertial_eskf_simulation(options)` remains a compatibility alias for
the name advertised by the old README. Both file names now match their public
functions.

For actual C++ fusion, build the `pose3_eskf_csv_replay` target using the
repository's CMake test configuration, then pass its path:

```matlab
options = struct('seed', 20261010, 'plot', false, ...
    'output_dir', fullfile(pwd, 'simulation-results'), ...
    'replay_executable', fullfile(pwd, 'build', 'pose3_eskf_csv_replay'));
results = run_inertial_pose_eskf_simulation(options);
assert(results.cpp_replay_available);
disp(results.metrics);
```

Append `.exe` on Windows. When the executable option is omitted, the script
looks only in `<repo>/build/pose3_eskf_csv_replay[.exe]`. An explicit empty path
disables native replay; an explicit missing executable is an error. With
`output_dir` set, `options.plot = true` saves a `trajectory.png` plot in every
case directory using invisible figures that are closed automatically. The
simulation therefore remains usable on headless machines when plotting.

The caller's RNG state is restored on return or error. Each case uses the
specified seed plus its zero-based case index. The numerical fixtures are
repeatable within the same MATLAB/Octave environment; normal-random generation
is not claimed to be bitwise identical between the two runtimes. Export the
CSV fixtures when comparing versions or environments.

## Scenarios and independent truth

All cases start at rest at the origin with identity attitude. IMU acceleration
is **body-frame specific force**, not world acceleration: stationary input is
`[0, 0, 9.8066]` m/s² and world gravity is `[0, 0, -9.8066]`. Gyroscope units are
rad/s; positions and velocities use meters and m/s. Quaternions are `[w x y z]`
inside MATLAB helpers and **`[x y z w]` in every CSV**.

| Case | Duration | IMU / pose rate | Disturbance or purpose |
|---|---:|---:|---|
| `stationary` | 2 s | 200 / 50 Hz | Gravity cancellation and unit quaternion |
| `constant_acceleration` | 3 s | 200 / 50 Hz | Exact `p = a t²/2`, `v = a t` in 3D |
| `constant_yaw_turn` | 4 s | 200 / 50 Hz | Rotating body force and analytic curved path |
| `noisy_turn_100hz` | 6 s | 100 / 20 Hz | Sensor noise and constant IMU biases |
| `noisy_turn_200hz` | 6 s | 200 / 25 Hz | A second sensor-rate ratio |
| `pose_dropout` | 6 s | 200 / 25 Hz | No pose samples during `[2.0, 3.2]` s; recovery |
| `delayed_pose` | 6 s | 200 / 25 Hz | Pose arrival delayed 40 ms; acquisition stamp retained |

For constant acceleration, `a = [0.3, -0.15, 0.08]` m/s². For the turning
trajectory, body forward acceleration `a = 0.2` m/s² and yaw rate `w = 0.35`
rad/s give these analytic world coordinates:

```text
p_x(t) = a (1 - cos(w t)) / w²
p_y(t) = a (t / w - sin(w t) / w²)
v_x(t) = a sin(w t) / w
v_y(t) = a (1 - cos(w t)) / w
yaw(t) = w t
```

Noisy cases use independent Gaussian **per-sample** standard deviations of
0.06 m/s² for accelerometer readings, 0.003 rad/s for gyroscope readings,
0.008 m for pose position, and 0.004 rad for each component of the pose
rotation-vector perturbation. Constant sensor biases are
`b_a = [0.04, -0.03, 0.02]` m/s² and
`b_g = [0.001, -0.0008, 0.002]` rad/s. The initial pose is exact. C++ replay uses
its configured process-noise density defaults; these synthetic per-sample
noise levels are not claimed to be calibrated to the filter's covariance.

The reference integrator uses the mean of consecutive IMU readings and the
midpoint attitude. It is intentionally independent of C++ pose-update code.
Native output rows are recorded immediately after each IMU propagation,
**before** any pose correction at the same receive time. The initial IMU
precedes initialization and has no output row, so native rows must number
`IMU samples - 1`; timestamps must match analytic truth exactly to 1e-8 s.
Delayed poses are sorted by receive time and fused using their original
acquisition timestamps; no artificial delay parameter is applied twice.

## Metrics and failure behavior

Every scenario reports position, velocity, and sign-invariant geodesic
orientation RMSE, peak/final position error, and quaternion normalization
error. Dropout also reports errors during the gap and after 0.5 s of recovery.
Unprefixed error fields describe the **strapdown baseline**; `cpp_*` fields
describe native ESKF output. They contain `NaN` when native replay is skipped.

Assertions make execution fail on non-finite output, sample-count or clock
changes, non-unit quaternions, loss of noiseless analytic accuracy, excessive
noisy estimation error, or unsuccessful dropout recovery. Native noisy gates
are position RMSE below 0.12 m, velocity RMSE below 0.25 m/s, orientation RMSE
below 0.04 rad, and improved position RMSE relative to the biased IMU baseline.
After dropout the recovery position RMSE must be below 0.04 m. These are
physical regression bounds for these fixtures, not a general accuracy claim.

With `output_dir` set, results are written to
`<output_dir>/inertial_pose_eskf/<case>/`:

- `events.csv`: exact receive-ordered input to the native executable;
- `truth.csv`: analytic state at every IMU timestamp;
- `baseline.csv`: MATLAB strapdown state at every IMU timestamp;
- `cpp_state.csv`: native state output, only when replay runs;
- `cpp_replay.log`: native replay output and acceptance/rejection counters;
- `trajectory.png`: optional headless trajectory/error plot (`plot = true`);
- `metrics.csv`: numeric metrics, actual seeds, rates, sample counts, and delay/dropout settings.

The parent `inertial_pose_eskf/metrics.csv` collects all cases. State CSVs have
no header and follow the 11-column layout above; metrics CSVs have a header.
Reruns overwrite generated results. Without `output_dir`, temporary CSVs are
cleaned up, and the returned struct still contains truth, baseline, native
states, configs, and metrics for inspection.

## Native C++ regression coverage

`test/pose3_eskf_scenarios_test.cpp` checks public-API behavior directly:

- Equivalent initialization from a queued IMU and a separately supplied IMU,
  including a varying first sample pair that detects the input-alias bug;
- Rotated/translating world and marker frames, fixed extrinsics, and config fallbacks;
- Stationary and constant-acceleration exact trajectories at 50/100/200 Hz;
- A long IMU interval integrated in substeps, with state/covariance agreement;
- Second-order midpoint convergence on an analytic turn;
- Continuous-density versus sample-held acceleration-noise covariance formulas
  across sample rates, including position/velocity cross covariance;
- Reproducible noisy fusion, pose dropout, and recovery at two IMU rates;
- Delayed pose rewind/replay agreement with in-order fusion, including covariance;
- Invalid inputs, duplicate/backward/tiny/non-finite timestamps, raw jump gates,
  exact observation-window edges and one-ULP violations, evicted rewind
  history, and IMU time-jump reset/reinitialization;
- The velocity posterior against scalar Gaussian conditioning, plus unchanged
  state/covariance on rejected velocity measurements;
- Finite, symmetric, positive-semidefinite covariance and unit quaternions.

Checks throw on failure and return a nonzero process status, even when
`NDEBUG` is defined. The separate Jacobian regression executable continues to
test analytical derivatives and iterated update configurations. MATLAB does
not substitute for these covariance and boundary tests.
