# Filter and observer numerical simulations

`run_filter_observer_simulation.m` exercises the filter, Luenberger observer,
and scalar recursive least-squares contracts with deterministic inputs. It uses
base MATLAB operations and also runs under GNU Octave. The companion C++ suite
is `test/filter_observer_property_test.cpp`, which calls the public library
classes directly.

## Run

From the repository root in MATLAB:

```matlab
addpath('matlab/filter_observer');
results = run_filter_observer_simulation();
assert(results.pass);
disp(results.metrics);
```

Save numerical traces and a diagnostic figure:

```matlab
options = struct('plot', true, 'output_dir', fullfile(pwd, 'build', 'matlab-filter-observer'));
results = run_filter_observer_simulation(options);
```

The default is `plot=false` and `output_dir=''`: assertions run without a
graphics backend or file writes. A nonempty output directory writes seven CSV
files; `plot=true` also writes `filter_observer_diagnostics.png`. With plotting
enabled and no output directory, the figure is displayed interactively.
Existing files with these names in the chosen output directory are overwritten.

Headless Octave, without plotting:

```bash
octave-cli --quiet --no-history --eval "addpath('matlab/filter_observer'); r = run_filter_observer_simulation(struct('plot', false, 'output_dir', 'build/matlab-filter-observer')); assert(r.pass); disp(r.metrics);"
```

Plotting in headless Octave additionally needs a graphics backend such as
`gnuplot-nox` and usable fonts. MATLAB does not need an additional toolbox.
Successful Octave execution establishes Octave compatibility; native MATLAB
execution is a separate validation.

## What is checked

| Scenario | Independent reference or invariant | Failure detected |
| --- | --- | --- |
| Exponential filter | Exact constant-input solution `u + (y0-u)*exp(-2*pi*fc*t)` over four cutoffs and unequal time steps; elapsed-time partition identity | Incorrect time units, cutoff scaling, discretization or update timing |
| Butterworth sweep | Complex analog prototype `1/(s^2 + sqrt(2)*s + 1)` evaluated under the bilinear frequency map | Gain or phase errors, coefficient signs, incorrect Nyquist clamping |
| Butterworth step | Unity DC gain and the expected bounded second-order overshoot at two cutoffs | Unstable poles, state update order, missing second-order response |
| Slew limiter | Every update reaches the closest feasible command under `abs(delta) <= rate*dt`; total variation and elapsed-time partition bounds, including reachable targets | Treating rate as a per-sample bound or allowing overshoot with changing `dt` |
| Position/velocity observer | Analytic constant-acceleration position and velocity with 15 gated outliers; convergence from perturbed initial position/velocity | Freezing prediction on a rejected measurement, incorrect acceleration term, disabled correction, failure to resume updates |
| Angular observer | Analytic accelerated angle over at least 10 branch crossings, explicit +0.8 rad outlier residuals, and a nonzero +0.04 rad short innovation with exact one-step posterior | Velocity spikes at the branch cut, wrong shortest-angle residual, dropped prediction, disabled correction or missing posterior normalization |
| Forgetting RLS | Full weighted batch least squares, including the decayed nonzero prior, on every sample; parameter switch from 1.8 to -0.7 | Incorrect forgetting factor placement, covariance update, prior handling or adaptation |

The Butterworth frequency test covers three requested cutoffs and five input
frequencies at 200 Hz. A requested cutoff of 80 Hz deliberately exceeds the
library's 45 percent of Nyquist limit; the analytic oracle uses the effective
45 Hz cutoff. A 20-second settling interval precedes a 20-second measurement
interval, which contains an integer number of cycles at every tested frequency.
The MATLAB numerical realization uses base `filter`; the frequency-domain
oracle does not reuse its recurrence coefficients.

The angular correction check deliberately starts with an imperfect prediction:
`angle = pi-0.02`, `rate = 0.2`, and `dt = 0.05` predict `pi-0.01`, while the
measurement is `-pi+0.03`. The independent physical residual is **+0.04 rad**,
which must pass the 0.2 rad gate. Gains `alpha = 0.35`, `beta = 0.08` give the
explicit posterior `angle = -pi+0.004`, `rate = 0.264`. This catches omitted
measurement correction and incorrect use of the long angle difference, even
if an exact-initial-state propagation scenario still passes.

RLS uses a forgetting factor of 0.985, initial parameter -0.6 and initial
covariance 7. Its batch oracle minimizes

```text
J(theta) = lambda^n/P0 * (theta-theta0)^2
         + sum(j=1..n) lambda^(n-j) * (y_j-phi_j*theta)^2.
```

The parameter and covariance are checked against the batch optimum and inverse
normal matrix. A second estimate with `lambda=1` provides a control: it retains
the old parameter's influence after the switch, while the forgetting estimate
must settle within 0.02 of the new parameter. All inputs use explicit formulas
and fixed time schedules so MATLAB and C++ scenarios can be compared without
differences between random-number implementations. The entry point fixes seed
1729 and restores the caller's RNG state when it returns or raises an error.

## Outputs

`results.pass` is true only after all assertions complete. A failed assertion
raises `xgc2_math:filter_observer:checkFailed`; no success result is returned.
`results.metrics` contains maximum oracle errors, gate and wrap counts, slew
constraint margins, step peaks, and RLS tracking errors. `results.traces`
contains the same numeric matrices written to CSV, each accompanied by its
column header. `results.files` lists the files created by the call.

| CSV | Contents |
| --- | --- |
| `exponential.csv` | Cutoff, elapsed time, `dt`, command, numerical and analytic output |
| `frequency_response.csv` | Requested/effective cutoff, input frequency, measured and analytic complex gain |
| `butterworth_step.csv` | Time and step output for both cutoffs |
| `slew.csv` | Target, limited output, elapsed time, `dt`, observed rate and configured limit |
| `observer.csv` | Analytic and estimated position/velocity, measurement, residual, acceptance and outlier flags |
| `angular.csv` | Unwrapped truth, wrapped estimate, angular rate, residual and gate flags |
| `rls.csv` | Regressor, measurement, true parameter, forgetting/cumulative/batch estimates and covariance |

These MATLAB simulations are numerical references. They do not by themselves
execute or prove parity with the C++ library; the companion C++ property suite
tests the library implementations against the corresponding independent
oracles. Its named cases can be selected with a positional argument, and
`--list` prints all cases. No argument runs the whole suite. C++ checks use
exceptions and remain active with `NDEBUG`.
