function results = run_filter_observer_simulation(options)
%RUN_FILTER_OBSERVER_SIMULATION Deterministic filter and observer references.
%   RESULTS = RUN_FILTER_OBSERVER_SIMULATION() runs assertions without plots.
%   OPTIONS.plot enables a six-panel diagnostic figure (default false).
%   OPTIONS.output_dir writes trace CSVs and, when plot=true, a PNG.
%   No toolbox is required. Tests use explicit time schedules and signals;
%   a fixed RNG seed is set and the caller's RNG state is restored on exit.

if nargin < 1
    options = struct();
end
options = checked_options(options);
previous_rng = rng;
rng_cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(1729, 'twister');

[exponential_metrics, exponential] = exponential_case();
[butterworth_metrics, frequency_response, butterworth_step] = butterworth_case();
[slew_metrics, slew] = slew_case();
[observer_metrics, observer] = observer_case(false);
[angular_metrics, angular] = observer_case(true);
[rls_metrics, rls] = rls_case();

results = struct();
results.name = 'filter_observer';
results.seed = 1729;
results.pass = true;
results.metrics = struct();
groups = {exponential_metrics, butterworth_metrics, slew_metrics, ...
    observer_metrics, angular_metrics, rls_metrics};
for i = 1:numel(groups)
    fields = fieldnames(groups{i});
    for j = 1:numel(fields)
        results.metrics.(fields{j}) = groups{i}.(fields{j});
    end
end
results.case_names = {'exponential_partition', 'butterworth_response', ...
    'slew_variable_dt', 'observer_acceleration_gating', 'angular_wrap', ...
    'rls_parameter_change'};
results.traces = struct('exponential', exponential, ...
    'frequency_response', frequency_response, 'butterworth_step', butterworth_step, ...
    'slew', slew, 'observer', observer, 'angular', angular, 'rls', rls);
results.files = {};

if ~isempty(options.output_dir)
    if exist(options.output_dir, 'dir') ~= 7
        [created, message] = mkdir(options.output_dir);
        require(created, ['Cannot create output directory: ' message]);
    end
    names = fieldnames(results.traces);
    for i = 1:numel(names)
        trace = results.traces.(names{i});
        path = fullfile(options.output_dir, [names{i} '.csv']);
        write_csv(path, trace.header, trace.data);
        results.files{end + 1} = path; %#ok<AGROW>
    end
end
if options.plot
    figure_handle = plot_diagnostics(results.traces, ~isempty(options.output_dir));
    if ~isempty(options.output_dir)
        figure_cleanup = onCleanup(@() close(figure_handle)); %#ok<NASGU>
        path = fullfile(options.output_dir, 'filter_observer_diagnostics.png');
        print(figure_handle, path, '-dpng', '-r130');
        results.files{end + 1} = path;
    end
end
end

function options = checked_options(options)
require(isstruct(options) && isscalar(options), 'options must be a scalar struct');
if ~isfield(options, 'plot')
    options.plot = false;
end
require(isscalar(options.plot) && (islogical(options.plot) || isnumeric(options.plot)) ...
    && isfinite(options.plot) && (options.plot == 0 || options.plot == 1), ...
    'options.plot must be a scalar logical');
options.plot = logical(options.plot);
if ~isfield(options, 'output_dir')
    options.output_dir = '';
end
if isstring(options.output_dir) && isscalar(options.output_dir)
    options.output_dir = char(options.output_dir);
end
require(ischar(options.output_dir) && (isempty(options.output_dir) || isrow(options.output_dir)), ...
    'options.output_dir must be a character row or string scalar');
end

function [metrics, trace] = exponential_case()
cutoffs = [0.02, 0.7, 8.0, 30.0];
steps = [1.0e-7, 0.001, 0.007, 0.013, 0.037, 0.09];
initial = -1.7;
target = 2.3;
data = zeros(600 * numel(cutoffs), 6);
max_error = 0;
semigroup_error = 0;
row = 0;
for cutoff = cutoffs
    time = 0;
    value = initial;
    for i = 1:600
        dt = steps(mod(i - 1, numel(steps)) + 1);
        time = time + dt;
        previous = value;
        value = value - expm1(-2 * pi * cutoff * dt) * (target - value);
        exact = target + (initial - target) * exp(-2 * pi * cutoff * time);
        max_error = max(max_error, abs(value - exact));
        require(value >= previous - 1.0e-14 && value <= target + 1.0e-14, ...
            'Exponential step must be monotone without overshoot');
        row = row + 1;
        data(row, :) = [cutoff, time, dt, target, value, exact];
    end
    semigroup_error = max(semigroup_error, abs(value - ...
        (target + (initial - target) * exp(-2 * pi * cutoff * time))));
    for first_dt = [1.0e-12, 0.004, 0.17, 10.0]
        split = initial - expm1(-2 * pi * cutoff * first_dt) * (target - initial);
        split = split - expm1(-2 * pi * cutoff * 1.0e-8) * (target - split);
        whole = initial - expm1(-2 * pi * cutoff * (first_dt + 1.0e-8)) * (target - initial);
        semigroup_error = max(semigroup_error, abs(split - whole));
    end
end
require(max_error < 8.0e-13, 'Exponential response differs from the exact constant-input ODE solution');
require(semigroup_error < 8.0e-13, 'Exponential result changes when elapsed time is partitioned');
metrics = struct('exponential_max_abs_error', max_error, ...
    'exponential_semigroup_max_abs_error', semigroup_error);
trace = struct('header', 'cutoff_hz,time_s,dt_s,input,output,exact', 'data', data);
end

function [metrics, response_trace, step_trace] = butterworth_case()
sample_hz = 200;
dt = 1 / sample_hz;
cutoffs = [2.5, 12, 80];
frequencies = [0.5, 2.5, 12, 30, 80];
settle_samples = 4000;
measured_samples = 4000;
sample = (0:(settle_samples + measured_samples - 1))';
response = zeros(numel(cutoffs) * numel(frequencies), 9);
max_gain_error = 0;
max_complex_error = 0;
row = 0;
for cutoff = cutoffs
    effective_cutoff = min(cutoff, 0.225 * sample_hz);
    [b, a] = butterworth_coefficients(effective_cutoff, sample_hz);
    for frequency = frequencies
        phase = 2 * pi * frequency * sample * dt;
        % Base MATLAB/Octave filter is an independent numerical realization.
        output = filter(b, a, sin(phase));
        selected = (settle_samples + 1):(settle_samples + measured_samples);
        sine_gain = 2 * mean(output(selected) .* sin(phase(selected)));
        cosine_gain = 2 * mean(output(selected) .* cos(phase(selected)));
        measured = complex(sine_gain, cosine_gain);
        % Analog prototype plus bilinear frequency map; independent of b/a.
        ratio = tan(pi * frequency / sample_hz) / tan(pi * effective_cutoff / sample_hz);
        expected = 1 / complex(1 - ratio^2, sqrt(2) * ratio);
        max_gain_error = max(max_gain_error, abs(abs(measured) - abs(expected)));
        max_complex_error = max(max_complex_error, abs(measured - expected));
        row = row + 1;
        response(row, :) = [cutoff, effective_cutoff, frequency, abs(measured), ...
            abs(expected), real(measured), imag(measured), real(expected), imag(expected)];
    end
end
require(max_gain_error < 2.0e-10 && max_complex_error < 3.0e-10, ...
    'Butterworth sweep differs from the analytic complex transfer function');

step_data = zeros(4000, 3);
peaks = zeros(1, 2);
final_error = 0;
for i = 1:2
    cutoff = cutoffs(i);
    [b, a] = butterworth_coefficients(cutoff, sample_hz);
    output = filter(b, a, ones(2000, 1));
    peaks(i) = max(output);
    require(all(isfinite(output)) && min(output) >= -1.0e-12 ...
        && peaks(i) > 1.03 && peaks(i) < 1.06, ...
        'Butterworth step lost its stable second-order overshoot');
    final_error = max(final_error, abs(output(end) - 1));
    rows = (i - 1) * 2000 + (1:2000);
    step_data(rows, :) = [repmat(cutoff, 2000, 1), (0:1999)' * dt, output];
end
require(final_error < 2.0e-12, 'Butterworth DC gain is not one');
metrics = struct('butterworth_max_gain_error', max_gain_error, ...
    'butterworth_max_complex_error', max_complex_error, ...
    'butterworth_step_max_peak', max(peaks), 'butterworth_step_final_error', final_error);
response_trace = struct('header', ['cutoff_hz,effective_cutoff_hz,frequency_hz,' ...
    'measured_gain,expected_gain,measured_real,measured_imag,expected_real,expected_imag'], ...
    'data', response);
step_trace = struct('header', 'cutoff_hz,time_s,output', 'data', step_data);
end

function [b, a] = butterworth_coefficients(cutoff, sample_hz)
warped = tan(pi * cutoff / sample_hz);
denominator = 1 + sqrt(2) * warped + warped^2;
b = warped^2 / denominator * [1, 2, 1];
a = [1, 2 * (warped^2 - 1) / denominator, ...
    (1 - sqrt(2) * warped + warped^2) / denominator];
end

function [metrics, trace] = slew_case()
rate = 0.75;
steps = [1.0e-8, 0.001, 0.007, 0.025, 0.12];
value = -0.2;
time = 0;
variation = 0;
max_violation = 0;
data = zeros(1200, 6);
for i = 1:1200
    index = i - 1;
    dt = steps(mod(index, numel(steps)) + 1);
    if mod(floor(index / 71), 2) == 0
        target = 1.5;
    else
        target = -1.2;
    end
    target = target + 0.2 * sin(0.31 * index);
    previous = value;
    value = min(max(target, previous - rate * dt), previous + rate * dt);
    movement = abs(value - previous);
    max_violation = max(max_violation, movement - rate * dt);
    require(value >= min(previous, target) - 3.0e-16 && value <= max(previous, target) + 3.0e-16, ...
        'Slew limiter moved away from or past its target');
    require(abs(movement - min(abs(target - previous), rate * dt)) < 3.0e-16, ...
        'Slew limiter failed to reach the closest feasible value');
    time = time + dt;
    variation = variation + movement;
    data(i, :) = [time, dt, target, value, (value - previous) / dt, rate];
end
require(max_violation < 3.0e-16, 'Slew limiter violated its physical rate constraint');
require(variation <= rate * time + 1.0e-11, 'Slew total variation exceeds rate times elapsed time');
partition_error = 0;
for target = [-8, -0.25, 0.4, 0.45, 12]
    split = 0.4;
    for dt = steps
        split = min(max(target, split - rate * dt), split + rate * dt);
    end
    whole = min(max(target, 0.4 - rate * sum(steps)), 0.4 + rate * sum(steps));
    partition_error = max(partition_error, abs(split - whole));
end
require(partition_error < 3.0e-16, 'Slew result changes when constant-target elapsed time is partitioned');
metrics = struct('slew_max_step_constraint_violation', max(0, max_violation), ...
    'slew_total_variation', variation, 'slew_total_variation_bound', rate * time, ...
    'slew_partition_max_abs_error', partition_error);
trace = struct('header', 'time_s,dt_s,target,output,rate_per_s,rate_limit_per_s', 'data', data);
end

function [metrics, trace] = observer_case(angular)
if angular
    initial_position = pi - 0.08;
    initial_velocity = 1.7;
    acceleration = 0.15;
    steps = [0.007, 0.012, 0.031];
    count = 1200;
    gate = 0.2;
    period = 83;
    phase = 17;
    outlier_offset = 0.8;
else
    initial_position = -0.7;
    initial_velocity = 0.4;
    acceleration = 0.8;
    steps = [0.003, 0.007, 0.013, 0.027];
    count = 1000;
    gate = 0.15;
    period = 67;
    phase = 31;
    outlier_offset = 3.0;
end
state = [initial_position; initial_velocity];
time = 0;
max_position_error = 0;
max_velocity_error = 0;
rejected = 0;
wrap_count = 0;
previous_wrapped = initial_position;
data = zeros(count, 10);
for i = 1:count
    index = i - 1;
    dt = steps(mod(index, numel(steps)) + 1);
    time = time + dt;
    true_position = initial_position + initial_velocity * time + 0.5 * acceleration * time^2;
    true_velocity = initial_velocity + acceleration * time;
    measurement = true_position;
    if angular
        circular_measurement = atan2(sin(true_position), cos(true_position));
        wrap_count = wrap_count + (abs(circular_measurement - previous_wrapped) > pi);
        previous_wrapped = circular_measurement;
        measurement = circular_measurement + 2 * pi * (mod(index, 5) - 2);
    end
    outlier = mod(index, period) == phase;
    measurement = measurement + outlier_offset * outlier;
    [state, residual, accepted] = observer_step(state, measurement, dt, acceleration, gate, angular);
    position_error = state(1) - true_position;
    if angular
        position_error = atan2(sin(position_error), cos(position_error));
        require(abs(state(1)) <= pi + 1.0e-14, 'Angular estimate left the principal interval');
    end
    max_position_error = max(max_position_error, abs(position_error));
    max_velocity_error = max(max_velocity_error, abs(state(2) - true_velocity));
    require(accepted ~= outlier, 'Observer gate accepted the wrong samples');
    if outlier
        rejected = rejected + 1;
        if angular
            require(abs(residual - 0.8) < 1.0e-11, ...
                'Angular outlier must report the positive short residual of 0.8 rad');
        else
            require(abs(residual - outlier_offset) < 1.0e-11, ...
                'Outlier residual must be computed after model prediction');
        end
    end
    data(i, :) = [time, dt, true_position, true_velocity, measurement, ...
        state(1), state(2), residual, double(accepted), double(outlier)];
end
require(max_position_error < 1.0e-11 && max_velocity_error < 1.0e-11, ...
    'Observer differs from analytic constant-acceleration motion');
held = state;
[state, ~, accepted] = observer_step(state, 0, 0.2, acceleration, gate, angular);
require(~accepted && isequal(state, held), 'Invalid dt must hold both observer states');
if angular
    require(wrap_count >= 10, 'Angular scenario did not exercise repeated full rotations');
    % Independent one-step posterior for a nonzero innovation across +/-pi.
    % Prediction: pi-0.01. Measurement: -pi+0.03. Short residual: +0.04.
    % The alpha/beta correction must wrap the posterior to -pi+0.004
    % and change the rate from 0.2 to 0.2+0.08*0.04/0.05 = 0.264.
    [corrected, short_residual, accepted] = observer_step([pi - 0.02; 0.2], ...
        -pi + 0.03, 0.05, 0, gate, true);
    require(accepted, 'Small physical innovation across the angle branch cut must be accepted');
    require(abs(short_residual - 0.04) < 2.0e-14, ...
        'Angular correction used the long or wrong-sign residual');
    posterior_position_error = abs(corrected(1) - (-pi + 0.004));
    posterior_velocity_error = abs(corrected(2) - 0.264);
    require(posterior_position_error < 2.0e-14, ...
        'Angular posterior position omitted correction or principal-angle normalization');
    require(posterior_velocity_error < 2.0e-14, ...
        'Angular posterior velocity omitted or mis-scaled correction');
    metrics = struct('angular_max_position_error_rad', max_position_error, ...
        'angular_max_velocity_error_rad_s', max_velocity_error, ...
        'angular_wrap_count', wrap_count, 'angular_rejected_samples', rejected, ...
        'angular_correction_short_residual_rad', short_residual, ...
        'angular_correction_position_error_rad', posterior_position_error, ...
        'angular_correction_velocity_error_rad_s', posterior_velocity_error);
    header = ['time_s,dt_s,true_angle_unwrapped_rad,true_rate_rad_s,measurement_rad,' ...
        'estimated_angle_rad,estimated_rate_rad_s,residual_rad,accepted,injected_outlier'];
else
    require(rejected == 15, 'Observer scenario has the wrong number of injected outliers');
    % A perturbed initial state must converge, so prediction alone cannot pass.
    recovering = [initial_position + 0.1; initial_velocity - 0.2];
    recovery_dt = 0.02;
    for i = 1:160
        recovery_time = i * recovery_dt;
        recovery_truth = initial_position + initial_velocity * recovery_time ...
            + 0.5 * acceleration * recovery_time^2;
        [recovering, ~, accepted] = observer_step(recovering, recovery_truth, ...
            recovery_dt, acceleration, gate, false);
        require(accepted, 'Bounded initial observer error unexpectedly triggered the gate');
    end
    recovery_error = max(abs(recovering - [recovery_truth; initial_velocity + acceleration * recovery_time]));
    require(recovery_error < 1.0e-11, 'Observer correction did not remove the initial position/velocity error');
    metrics = struct('observer_max_position_error_m', max_position_error, ...
        'observer_max_velocity_error_m_s', max_velocity_error, ...
        'observer_rejected_samples', rejected, 'observer_recovery_final_max_error', recovery_error);
    header = ['time_s,dt_s,true_position_m,true_velocity_m_s,measurement_m,' ...
        'estimated_position_m,estimated_velocity_m_s,residual_m,accepted,injected_outlier'];
end
trace = struct('header', header, 'data', data);
end

function [state, residual, accepted] = observer_step(state, measurement, dt, acceleration, gate, angular)
residual = 0;
accepted = false;
if ~isfinite(measurement) || ~isfinite(acceleration) || ~isfinite(dt) || dt < 1.0e-6 || dt > 0.1
    return;
end
state = [1, dt; 0, 1] * state + [0.5 * dt^2; dt] * acceleration;
if angular
    state(1) = mod(state(1) + pi, 2 * pi) - pi;
    measurement = mod(measurement + pi, 2 * pi) - pi;
    residual = atan2(sin(measurement - state(1)), cos(measurement - state(1)));
else
    residual = measurement - state(1);
end
if abs(residual) > gate
    return; % Reject the measurement, but retain the model prediction.
end
state = state + [0.35; 0.08 / dt] * residual;
if angular
    state(1) = mod(state(1) + pi, 2 * pi) - pi;
end
accepted = true;
end

function [metrics, trace] = rls_case()
lambda = 0.985;
initial_covariance = 7.0;
prior = -0.6;
theta = prior;
covariance = initial_covariance;
cumulative_theta = prior;
cumulative_covariance = initial_covariance;
regressors = zeros(1200, 1);
measurements = zeros(1200, 1);
data = zeros(1200, 9);
max_parameter_error = 0;
max_covariance_error = 0;
for i = 1:1200
    index = i - 1;
    phi = 0.8 + 0.3 * sin(0.17 * index);
    if i <= 600
        truth = 1.8;
    else
        truth = -0.7;
    end
    measurement = truth * phi + 0.02 * sin(0.41 * index) + 0.01 * cos(0.13 * index);
    regressors(i) = phi;
    measurements(i) = measurement;
    denominator = lambda + phi * covariance * phi;
    theta = theta + covariance * phi / denominator * (measurement - phi * theta);
    covariance = covariance / denominator;
    denominator = 1 + phi * cumulative_covariance * phi;
    cumulative_theta = cumulative_theta + cumulative_covariance * phi / denominator ...
        * (measurement - phi * cumulative_theta);
    cumulative_covariance = cumulative_covariance / denominator;

    % Independently solve the weighted batch objective from all observations.
    % J(theta) = lambda^i/P0*(theta-prior)^2
    %          + sum_j lambda^(i-j)*(y_j-phi_j*theta)^2.
    weights = lambda .^ ((i - 1):-1:0)';
    batch_precision = lambda^i / initial_covariance ...
        + sum(weights .* regressors(1:i).^2);
    batch_information = lambda^i * prior / initial_covariance ...
        + sum(weights .* regressors(1:i) .* measurements(1:i));
    batch_theta = batch_information / batch_precision;
    batch_covariance = 1 / batch_precision;
    max_parameter_error = max(max_parameter_error, abs(theta - batch_theta));
    max_covariance_error = max(max_covariance_error, abs(covariance - batch_covariance));
    data(i, :) = [index, phi, measurement, truth, theta, cumulative_theta, ...
        batch_theta, covariance, batch_covariance];
    if i == 600
        require(abs(theta - truth) < 0.02, 'RLS did not converge before the parameter change');
    end
end
require(max_parameter_error < 3.0e-12 && max_covariance_error < 3.0e-12, ...
    'RLS differs from the full weighted batch solution including the prior');
require(abs(theta + 0.7) < 0.02, 'Forgetting RLS did not converge after the parameter change');
require(abs(cumulative_theta + 0.7) > 0.7, ...
    'Parameter-change scenario failed to distinguish cumulative least squares');
metrics = struct('rls_batch_parameter_max_abs_error', max_parameter_error, ...
    'rls_batch_covariance_max_abs_error', max_covariance_error, ...
    'rls_final_parameter', theta, 'rls_final_abs_tracking_error', abs(theta + 0.7), ...
    'rls_no_forgetting_final_abs_error', abs(cumulative_theta + 0.7));
trace = struct('header', ['sample,regressor,measurement,true_parameter,forgetting_estimate,' ...
    'no_forgetting_estimate,batch_estimate,covariance,batch_covariance'], 'data', data);
end

function figure_handle = plot_diagnostics(traces, invisible)
visibility = 'on';
if invisible
    visibility = 'off';
end
figure_handle = figure('Visible', visibility, 'Color', 'w', 'Position', [100, 100, 1400, 1000]);

subplot(3, 2, 1);
data = traces.exponential.data;
selected = data(:, 1) == 0.7;
plot(data(selected, 2), data(selected, 5), 'b-', data(selected, 2), data(selected, 6), 'k--');
xlim([0, 2]); grid on;
xlabel('Time (s)'); ylabel('Output'); title('Exponential filter: exact variable-step response');
legend('Filter', 'Analytic solution', 'Location', 'southeast');

subplot(3, 2, 2);
data = traces.frequency_response.data;
selected = data(:, 1) == 12;
semilogx(data(selected, 3), 20 * log10(data(selected, 5)), 'k-', ...
    data(selected, 3), 20 * log10(data(selected, 4)), 'bo');
grid on; xlabel('Frequency (Hz)'); ylabel('Gain (dB)');
title('Butterworth: prewarped transfer-function oracle');
legend('Analytic gain', 'Measured gain', 'Location', 'southwest');

subplot(3, 2, 3);
data = traces.slew.data;
plot(data(:, 1), data(:, 3), 'Color', [0.65, 0.65, 0.65]); hold on;
plot(data(:, 1), data(:, 4), 'b-'); grid on;
xlabel('Time (s)'); ylabel('Command'); title('Slew limiter: variable time steps');
legend('Target', 'Rate-limited output', 'Location', 'southwest');

subplot(3, 2, 4);
data = traces.observer.data;
plot(data(:, 1), data(:, 3), 'k--', data(:, 1), data(:, 6), 'b-'); hold on;
rejected = data(:, 10) == 1;
plot(data(rejected, 1), data(rejected, 5), 'rx'); grid on;
xlabel('Time (s)'); ylabel('Position (m)'); title('Observer: acceleration and rejected outliers');
legend('Truth', 'Estimate', 'Rejected input', 'Location', 'northwest');

subplot(3, 2, 5);
data = traces.angular.data;
plot(data(:, 1), atan2(sin(data(:, 3)), cos(data(:, 3))), 'k--', data(:, 1), data(:, 6), 'b-');
grid on; xlabel('Time (s)'); ylabel('Wrapped angle (rad)');
title('Angular observer: repeated branch crossings');
legend('Truth', 'Estimate', 'Location', 'southeast');

subplot(3, 2, 6);
data = traces.rls.data;
plot(data(:, 1), data(:, 4), 'k--', data(:, 1), data(:, 5), 'b-', ...
    data(:, 1), data(:, 6), 'Color', [0.8, 0.25, 0.1]);
grid on; xlabel('Sample'); ylabel('Parameter'); title('RLS: parameter change and forgetting');
legend('True parameter', 'Forgetting', 'No forgetting', 'Location', 'southwest');
end

function write_csv(path, header, data)
file = fopen(path, 'w');
require(file >= 0, ['Cannot open CSV output: ' path]);
file_cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
fprintf(file, '%s\n', header);
format = [repmat('%.17g,', 1, size(data, 2) - 1), '%.17g\n'];
fprintf(file, format, data');
end

function require(condition, message)
if ~isscalar(condition) || ~condition
    error('xgc2_math:filter_observer:checkFailed', '%s', message);
end
end
