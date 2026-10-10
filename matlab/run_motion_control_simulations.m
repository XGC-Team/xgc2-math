function results = run_motion_control_simulations(options)
%RUN_MOTION_CONTROL_SIMULATIONS Deterministic motion, control and trajectory checks.
% Base MATLAB / GNU Octave only. All units are SI; planar velocity is body-frame,
% translational acceleration is world-frame and excludes gravity. The SMC plant
% is an ideal acceleration inner loop, not a physical vehicle/safety guarantee.
if nargin < 1, options = struct(); end
if ~isfield(options, 'plot'), options.plot = false; end
if ~isfield(options, 'output_dir'), options.output_dir = tempname; end
if ~isfield(options, 'reference_executable'), options.reference_executable = ''; end
outdir = fullfile(options.output_dir, 'motion_control');
if ~exist(outdir, 'dir'), mkdir(outdir); end

[planar, planar_metrics] = planar_reference();
[delayed, delayed_metrics] = delayed_reference();
[tracking, tracking_metrics] = tracking_reference();
[trajectories, trajectory_metrics] = trajectory_reference();
data = {planar, delayed, tracking, trajectories};
names = {'planar', 'delayed', 'tracking', 'trajectories'};
metrics = {planar_metrics, delayed_metrics, tracking_metrics, trajectory_metrics};
results.cpp_reference_available = ~isempty(options.reference_executable);
for k = 1:numel(names)
    name = names{k};
    write_csv(fullfile(outdir, [name '_reference.csv']), data{k});
    metric = metrics{k};
    native_path = fullfile(outdir, [name '_cpp.csv']);
    % Only this invocation's native output is evidence. In particular, an
    % executable that exits zero without writing must not reuse an old CSV.
    if exist(native_path, 'file') == 2, delete(native_path); end
    if results.cpp_reference_available
        if ~exist(options.reference_executable, 'file')
            error('xgc2:MissingExecutable', 'C++ exporter does not exist: %s', options.reference_executable);
        end
        command = [shell_quote(options.reference_executable) ' ' name ' ' shell_quote(native_path)];
        [status, message] = system(command);
        assert(status == 0, ['C++ reference export failed: ' message]);
        assert(exist(native_path, 'file') == 2, ['C++ exporter did not create ' name ' CSV']);
        actual = dlmread(native_path, ',');
        expected = data{k};
        assert(isequal(size(actual), size(expected)), ['C++ ' name ' CSV shape mismatch']);
        assert(all(isfinite(actual(:))), ['C++ ' name ' emitted nonfinite values']);
        difference = actual - expected;
        if strcmp(name, 'planar')
            difference(:, 5) = atan2(sin(difference(:, 5)), cos(difference(:, 5)));
        end
        metric.cpp_max_abs_error = max(abs(difference(:)));
        tolerance = 2e-9;
        assert(metric.cpp_max_abs_error < tolerance, ...
            sprintf('C++ %s differs from MATLAB reference: %.12g >= %.12g', ...
            name, metric.cpp_max_abs_error, tolerance));
    end
    results.metrics.(name) = metric;
end
results.pass = true;
results.output_dir = outdir;
if options.plot
    plot_results(planar, delayed, tracking, trajectories, outdir);
end
end

function [rows, metrics] = planar_reference()
% Independent SE(2) matrix exponential, evaluated from the initial pose at each
% absolute time. The C++ implementation instead advances a sinc-based ZOH step.
inputs = [1.2 0 0.7; 0.4 0.8 -0.5; -0.9 -0.2 0.4; 1 0.3 1e-11; 0 0 1.1];
time = (0:400)' * 0.01;
rows = zeros(numel(time) * size(inputs, 1), 5);
initial = [cos(0.3) -sin(0.3) 0.4; sin(0.3) cos(0.3) -0.7; 0 0 1];
for scenario = 1:size(inputs, 1)
    u = inputs(scenario, :);
    generator = [0 -u(3) u(1); u(3) 0 u(2); 0 0 0];
    for k = 1:numel(time)
        pose = initial * expm(time(k) * generator);
        index = (scenario - 1) * numel(time) + k;
        rows(index, :) = [scenario - 1, time(k), pose(1:2, 3)', atan2(pose(2, 1), pose(1, 1))];
        assert(norm(pose(1:2, 1:2)' * pose(1:2, 1:2) - eye(2), 'fro') < 1e-12);
    end
end
spin = rows(rows(:, 1) == 4, 3:4);
assert(max(max(abs(spin - repmat([0.4 -0.7], size(spin, 1), 1)))) < 1e-12);
metrics.scenarios = size(inputs, 1);
metrics.samples = size(rows, 1);
end

function [rows, metrics] = delayed_reference()
% Superposition of two continuous-time unit-step responses, including a pure
% delay of 35 ms that does not coincide with the 10 ms output grid.
time = (0:200)' * 0.01;
first = [1.2 -0.6];
second = [-0.4 0.3];
tau = [0.12 0.07];
value = zeros(numel(time), 2);
for axis = 1:2
    on = max(0, time - 0.035);
    changed = max(0, time - 0.665);
    value(:, axis) = first(axis) * (-expm1(-on / tau(axis))) ...
        + (second(axis) - first(axis)) * (-expm1(-changed / tau(axis)));
end
rows = [time value];
assert(all(all(value(time < 0.035, :) == 0)));
assert(max(abs(value(end, :) - second)) < 3e-5);
metrics.delay_s = 0.035;
metrics.final_error = max(abs(value(end, :) - second));
end

function [rows, metrics] = tracking_reference()
% Four sampled closed loops: accelerating and sinusoidal references, two
% controller rates, and a bounded disturbance that reverses at 15 s.
rows = cell(4, 1);
final_error = zeros(4, 2);
max_feedback = 0;
for scenario = 0:3
    dt = 0.005;
    if mod(scenario, 2) == 1, dt = 0.01; end
    time = (0:round(30 / dt))' * dt;
    p = [1; -1; 0.4];
    v = [-0.5; 0.5; 0];
    output = zeros(numel(time), 14);
    for k = 1:numel(time)
        t = time(k);
        [pr, vr, ar] = tracking_target(t, scenario);
        ep = p - pr;
        ev = v - vr;
        z = ev .* abs(ev) + 0.5 * atan(ep);
        sliding = ev + sign(z) .* sqrt(abs(z));
        feedback = -0.5 * max(-1, min(1, sliding / 0.01));
        acceleration = ar + feedback;
        output(k, :) = [scenario t p' v' acceleration' feedback'];
        max_feedback = max(max_feedback, max(abs(feedback)));
        disturbance = [0.1; -0.05; 0.02];
        if t >= 15, disturbance = -disturbance; end
        applied = acceleration + disturbance;
        % Exact constant-acceleration plant under this held control interval.
        p = p + v * dt + 0.5 * applied * dt^2;
        v = v + applied * dt;
    end
    [pr, vr] = tracking_target(30, scenario);
    final_error(scenario + 1, :) = [norm(output(end, 3:5)' - pr), norm(output(end, 6:8)' - vr)];
    assert(final_error(scenario + 1, 1) < 0.008, 'SMC final position error exceeds 8 mm');
    assert(final_error(scenario + 1, 2) < 0.015, 'SMC final velocity error exceeds 15 mm/s');
    rows{scenario + 1} = output;
end
rows = vertcat(rows{:});
assert(max_feedback <= 0.5 + 1e-14);
assert(all(isfinite(rows(:))));
metrics.final_position_error_m = final_error(:, 1)';
metrics.final_velocity_error_mps = final_error(:, 2)';
metrics.max_feedback_mps2 = max_feedback;
metrics.scenarios = 4;
end

function [p, v, a] = tracking_target(t, scenario)
if scenario < 2
    a = [0.2; -0.1; 0.03];
    p = 0.5 * t^2 * a;
    v = t * a;
else
    amplitude = [0.8; 0.6; 0.3];
    frequency = [0.4; 0.3; 0.2];
    p = amplitude .* sin(frequency * t);
    v = amplitude .* frequency .* cos(frequency * t);
    a = -amplitude .* frequency.^2 .* sin(frequency * t);
end
end

function [rows, metrics] = trajectory_reference()
% Differentiate harmonic coefficients generically rather than copying the
% individual position/velocity/acceleration/jerk/snap expressions in C++.
time = (0:500)' * 0.02;
rows = zeros(4 * numel(time), 17);
for scenario = 0:3
    values = zeros(numel(time), 15);
    for order = 0:4
        derivative = harmonic_curve(scenario, time, order);
        values(:, 3*order + (1:3)) = derivative;
    end
    offset = scenario * numel(time);
    rows(offset + (1:numel(time)), :) = [scenario * ones(numel(time), 1), time, values];
end
assert(all(isfinite(rows(:))));
metrics.curves = 4;
metrics.derivative_orders = 5;
metrics.samples = size(rows, 1);
end

function xyz = harmonic_curve(scenario, t, derivative)
% Each term is [axis, amplitude, frequency, phase] for A*sin(w*t+phase).
if scenario == 0
    terms = [1 3 1 pi/2; 2 3 1 0];
    offset = [0 0 3]; slope = [0 0 0];
elseif scenario == 1
    terms = [1 3 1 0; 2 1.5 2 0];
    offset = [0 0 3]; slope = [0 0 0];
elseif scenario == 2
    terms = [1 1 0.9 pi/2; 2 1 0.9 0];
    offset = [0 0 0]; slope = [0 0 0.1];
else
    terms = [1 0.3 0.3 0; 1 0.6 0.6 0; 2 0.3 0.3 pi/2; 2 -0.6 0.6 pi/2; 3 0.3 0.9 0];
    offset = [0 0 1.2]; slope = [0 0 0];
end
xyz = zeros(numel(t), 3);
for k = 1:size(terms, 1)
    axis = terms(k, 1); amplitude = terms(k, 2); omega = terms(k, 3); phase = terms(k, 4);
    xyz(:, axis) = xyz(:, axis) + amplitude * omega^derivative * sin(omega * t + phase + derivative * pi/2);
end
if derivative == 0
    xyz = xyz + repmat(offset, numel(t), 1) + t * slope;
elseif derivative == 1
    xyz = xyz + repmat(slope, numel(t), 1);
end
end

function write_csv(path, data)
dlmwrite(path, data, 'delimiter', ',', 'precision', '%.17g');
end

function value = shell_quote(value)
if ispc
    assert(isempty(regexp(value, '["%\r\n]', 'once')), 'Unsupported character in executable/output path');
    value = ['"' value '"'];
else
    quote = char(39);
    value = [quote strrep(value, quote, [quote char(34) quote char(34) quote]) quote];
end
end

function plot_results(planar, delayed, tracking, trajectories, outdir)
figure_handle = figure('Visible', 'off', 'Position', [100 100 1100 760]);
cleanup = onCleanup(@() close(figure_handle)); %#ok<NASGU>
subplot(2, 2, 1); hold on;
for k = 0:4
    segment = planar(planar(:, 1) == k, :);
    if k == 4
        plot(segment(1, 3), segment(1, 4), 'o', 'MarkerSize', 7, ...
            'MarkerFaceColor', [0.47 0.67 0.19], 'MarkerEdgeColor', [0.2 0.4 0.1], 'LineWidth', 1.2);
    else
        plot(segment(:, 3), segment(:, 4), 'LineWidth', 1.4);
    end
end
axis equal; grid on; xlabel('World x (m)'); ylabel('World y (m)'); title('Body velocity: five planar motions');
legend('Unicycle', 'Mecanum', 'Reverse', 'Tiny yaw rate', 'Spin', 'Location', 'northwest');
subplot(2, 2, 2); plot(delayed(:, 1), delayed(:, 2:3), 'LineWidth', 1.4); grid on;
xlabel('Time (s)'); ylabel('Velocity (m/s or rad/s)'); title('35 ms delay and actuator lag');
legend('Forward velocity', 'Yaw rate', 'Location', 'best');
subplot(2, 2, 3); hold on;
for k = 0:3
    segment = tracking(tracking(:, 1) == k, :);
    target = zeros(size(segment, 1), 3);
    for j = 1:size(segment, 1)
        p = tracking_target(segment(j, 2), k);
        target(j, :) = p';
    end
    semilogy(segment(:, 2), max(1e-8, sqrt(sum((segment(:, 3:5) - target).^2, 2))), 'LineWidth', 1.2);
end
grid on; xlabel('Time (s)'); ylabel('Position error (m)'); title('Sampled SMC: disturbance reverses at 15 s');
legend('Accel 200 Hz', 'Accel 100 Hz', 'Sine 200 Hz', 'Sine 100 Hz', 'Location', 'best');
subplot(2, 2, 4); hold on;
for k = 0:3
    segment = trajectories(trajectories(:, 1) == k, :);
    plot3(segment(:, 3), segment(:, 4), segment(:, 5), 'LineWidth', 1.3);
end
grid on; axis tight; pbaspect([1.4 1 1]); view(35, 25); xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
title('Analytic trajectories: derivatives through snap');
trajectory_legend = legend('Circle', 'Figure eight', 'Helix', 'Torus knot', 'Location', 'eastoutside');
set(trajectory_legend, 'FontSize', 9);
print(figure_handle, fullfile(outdir, 'motion_control.png'), '-dpng', '-r140');
end
