function results = run_inertial_pose_eskf_simulation(options)
%RUN_INERTIAL_POSE_ESKF_SIMULATION Analytic IMU fixtures and native ESKF replay.
%   A toolbox-free, headless midpoint strapdown reference, NOT a MATLAB ESKF
%   clone. Set options.replay_executable to the built pose3_eskf_csv_replay
%   executable to evaluate the real C++ estimator on the generated samples.
%   options: seed (20261010), plot (false), output_dir (''),
%            replay_executable (auto-detect <repo>/build if omitted).
%   State CSV: time,p_x,p_y,p_z,q_x,q_y,q_z,q_w,v_x,v_y,v_z.
%   IMU acceleration is body-frame specific force; world gravity is -Z.

if nargin < 1, options = struct(); end
options = resolve_options(options);
saved_rng = rng();
rng_cleanup = onCleanup(@() rng(saved_rng)); %#ok<NASGU>
if isempty(options.output_dir)
    work_dir = tempname(); mkdir(work_dir);
    work_cleanup = onCleanup(@() rmdir(work_dir, 's')); %#ok<NASGU>
else
    work_dir = fullfile(options.output_dir, 'inertial_pose_eskf');
    if exist(work_dir, 'dir') ~= 7, mkdir(work_dir); end
end
cases = scenario_definitions();
results = struct('pass', true, 'seed', options.seed, ...
    'cpp_replay_available', ~isempty(options.replay_executable), ...
    'reference_scope', 'Analytic truth and midpoint strapdown; no MATLAB pose fusion or covariance model');
results.scenarios = cell(numel(cases), 1);
results.metrics = struct([]);
for i = 1:numel(cases)
    rng(options.seed + i - 1, 'twister');
    cfg = cases{i};
    cfg.seed = options.seed + i - 1;
    scenario_dir = fullfile(work_dir, cfg.name);
    if exist(scenario_dir, 'dir') ~= 7, mkdir(scenario_dir); end
    fixture = make_fixture(cfg);
    reference = integrate_strapdown(fixture);
    metrics = trajectory_metrics(reference, fixture.truth, cfg);
    metrics.seed = cfg.seed;
    metrics.duration_s = cfg.duration;
    metrics.pose_arrival_delay_s = cfg.arrival_delay_s;
    metrics.pose_dropout_begin_s = cfg.dropout(1);
    metrics.pose_dropout_end_s = cfg.dropout(2);
    metrics.imu_samples = numel(fixture.time);
    metrics.pose_samples = sum(fixture.events(:, 1) == 1);
    metrics.cpp_replay_exercised = false;
    metrics.cpp_position_rmse_m = NaN;
    metrics.cpp_velocity_rmse_mps = NaN;
    metrics.cpp_orientation_rmse_rad = NaN;
    metrics.cpp_dropout_position_rmse_m = NaN;
    metrics.cpp_recovery_position_rmse_m = NaN;
    metrics.cpp_rows = 0;
    assert(all(isfinite(reference(:))), 'Strapdown reference produced non-finite states.');
    assert(metrics.max_quaternion_norm_error < 1e-12, 'Reference quaternion lost unit norm.');
    if ~cfg.noisy
        assert(metrics.position_rmse_m < 2e-5, 'Noiseless position disagrees with analytic truth.');
        assert(metrics.velocity_rmse_mps < 2e-5, 'Noiseless velocity disagrees with analytic truth.');
        assert(metrics.orientation_rmse_rad < 1e-7, 'Noiseless attitude disagrees with analytic truth.');
    end
    events_path = fullfile(scenario_dir, 'events.csv');
    state_path = fullfile(scenario_dir, 'cpp_state.csv');
    log_path = fullfile(scenario_dir, 'cpp_replay.log');
    write_events(events_path, fixture.events);
    write_matrix(fullfile(scenario_dir, 'truth.csv'), fixture.truth);
    write_matrix(fullfile(scenario_dir, 'baseline.csv'), reference);
    native_states = zeros(0, 11);
    if exist(state_path, 'file') == 2
        % A successful process exit is insufficient: the executable must
        % create fresh output, not accidentally validate an earlier run.
        delete(state_path);
    end
    if exist(log_path, 'file') == 2, delete(log_path); end
    if results.cpp_replay_available
        % Delayed arrivals retain their acquisition stamps; delay CLI stays 0.
        command = [shell_quote(options.replay_executable), ' ', ...
            shell_quote(events_path), ' ', shell_quote(state_path), ' 2>&1'];
        [status, replay_log] = system(command);
        write_text(log_path, replay_log);
        assert(status == 0, ['C++ ESKF replay failed: ', replay_log]);
        assert(exist(state_path, 'file') == 2, 'C++ replay did not create a state CSV.');
        native_states = dlmread(state_path, ',');
        assert(size(native_states, 2) >= 11, 'C++ state CSV needs at least 11 columns.');
        native_states = native_states(:, 1:11);
        assert(size(native_states, 1) == numel(fixture.time) - 1, 'Unexpected C++ replay row count.');
        assert(all(isfinite(native_states(:))), 'C++ replay produced non-finite states.');
        native_truth = fixture.truth(2:end, :);
        assert(max(abs(native_states(:, 1) - native_truth(:, 1))) < 1e-8, 'C++ replay changed the IMU clock.');
        native_metrics = trajectory_metrics(native_states, native_truth, cfg);
        assert(native_metrics.max_quaternion_norm_error < 1e-10, 'C++ quaternion lost unit norm.');
        metrics.cpp_replay_exercised = true;
        metrics.cpp_position_rmse_m = native_metrics.position_rmse_m;
        metrics.cpp_velocity_rmse_mps = native_metrics.velocity_rmse_mps;
        metrics.cpp_orientation_rmse_rad = native_metrics.orientation_rmse_rad;
        metrics.cpp_dropout_position_rmse_m = native_metrics.dropout_position_rmse_m;
        metrics.cpp_recovery_position_rmse_m = native_metrics.recovery_position_rmse_m;
        metrics.cpp_rows = size(native_states, 1);
        if cfg.noisy
            % Physical regression bounds, not a tuned benchmark claim.
            assert(native_metrics.position_rmse_m < 0.12, 'Noisy native position RMSE exceeded 12 cm.');
            assert(native_metrics.velocity_rmse_mps < 0.25, 'Noisy native velocity RMSE exceeded 0.25 m/s.');
            assert(native_metrics.orientation_rmse_rad < 0.04, 'Noisy native attitude RMSE exceeded 0.04 rad.');
            assert(native_metrics.position_rmse_m < metrics.position_rmse_m, ...
                'Pose fusion did not improve the biased inertial baseline.');
        else
            assert(native_metrics.position_rmse_m < 2e-5, 'Noiseless native position differs from analytic truth.');
            assert(native_metrics.velocity_rmse_mps < 2e-5, 'Noiseless native velocity differs from analytic truth.');
            assert(native_metrics.orientation_rmse_rad < 1e-7, 'Noiseless native attitude differs from analytic truth.');
        end
        if all(isfinite(cfg.dropout))
            assert(native_metrics.recovery_position_rmse_m < 0.04, 'Native ESKF did not recover after pose dropout.');
        end
    end
    write_metrics(fullfile(scenario_dir, 'metrics.csv'), metrics);
    item = struct('name', cfg.name, 'config', cfg, 'truth', fixture.truth, ...
        'baseline', reference, 'cpp_states', native_states, 'metrics', metrics);
    results.scenarios{i} = item;
    results.metrics(i) = metrics;
    fprintf('[ESKF] %-22s IMU=%3g Hz pose=%2g Hz baseline_p_rmse=%.6g m cpp_p_rmse=%.6g m\n', ...
        cfg.name, cfg.imu_hz, cfg.pose_hz, metrics.position_rmse_m, metrics.cpp_position_rmse_m);
    if options.plot, plot_scenario(item, scenario_dir); end
end
if ~results.cpp_replay_available
    fprintf('[ESKF] C++ replay SKIPPED: set options.replay_executable to validate native pose fusion.\n');
end
write_metrics(fullfile(work_dir, 'metrics.csv'), results.metrics);
results.scenario_count = numel(cases);
if isempty(options.output_dir), results.output_dir = ''; else, results.output_dir = work_dir; end
end

function options = resolve_options(options)
assert(isstruct(options) && isscalar(options), 'options must be a scalar struct.');
if ~isfield(options, 'seed'), options.seed = 20261010; end
if ~isfield(options, 'plot'), options.plot = false; end
if ~isfield(options, 'output_dir'), options.output_dir = ''; end
if ~isfield(options, 'replay_executable')
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    candidate = fullfile(repo_root, 'build', 'pose3_eskf_csv_replay');
    if ispc, candidate = [candidate, '.exe']; end
    if exist(candidate, 'file') == 2, options.replay_executable = candidate;
    else, options.replay_executable = ''; end
end
assert(isnumeric(options.seed) && isscalar(options.seed) && isfinite(options.seed) && ...
    options.seed >= 0 && options.seed == floor(options.seed) && options.seed <= 2^32 - 100, 'Invalid RNG seed.');
assert(isscalar(options.plot) && (islogical(options.plot) || isnumeric(options.plot)) && ...
    any(options.plot == [0, 1]), 'plot must be true or false.');
if isstring(options.output_dir), options.output_dir = char(options.output_dir); end
if isstring(options.replay_executable), options.replay_executable = char(options.replay_executable); end
assert(ischar(options.output_dir) && size(options.output_dir, 1) <= 1, 'output_dir must be a path.');
assert(ischar(options.replay_executable) && size(options.replay_executable, 1) <= 1, 'replay_executable must be a path.');
if ~isempty(options.replay_executable)
    assert(exist(options.replay_executable, 'file') == 2, 'Requested C++ replay executable does not exist.');
end
end

function cases = scenario_definitions()
base = struct('name', '', 'motion', 'stationary', 'duration', 2, ...
    'imu_hz', 200, 'pose_hz', 50, 'noisy', false, 'dropout', [NaN, NaN], ...
    'arrival_delay_s', 0, 'gravity_mps2', 9.8066, 'yaw_rate_radps', 0.35, ...
    'body_acceleration_mps2', 0.2, 'accel_noise_std_per_sample', 0.06, ...
    'gyro_noise_std_per_sample', 0.003, 'position_noise_std_m', 0.008, ...
    'orientation_noise_std_rad', 0.004);
cases = cell(7, 1);
cfg = base; cfg.name = 'stationary'; cases{1} = cfg;
cfg = base; cfg.name = 'constant_acceleration'; cfg.motion = 'acceleration'; cfg.duration = 3; cases{2} = cfg;
cfg = base; cfg.name = 'constant_yaw_turn'; cfg.motion = 'turn'; cfg.duration = 4; cases{3} = cfg;
cfg = base; cfg.name = 'noisy_turn_100hz'; cfg.motion = 'turn'; cfg.duration = 6;
cfg.imu_hz = 100; cfg.pose_hz = 20; cfg.noisy = true; cases{4} = cfg;
cfg = base; cfg.name = 'noisy_turn_200hz'; cfg.motion = 'turn'; cfg.duration = 6;
cfg.pose_hz = 25; cfg.noisy = true; cases{5} = cfg;
cfg.name = 'pose_dropout'; cfg.dropout = [2.0, 3.2]; cases{6} = cfg;
cfg.name = 'delayed_pose'; cfg.dropout = [NaN, NaN]; cfg.arrival_delay_s = 0.04; cases{7} = cfg;
end

function fixture = make_fixture(cfg)
count = round(cfg.duration * cfg.imu_hz) + 1;
time = (0:count - 1)' / cfg.imu_hz;
[position, velocity, quaternion, gyro, accel] = analytic_motion(time, cfg);
truth = [time, position, quaternion(:, [2, 3, 4, 1]), velocity];
if cfg.noisy
    gyro = gyro + cfg.gyro_noise_std_per_sample * randn(size(gyro)) + repmat([0.001, -0.0008, 0.002], count, 1);
    accel = accel + cfg.accel_noise_std_per_sample * randn(size(accel)) + repmat([0.04, -0.03, 0.02], count, 1);
end
stride = cfg.imu_hz / cfg.pose_hz;
assert(abs(stride - round(stride)) < 1e-12, 'Scenario requires an integer IMU/pose rate ratio.');
pose_indices = (1:round(stride):count)';
pose_times = time(pose_indices);
delay_ticks = round(cfg.arrival_delay_s * cfg.imu_hz);
assert(abs(delay_ticks - cfg.arrival_delay_s * cfg.imu_hz) < 1e-12, ...
    'Scenario arrival delay must be an integer number of IMU ticks.');
keep = ~(pose_times >= cfg.dropout(1) & pose_times <= cfg.dropout(2));
keep = keep & (pose_indices - 1 + delay_ticks <= count - 1 | pose_indices == 1);
pose_indices = pose_indices(keep); pose_times = time(pose_indices);
pose_position = position(pose_indices, :); pose_quaternion = quaternion(pose_indices, :);
if cfg.noisy
    % The initial pose is exact; subsequent independent noise is explicit.
    pose_position(2:end, :) = pose_position(2:end, :) + cfg.position_noise_std_m * randn(numel(pose_indices) - 1, 3);
    for j = 2:numel(pose_indices)
        pose_quaternion(j, :) = quat_multiply(pose_quaternion(j, :), quat_exp(cfg.orientation_noise_std_rad * randn(1, 3)));
    end
end
% Compute both clocks from the same integer grid. Adding two binary floating
% times can spuriously order a mathematically tied pose before its IMU event.
receive_times = (pose_indices - 1 + delay_ticks) / cfg.imu_hz; receive_times(1) = 0;
imu_events = [zeros(count, 1), time, time, gyro, accel, zeros(count, 1)];
pose_events = [ones(numel(pose_indices), 1), receive_times, pose_times, pose_position, pose_quaternion(:, [2, 3, 4, 1])];
% Native rows are sampled after IMU propagation, before a tied pose update.
events = sortrows([imu_events; pose_events], [2, 1]);
fixture = struct('time', time, 'gyro', gyro, 'accel', accel, 'truth', truth, ...
    'events', events, 'gravity_mps2', cfg.gravity_mps2);
end

function [position, velocity, q, gyro, accel] = analytic_motion(time, cfg)
n = numel(time); position = zeros(n, 3); velocity = zeros(n, 3);
q = repmat([1, 0, 0, 0], n, 1); gyro = zeros(n, 3);
accel = repmat([0, 0, cfg.gravity_mps2], n, 1);
switch cfg.motion
    case 'stationary'
        return;
    case 'acceleration'
        acceleration = [0.3, -0.15, 0.08];
        position = 0.5 * time.^2 * acceleration; velocity = time * acceleration;
        accel = accel + repmat(acceleration, n, 1);
    case 'turn'
        w = cfg.yaw_rate_radps; a = cfg.body_acceleration_mps2; theta = w * time;
        position(:, 1) = a * (1 - cos(theta)) / w^2;
        position(:, 2) = a * (time / w - sin(theta) / w^2);
        velocity(:, 1) = a * sin(theta) / w; velocity(:, 2) = a * (1 - cos(theta)) / w;
        q = [cos(theta / 2), zeros(n, 2), sin(theta / 2)]; gyro(:, 3) = w; accel(:, 1) = a;
    otherwise
        error('Unknown analytic scenario.');
end
end

function states = integrate_strapdown(fixture)
% Nominal p/v/q only: no fusion, bias estimation, covariance, gate or replay.
time = fixture.time; states = zeros(numel(time), 11); states(:, 1) = time;
p = [0, 0, 0]; v = [0, 0, 0]; q = [1, 0, 0, 0];
states(1, :) = [time(1), p, q([2, 3, 4, 1]), v]; gravity = [0, 0, -fixture.gravity_mps2];
for i = 2:numel(time)
    dt = time(i) - time(i - 1);
    assert(dt > 0 && isfinite(dt), 'IMU timestamps must increase.');
    omega = 0.5 * (fixture.gyro(i - 1, :) + fixture.gyro(i, :));
    force = 0.5 * (fixture.accel(i - 1, :) + fixture.accel(i, :));
    mid_q = quat_multiply(q, quat_exp(omega * dt / 2));
    acceleration = (quat_rotation(mid_q) * force')' + gravity;
    p = p + v * dt + 0.5 * acceleration * dt^2; v = v + acceleration * dt;
    q = quat_multiply(q, quat_exp(omega * dt));
    states(i, :) = [time(i), p, q([2, 3, 4, 1]), v];
end
end

function metrics = trajectory_metrics(states, truth, cfg)
assert(isequal(size(states), size(truth)), 'State and truth shapes differ.');
pe = sqrt(sum((states(:, 2:4) - truth(:, 2:4)).^2, 2));
ve = sqrt(sum((states(:, 9:11) - truth(:, 9:11)).^2, 2));
q = states(:, 5:8); qt = truth(:, 5:8); qnorm = sqrt(sum(q.^2, 2));
q = bsxfun(@rdivide, q, qnorm); qt = bsxfun(@rdivide, qt, sqrt(sum(qt.^2, 2)));
oe = zeros(size(q, 1), 1);
for i = 1:size(q, 1)
    a = qt(i, [4, 1, 2, 3]); b = q(i, [4, 1, 2, 3]);
    d = quat_multiply([a(1), -a(2:4)], b);
    oe(i) = 2 * atan2(norm(d(2:4)), abs(d(1)));
end
metrics = struct('name', cfg.name, 'imu_hz', cfg.imu_hz, 'pose_hz', cfg.pose_hz, ...
    'position_rmse_m', sqrt(mean(pe.^2)), 'velocity_rmse_mps', sqrt(mean(ve.^2)), ...
    'orientation_rmse_rad', sqrt(mean(oe.^2)), 'max_position_error_m', max(pe), ...
    'final_position_error_m', pe(end), 'max_quaternion_norm_error', max(abs(qnorm - 1)), ...
    'dropout_position_rmse_m', NaN, 'recovery_position_rmse_m', NaN);
if all(isfinite(cfg.dropout))
    dropped = states(:, 1) >= cfg.dropout(1) & states(:, 1) <= cfg.dropout(2);
    recovered = states(:, 1) >= cfg.dropout(2) + 0.5;
    assert(any(dropped) && any(recovered), 'Dropout/recovery windows are empty.');
    metrics.dropout_position_rmse_m = sqrt(mean(pe(dropped).^2));
    metrics.recovery_position_rmse_m = sqrt(mean(pe(recovered).^2));
end
end

function q = quat_exp(v)
angle = norm(v);
if angle < 1e-12, q = [1, 0.5 * v]; else, q = [cos(angle / 2), sin(angle / 2) * v / angle]; end
q = q / norm(q);
end

function q = quat_multiply(a, b)
q = [a(1) * b(1) - dot(a(2:4), b(2:4)), a(1) * b(2:4) + b(1) * a(2:4) + cross(a(2:4), b(2:4))];
q = q / norm(q);
if q(1) < 0, q = -q; end
end

function rotation = quat_rotation(q)
w = q(1); x = q(2); y = q(3); z = q(4);
rotation = [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w); ...
    2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w); ...
    2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)];
end

function write_events(path, events)
fid = fopen(path, 'w'); assert(fid >= 0, ['Cannot write ', path]);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# receive-ordered events; specific force m/s^2; quaternion xyzw\n');
for i = 1:size(events, 1)
    if events(i, 1) == 0
        fprintf(fid, 'imu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', events(i, 2:9));
    else
        fprintf(fid, 'vrpn,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', events(i, 2:10));
    end
end
end

function write_matrix(path, data)
fid = fopen(path, 'w'); assert(fid >= 0, ['Cannot write ', path]);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
format = [repmat('%.17g,', 1, size(data, 2) - 1), '%.17g\n']; fprintf(fid, format, data');
end

function write_text(path, content)
fid = fopen(path, 'w'); assert(fid >= 0, ['Cannot write ', path]);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', content);
end

function write_metrics(path, metrics)
fid = fopen(path, 'w'); assert(fid >= 0, ['Cannot write ', path]);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
names = fieldnames(metrics); fprintf(fid, '%s', names{1});
for j = 2:numel(names), fprintf(fid, ',%s', names{j}); end
fprintf(fid, '\n');
for i = 1:numel(metrics)
    fprintf(fid, '%s', metrics(i).(names{1}));
    for j = 2:numel(names), fprintf(fid, ',%.17g', metrics(i).(names{j})); end
    fprintf(fid, '\n');
end
end

function quoted = shell_quote(value)
assert(~any(value == char(10)) && ~any(value == char(13)), 'Newlines are not allowed in command paths.');
if ispc
    assert(~any(value == '"') && ~any(value == '%'), 'Unsupported Windows command path character.');
    quoted = ['"', value, '"'];
else
    quoted = [char(39), strrep(value, char(39), [char(39), char(34), char(39), char(34), char(39)]), char(39)];
end
end

function plot_scenario(item, scenario_dir)
fig = figure('Name', ['Inertial pose: ', item.name], 'Visible', 'off', ...
    'Color', 'w', 'Position', [100, 100, 1100, 760]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
baseline_color = [0.2, 0.45, 0.7]; native_color = [0.8, 0.3, 0.1];
subplot(2, 2, 1);
h = plot(item.truth(:, 2), item.truth(:, 3), 'k-', 'LineWidth', 1.4); hold on;
h(end + 1) = plot(item.baseline(:, 2), item.baseline(:, 3), '--', 'Color', baseline_color, 'LineWidth', 1.2);
labels = {'Analytic truth', 'Strapdown baseline'};
if ~isempty(item.cpp_states)
    h(end + 1) = plot(item.cpp_states(:, 2), item.cpp_states(:, 3), '-', 'Color', native_color, 'LineWidth', 1.2);
    labels{end + 1} = 'Native C++ ESKF';
end
axis equal; grid on; xlabel('x (m)'); ylabel('y (m)'); legend(h, labels, 'Location', 'northwest');
title([strrep(item.name, '_', ' '), ': horizontal trajectory']);
subplot(2, 2, 2);
plot(item.truth(:, 1), item.truth(:, 4), 'k-', 'LineWidth', 1.4); hold on;
plot(item.baseline(:, 1), item.baseline(:, 4), '--', 'Color', baseline_color, 'LineWidth', 1.2);
if ~isempty(item.cpp_states)
    plot(item.cpp_states(:, 1), item.cpp_states(:, 4), '-', 'Color', native_color, 'LineWidth', 1.2);
end
grid on; xlabel('Time (s)'); ylabel('z (m)'); title('Vertical position');
subplot(2, 2, [3, 4]);
pe = sqrt(sum((item.baseline(:, 2:4) - item.truth(:, 2:4)).^2, 2));
upper = max(1e-9, 1.1 * max(pe));
if all(isfinite(item.config.dropout))
    gap = item.config.dropout;
    patch([gap(1), gap(2), gap(2), gap(1)], [0, 0, upper, upper], ...
        [0.94, 0.94, 0.94], 'EdgeColor', 'none'); hold on;
    text(mean(gap), 0.93 * upper, 'Pose dropout', 'HorizontalAlignment', 'center', 'Color', [0.35, 0.35, 0.35]);
end
he = plot(item.truth(:, 1), pe, '--', 'Color', baseline_color, 'LineWidth', 1.2); hold on;
error_labels = {'Strapdown baseline'};
if ~isempty(item.cpp_states)
    pe = sqrt(sum((item.cpp_states(:, 2:4) - item.truth(2:end, 2:4)).^2, 2));
    he(end + 1) = plot(item.cpp_states(:, 1), pe, '-', 'Color', native_color, 'LineWidth', 1.2);
    error_labels{end + 1} = 'Native C++ ESKF';
    upper = max(upper, 1.1 * max(pe));
end
xlabel('Time (s)'); ylabel('Position error (m)'); grid on; ylim([0, upper]);
legend(he, error_labels, 'Location', 'northwest');
print(fig, fullfile(scenario_dir, 'trajectory.png'), '-dpng', '-r130');
end
