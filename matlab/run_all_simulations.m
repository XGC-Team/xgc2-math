function results = run_all_simulations(options)
%RUN_ALL_SIMULATIONS Reproducible numerical validation without extra toolboxes.
% options.output_dir, plot, seed, replay_executable, reference_executable.
% Set require_cpp=true in automated parity jobs so a missing native executable
% cannot silently turn a C++ validation run into a reference-only run.
if nargin < 1, options = struct(); end
assert(isstruct(options) && isscalar(options), 'options must be a scalar struct');
if ~isfield(options, 'output_dir'), options.output_dir = tempname; end
if ~isfield(options, 'plot'), options.plot = false; end
if ~isfield(options, 'seed'), options.seed = 20261010; end
if ~isfield(options, 'replay_executable'), options.replay_executable = ''; end
if ~isfield(options, 'reference_executable'), options.reference_executable = ''; end
if ~isfield(options, 'require_cpp'), options.require_cpp = false; end
if options.require_cpp
    assert(exist(options.replay_executable, 'file') == 2, 'C++ ESKF replay executable is required');
    assert(exist(options.reference_executable, 'file') == 2, 'C++ simulation exporter is required');
end
root = fileparts(mfilename('fullpath'));
original_path = path;
restore_path = onCleanup(@() path(original_path)); %#ok<NASGU>
original_rng = rng;
restore_rng = onCleanup(@() rng(original_rng)); %#ok<NASGU>
addpath(fullfile(root, 'inertial_pose_eskf'));
addpath(fullfile(root, 'filter_observer'));
if ~exist(options.output_dir, 'dir'), mkdir(options.output_dir); end
% Do not leave a previous success report when this invocation fails partway.
for report = {'results.mat', 'summary.txt'}
    previous_report = fullfile(options.output_dir, report{1});
    if exist(previous_report, 'file') == 2, delete(previous_report); end
end
rng(options.seed, 'twister');
results.engine_version = version;
results.seed = options.seed;
results.cpp_validation_requested = options.require_cpp;
results.filter_observer = run_filter_observer_simulation(options);
results.motion_control = run_motion_control_simulations(options);
results.inertial_pose_eskf = run_inertial_pose_eskf_simulation(options);
results.pass = results.filter_observer.pass && results.motion_control.pass && results.inertial_pose_eskf.pass;
assert(results.pass, 'A simulation group failed');
save(fullfile(options.output_dir, 'results.mat'), 'results', '-v7');
report_path = fullfile(options.output_dir, 'summary.txt');
fid = fopen(report_path, 'w');
assert(fid >= 0, 'Cannot create simulation summary');
close_report = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'xgc2-math numerical simulations\nEngine: %s\nSeed: %d\n', version, options.seed);
fprintf(fid, 'Filter/observer: PASS\nMotion/control/trajectories: PASS\nInertial pose: PASS\n');
fprintf(fid, 'Native motion parity available: %d\nNative ESKF replay available: %d\n', ...
    results.motion_control.cpp_reference_available, results.inertial_pose_eskf.cpp_replay_available);
fprintf('PASS all simulation groups; outputs: %s\n', options.output_dir);
end
