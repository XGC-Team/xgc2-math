function results = run_pose3_inertial_eskf_simulation(options)
%RUN_POSE3_INERTIAL_ESKF_SIMULATION Compatibility alias for the old README.
if nargin < 1, options = struct(); end
results = run_inertial_pose_eskf_simulation(options);
end
