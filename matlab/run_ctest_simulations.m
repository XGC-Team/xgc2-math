function run_ctest_simulations()
%RUN_CTEST_SIMULATIONS Environment-based entry point avoids quoting repo paths
% inside MATLAB code passed via --eval/-batch (including spaces/apostrophes).
options.output_dir = getenv('XGC2_MATH_TEST_OUTPUT_DIR');
options.plot = false;
options.seed = 20261010;
options.replay_executable = getenv('XGC2_MATH_REPLAY');
options.reference_executable = getenv('XGC2_MATH_REFERENCE');
options.require_cpp = strcmp(getenv('XGC2_MATH_REQUIRE_CPP'), '1');
run_all_simulations(options);
end
