cmake_minimum_required(VERSION 3.10)
project(xgc2_math VERSION 0.5.12 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(XGC2_MATH_BUILD_TESTING "Build xgc2_math tests" ON)
option(XGC2_MATH_TEST_SIMULATIONS "Run MATLAB/Octave reference and C++ parity simulations" OFF)
set(XGC2_MATH_SIMULATION_ENGINE "Octave" CACHE STRING "Simulation engine: Octave or MATLAB")
set_property(CACHE XGC2_MATH_SIMULATION_ENGINE PROPERTY STRINGS Octave MATLAB)

find_package(Eigen3 REQUIRED)

function(add_xgc2_math_component component)
  set(target "xgc2_math_${component}")
  add_library("${target}" INTERFACE)
  add_library("xgc2_math::${component}" ALIAS "${target}")
  set_target_properties("${target}" PROPERTIES EXPORT_NAME "${component}")
  target_compile_features("${target}" INTERFACE cxx_std_17)
target_include_directories("${target}"
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_link_libraries("${target}" INTERFACE Eigen3::Eigen ${ARGN})
  install(TARGETS "${target}" EXPORT xgc2_mathTargets)
endfunction()

add_xgc2_math_component(algebra)
add_xgc2_math_component(utils)
add_xgc2_math_component(geometry xgc2_math::algebra xgc2_math::utils)
add_xgc2_math_component(filter xgc2_math::algebra xgc2_math::utils)
add_xgc2_math_component(observer xgc2_math::algebra xgc2_math::utils xgc2_math::filter)
add_xgc2_math_component(estimation xgc2_math::algebra xgc2_math::utils xgc2_math::geometry xgc2_math::filter xgc2_math::observer)
add_xgc2_math_component(trajectory xgc2_math::utils xgc2_math::geometry)
add_xgc2_math_component(optimization xgc2_math::utils xgc2_math::trajectory)
add_xgc2_math_component(control xgc2_math::utils)
add_xgc2_math_component(scene xgc2_math::geometry)

add_library(xgc2_math INTERFACE)
add_library(xgc2_math::math ALIAS xgc2_math)
set_target_properties(xgc2_math PROPERTIES EXPORT_NAME math)
target_compile_features(xgc2_math INTERFACE cxx_std_17)
target_include_directories(xgc2_math
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(xgc2_math
  INTERFACE
    xgc2_math::control
    xgc2_math::algebra
    xgc2_math::utils
    xgc2_math::estimation
    xgc2_math::filter
    xgc2_math::geometry
    xgc2_math::observer
    xgc2_math::optimization
    xgc2_math::scene
    xgc2_math::trajectory
)
install(TARGETS xgc2_math EXPORT xgc2_mathTargets)

install(DIRECTORY include/xgc2_math/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xgc2_math
)

install(FILES README.md NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/libxgc2-math-dev
)

install(DIRECTORY matlab/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/libxgc2-math-dev/matlab
)

configure_package_config_file(
  cmake/xgc2_mathConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/xgc2_mathConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgc2_math
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/xgc2_mathConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/xgc2_mathConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/xgc2_mathConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgc2_math
)

install(EXPORT xgc2_mathTargets
  NAMESPACE xgc2_math::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgc2_math
)

if(XGC2_MATH_BUILD_TESTING)
  include(CTest)
  if(BUILD_TESTING)
    add_executable(math_header_test
      test/math_header_test.cpp
    )
    target_link_libraries(math_header_test
      PRIVATE
        xgc2_math::math
    )
    add_test(NAME math_header_test COMMAND math_header_test)

    add_executable(lbfgs_standard_test test/lbfgs_standard_test.cpp)
    target_link_libraries(lbfgs_standard_test PRIVATE xgc2_math::optimization)
    add_test(NAME lbfgs_standard_test COMMAND lbfgs_standard_test)

    add_executable(pose3_eskf_jacobian_test
      test/pose3_eskf_jacobian_test.cpp
    )
    target_link_libraries(pose3_eskf_jacobian_test
      PRIVATE xgc2_math::estimation
    )
    add_test(NAME pose3_eskf_jacobian_test COMMAND pose3_eskf_jacobian_test)

    add_executable(numerical_regression_test
      test/numerical_regression_test.cpp
    )
    target_link_libraries(numerical_regression_test
      PRIVATE xgc2_math::math
    )
    add_test(NAME numerical_regression_test COMMAND numerical_regression_test)

    add_executable(delayed_planar_velocity_test test/delayed_planar_velocity_test.cpp)
    target_link_libraries(delayed_planar_velocity_test PRIVATE xgc2_math::control)
    add_test(NAME delayed_planar_velocity_test COMMAND delayed_planar_velocity_test)

    add_executable(simulation_kinematics_test test/simulation_kinematics_test.cpp)
    target_link_libraries(simulation_kinematics_test PRIVATE xgc2_math::geometry)
    add_test(NAME simulation_kinematics_test COMMAND simulation_kinematics_test)

    add_executable(smc_tracking_test test/smc_tracking_test.cpp)
    target_link_libraries(smc_tracking_test PRIVATE xgc2_math::control)
    add_test(NAME smc_tracking_test COMMAND smc_tracking_test)

    # Keep these checks active in all build types. The test implementations use
    # explicit require/throw checks rather than the NDEBUG-controlled assert.
    add_executable(implicit_wheel_contact_test test/implicit_wheel_contact_test.cpp)
    target_link_libraries(implicit_wheel_contact_test PRIVATE xgc2_math::control)
    add_test(NAME implicit_wheel_contact_test COMMAND implicit_wheel_contact_test)
    set_tests_properties(implicit_wheel_contact_test PROPERTIES LABELS "control;numerical" TIMEOUT 180)

    function(xgc2_math_property_tests target component)
      add_executable(${target} test/${target}.cpp)
      target_link_libraries(${target} PRIVATE xgc2_math::${component})
      foreach(case_name IN LISTS ARGN)
        add_test(NAME ${target}.${case_name} COMMAND ${target} ${case_name})
        set_tests_properties(${target}.${case_name} PROPERTIES LABELS "${component};numerical" TIMEOUT 120)
      endforeach()
    endfunction()

    xgc2_math_property_tests(filter_observer_property_test estimation
      exponential_partition butterworth_response slew_variable_dt
      observer_acceleration_gating angular_wrap rls_parameter_change)
    xgc2_math_property_tests(geometry_property_test geometry
      se2_group se3_group rotation_maps body_velocity_flow body_velocity_quadrature
      acceleration_flow support_maps sphere_distance box_sphere_distance box_box_distance)
    xgc2_math_property_tests(trajectory_property_test optimization
      analytic_2d_derivatives analytic_3d_derivatives analytic_time_domains analytic_periodicity
      line_boundaries entry_continuity torus_entry_yaw septic_waypoint_continuity
      sampled_interpolation flatness_dynamics trajectory_validation minco_s2_oracle
      minco_s3_oracle minco_s2_gradients minco_s3_gradients)

    xgc2_math_property_tests(scene_geometry_test scene
      obstacle_validation polytope_hull surface_sampling ray_analytic ray_against_brute_force
      bvh_build_and_refit bvh_degenerate_scenes sample_index scene_rebuild scene_map_layout)
    xgc2_math_property_tests(scene_lidar_test scene
      raycast_occlusion penetrating_back_faces on_surface_and_in_range field_of_view noise_and_global_map
      noise_stream beams depth_frustum heading_crop vehicle_bodies shared_scene_index penetrating_brute_force
      packed_cloud fleet_threads config_validation scan_requests beam_patterns)
    find_package(Threads REQUIRED)
    target_link_libraries(scene_lidar_test PRIVATE Threads::Threads)
    xgc2_math_property_tests(scene_allocation_test scene
      counter_sees_allocations steady_state_scans steady_state_queries scan_scratch_is_bounded)

    add_executable(pose3_eskf_scenarios_test test/pose3_eskf_scenarios_test.cpp)
    target_link_libraries(pose3_eskf_scenarios_test PRIVATE xgc2_math::estimation)
    add_test(NAME pose3_eskf_scenarios_test COMMAND pose3_eskf_scenarios_test)
    set_tests_properties(pose3_eskf_scenarios_test PROPERTIES LABELS "estimation;numerical" TIMEOUT 180)

    add_executable(pose3_eskf_csv_replay test/pose3_eskf_csv_replay.cpp)
    target_link_libraries(pose3_eskf_csv_replay PRIVATE xgc2_math::estimation)
    # GCC 7 uses experimental::filesystem; GCC 8 still requires its support
    # archive. This only affects the offline test executable, not the library.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
      target_link_libraries(pose3_eskf_csv_replay PRIVATE stdc++fs)
    endif()
    add_executable(simulation_reference_export test/simulation_reference_export.cpp)
    target_link_libraries(simulation_reference_export PRIVATE xgc2_math::control xgc2_math::trajectory)
    foreach(case_name valid malformed_number trailing_number missing_column extra_column
        unknown_event empty no_pose invalid_quaternion invalid_sensor_after_init
        bad_argument bad_boolean bad_iterations same_path alias_same_path)
      add_test(NAME replay_cli.${case_name}
        COMMAND ${CMAKE_COMMAND}
          "-DREPLAY=$<TARGET_FILE:pose3_eskf_csv_replay>"
          "-DCASE=${case_name}"
          "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/replay-cli/${case_name}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/test/replay_cli_test.cmake")
      set_tests_properties(replay_cli.${case_name} PROPERTIES LABELS "estimation;cli" TIMEOUT 45)
    endforeach()

    if(XGC2_MATH_TEST_SIMULATIONS)
      if(XGC2_MATH_SIMULATION_ENGINE STREQUAL "Octave")
        find_program(XGC2_MATH_OCTAVE_EXECUTABLE NAMES octave-cli octave)
        if(NOT XGC2_MATH_OCTAVE_EXECUTABLE)
          message(FATAL_ERROR "XGC2_MATH_TEST_SIMULATIONS requires Octave, or select MATLAB explicitly")
        endif()
        set(simulation_command "${XGC2_MATH_OCTAVE_EXECUTABLE}" --no-gui --quiet --eval)
      elseif(XGC2_MATH_SIMULATION_ENGINE STREQUAL "MATLAB")
        find_program(XGC2_MATH_MATLAB_EXECUTABLE NAMES matlab)
        if(NOT XGC2_MATH_MATLAB_EXECUTABLE)
          message(FATAL_ERROR "XGC2_MATH_SIMULATION_ENGINE=MATLAB requires a matlab executable")
        endif()
        set(simulation_command "${XGC2_MATH_MATLAB_EXECUTABLE}" -batch)
      else()
        message(FATAL_ERROR "XGC2_MATH_SIMULATION_ENGINE must be Octave or MATLAB")
      endif()
      foreach(mode reference_only cpp_parity)
        set(replay_path "")
        set(reference_path "")
        set(require_cpp 0)
        if(mode STREQUAL "cpp_parity")
          set(replay_path "$<TARGET_FILE:pose3_eskf_csv_replay>")
          set(reference_path "$<TARGET_FILE:simulation_reference_export>")
          set(require_cpp 1)
        endif()
        add_test(NAME simulation.${mode}
          COMMAND ${CMAKE_COMMAND} -E env
            "XGC2_MATH_MATLAB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/matlab"
            "XGC2_MATH_TEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/simulation-${mode}"
            "XGC2_MATH_REPLAY=${replay_path}"
            "XGC2_MATH_REFERENCE=${reference_path}"
            "XGC2_MATH_REQUIRE_CPP=${require_cpp}"
            ${simulation_command}
            "addpath(getenv('XGC2_MATH_MATLAB_DIR')); run_ctest_simulations();")
        set_tests_properties(simulation.${mode} PROPERTIES LABELS "simulation;${XGC2_MATH_SIMULATION_ENGINE}" TIMEOUT 300)
      endforeach()
    endif()
  endif()
endif()

if(XGC2_MATH_TEST_SIMULATIONS AND (NOT XGC2_MATH_BUILD_TESTING OR NOT BUILD_TESTING))
  message(FATAL_ERROR "Simulation testing requires XGC2_MATH_BUILD_TESTING=ON and BUILD_TESTING=ON")
endif()
