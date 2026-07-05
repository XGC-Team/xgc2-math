cmake_minimum_required(VERSION 3.16)
project(xgc2_math VERSION 0.5.6 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(XGC2_MATH_BUILD_TESTING "Build xgc2_math tests" ON)

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
add_xgc2_math_component(optimization xgc2_math::utils)
add_xgc2_math_component(trajectory xgc2_math::utils xgc2_math::geometry xgc2_math::optimization)
add_xgc2_math_component(control xgc2_math::utils)

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
    xgc2_math::trajectory
)
install(TARGETS xgc2_math EXPORT xgc2_mathTargets)

install(DIRECTORY include/xgc2_math/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xgc2_math
)

install(FILES README.md
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
    find_package(GTest REQUIRED)

    add_executable(math_header_test
      test/math_header_test.cpp
    )
    target_link_libraries(math_header_test
      PRIVATE
        xgc2_math::math
        GTest::gtest
    )
    add_test(NAME math_header_test COMMAND math_header_test)
  endif()
endif()
