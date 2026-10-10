cmake_minimum_required(VERSION 3.16)
project(dfbc_runtime_test_fixture LANGUAGES CXX)

include(GNUInstallDirs)
find_package(xgc2_math REQUIRED CONFIG)
find_package(XgcRuntimeSDK REQUIRED CONFIG)
find_package(XgcRoboticsInterfaces REQUIRED CONFIG)
find_package(RigidStateNativeWire 1.0 REQUIRED CONFIG)
find_package(ReferenceTrajectoryNative REQUIRED CONFIG)

add_library(ctl_dfbc SHARED ctl_dfbc.cpp)
add_executable(ctl_dfbc_reference reference_replay.cpp)
foreach(target ctl_dfbc ctl_dfbc_reference)
  target_link_libraries(${target} PRIVATE
    xgc2_math::control
    XgcRuntime::SDK
    XgcRoboticsInterfaces::Interfaces
    RigidStateNative::Wire
    ReferenceTrajectoryNative::Wire)
endforeach()
set_target_properties(ctl_dfbc PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS ctl_dfbc ctl_dfbc_reference
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
