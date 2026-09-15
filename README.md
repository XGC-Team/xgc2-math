# XGC2 Math

`libxgc2-math-dev` is the umbrella development package for the XGC2
header-only C++ math library. The library owns ROS-independent geometry,
filters, observers, estimators, and control math primitives used by runtime
products.

Install the meta package for normal development:

```bash
sudo apt update
sudo apt install libxgc2-math-dev
```

The package exports a CMake config package:

```cmake
find_package(xgc2_math REQUIRED CONFIG)
xgc2_math_require()

target_link_libraries(my_target PRIVATE xgc2_math::math)
```

Component targets are also exported:

- `xgc2_math::utils`
- `xgc2_math::algebra`
- `xgc2_math::geometry`
- `xgc2_math::filter`
- `xgc2_math::observer`
- `xgc2_math::estimation`
- `xgc2_math::optimization`
- `xgc2_math::trajectory`
- `xgc2_math::control`

Public headers are grouped by domain:

```text
/usr/include/xgc2_math/utils/
/usr/include/xgc2_math/algebra/
/usr/include/xgc2_math/geometry/
/usr/include/xgc2_math/filter/
/usr/include/xgc2_math/observer/
/usr/include/xgc2_math/estimation/
/usr/include/xgc2_math/optimization/
/usr/include/xgc2_math/trajectory/
/usr/include/xgc2_math/control.hpp
/usr/include/xgc2_math/types.hpp
/usr/include/xgc2_math/math.hpp
```

Small deb packages contain domain headers. The meta package
`libxgc2-math-dev` depends on all small packages and owns the CMake entrypoint,
the aggregate headers, and Matlab algorithm validation assets.

## Algorithm Coverage

Estimation algorithms are ROS-independent and live under
`include/xgc2_math/estimation`. Matlab references for nonlinear estimator
simulation live under `matlab/`, and C++ tests cover SE3 operations, filtering,
RLS, observers, and inertial pose estimation edge cases.

`control/wheel_drive.hpp` provides calibrated differential-drive wheel allocation
and an incremental I-P wheel velocity controller. Inputs use metres, radians,
seconds and newton metres; controller state and time steps are explicit. The
existing `filter/slew_rate_limiter.hpp` supplies finite rate limits. These headers
have no ROS or Gazebo dependency. Adapters own configuration, clocks, message and
joint I/O; tire contact and collision integration belong to the physics engine.
