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

`geometry/kinematics.hpp` provides exact constant-acceleration translation and
constant-body-velocity planar motion. Callers supply elapsed time and actual
control inputs; the functions do not add a tracking controller or collision response.

## LBFGS source

`optimization/lbfgs.hpp` exposes the unmodified official LBFGS-Lite v2.3
through `xgc2_math::optimization::lbfgs`. Its Eigen API, default parameters
and Lewis–Overton line search are the upstream implementation; no legacy
raw-pointer API or alternative solver is provided.

The fixed upstream source, MIT license and SHA-256 record live together in
`include/xgc2_math/third_party/lbfgs_lite/`. They ship with the optimization
Debian package. See `NOTICE` for attribution scope. Upstream code is excluded
from project formatting/static-style rules so its checked-in bytes remain
verifiable; project callers are still compiled and tested.

## Evaluation and planning headers

`xgc2_math/trajectory.hpp` and `xgc2_math::trajectory` provide polynomial,
sampled and analytic evaluation, including circle and torus-knot entry curves.
They do not include or depend on LBFGS/MINCO. Fixed-time planar waypoints use
`SepticWaypointInterpolator2`; it constructs seventh-order boundary segments
and finite-difference interior velocities without optimization or region constraints.

Optimization consumers explicitly include `xgc2_math/optimization/waypoint3.hpp`
or `xgc2_math/optimization/se2_target_trajectory.hpp` and link
`xgc2_math::optimization`. The official LBFGS-Lite and MINCO headers remain
available for research reproduction. ROS interfaces remain in their owning products.
