# XGC2 Math

Header-only C++17 math library for geometry, filtering, estimation, trajectory
evaluation, optimization and control.

```bash
sudo apt update
sudo apt install libxgc2-math-dev
```

```cmake
find_package(xgc2_math REQUIRED CONFIG)
xgc2_math_require()
target_link_libraries(my_target PRIVATE xgc2_math::math)
```

```cpp
#include <xgc2_math/math.hpp>
```

Component headers and targets are also available, such as
`<xgc2_math/control.hpp>` and `xgc2_math::control`. The scene component
(`<xgc2_math/scene.hpp>`, `xgc2_math::scene`: scene modeling, ray casting and
LiDAR / depth-camera models) is described in
[include/xgc2_math/scene/README.md](include/xgc2_math/scene/README.md).
