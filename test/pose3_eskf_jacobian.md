# Pose3 ESKF residual Jacobian: native verification, 2026-09-05

## Scope and provenance

This is a verification record, not a claim that the complete estimator or a deployed robot is validated.

- Baseline repository commit: `9d49f048b7030b53f0e35e458bd70e236fc40b10`.
- Baseline `include/xgc2_math/estimation/pose3_inertial_eskf.hpp` Git blob: `4ed4369fc4fb8b1cfc73399cad2374fbe54535fa` (51,250 bytes).
- Fixed header commit: `723963d6bbbdd94cf387a475f1634bcb66437187`.
- Fixed header Git blob: `d5a4bc886e9e57541b101d3349ee7a9380161e75` (52,362 bytes).
- Unchanged `geometry/se3.hpp` Git blob: `302e9cf75a1bbebd524c31906023615335cac1d3`.
- Unchanged `estimation/health.hpp` Git blob: `a5cef1141319a8d2230e1581e7883064b5b2f76b`.
- Regression source: `test/pose3_eskf_jacobian_test.cpp`, Git blob `779c846d5392922264f17864eed48215e4ed0ced`.
- CTest registration commit: `0bf789aab5e1ce138fe273c21849f42acb63c0f0`.

The three production headers were reconstructed from connector reads and their complete bytes checked against the Git blob identities before testing. The fixed header returned by GitHub has the same blob identity as the locally compiled header. The test calls the actual production residual, injection, prediction, Jacobian and public update methods through the already-declared test-access friend; it does not reimplement an estimator and call that a product test.

Execution environment: x86_64, GNU C++ 14.2.0 (Debian 14.2.0-19), Eigen 3.4.0, C++17, `-O1`. Eigen headers came from the preinstalled CasADi distribution; no CasADi solver is used. This is not the product's Focal/ARM64 build matrix.

## Implementation-to-formula derivation

Let body pose be `(R_b, p_b)` and fixed body-to-marker extrinsic be `(R_bm, t_bm)`. The predicted marker is

\[
R_p=R_bR_{bm},\qquad p_p=p_b+R_bt_{bm}.
\]

The production `se3Error(predicted, measured)` returns

\[
r_p=R_p^T(p_m-p_p),\qquad r_R=\operatorname{Log}(R_p^TR_m).
\]

This is **relative translation plus SO(3) Log**, not the full SE(3) logarithm: translation is not premultiplied by the inverse SO(3) left Jacobian.

`injectError` uses world-additive position and body-right attitude perturbations:

\[
p_b'=p_b+\delta p,\qquad R_b'=R_b\operatorname{Exp}(\delta\theta).
\]

Therefore

\[
R_p'=R_p\operatorname{Exp}(R_{bm}^T\delta\theta),
\]

\[
r_R'=\operatorname{Log}\left(\operatorname{Exp}(-R_{bm}^T\delta\theta)\operatorname{Exp}(r_R)\right),
\qquad H_{R,\theta}=-J_l^{-1}(r_R)R_{bm}^T.
\]

For the existing 15D ordering `[p,v,theta,bg,ba]`, the complete residual Jacobian is

\[
H=\begin{bmatrix}
-R_p^T&0&[r_p]_\times R_{bm}^T+R_{bm}^T[t_{bm}]_\times&0&0\\
0&0&-J_l^{-1}(r_R)R_{bm}^T&0&0
\end{bmatrix}.
\]

The old rotational block `-R_bm^T` is exact at zero residual, but not at a general finite residual. The translation blocks were left unchanged and are included in the regression.

The inverse uses

\[
J_l^{-1}(\phi)=I-\tfrac12[\phi]_\times+c(\theta)[\phi]_\times^2,
\quad c(\theta)=\frac{1-(\theta/2)\cot(\theta/2)}{\theta^2}.
\]

For `theta^2 < 1e-6`, the implementation uses `c = 1/12 + theta^2/720 + theta^4/30240` to avoid cancellation. Else it uses the half-angle cotangent form, which avoids a division by `sin(theta)` near pi. The principal Log remains discontinuous at its pi branch cut; no global differentiability or chart-switching policy is claimed.

Primary cross-check: [GTSAM SO3, inverse left Jacobian](https://borglab.github.io/gtsam/so3/) and [Sophus SO3 leftJacobianInverse](https://github.com/strasdat/Sophus/blob/main/sophus/so3.hpp). The derivation above is specialized to this repository's residual and injection rather than inferred from a similarly named library method.

## Native red/green result

The same test source was compiled and executed against the baseline and fixed headers. Baseline exited **1** on the Jacobian assertion; fixed exited **0**. Baseline already passed the public-update smoke checks, demonstrating why those checks alone would not detect this defect.

There are 11 residual magnitudes, 32 seeded nonidentity pose/extrinsic/lever-arm fixtures per magnitude, and three central-difference steps (`1e-5`, `1e-6`, `1e-7`): **1,056 full 6x15 matrix comparisons**, using 31,680 perturbed residual evaluations. Seed: `20260905`. Acceptance threshold: maximum absolute matrix-entry difference `< 1e-6`.

| Residual angle (rad) | Baseline maximum absolute error | Fixed maximum absolute error |
| ---: | ---: | ---: |
| 0 | 4.14064565274e-09 | 4.14064565274e-09 |
| 1e-9 | 3.76013933878e-09 | 3.76013933878e-09 |
| 1e-6 | 4.94866301617e-07 | 4.15997020164e-09 |
| 0.000999 | 0.000482800826586 | 4.49446184436e-09 |
| 0.001001 | 0.000483944333441 | 3.14415554703e-09 |
| 0.05 | 0.0242647209615 | 4.6817614785e-09 |
| 0.3 | 0.146460537339 | 4.40442506078e-09 |
| 0.8 | 0.401230677634 | 3.81759779344e-09 |
| 1.5 | 0.769475185813 | 5.47930525951e-09 |
| 2.6 | 1.43609290489 | 5.89410303897e-09 |
| pi - 1e-4 | 1.83299942266 | 6.11063688538e-09 |

These are derivative errors, not trajectory or flight-accuracy metrics. Near-pi fixtures exercise the mathematical helper/residual chart without changing the product's raw-measurement jump gate.

Additional checks pass for quaternion sign invariance, public pose updates with 1/3/8 iterations and covariance-floor off/on, finite symmetric positive-semidefinite covariance in those fixtures, unit quaternion, raw position-jump rejection, delayed pose replay preserving the IMU endpoint, and non-finite pose rejection. They do not establish statistical covariance consistency or arbitrary out-of-order measurement correctness.

## Reproduce

With the normal product dependencies installed:

```bash
cmake -S . -B build -DXGC2_MATH_BUILD_TESTING=ON
cmake --build build --target pose3_eskf_jacobian_test
ctest --test-dir build -R '^pose3_eskf_jacobian_test$' --output-on-failure
```

The actual local verification used direct compilation (this test itself requires only Eigen):

```bash
c++ -std=c++17 -O1 -Iinclude -isystem /usr/include/eigen3 \
  test/pose3_eskf_jacobian_test.cpp -o /tmp/pose3_eskf_jacobian_test
/tmp/pose3_eskf_jacobian_test
```

Select the installed Eigen include location explicitly on other machines. To reproduce the red case without modifying the working tree, put the baseline header obtained with `git show 9d49f048b7030b53f0e35e458bd70e236fc40b10:include/xgc2_math/estimation/pose3_inertial_eskf.hpp` in a temporary matching include tree and put that include root before `-Iinclude`. Use this same test source; expect exit status 1 and the Jacobian failure, not a compiler error.

## Limits and remaining gates

The source defect and this native numerical regression are closed. Full `math_header_test`, the full CMake build, product CI matrix, ROS consumer integration, Debian release, installed-package checks and physical/bag regression were not executed in this verification environment. CTest wiring is committed, not represented as an observed remote CI success.

An attempted ASan+UBSan build exceeded the execution time limit before producing an executable. It is **unverified**, not a passing sanitizer run and not an observed runtime sanitizer defect.

No runtime deployment, package publication, parameter retuning, rejection-policy change, covariance-reset change or fixed-lag redesign is included. In particular, the existing `updateVelocity` path gates timestamps but updates the current state without pose-style rewind; mixed delayed pose/velocity history and covariance tangent-reset semantics require separate derivation and tests.
