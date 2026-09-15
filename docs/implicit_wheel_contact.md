# Coupled compliant wheel-contact and bounded motor model

Status: mathematically checked **candidate**, not calibrated Scout ground truth.
This is an active force/torque model, not a slip diagnostic and not a chassis
velocity prescription. The existing allocation and I-P implementation are
unchanged for the legacy backend. The new backend replaces their physical
control/contact realization when explicitly selected.

## State and work conjugacy

At a frozen planar step, use world-frame generalized velocity
`q=(vx,vy,r,Omega_FL,Omega_FR,Omega_RL,Omega_RR)` with diagonal mass
`M=diag(m,m,Iz,J_FL,J_FR,J_RL,J_RR)>0`. For wheel position `(x,y)`, radius `R`
and rolling unit vector `t`, its surface slip is

```
w_i = A_i q = (vx-r*y-R*t_x*Omega_i, vy+r*x-R*t_y*Omega_i).
```

The same `A_i^T` maps contact force to vehicle translation, yaw and wheel reaction
moment. Thus `q^T A_i^T f_i = w_i^T f_i`; applying both contact-at-wheel force AND
an extra `-R*Fx` joint torque would double-count traction. A Gazebo adapter must
use actual joint axis signs and contact lever arms, not mesh/joint names.

## Tread: elastic memory, renewal, combined friction

`e_i` is world-plane mean tread deflection, `K_i=k_i I>0` and
`gamma_i=abs(R*Omega_i)/ell_i>=0`. Positive renewal length represents finite
residence in a contact patch. Explicit `ell=0` disables renewal for invariant
checks; it does not divide by zero. The input normal force `N_i>=0` is frozen
for a step. The admissible set is ONE disk `||f_i||<=mu_i*N_i`, not two independent
friction intervals. Backward Euler with return mapping gives

```
f_i = projection_disk[-k_i*(e_i_old+h*w_i_new)/(1+h*gamma_i)].
e_i_new = -f_i/k_i.
```

Nonzero static force at zero slip is possible through stored deformation; the
model is compliant, not perfectly rigid static friction at zero deflection.
At separation `N=0`, force and stored tread deformation are zero. A changing
normal force shrinks or expands the admissible force disk without introducing
negative normal traction. No slip ratio or slip angle division by forward
speed is used, so reversing and pivoting are defined at zero vehicle speed.

This is a reduced isotropic elastic-plastic brush model, NOT a claim to implement
TMeasy, Pacejka or a fully resolved pneumatic tire. Parameters shown in the header
are initial engineering hypotheses, not uniquely identified material data.

## Motor: PI in the unsaturated regime, different saturation dynamics

Let `a` be elastic motor tracking error, `kp>0`, `ki>0`,
`d=kp+h*ki`, and `Omega_ref` the wheel-speed reference. Solve

```
tau = clip[-ki*a_old - d*(Omega_new-Omega_ref), -tau_max, tau_max]
a_new = (kp*a_old-h*tau)/d.
```

Without saturation, `a_new=a_old+h*(Omega_new-Omega_ref)` and
`tau=-ki*a_new-kp*(Omega_new-Omega_ref)`: a PI speed servo discretized implicitly
WITH the wheel and contact dynamics. It is deliberately not the previous
explicit incremental I-P controller. A reference change has a bounded
proportional kick; there is NO added command filter. The same old numerical
P/I values do not imply the same real-vehicle response.

The anti-windup is a series yielding Kelvin-Voigt motor impedance. During
saturation it updates elastic error from the bounded torque, rather than
integrating unattainable target error indefinitely. At a stalled wheel under a
constant saturated reference, `|a|` approaches `tau_max/ki`, not infinity.
An isolated wheel's implicit unsaturated update has negative-feedback damping;
full stability is tested through the coupled energy identity, not an assumed
continuous single-wheel pole at a 250 Hz sample rate.

## One strictly convex solve

Stack twelve impulses `p=(h*f_1,...,h*f_4,h*tau_1,...,h*tau_4)` and the
work-conjugate columns `D=[A^T,E]`. Optional physical drag `B>=0` is implicit.
Let `W=(M+h*B)^-1` and `q_free=W*(M*q_old+h*F_external)`.

```
q_new = q_free + W*D*p
H = D^T*W*D + diag(c)
contact c_i=(1+h*gamma_i)/(h*h*k_i)
motor c_i=1/(h*d)
contact b_i=A_i*q_free + e_i_old/h
motor b_i=Omega_free-Omega_ref + ki*a_old/d
minimize 0.5*p^T*H*p + b^T*p
subject to ||p_contact_i|| <= h*mu_i*N_i, |p_motor_i| <= h*tau_max.
```

All diagonal compliances are positive. Therefore `H` is positive definite and
the feasible compact convex problem has a unique minimizer. This is an SOCP-type
quadratic problem with disk constraints, not a polyhedral QP approximation.
The implementation uses exact 2D constrained block minimizers (including the
non-diagonal effective inertia) and scalar motor projections. It checks the
projected-gradient KKT residual. A failure to converge throws before the adapter
can apply any force; it is not silently accepted as a physical slip response.

## Discrete passivity: include stored elastic energy

The correct storage is

```
E = 0.5*q^T*M*q + sum(0.5*k_i*||e_i||^2) + sum(0.5*ki*a_i^2).
```

It is NOT generally correct to demand `f dot w <= 0` instantaneously for an
elastic contact: a stretched tread may return stored energy.

For one tire the optimality condition has a normal-cone term `eta` with
`f dot eta >= 0`. Substituting the update gives

```
h*f dot w_new + Delta(0.5*k*||e||^2)
 = -0.5*k*||e_new-e_old||^2 - h*gamma*k*||e_new||^2 - h*f dot eta <= 0.
```

For the motor define `a_dot=(a_new-a_old)/h`. Its optimality condition gives
`Omega_new-Omega_ref=a_dot-eta`, with `tau*eta>=0`, hence

```
h*tau*(Omega_new-Omega_ref) + Delta(0.5*ki*a^2)
 = -0.5*ki*(a_new-a_old)^2 - h*kp*a_dot^2 - h*tau*eta <= 0.
```

Multiplying the momentum update by `q_new` adds the nonnegative backward-Euler
kinetic dissipation. The result, up to solve tolerance, is

```
E_new-E_old <= h*sum(tau_i*Omega_ref_i) + h*F_external dot q_new.
```

At zero reference and zero external work the numerical plant cannot create
energy. Resetting a contact on lift-off discards nonnegative stored energy.
This establishes consistency for the frozen reduced numerical system. It does
NOT prove passivity of a Gazebo/ODE co-simulation using delayed normal feedback,
all 3D constraints, state-dependent geometry and a separate integrator.

## Executed verification

`test/implicit_wheel_contact_test.cpp` was compiled and executed with C++17,
`-Wall -Wextra -Werror -pedantic`, both optimized and ASan/UBSan builds.
12,478 assertions passed, including 1,200 randomized energy/work cases, combined
friction/torque bounds, exact preloaded static equilibrium, lift-off, force/power
identity, rotation/mirror invariance, high stiffness, stalled anti-windup,
invalid-input and nonconvergence rejection.

A synthetic constant -8 N longitudinal load with four 5 rad/s wheel targets and
R=.08 m converged to wheel speed within 1e-5 rad/s of target. At 1/2/4 ms the
forward speed was .398333 m/s in all three runs. The difference from .4 m/s is
modeled traction slip, not wheel tracking error. These are synthetic numerical
results, NOT Gazebo execution or experimental Scout validation.

No new field bag was replayed; no claim of lower real/sim RMS, identified tire
stiffness, real firmware structure, full product stability or terrain fidelity.
