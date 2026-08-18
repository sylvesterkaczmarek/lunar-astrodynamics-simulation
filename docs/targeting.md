# Orbit targeting, sensitivity and station-keeping

This module extends the frozen-orbit search from candidate discovery into local preliminary mission-design analysis. It provides numerical state and orbital-parameter sensitivities, a local differential corrector, and a simple impulsive orbit-maintenance simulation.

The implementation is deliberately transparent. A converged corrector is identified as converged. A stalled, rank-deficient, numerically unstable, over-limit or impacted trial is reported as a failure rather than returned as a plausible-looking solution.

## Scope

The tools are intended for:

- local sensitivity around a propagated reference orbit;
- refinement of a promising candidate returned by the stability search;
- preliminary targeting of osculating apsides and other explicit trajectory constraints;
- rough impulsive station-keeping frequency and delta-v studies;
- comparison of candidate robustness under different force-model fidelities.

They are not intended for operational guidance, navigation or maneuver planning.

## Finite-difference sensitivity

`finite_difference_jacobian(...)` implements a central finite-difference derivative

\[
\frac{\partial y}{\partial x_j}\approx
\frac{y(x+h_j e_j)-y(x-h_j e_j)}{2h_j}.
\]

One perturbation size is not accepted without a numerical consistency check. Every column is evaluated at:

- `0.5 h`;
- `h`;
- `2 h`.

The half/base and base/double derivative differences are compared in norm. The derivative from the tighter adjacent pair is retained and the disagreement is stored in `DerivativeColumnDiagnostic`.

A column is marked stable only when its selected-pair disagreement is below `FiniteDifferenceSettings.max_relative_disagreement`.

This is a numerical consistency test, not proof that the derivative is physically useful. A finite-difference derivative can still be misleading near discontinuities, active inequality boundaries, impacts, event topology changes or strongly chaotic dynamics.

NASA sensitivity literature has long emphasized that finite-difference step selection is a numerical analysis problem rather than an arbitrary user constant. See NASA NTRS 19850025225, *Selecting step sizes in sensitivity analysis by finite differences*:

https://ntrs.nasa.gov/citations/19850025225

## Cartesian state-transition approximation

`finite_difference_state_transition(...)` returns

\[
\Phi(t_f,t_0)\approx\frac{\partial x(t_f)}{\partial x(t_0)}
\]

by propagating positive and negative perturbations of each initial Cartesian state component.

The API exposes separate perturbation sizes for:

- initial position, in metres;
- initial velocity, in metres per second.

The output contains:

- nominal initial and final state;
- the full `6 x 6` numerical state-transition matrix;
- one step-size diagnostic for each initial-state component;
- force-model provenance from `SearchDynamics`.

The CI suite first validates this implementation against analytically tractable zero-acceleration motion, where

\[
\Phi=\begin{bmatrix}
I & \Delta t I\\
0 & I
\end{bmatrix}.
\]

It then checks finite-difference consistency for a lunar two-body propagation.

For applications that require repeated STM evaluation over long arcs, integrated variational equations may be more efficient. The current implementation deliberately uses black-box finite differencing because it works with the repository's composable high-degree gravity, ephemeris and SRP force stack without requiring a separate analytical force Jacobian for every component.

## Propagated orbital-parameter sensitivity

`orbit_parameter_sensitivity(...)` differentiates propagated analysis outputs with respect to selected initial orbital parameters.

Supported initial variables are:

- semimajor axis;
- eccentricity;
- inclination;
- RAAN;
- argument/longitude-of-periapsis parameter used by `OrbitSearchPoint`;
- initial anomaly.

Supported outputs currently include:

- final osculating periselene altitude;
- final osculating aposelene altitude;
- eccentricity-vector linear-drift norm over the propagation interval;
- minimum reference-radius altitude;
- minimum terrain clearance when terrain is supplied.

The same half/base/double finite-difference validation is applied to every parameter column.

## Differential correction

`differential_correct(...)` is the common local nonlinear corrector.

Each design variable has:

- a finite-difference perturbation size;
- a correction scale used to nondimensionalize the Newton step;
- optional lower/upper bounds;
- optional periodic wrapping.

At each iteration the algorithm:

1. evaluates the normalized target residual;
2. builds a central finite-difference Jacobian with step-size diagnostics;
3. rejects a step-size-sensitive Jacobian when configured to do so;
4. scales the Jacobian by the design-variable correction scales;
5. solves a damped linear least-squares Newton step;
6. limits the maximum scaled update;
7. performs a monotonic residual-reducing line search;
8. records rank, condition number, accepted line-search factor and residual reduction.

A result contains `converged=False` and an explicit reason when:

- the target is locally insensitive to all selected variables;
- the finite-difference Jacobian fails its configured step-size test;
- the useful Jacobian subspace is too ill-conditioned;
- every line-search trial fails to reduce the residual;
- the maximum iteration count is reached.

The corrector never changes a failed status into success simply because the last iterate looks reasonable.

## Cartesian terminal targeting

`target_initial_state(...)` is a small wrapper around the generic corrector. It changes selected components of the initial Cartesian state to satisfy selected terminal Cartesian-state components.

This is used in CI to validate differential correction against straight-line dynamics with an exact analytical solution before using the same numerical machinery on lunar orbit constraints.

## Lunar orbit-parameter targeting

`target_orbit_parameters(...)` refines an `OrbitSearchPoint` over a user-selected subset of its parameters.

`OrbitTargetSpecification` supports:

- desired final osculating periselene altitude;
- desired final osculating aposelene altitude;
- an upper bound on eccentricity-vector linear drift over the arc;
- desired sampled body-fixed periapsis latitude;
- desired sampled body-fixed periapsis east longitude;
- a minimum terrain-clearance constraint.

Exact targets use signed normalized residuals. Upper/lower bounds use a one-sided residual that is zero after the inequality is satisfied.

### Periapsis location convention

The geographic periapsis target is based on the trajectory sample with minimum reference-radius altitude. The inertial position at that sample is transformed through the caller-supplied body-fixed rotation.

This is suitable for coarse/local preliminary targeting. It is not an event-refined geodetic periapsis solver. Increase the sampling density for tighter work, or add a dedicated radial-minimum event/refinement if sub-sample location accuracy matters.

The frame is deliberately caller supplied. A GRGM1200A study must use an orientation context compatible with that gravity solution rather than silently substituting another lunar body-fixed frame.

## Terrain constraint

A minimum-terrain-clearance target requires all of:

- a `TerrainShapeModel`;
- the explicit terrain body-fixed rotation provider;
- the terrain frame name.

The targeting evaluation uses the existing terrain-aware propagation and refined minimum-clearance search. If no terrain model is supplied, a terrain-clearance target fails explicitly.

Reference-radius altitude and terrain clearance remain different quantities.

## Impulsive station-keeping model

`simulate_impulsive_stationkeeping(...)` implements a deliberately simple threshold controller.

At fixed control intervals it evaluates the current osculating orbit and may trigger on:

- minimum allowed osculating periselene altitude;
- maximum semimajor-axis deviation from the initial reference orbit;
- maximum eccentricity-vector deviation from the initial reference orbit.

When a trigger fires, the controller constructs a local RTN basis and uses the same differential-correction machinery to solve for an instantaneous velocity impulse that restores configured osculating targets.

The default correction variables are radial and transverse delta-v components. Normal delta-v can also be enabled. The correction targets are:

- target osculating periselene altitude;
- target osculating aposelene altitude;
- optionally the reference eccentricity-vector components.

Each maneuver records:

- elapsed simulation time;
- optional UTC report timestamp;
- trigger reason(s);
- inertial delta-v vector;
- RTN correction components;
- delta-v magnitude;
- pre/post osculating apsides;
- pre/post eccentricity vector;
- complete differential-correction diagnostics.

`StationKeepingResult` reports:

- maneuver count;
- total delta-v;
- maximum individual delta-v;
- all maneuver times;
- achieved duration;
- impact state/time if applicable;
- early-termination reason when applicable.

### Absolute dynamics time

Station-keeping propagation is segmented between control checks. The acceleration callback receives **absolute elapsed simulation time**, not a reset zero at the beginning of every segment. This is required for ephemeris-driven Earth/Sun/SRP forces.

If `start_epoch_utc` is supplied, it is used only to provide human-readable maneuver timestamps. The caller remains responsible for ensuring that this UTC label corresponds to the epoch used by the supplied SPICE/force-model context.

## What this controller does not model

The current station-keeping estimate excludes:

- orbit-determination and navigation error;
- maneuver execution error;
- finite burn duration;
- thrust magnitude and attitude slew limits;
- minimum impulse bit;
- propulsion-system duty cycles;
- ground-station and operations windows;
- occultation or communications restrictions during burns;
- maneuver dispersions and covariance;
- missed or delayed burns;
- autonomous onboard guidance logic;
- propellant mass depletion;
- finite spacecraft geometry.

The controller is reactive at fixed check epochs. It does not perform global optimal control or predict the best maneuver epoch over an extended future window.

Consequently, its delta-v is a **preliminary analysis estimate**, not an operational budget.

## Relation to lunar mission practice

NASA's LRO mission-design and station-keeping studies explicitly considered controlling lunar-orbit altitude/eccentricity behavior while minimizing station-keeping delta-v. These references provide useful context for why periselene, eccentricity and delta-v are appropriate preliminary design quantities:

- NASA NTRS 20080012683, *Stationkeeping for the Lunar Reconnaissance Orbiter*  
  https://ntrs.nasa.gov/citations/20080012683
- NASA NTRS 20070021535, *Mission Design for the Lunar Reconnaissance Orbiter*  
  https://ntrs.nasa.gov/citations/20070021535
- NASA NTRS 20230010952, long-term LRO orbit/station-keeping analysis  
  https://ntrs.nasa.gov/citations/20230010952

The repository does not attempt to reproduce LRO's operational controller.

## Suggested workflow

A practical preliminary workflow is:

1. run the coarse-to-fine frozen-orbit search;
2. take several promising candidates rather than only the top combined score;
3. run Cartesian and orbital-parameter sensitivity analysis;
4. verify derivative step-size stability;
5. locally target useful apsis/clearance/location constraints;
6. reject candidates whose corrector is ill-conditioned or non-convergent;
7. run uncontrolled long-arc propagation;
8. run the simple station-keeping estimate with mission-specific thresholds;
9. compare maneuver count and delta-v across candidates;
10. repeat with higher harmonic degree, third bodies, SRP, terrain and gravity ensembles;
11. perform independent trajectory/operations validation before mission use.

## Example

Run the self-contained example:

```bash
python examples/targeting_stationkeeping.py
```

For a short deterministic smoke run:

```bash
python examples/targeting_stationkeeping.py --quick --output /tmp/targeting_stationkeeping.json
```

The example reports the numerical sensitivity Jacobian and step diagnostics, targeting convergence history, and uncontrolled versus maintained periselene behavior with maneuver count and delta-v.
