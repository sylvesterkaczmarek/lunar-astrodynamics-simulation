# Gravity convergence and force-model fidelity

## Purpose

High-degree lunar gravity is expensive, and the largest available degree is not automatically the right propagation setting for every analysis. `fidelity.py` provides an explicit error/performance workflow for choosing harmonic degree and order against a selected reference over the positions or trajectory that matter to the study.

The library deliberately does **not** provide a rule such as "100 km altitude requires degree N". Harmonic sensitivity depends on altitude, latitude/longitude history, body-fixed ground track, epoch/frame orientation, propagation duration, the gravity solution itself, additional perturbations and the output accuracy that matters to the mission.

This follows the way lunar-gravity fidelity has historically been assessed. NASA/JPL studies have compared radial accelerations and propagated trajectories between gravity fields, and the PDS spherical-harmonic documentation states that truncation depends on the gravity solution and spacecraft orbits. GRAIL model strength is also geographically nonuniform, so local/trajectory context matters.

Primary context:

- NASA NTRS 19920060682, *Simulations of lunar gravity field determination for Lunar Observer*: https://ntrs.nasa.gov/citations/19920060682
- NASA/PDS SHADR SIS: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/document/shadr.htm
- NASA NTRS 20160005754, *GRGM900C: A Degree 900 Lunar Gravity Model from GRAIL Primary and Extended Mission Data*: https://ntrs.nasa.gov/citations/20160005754

## Harmonic truncation

A model choice is represented by:

```python
HarmonicTruncation(degree=N, order=M)
```

with

```text
0 <= M <= N <= gravity_model.max_degree
```

Degree and order are independent inputs. `60x20` and `60x60` are therefore distinct model choices.

The convenience ladder

```python
default_harmonic_truncations(model)
```

tries the available members of

```text
2x2, 10x10, 20x20, 40x40, 60x60, 120x120, 300x300
```

and adds the model maximum. Users can pass any explicit degree/order combinations instead.

## Reference model

Every convergence calculation has an explicit `reference` truncation. The default is the highest available degree with the highest represented order in the loaded model.

A reference is a numerical comparison baseline, not absolute truth. A `300x300` reference can be useful for a screening study even when a `1200x1200` product exists, provided the study clearly records that choice. Conversely, if the required accuracy is sensitive to terms above degree 300, a 300-degree reference is not adequate.

The reference gravity solution, frame, epoch and rotation provider are therefore part of the reproducibility record.

## Pointwise acceleration convergence

Use:

```python
report = compare_harmonic_accelerations(
    model,
    body_fixed_from_inertial,
    positions,
    times_s=times,
    velocities_m_s=velocities,
    truncations=(
        HarmonicTruncation(20, 20),
        HarmonicTruncation(40, 40),
        HarmonicTruncation(60, 60),
        HarmonicTruncation(120, 120),
    ),
    reference=HarmonicTruncation(300, 300),
)
```

For each supplied position, the code evaluates

```text
delta_a = a_candidate - a_reference
absolute error = |delta_a|
relative error = |delta_a| / |a_reference|
```

The report retains every sample and aggregates maximum and RMS absolute/relative errors.

### RTN components

Radial error is always defined from the position vector.

If a nondegenerate velocity vector is also supplied, the acceleration-error vector is decomposed into the local orbital RTN frame:

```text
R = r / |r|
N = (r x v) / |r x v|
T = N x R
```

and the signed components are

```text
delta_a_R = delta_a dot R
delta_a_T = delta_a dot T
delta_a_N = delta_a dot N
```

The API names T `along_track` and N `cross_track`. If velocity is absent or `r x v` is degenerate, along-track and cross-track values are returned as unavailable rather than manufactured.

## Actual acceleration runtime

The acceleration report times real harmonic evaluations with `time.perf_counter()`.

For each truncation it records:

- median wall-clock time across the configured repetitions for the complete supplied position set;
- median seconds per acceleration evaluation.

This is an empirical machine-specific benchmark. It is useful for comparing truncations on the same hardware/runtime but is not a universal performance constant.

## Trajectory convergence

Use:

```python
trajectory = compare_harmonic_trajectories(
    model,
    body_fixed_from_inertial,
    initial_state,
    duration_s,
    truncations=truncations,
    reference=reference,
    sample_count=257,
    propagation=settings,
)
```

All cases start from the same Cartesian state and use the same integration settings. Optional Earth/Sun/SRP force components can be supplied identically to every harmonic truncation so the comparison isolates gravity truncation while retaining the larger force environment.

Each candidate records:

- measured propagation runtime;
- final position difference when candidate and reference reach the same terminal epoch;
- final velocity difference under the same condition;
- maximum position and velocity separation over common sampled epochs;
- osculating periselene-altitude peak-to-peak variation and its difference from the reference;
- eccentricity peak-to-peak variation and its difference from the reference;
- minimum reference-radius altitude;
- optional minimum terrain clearance and its difference from the reference;
- predicted impact state;
- impact-free achieved lifetime;
- impact agreement with the reference;
- lifetime difference.

If a candidate impacts before the reference terminal epoch, the API does not invent a same-epoch final-state difference. Final position/velocity difference is unavailable and the impact/lifetime metrics carry the model discrepancy instead.

## Terrain-aware fidelity

`compare_harmonic_trajectories(...)` accepts the existing terrain abstraction plus an explicit terrain frame rotation.

When terrain is supplied, each run uses `propagate_with_terrain(...)` and therefore includes refined minimum terrain clearance and terrain-impact detection. The fidelity report keeps minimum terrain clearance distinct from reference-radius altitude.

Terrain resolution and terrain-frame compatibility remain separate model choices. A gravity truncation that meets position-error tolerances can still be insufficient for a low-clearance study if it moves the trajectory enough to change terrain clearance or predicted impact.

## Selecting the lowest tested truncation

`FidelityTolerance` makes the accuracy requirement explicit. Example:

```python
tolerance = FidelityTolerance(
    maximum_final_position_difference_m=100.0,
    maximum_final_velocity_difference_m_s=0.05,
    maximum_periselene_variation_difference_m=50.0,
)

selection = select_lowest_harmonic_truncation(
    trajectory,
    tolerance,
)
```

The selector sorts the **tested** models by degree and then order and returns the lowest tested truncation satisfying every requested metric. It does not interpolate an untested degree and it does not convert the result into an altitude lookup table.

Pointwise acceleration selection can instead use:

```python
FidelityTolerance(
    maximum_absolute_acceleration_error_m_s2=...,
    maximum_relative_acceleration_error=...,
)
```

The selected degree is only defensible for the position set or propagated trajectory represented by that report. A different ground track, epoch, altitude envelope or duration requires revalidation.

## Fidelity-versus-runtime study

`GravityFidelityStudy` can combine the acceleration and trajectory reports:

```python
study = GravityFidelityStudy(acceleration, trajectory)
study.write_json("results/gravity_fidelity.json")
study.write_runtime_csv("results/gravity_fidelity_runtime.csv")
```

The CSV places error and runtime on the same row for each truncation, including:

```text
degree, order
seconds per acceleration evaluation
maximum acceleration error
maximum relative acceleration error
trajectory runtime
final position/velocity difference
periselene/eccentricity variation difference
terrain-clearance difference
impact agreement
lifetime difference
```

This is intended to support an engineering decision such as "60x60 is within the stated mission-analysis tolerances and is materially cheaper than 300x300 for this trajectory", rather than "degree 60 is enough at this altitude".

## Force-model fidelity ladder

Gravity truncation is only one modelling choice. `build_force_model_ladder(...)` can construct the optional ladder:

```text
central gravity
J2 zonal gravity
truncated GRAIL
high-degree GRAIL
high-degree GRAIL + third bodies
high-degree GRAIL + third bodies + SRP
```

Example:

```python
cases = build_force_model_ladder(
    gravity_model,
    body_fixed_from_inertial,
    truncated=HarmonicTruncation(60, 60),
    high_degree=HarmonicTruncation(300, 300),
    third_body_forces=(earth, sun),
    srp=srp,
)

report = compare_force_model_ladder(
    initial_state,
    duration_s,
    cases,
)
```

The last case is the default reference, or another case can be selected explicitly. The comparison uses the same trajectory metrics as harmonic convergence.

`compare_force_model_ladder(...)` also accepts arbitrary named `SearchDynamics` cases, so mission-specific model ladders are possible without changing the fidelity engine.

## Choosing a degree in practice

A defensible preliminary workflow is:

1. Load the gravity solution and establish a scientifically compatible body-fixed frame and epoch workflow.
2. Choose a reference truncation appropriate to the study.
3. Sample representative positions over the actual latitude/longitude/altitude envelope, not only one convenient point.
4. Compare acceleration truncations and inspect worst-case as well as RMS error.
5. Propagate the same initial condition with the candidate truncations.
6. Set tolerances in mission-relevant outputs such as position, velocity, periselene stability or terrain clearance.
7. Select the lowest **tested** truncation satisfying those tolerances.
8. Inspect runtime savings.
9. Repeat for longer duration, uncertainty realizations or different ground tracks if those are part of the mission envelope.
10. Preserve the full report with the simulation provenance.

The chosen degree is a property of this evidence and accuracy requirement, not an intrinsic property of the altitude.

## Example

Run the self-contained workflow regression:

```bash
python examples/gravity_fidelity.py --quick
```

For a real external gravity field:

```bash
python -m pip install -e .[spice]
python examples/gravity_fidelity.py \
  --gravity-model data/gggrx_1200a_sha.tab \
  --kernel path/to/leapseconds.tls \
  --kernel path/to/compatible_lunar_orientation_kernel \
  --body-fixed-frame YOUR_COMPATIBLE_GRAVITY_FRAME \
  --reference-degree 300 \
  --duration-days 2
```

The example deliberately requires the caller to identify the body-fixed SPICE frame. It does not assume that `IAU_MOON`, a DE421 PA frame or a DE440 PA frame is interchangeable with the DE430 context associated with GRGM1200A.

## Validation

Synthetic tests include a pure normalized `C20` field. When the field is truncated to `0x0`, the omitted acceleration is compared against the repository's independent closed-form J2 implementation. The error vector is exactly the negative of the J2 perturbing acceleration to numerical precision.

A separate C22/S22 fixture checks that order truncation is independently detected. Additional tests cover:

- standard ladder clipping;
- acceleration tolerance selection;
- trajectory convergence for a central-only model represented in a higher-degree coefficient array;
- terrain-clearance comparison;
- force-model ladder construction;
- arbitrary `SearchDynamics` model ladders;
- real runtime fields being positive.

## Limitations

- The selected high-degree reference is still a numerical model, not truth.
- Acceleration error at a finite sample set does not bound error everywhere between samples.
- Trajectory differences depend on integration settings as well as force-model fidelity.
- Runtime measurements are hardware, Python/NumPy/SciPy and workload dependent.
- The current harmonic evaluator is CPU/NumPy based and direct synthesis scales approximately quadratically with degree.
- No automatic adaptive degree switching occurs during a propagation.
- No analytic bound is claimed between altitude and required degree.
- Terrain-clearance comparison is only as good as the supplied terrain product and frame transformation.
- Force-model ladder comparisons inherit the limitations of each included force component.
- A convergence result is preliminary mission-analysis evidence, not flight certification.
