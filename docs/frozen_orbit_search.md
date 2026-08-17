# Automated lunar stability and frozen-orbit search

## Purpose

The search API is intended to find numerical low-lunar-orbit candidates whose geometry remains comparatively stable under a stated force model and propagation horizon. It does **not** define a frozen orbit as merely an orbit that avoids impact.

NASA low-lunar-orbit experience motivates the metrics used here:

- Mesarch, *An Observational Approach to Low Lunar Frozen Orbit Design*, AAS-23-238, NASA Goddard: https://ntrs.nasa.gov/citations/20230011564
- Folta, *Lunar Frozen Orbits*, AAS/AIAA Space Flight Mechanics Meeting: https://ntrs.nasa.gov/citations/20080041459
- Wallace, Sweetser and Roncoli, *Low Lunar Orbit Design via Graphical Manipulation of Eccentricity Vector Evolution*: https://ntrs.nasa.gov/citations/20130000242
- Beckman and Lamb, *Stationkeeping for the Lunar Reconnaissance Orbiter*: https://ntrs.nasa.gov/citations/20080012683

The recent LRO frozen-orbit work specifically describes high-fidelity numerical targeting that suppresses eccentricity and argument-of-periapsis growth, and an observational search based on minimizing spread in periselene-altitude evolution. The GRAIL work treats eccentricity-vector evolution directly as an orbit-design variable. The repository therefore reports both geometric altitude/apsis behaviour and nonsingular eccentricity-vector behaviour.

## Stability metrics

Each `StabilityCandidate` stores all individual metrics. The combined `ranking.penalty` is only a configurable ordering aid.

### Periselene altitude spread

`periselene_altitude_peak_to_peak_m` is the peak-to-peak variation of the osculating periselene altitude over the propagated history.

This is the most strongly weighted default ranking term because low-lunar frozen-orbit work has used bounded periselene evolution as a practical observable. A small spread is useful but is not sufficient by itself.

### Aposelene altitude spread

`aposelene_altitude_peak_to_peak_m` measures the corresponding osculating aposelene variation. It helps distinguish a stable low-side envelope from a trajectory whose overall eccentricity/energy is changing substantially.

### Eccentricity-vector evolution

Three quantities are retained:

- `eccentricity_vector_final_change_norm`: direct first-to-last displacement in eccentricity-vector space;
- `eccentricity_vector_linear_drift_norm`: norm of the least-squares secular drift over the propagated time span;
- `eccentricity_vector_detrended_max_radius`: maximum distance of the eccentricity vector from its fitted linear trend.

The drift term is strongly weighted by default. The detrended radius describes bounded oscillation around the secular trend and is intentionally reported separately.

This vector representation remains meaningful for nearly circular orbits where a classical argument of periapsis becomes numerically fragile or undefined.

### Eccentricity magnitude variation

`eccentricity_peak_to_peak` reports the range of osculating eccentricity. A candidate with a slowly translating eccentricity vector can have a different signature from one with a large bounded eccentricity cycle, so vector drift and magnitude variation are kept as distinct metrics.

### Apsidal orientation evolution

`apsidal_direction_max_change_rad` measures the maximum change of the eccentricity-vector direction from the first sample for which the apsis is defined.

The apsis is deliberately treated as undefined below `apsis_eccentricity_threshold` rather than manufacturing an angle from numerical noise. The default search threshold is `1e-6`; users can change it in `StabilitySearchSettings`.

### Orbital-plane evolution

`orbital_plane_max_change_rad` and `orbital_plane_final_change_rad` measure changes in the specific-angular-momentum direction. This is a coordinate-safe measure of plane drift and avoids depending on RAAN near equatorial singularities.

### Altitude and terrain

The search always records reference-radius altitude separately from terrain clearance:

- `minimum_reference_altitude_m` and `maximum_reference_altitude_m` are radial altitude above `SearchDynamics.analysis_reference_radius_m`;
- `minimum_terrain_clearance_m` is present only when an explicit `TerrainShapeModel`, matching body-fixed rotation and frame name are supplied.

When terrain is supplied, propagation uses the terrain-aware impact event and the refined terrain-clearance minimum from `propagate_with_terrain(...)`.

### Lifetime

`impact_free_lifetime_s` and `survived_duration_fraction` are necessary safety/stability screening quantities. They are not the frozen-orbit definition. A trajectory can survive the full run while showing large periselene, eccentricity-vector or plane evolution and rank poorly.

## Ranking

`StabilityRankingPolicy` turns the named metrics into a transparent normalized penalty. Lower is better.

Default normalized terms are:

```text
periselene spread / 10 km
aposelene spread / 10 km
eccentricity-vector secular drift / 1e-3
eccentricity variation / 1e-3
apsidal change / 30 deg, when defined
plane change / 1 deg
lifetime shortfall fraction
optional terrain-clearance shortfall fraction
```

The default weights are:

```text
periselene spread              3
eccentricity-vector drift      3
aposelene spread               1
eccentricity variation         1
apsidal change                 1
plane change                   1
lifetime shortfall             6
terrain-clearance shortfall    4, only when a clearance target is configured
```

The penalty is the weighted mean of applicable normalized terms. `RankingBreakdown.normalised_terms` and `weighted_contributions` are stored for every candidate, so the ranking can always be reconstructed.

These weights are a screening convention, not a universal physical definition of frozen orbit. For a particular mission, set explicit `StabilityConstraints` and adjust the ranking scales/weights to match operational requirements.

## Search parameterization

`StabilitySearchSpace` supports deterministic ranges of:

- semimajor-axis altitude above the analysis reference radius, or absolute semimajor axis;
- eccentricity;
- inclination;
- RAAN;
- argument of periapsis **or** longitude of periapsis;
- initial true anomaly.

The default helper is intended as a broad low, near-polar screen rather than a mission design.

For circular/equatorial combinations, classical angles can describe the same physical Cartesian state multiple times. Before propagation the search converts the state to modified equinoctial elements and removes physically duplicate initial states. This prevents a singular classical parameterization from biasing the search simply by duplicating equivalent candidates.

## Coarse-to-fine search

`run_coarse_to_fine_search(...)` performs two deterministic stages:

1. evaluate and rank the coarse grid;
2. select the best feasible coarse candidates, then construct independent local sub-grids around each seed.

`RefinementSettings` controls the number of seed candidates, number of points per refined axis, spacing fraction and which parameters are refined.

Refinement is local to each selected basin. It does not form one large cross-product of values taken from unrelated coarse winners.

Example:

```python
search = run_coarse_to_fine_search(
    space,
    dynamics,
    settings=settings,
    refinement=RefinementSettings(
        top_candidates=4,
        points_per_axis=3,
        spacing_fraction=0.5,
        refine_axes=(
            "semi_major_axis_m",
            "eccentricity",
            "inclination_rad",
            "raan_rad",
            "periapsis_parameter_rad",
        ),
    ),
)
```

No random sampling is used by the grid or refinement algorithm, so the same search definition, dynamics and numerical settings produce the same candidate ordering subject to the deterministic behaviour of the supplied force/ephemeris providers.

## Force-model fidelity

Search is separated from the force model through `SearchDynamics`.

### Central/J2 screening

```python
dynamics = j2_search_dynamics(include_j2=True)
```

This is useful for fast testing and workflow development. Without an explicit body-fixed rotation the simple J2 symmetry axis is the inertial z-axis, so it is not a substitute for high-degree lunar gravity.

### High-degree gravity

```python
dynamics = harmonic_search_dynamics(
    gravity_model,
    body_fixed_from_inertial,
    max_degree=120,
    max_order=120,
    additional_forces=(earth, sun, srp),
)
```

The selected harmonic degree/order, gravity model, frame and extra-force provenance are included in the search result.

High-degree GRAIL searches must use a lunar body-fixed transformation compatible with the gravity solution. A frame with a similar name is not automatically compatible.

Earth/Sun/SRP components from `forces.py` can be supplied through `additional_forces`. SRP remains optional and should only be enabled with defensible spacecraft mass/area/optical parameters.

## Parallel execution

Set `StabilitySearchSettings(workers=N)` to evaluate candidate states concurrently with a thread pool.

Parallel execution is permitted only when all supplied `SearchDynamics` objects declare `parallel_safe=True`. Pure central/J2 and pure NumPy harmonic dynamics are marked safe. Dynamics containing caller-supplied extra force providers default to unsafe because SPICE/global-kernel contexts and other external providers should not silently be assumed thread-safe.

For expensive full-degree searches, process-level batch scheduling outside this library may provide better CPU scaling than Python threads.

## Gravity uncertainty

Convert gravity realizations to full search dynamics with:

```python
ensemble = harmonic_ensemble_dynamics(
    gravity_realizations,
    body_fixed_from_inertial,
    max_degree=120,
)

result = run_stability_search(
    space,
    nominal_dynamics,
    settings=settings,
    uncertainty_dynamics=ensemble,
)
```

Every realization's full `StabilityMetrics` is retained in `candidate.uncertainty_metrics`.

`UncertaintyStabilitySummary` reports:

- impact fraction;
- minimum lifetime across realizations;
- worst minimum terrain clearance, if terrain is present;
- configured adverse percentile of periselene/aposelene spread;
- adverse percentile of eccentricity variation and eccentricity-vector drift;
- adverse percentile of apsidal and plane evolution.

By default the ranking uses the adverse uncertainty metrics when available, and hard constraints are also applied to uncertainty. This behaviour is explicit in `StabilityRankingPolicy` and `StabilityConstraints`.

A gravity clone ensemble describes gravity-field uncertainty represented by those supplied realizations. It is not a complete mission covariance model.

## Terrain constraints

Terrain can be passed directly to the search:

```python
result = run_stability_search(
    space,
    dynamics,
    settings=StabilitySearchSettings(
        constraints=StabilityConstraints(
            minimum_terrain_clearance_m=10_000.0,
        )
    ),
    terrain=lola,
    terrain_body_fixed_from_inertial=terrain_rotation,
    terrain_frame=lola.frame,
)
```

The existing terrain module enforces the declared frame match. Terrain clearance and reference-radius altitude are never treated as interchangeable.

## Structured output

`StabilitySearchResult` can be serialized directly:

```python
result.write_json("results/search.json")
result.write_csv("results/search.csv")
```

JSON retains nested ranking, uncertainty and provenance details. CSV flattens the primary candidate parameters, metrics, ranking terms and uncertainty summary for analysis in external tools.

`CoarseToFineSearchResult.write_json(...)` stores both stages and the selected coarse seed ids.

## Stability maps

`make_stability_map(...)` creates a 2-D numerical slice suitable for plotting:

```python
map_data = make_stability_map(
    result,
    "semi_major_axis_altitude_m",
    "inclination_deg",
    metric="periselene_altitude_peak_to_peak_m",
)
```

If other parameters remain free in a cell, the best-ranked candidate for that x/y cell is selected and its candidate id is stored. This reduction is therefore auditable rather than hidden.

## Example

Run the self-contained coarse-to-fine workflow:

```bash
python examples/frozen_orbit_search.py
```

Routine CI uses:

```bash
python examples/frozen_orbit_search.py --quick \
  --output /tmp/frozen_search.json \
  --csv /tmp/frozen_search.csv \
  --map-csv /tmp/frozen_map.csv
```

The self-contained mode uses low-degree J2 and is a search-workflow demonstration.

For high-degree work, provide an external SHADR gravity model and caller-controlled SPICE frame context:

```bash
python -m pip install -e .[spice]
python examples/frozen_orbit_search.py \
  --gravity-model data/gggrx_1200a_sha.tab \
  --degree 120 \
  --kernel data/spice/naif0012.tls \
  --kernel <compatible-lunar-orientation-kernel> \
  --body-fixed-frame <compatible-lunar-PA-frame> \
  --epoch-utc 2026-08-17T00:00:00
```

The program intentionally requires the body-fixed frame to be named by the caller. It does not guess that a loaded lunar frame is compatible with GRGM1200A.

## Interpretation and limitations

A numerically favourable candidate means only that it performed well under the stated:

- gravity realization and harmonic truncation;
- lunar orientation model;
- Earth/Sun/SRP or other enabled force components;
- terrain resolution;
- epoch;
- initial state;
- integration settings;
- propagation duration;
- search grid and refinement resolution;
- stability metrics, constraints and ranking policy.

A short screening propagation can miss long-period instability. A coarse parameter grid can miss a narrow stable basin. A gravity-only ensemble omits state/OD, maneuver, attitude, SRP-parameter, ephemeris, terrain and other uncertainty sources.

The output is therefore a set of numerical candidate orbits for further study, not a flight-certified orbit and not a guarantee of stationkeeping-free operation. Promising candidates should be re-propagated for substantially longer horizons at higher force-model fidelity, through appropriate uncertainty ensembles, with mission-specific navigation and maneuver constraints before operational use.
