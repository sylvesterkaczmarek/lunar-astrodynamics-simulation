# Configuration-driven mission analysis

The configuration workflow is a thin layer over the package scientific APIs. It does not replace the force, propagation, stability, uncertainty, fidelity or access models. A TOML file records the choices required to reconstruct a run, and the command-line interface calls the same Python functions that can be used directly from an application or notebook.

## Recommended entry points

Python remains the primary interface:

```python
from lunar_astrodynamics.mission import (
    build_mission_context,
    load_mission_config,
    run_access_workflow,
    run_mission,
)

config = load_mission_config("examples/missions/polar_quick.toml")
context = build_mission_context(config)

try:
    mission = run_mission(context)
    access = run_access_workflow(context, mission)
    print(mission.human_summary())
finally:
    context.close()
```

The installed CLI is useful for reproducible runs:

```bash
lunar-mission analyse examples/missions/polar_quick.toml \
  --output-dir results/polar_quick
```

Available subcommands are:

- `propagate`: propagate and write `mission.json`, `trajectory.csv` and `summary.txt`;
- `analyse`: propagate, analyse nonsingular orbital evolution and add standard SVG plots;
- `search`: run the configured stability/frozen-orbit search;
- `uncertainty`: propagate the configured gravity-field ensemble;
- `fidelity`: compare gravity truncations and runtime;
- `access`: calculate ground track, site access, coverage and optional Earth visibility;
- `all`: run the nominal mission and every downstream workflow whose TOML section is present.

The CLI is orchestration only. Propagation still goes through `propagate_with_acceleration` or `propagate_with_terrain`; orbital evolution still uses `orbit_history`; frozen-orbit search still uses `run_stability_search` / `run_coarse_to_fine_search`; and fidelity still uses `compare_harmonic_accelerations` and `compare_harmonic_trajectories`.

## TOML structure

TOML was chosen because Python 3.11 and newer include `tomllib`. Python 3.10 uses the small compatibility dependency `tomli`.

### Mission and output cadence

```toml
[mission]
name = "Polar lunar orbiter"
epoch_utc = "2015-07-13T17:05:30"
duration_s = 21600.0
output_cadence_s = 300.0
```

`epoch_utc` becomes dynamically active when SPICE is enabled. Without SPICE it remains provenance for the self-contained screening context.

### Initial Cartesian state

Positions are metres and velocities metres per second:

```toml
[state]
kind = "cartesian"
cartesian = [1837400.0, 0.0, 0.0, 0.0, 0.0, 1633.0]
```

### Initial classical elements

Angles are degrees. Semimajor axis may be supplied directly or as altitude above the 1737.4 km analysis reference sphere:

```toml
[state]
kind = "elements"

[state.elements]
semi_major_axis_altitude_m = 100000.0
eccentricity = 0.005
inclination_deg = 90.0
raan_deg = 27.0
argument_of_periapsis_deg = 270.0
true_anomaly_deg = 0.0
```

For SHADR missions, the full coefficient field is retained in the mission context even when the nominal propagation uses a lower configured degree/order. This allows the fidelity workflow to compare that propagation choice with a higher-degree reference from the same gravity product.

The underlying classical-element singularity policy is unchanged. Downstream trajectory analysis uses the nonsingular eccentricity-vector and modified-equinoctial representations already implemented by the library.

## Gravity models

### Central or J2 screening

```toml
[gravity]
model = "j2"
```

or

```toml
[gravity]
model = "central"
```

These are useful for self-contained workflow tests and early screening. They are not substitutes for GRAIL harmonics in low-altitude science studies.

### PDS SHADR gravity

```toml
[gravity]
model = "shadr"
path = "../../data/example_mission/gggrx_0900c_sha.tab"
name = "GRGM900C"
degree = 60
order = 60
frame = "MOON_PA_DE421"
```

A SHADR mission requires explicit SPICE frame configuration. The workflow does not infer that similarly named lunar body-fixed frames are interchangeable.

## SPICE

```toml
[spice]
enabled = true
kernels = [
  "../../data/example_mission/naif0012.tls",
  "../../data/example_mission/de421.bsp",
  "../../data/example_mission/moon_pa_de421_1900-2050.bpc",
  "../../data/example_mission/moon_080317.tf",
]
inertial_frame = "J2000"
gravity_frame = "MOON_PA_DE421"
surface_frame = "MOON_ME_DE421"
observer = "MOON"
```

The workflow furnishes these kernels, creates a geometric `SpiceEphemeris`, and records the loaded kernel pool in result provenance. Kernel paths remain user-controlled and are never silently downloaded during a normal analysis run.

## Earth and Sun perturbations

```toml
[perturbations]
earth = true
sun = true
```

These create the existing Moon-centred differential point-mass `ThirdBodyGravity` components using the configured SPICE ephemeris.

## Solar radiation pressure

```toml
[srp]
enabled = true
spacecraft_mass_kg = 250.0
illuminated_area_m2 = 4.0
reflectivity_coefficient = 1.4
lunar_shadow = true
```

This uses the existing cannonball SRP and finite apparent Sun/Moon eclipse model. It does not add attitude-dependent optical properties, articulated surfaces or thermal reradiation.

## Terrain

Prepared NPZ:

```toml
[terrain]
kind = "npz"
path = "terrain.npz"
```

NASA Goddard MOON_PA GMT/netCDF:

```toml
[terrain]
kind = "lola_moon_pa"
path = "LDEM64_PA_gridline_202405.grd"
registration = "gridline"
stride = 8
frame = "MOON_PA_DE421"
```

PDS LOLA global cylindrical image:

```toml
[terrain]
kind = "lola_pds"
image_path = "../../data/example_mission/LDEM_4.IMG"
label_path = "../../data/example_mission/LDEM_4.LBL"
frame = "MOON_ME_DE421"
```

For the standard PDS `LDEM_4` product the loader reads the archived DE421 Mean Earth/Polar Axis frame description. The workflow permits the documented SPICE alias `MOON_ME_DE421` while retaining the underlying terrain source in provenance. No other frame alias is invented.

## Surface sites and access

```toml
[access]
minimum_elevation_deg = 10.0
terrain_aware = false

[[sites]]
name = "Apollo 11 Tranquility Base"
latitude_deg = 0.67409
longitude_deg_east = 23.47298
use_terrain_elevation = true
coordinate_source = "NASA Apollo 11 Lunar Surface Journal"
```

`use_terrain_elevation = true` obtains site height from the configured terrain grid. Individual site access can optionally use sampled terrain-aware line of sight. Coverage remains the documented local-horizon/spherical-limb grid analysis rather than a full terrain ray trace for every cell.

## Coverage

```toml
[coverage]
enabled = true
latitude_min_deg = -30.0
latitude_max_deg = 30.0
latitude_step_deg = 10.0
longitude_min_deg_east = 0.0
longitude_max_deg_east = 60.0
longitude_step_deg = 10.0
minimum_elevation_deg = 10.0
```

Output includes dwell time, access count and revisit statistics per configured grid point. `covered_fraction` remains the fraction of configured grid points, not an equal-area surface fraction.

## Integration settings

```toml
[integration]
method = "DOP853"
rtol = 1e-11
position_atol_m = 1e-4
velocity_atol_m_s = 1e-7
max_step_s = 120.0
```

These values are recorded in provenance. Users should tighten and convergence-test settings when analysis requirements justify it.

## Frozen-orbit search

```toml
[search]
semi_major_axis_altitudes_km = [90.0, 110.0]
eccentricities = [0.005, 0.01]
inclinations_deg = [85.0, 90.0, 95.0]
raan_deg = [27.0]
periapsis_deg = [90.0, 270.0]
initial_anomaly_deg = [0.0]
periapsis_parameterization = "longitude_of_periapsis"
duration_s = 10800.0
sample_count = 65
workers = 1
refine = false
minimum_reference_altitude_km = 20.0
minimum_terrain_clearance_km = 5.0
```

Set `refine = true` to use the existing deterministic coarse-to-fine search and optionally configure `top_candidates`, `points_per_axis`, `spacing_fraction` and `refine_axes`.

If a compatible `[uncertainty]` section is present, `use_uncertainty = true` causes the frozen-orbit search to evaluate supplied gravity realizations through the existing uncertainty-aware stability machinery.

## Gravity uncertainty

Diagonal coefficient sampling:

```toml
[uncertainty]
mode = "diagonal"
seed = 20260818
samples = 8
sigma_scale = 1.0
assume_independent = true
include_mu = false
duration_s = 3600.0
sample_count = 65
percentiles = [5.0, 50.0, 95.0]
```

This is deliberately explicit about the independence approximation. SHADR coefficient uncertainty fields do not contain the full off-diagonal covariance.

For GRGM1200A clone products use:

```toml
[uncertainty]
mode = "clones"
clone_paths = ["clone0001.tab", "clone0002.tab"]
```

The uncertainty CLI uses `propagate_gravity_ensemble`, so its result isolates gravity-field uncertainty. It does not silently add deterministic Earth/Sun/SRP forces to that ensemble product. Uncertainty-aware frozen search can, separately, compose the gravity realizations with the mission's deterministic perturbation forces.

## Gravity fidelity

```toml
[fidelity]
degrees = [2, 10, 20, 40, 60]
reference_degree = 60
reference_order = 60
benchmark_repetitions = 2
trajectory_duration_s = 3600.0
trajectory_sample_count = 49

maximum_relative_acceleration_error = 0.0001
maximum_final_position_difference_m = 500.0
maximum_final_velocity_difference_m_s = 0.5
maximum_periselene_variation_difference_m = 250.0
require_impact_match = true
```

The output stores both pointwise acceleration convergence and propagated-trajectory convergence with measured runtime. Any selected degree applies only to the configured samples, trajectory, duration, frame, force model and tolerance.

## Machine-readable results

A normal `analyse` run writes:

- `mission.json`
- `trajectory.csv`
- `summary.txt`
- `plots/altitude.svg`
- `plots/apsides.svg`
- `plots/eccentricity.svg`

The JSON contains configuration source, epoch, initial state, complete dynamics provenance, harmonic degree/order, force components and their parameters, SPICE kernel records, surface and terrain frames, site coordinates and provenance, integration settings, software versions, a concise orbital summary, and sampled trajectory/analysis quantities.

Other subcommands add their own JSON/CSV products.

## Complete examples

### Self-contained screening workflow

```bash
python -m pip install -e .[dev]
lunar-mission all examples/missions/polar_quick.toml \
  --output-dir results/polar_quick
```

This requires no external data and is suitable for CI and for learning the workflow. It deliberately uses low-degree J2 screening dynamics and synthetic reference sites.

### Public-data science workflow

Install SPICE support and retrieve the public products:

```bash
python -m pip install -e .[spice]
python scripts/download_example_mission_data.py
```

Then run:

```bash
lunar-mission all examples/missions/polar_high_fidelity.toml \
  --output-dir results/polar_high_fidelity
```

This example uses PDS GRGM900C gravity, NAIF DE421 ephemerides and lunar DE421 frame kernels, Earth/Sun differential perturbations, cannonball SRP with lunar eclipse, PDS LOLA `LDEM_4` terrain, NASA-published Apollo 11 and Apollo 17 coordinates, access, regional coverage, frozen-orbit search, gravity uncertainty and degree/order fidelity workflows.

The downloader writes `data/example_mission/manifest.json` with source URLs, sizes and SHA-256 hashes.

## Interpretation

This package is intended for research and preliminary lunar mission analysis. Configuration makes model choices easier to reproduce; it does not make them automatically correct for a particular mission.

A mission study still needs to justify gravity solution and truncation, lunar frame compatibility, ephemeris/kernel versions, terrain resolution, non-gravitational force assumptions, integration tolerances, uncertainty model, search duration and constraints, site coordinate sources, and operational/navigation assumptions.

The independent validation campaign in `docs/independent_validation.md` provides external evidence for the implemented high-fidelity propagation stack. It does not replace mission-specific orbit determination, navigation analysis or operational verification.
