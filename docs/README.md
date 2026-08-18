# Documentation

This is the starting point for researchers and preliminary lunar mission designers using Lunar Astrodynamics Simulation. The package exposes a configuration-driven mission workflow for common studies while keeping the lower-level mathematical and numerical APIs available for custom research.

## Choose an interface

| Interface | Use it when |
|---|---|
| `lunar_astrodynamics.mission` Python API | You want to define and compose a reproducible lunar mission study from Python. This is the recommended interface for most users. |
| `lunar-mission` CLI | You want repeatable command-line runs from a TOML mission configuration, with JSON/CSV outputs and concise summaries. |
| Lower-level Python APIs | You need direct control of gravity, frames, forces, propagation, orbital analysis, search, targeting, fidelity, uncertainty or access calculations for research code. |

The mission API and CLI call the same scientific functions used by the lower-level interfaces. See [Configuration-driven mission analysis](mission_workflow.md) for the TOML schema and composition rules.

## Find the right document

| I want to... | Read | What it covers |
|---|---|---|
| Run a complete mission analysis | [Mission workflow](mission_workflow.md) | TOML configuration, Python orchestration, CLI commands, outputs and provenance. |
| Configure GRAIL gravity | [Gravity model](model.md) | SHADR conventions, high-degree harmonics, frames, truncation and gravity limitations. |
| Use LOLA terrain | [Terrain](terrain.md) | LOLA products, terrain frames, clearance, impact and interpolation. |
| Add Earth, Sun or SRP perturbations | [Force models](forces.md) | Moon-centred third bodies, SPICE ephemerides, SRP and eclipse geometry. |
| Analyse near-circular or singular orbits safely | [Orbital analysis](orbital_analysis.md) | Eccentricity vectors, modified equinoctial elements and secular/bounded evolution. |
| Search for frozen or stable lunar orbits | [Frozen-orbit search](frozen_orbit_search.md) | Stability metrics, ranking, constraints, coarse-to-fine search and uncertainty. |
| Run sensitivity, targeting or station-keeping analysis | [Targeting](targeting.md) | Finite-difference sensitivity, differential correction and preliminary impulsive maintenance. |
| Choose a defensible gravity degree/order | [Fidelity](fidelity.md) | Acceleration and trajectory convergence, runtime comparison and tolerance-based selection. |
| Compute ground tracks, site access or coverage | [Access and coverage](access.md) | Body-fixed ground tracks, surface-site visibility, revisit, dwell and Earth visibility. |
| Propagate gravity-field uncertainty | [Uncertainty](uncertainty.md) | SHADR sigma fields, independent sampling and covariance-derived gravity clones. |
| Reproduce the external validation campaign | [Independent validation](independent_validation.md) | Independent gravity/frame/terrain/third-body checks and reconstructed LRO comparison. |
| Reproduce a run or check conventions | [Reproducibility](reproducibility.md) and [Scientific conventions](conventions.md) | Required provenance, units, frames, numerical settings and interpretation conventions. |

Additional validation records are available in [LOLA PA validation](lola_pa_validation.md) and [ground-track SPICE validation](groundtrack_spice_validation.md).

## Recommended first workflow

Start with the self-contained configuration before downloading external data:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]

lunar-mission analyse examples/missions/polar_quick.toml \
  --output-dir results/polar_quick
```

Inspect `summary.txt`, `mission.json`, `trajectory.csv` and the SVG plots in the output directory. The quick mission uses low-degree screening dynamics so the workflow can be exercised without external data.

Then run the public-data example:

```bash
python -m pip install -e .[spice]
python scripts/download_example_mission_data.py

lunar-mission all examples/missions/polar_high_fidelity.toml \
  --output-dir results/polar_high_fidelity
```

That configuration uses public GRAIL gravity, SPICE ephemerides/frames, LOLA terrain, Earth/Sun perturbations, SRP, surface sites and downstream analysis workflows. After that, move to the specialised search, fidelity, uncertainty, targeting or access documents above as required by the study.

## Before interpreting results

Read [Scientific conventions](conventions.md) and preserve the model provenance written with each run. In particular, keep gravity and terrain frame realisations explicit, distinguish reference-radius altitude from terrain clearance, and treat screening/default tolerances as analysis choices rather than universal lunar-orbit requirements.
