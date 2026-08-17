# Reproducibility

## Local checks

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
python examples/j2_precession.py --orbits 40
python examples/harmonic_validation.py
python examples/gravity_uncertainty.py --samples 16 --seed 20260817 --duration-days 1
python examples/terrain_clearance.py
```

The terrain example is self-contained and compares a mean-radius collision model with a synthetic gridded surface containing a 6 km mountain.

## External GRGM1200A data

Download the official NASA PDS file with:

```bash
python scripts/download_grgm1200a.py
```

The nominal SHADR table remains external to Git. Covariance-derived clone perturbations can be downloaded selectively with `scripts/download_grgm1200a_clones.py`.

## External LOLA terrain data

The recommended global gravity/topography source is NASA Goddard's 2024 LOLA MOON_PA global shape grid. Download the 64 pixels/degree gridline product with:

```bash
python scripts/download_lola_pa_shape.py --registration gridline
```

For GMT/netCDF support install:

```bash
python -m pip install -e .[terrain]
```

A smaller explicit-resolution working grid can be prepared with:

```bash
python scripts/prepare_lola_pa_grid.py \
  data/LDEM64_PA_gridline_202405.grd \
  --stride 8 \
  --output data/lola_moon_pa_8ppd.npz
```

The prepared NPZ preserves terrain frame, reference radius, registration and source metadata.

## Independent LOLA PDS reference check

The repository contains an external validation utility for the standard NASA/PDS `LDEM_4` global product:

```bash
python scripts/validate_lola_pds_reference.py
```

It downloads the official `LDEM_4.IMG` and `LDEM_4.LBL`, decodes them using `load_lola_pds_global_gdr(...)`, and prints selected reference grid values.

On 17 August 2026 the observed product metadata and values were:

```text
frame=MEAN EARTH/POLAR AXIS OF DE421
reference_radius_m=1737400.0
shape=(720, 1440)
lat=0.125,lon=0.125,elevation_m=-796.0
lat=0.125,lon=90.125,elevation_m=-3814.5
lat=0.125,lon=180.125,elevation_m=2432.0
lat=45.125,lon=45.125,elevation_m=-699.0
lat=-45.125,lon=315.125,elevation_m=-301.0
lat=89.875,lon=0.125,elevation_m=-119.5
lat=-89.875,lon=180.125,elevation_m=172.0
```

These values were copied into `tests/test_lola_reference.py` as an offline indexing/scaling regression. Normal CI therefore does not require NASA network access. The live validation script can be run when archive-level revalidation is desired.

The PDS LDEM_4 grid is in the DE421 Mean Earth/Polar Axis frame. It is used to validate the file reader, not as an implicit substitute for the recommended DE421 Principal Axes terrain grid.

## Frame provenance

A reproducible combined GRAIL/LOLA run should record both frame transformations independently. The recommended Goddard terrain grid is `MOON_PA_DE421`, while GRGM1200A is a DE430 gravity solution. The library requires the terrain rotation's declared frame to equal the terrain model frame and does not infer that one principal-axes realization can be substituted for another.

For science propagation, record:

- SPICE kernel filenames and versions;
- inertial frame;
- gravity body-fixed frame;
- terrain body-fixed frame;
- epoch/ephemeris time offset;
- gravity product and truncation;
- terrain product and prepared resolution;
- integration tolerances and maximum step.

## What CI verifies

Routine CI avoids the large external GRGM1200A and LOLA products. It verifies the implementations with synthetic and frozen-reference fixtures, including:

- two-body and J2 regressions;
- high-degree spherical-harmonic validation through degree 1200;
- pole-safe gravity evaluation;
- SHADR uncertainty retention and reproducible diagonal sampling;
- GRGM1200A clone perturbation semantics and ensemble metrics;
- terrain bilinear interpolation and periodic longitude;
- exact-pole and pixel-polar-cap behavior;
- explicit terrain-frame mismatch rejection;
- terrain-aware impact event root finding and impact geometry;
- GMT/netCDF terrain loading;
- PDS3 IMG/label scaling, byte order and coordinate reconstruction;
- selected reference elevations independently observed from the official PDS LDEM_4 product;
- prepared NPZ metadata round trips;
- end-to-end terrain-clearance smoke propagation.

This keeps source-code verification deterministic while retaining separate scripts for live archive validation.
