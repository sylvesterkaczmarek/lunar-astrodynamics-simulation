# Lunar ground track, site access and coverage

This module turns a propagated Moon-centred trajectory into quantities used in preliminary lunar observation, communications and surface-support studies.

It is a geometry and sampled-operations analysis layer. It does not replace orbit determination, an RF link budget, terrain-mesh ray tracing or flight mission-planning software.

## Frame convention

`ground_track_history(...)`, site access and coverage require an explicit body-fixed transformation and body-fixed frame name.

For a trajectory position `r_i(t)` in the propagation inertial frame,

```text
r_b(t) = R_b<-i(t) r_i(t)
```

The sub-spacecraft coordinates are planetocentric:

```text
latitude  = atan2(z, sqrt(x^2 + y^2))
longitude = atan2(y, x) mod 360 deg
altitude  = |r_b| - reference_radius
```

Longitudes are east-positive.

Longitude is undefined exactly at either pole. The API does not assign an arbitrary longitude there: `GroundTrackHistory.longitude_deg_east` stores `NaN`, and JSON output writes `null`. Wrapped longitudes are in `[0, 360)` and an additional unwrapped history is supplied for continuous-track analysis across the 0/360 seam.

The caller is responsible for choosing a frame consistent with the coordinate data being analysed. The worked science example uses NAIF's `MOON_ME_DE421` frame, with `moon_080317.tf` and its compatible `moon_pa_de421_1900-2050.bpc`. It does not silently substitute `IAU_MOON` or a DE440 lunar frame.

## Ground track

```python
track = ground_track_history(
    solution.t,
    solution.y[:3].T,
    body_fixed_from_inertial,
    body_fixed_frame="MOON_ME_DE421",
)
```

`GroundTrackHistory` contains:

- elapsed propagation time;
- planetocentric latitude;
- east-positive wrapped longitude;
- unwrapped longitude;
- altitude above the selected reference radius;
- optional clearance above a compatible radial terrain model.

Reference-radius altitude and terrain clearance are deliberately separate quantities.

## Surface sites

A site is represented by `LunarSurfaceSite`:

```python
site = LunarSurfaceSite(
    "example site",
    latitude_deg=10.0,
    longitude_deg_east=25.0,
    elevation_m=1000.0,
    frame="MOON_ME_DE421",
)
```

The site is a radial planetocentric point at

```text
radius = reference_radius + elevation
```

When a compatible terrain model is available, use `LunarSurfaceSite.from_terrain(...)` to obtain the local elevation from the terrain grid instead of entering it manually.

## Site visibility geometry

For the body-fixed site position `r_s` and spacecraft position `r_sc`, define

```text
rho = r_sc - r_s
z_hat = r_s / |r_s|
elevation = asin((rho / |rho|) dot z_hat)
```

A spacecraft is visible only when all selected conditions are satisfied:

1. elevation is at or above `minimum_elevation_deg`;
2. the site-to-spacecraft line segment is not blocked by the spherical lunar reference surface;
3. if terrain-aware LOS is enabled, the sampled line segment remains above the supplied terrain surface.

The spherical-limb test computes the minimum radius reached by the finite line segment. It therefore rejects a far-side spacecraft even if a numerical angle calculation alone were mishandled.

### Terrain-aware line of sight

For individual sites, `terrain_aware=True` samples the straight site-to-spacecraft ray and compares each point with `TerrainShapeModel.surface_radius_m(...)`.

```python
access = analyze_site_access(
    solution.t,
    solution.y[:3].T,
    site,
    body_fixed_from_inertial,
    body_fixed_frame=terrain.frame,
    terrain=terrain,
    terrain_aware=True,
    terrain_los_samples=256,
)
```

This is more useful than a purely local horizon when a mountain or crater rim lies between the site and spacecraft, but it remains a sampled radial-grid test. It is **not** a SPICE DSK or triangular-mesh ray trace, and it cannot recover relief below the terrain grid or LOS sampling resolution.

Gridded coverage can use terrain elevation to place grid sites at the local surface radius, but it intentionally does not perform an expensive intervening-terrain ray trace for every cell and every epoch. Use site-level terrain-aware analysis for surface locations where horizon masking matters.

## Access windows

`analyze_site_access(...)` evaluates the complete trajectory sample history and returns `SiteAccessResult`.

Each `AccessWindow` contains:

- access start and end elapsed time;
- duration;
- sampled maximum elevation and its time;
- sampled minimum slant range;
- sampled closest-approach time.

At ordinary elevation-mask crossings, the start/end time is linearly interpolated between neighbouring trajectory samples. If visibility changes because of a binary obstruction rather than a sign change in the elevation residual, the transition is placed at the sample-interval midpoint.

Therefore access boundaries, closest approach and maximum elevation are only as accurate as the propagation/sample cadence. For operational timing, rerun with a suitably dense output history or add dedicated root-finding around the events of interest.

The result also stores:

- total access time;
- access fraction over the sampled analysis span;
- start-to-start revisit intervals;
- end-to-next-start outage intervals;
- maximum elevation across all windows;
- minimum slant range across all windows.

## Multiple sites

```python
report = analyze_multiple_site_access(
    solution.t,
    solution.y[:3].T,
    (site_a, site_b, site_c),
    body_fixed_from_inertial,
    body_fixed_frame="MOON_ME_DE421",
    minimum_elevation_deg=10.0,
)
```

All sites must use the same explicit frame as the analysis. Results remain separate by site and can be exported as JSON and a flat access-window CSV.

## Coverage grid

`CoverageGrid` accepts arbitrary latitude and longitude arrays or can generate a regular region:

```python
grid = CoverageGrid.regular(
    latitude_min_deg=-90,
    latitude_max_deg=90,
    latitude_step_deg=10,
    longitude_min_deg_east=0,
    longitude_max_deg_east=360,
    longitude_step_deg=10,
    frame="MOON_ME_DE421",
)

coverage = coverage_analysis(
    solution.t,
    solution.y[:3].T,
    grid,
    body_fixed_from_inertial,
    body_fixed_frame="MOON_ME_DE421",
    minimum_elevation_deg=10.0,
)
```

Per grid point, the result reports:

- total dwell/access time;
- number of access windows;
- maximum sampled elevation;
- mean start-to-start revisit interval when at least two accesses exist;
- maximum start-to-start revisit interval.

Aggregate statistics include covered grid-point fraction and mean/median/minimum/maximum dwell time, plus revisit statistics across cells where revisit is defined.

`covered_fraction` is the fraction of configured **grid points**, not an equal-area fraction of lunar surface. A regular latitude/longitude grid oversamples high latitudes. For area-sensitive science products, choose an appropriate grid or apply area weighting downstream rather than interpreting the raw grid-point fraction as surface-area coverage.

## Earth visibility

`analyze_earth_visibility(...)` accepts a Moon-centred Earth position provider. For SPICE studies this should normally be created from the same `SpiceEphemeris` context as the propagation epoch:

```python
ephemeris = spice_ephemeris_from_utc(
    "2026-08-17T00:00:00",
    inertial_frame="J2000",
    observer="MOON",
)

earth = analyze_earth_visibility(
    solution.t,
    solution.y[:3].T,
    ephemeris.position_provider("EARTH"),
)
```

The apparent Earth and Moon are treated as circular disks as seen from the spacecraft. `earth_visibility_fraction(...)` returns the visible fraction of the Earth disk after lunar occultation:

- `1` means the complete Earth disk is visible;
- `0` means complete lunar occultation;
- intermediate values represent partial occultation.

The method uses the IAU nominal terrestrial equatorial radius and a spherical Moon. It does not include lunar limb topography, Earth oblateness in the apparent silhouette, atmosphere/refraction, antenna pointing or link-budget constraints.

When a `SpicePositionProvider` is supplied, its SPICE epoch/frame/kernel provenance is retained in the Earth-visibility report.

## Named-site worked example

The science workflow uses two NASA-published Apollo landing coordinates:

| Site | Latitude | East longitude | Source |
|---|---:|---:|---|
| Apollo 11 Tranquility Base | `0.67409 deg N` | `23.47298 deg E` | NASA Apollo 11 Lunar Surface Journal |
| Apollo 17 Taurus-Littrow | `20.1911 deg N` | `30.7769 deg E` | NASA Science Taurus-Littrow resource |

The example adopts those planetocentric east-positive coordinates in `MOON_ME_DE421` for a DE421-frame study. It does not claim sub-metre cross-frame equivalence. If a different lunar frame is required, transform/re-establish the site coordinates consistently rather than relabelling them.

Download the compatible NAIF kernel set:

```bash
python -m pip install -e .[spice]
python scripts/download_groundtrack_kernels.py
```

Then run a one-day polar-orbiter analysis:

```bash
python examples/groundtrack_access.py \
  --kernel-dir data/spice/de421 \
  --epoch-utc 2026-08-17T00:00:00 \
  --duration-days 1 \
  --altitude-km 100 \
  --minimum-elevation-deg 10
```

The four kernel files are:

- `naif0012.tls`;
- `de421.bsp`;
- `moon_pa_de421_1900-2050.bpc`;
- `moon_080317.tf`.

NAIF documents `moon_080317.tf` as the frame kernel defining the DE421 lunar Mean Earth and Principal Axes frames and shows it used with the compatible DE421 binary PCK.

Routine CI uses:

```bash
python examples/groundtrack_access.py --quick
```

The quick mode is deliberately synthetic and offline. It tests the same ground-track/access/coverage/Earth-occultation workflow without claiming that its synthetic sites or identity frame are physical lunar cartography.

## Structured output

- `GroundTrackHistory.write_csv(...)` writes the ground-track history.
- `MultiSiteAccessReport.write_json(...)` stores site definitions and access metrics.
- `MultiSiteAccessReport.write_windows_csv(...)` writes one row per access window.
- `CoverageResult.write_json(...)` stores the full coverage matrices and aggregate metrics.
- `CoverageResult.write_csv(...)` writes one row per grid point.
- `EarthVisibilityReport.write_json(...)` and `.write_csv(...)` preserve the occultation history.

The worked example writes a concise JSON mission-analysis summary plus the detailed CSV products.

## Interpretation limits

A computed access window means the stated geometric criteria were satisfied under the supplied sampled trajectory and frame model. It is not automatically a communications window, observation opportunity or operational commitment.

Not modelled here include antenna patterns, RF link margin, occultation by spacecraft structure, attitude/slew constraints, Sun-angle constraints, optical illumination requirements, surface multipath, navigation covariance, trajectory uncertainty, time-tag uncertainty, Earth ground-station availability and operational scheduling.

Terrain-aware LOS is only as reliable as the terrain product, its frame compatibility, grid resolution and LOS sampling. Coverage/revisit statistics are sampling-dependent and can miss short access windows if the trajectory cadence is too coarse.
