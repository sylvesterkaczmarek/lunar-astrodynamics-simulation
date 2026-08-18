# Terrain-aware lunar clearance

The library supports terrain-aware lunar clearance and impact analysis using external LOLA-derived global shape grids. The terrain model is deliberately separate from the gravity model because lunar body-fixed frame realizations are part of the numerical model and must not be silently mixed.

## Recommended global shape product

For gravity/topography studies, the recommended source is the NASA Goddard Planetary Geodesy Data Archive 2024 **LOLA MOON_PA gridded dataset** by Gregory A. Neumann:

- dataset page: https://pgda.gsfc.nasa.gov/products/95
- DOI: https://doi.org/10.60903/LOLA_PA
- 64 pixels/degree gridline GMT/netCDF grid: https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_gridline_202405.grd
- 64 pixels/degree pixel-registered GMT/netCDF grid: https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_pixel_202405.grd

NASA describes these products as global lunar surface elevations relative to a `1737.4 km` reference radius in the `MOON_PA DE421` principal-axes body-fixed frame. The 64 pixels/degree product has a nominal equatorial spacing of about `0.5 km`.

The Goddard page explains why the PA frame matters. The cartographic `MOON_ME` and geophysical `MOON_PA` frames differ by a constant rotation corresponding to about `875 m` of great-circle displacement at the lunar surface. At low orbital altitude, that is much too large to ignore.

## Gravity and terrain frames

GRGM1200A is a DE430-era lunar gravity solution, whereas the recommended Goddard terrain grid is explicitly `MOON_PA_DE421`. These frame realizations are related, but they are not interchangeable labels.

The terrain APIs therefore require a rotation provider whose declared frame exactly matches `terrain.frame`:

```python
terrain_rotation = spice_rotation_provider(
    "J2000",
    "MOON_PA_DE421",
    et_offset_s=et0,
)

result = propagate_with_terrain(
    initial_state,
    duration_s,
    acceleration,
    terrain,
    terrain_rotation,
    terrain_frame="MOON_PA_DE421",
)
```

If high-degree GRGM1200A gravity is evaluated in a DE430 principal-axes frame, use a separate gravity rotation provider for that frame. Do not reuse the terrain rotation merely because both frames are principal-axes frames.

NAIF lunar frame kernels use DE-dependent names such as `MOON_PA_DE421`. The exact available names depend on the kernels loaded by the caller. A reproducible science run should record the kernels, frame names, epoch and gravity/terrain data product versions.

## Terrain abstraction

`TerrainShapeModel` is the minimal surface interface. `RegularLatLonTerrain` is the supplied global-grid implementation.

A terrain grid carries:

- strictly increasing latitude coordinates;
- east-positive longitude coordinates;
- surface elevation above a stated reference sphere;
- the reference radius;
- a body-fixed frame identity;
- grid registration, either `gridline` or `pixel`;
- optional source provenance.

The surface radius is

```text
r_surface(phi, lambda) = R_reference + h(phi, lambda)
```

and radial terrain clearance is

```text
clearance = |r_spacecraft| - r_surface(phi, lambda)
```

where the spacecraft position is first transformed into the terrain model's own body-fixed frame.

## Grid registration and interpolation

For an interior point, `RegularLatLonTerrain` uses bilinear interpolation in latitude and east-positive longitude.

Longitude is periodic. Queries at negative longitude, values greater than 360 degrees, and the `+/-180 degree` antimeridian are wrapped consistently. Gridline products include the `0/360 degree` boundary; pixel products interpolate across the last-to-first column seam.

At an exact pole, longitude is mathematically undefined. The implementation therefore uses a longitude-independent polar value. For a gridline model this is the mean of the polar boundary row. For a pixel-registered model, values in the half-cell polar cap are linearly continued from the nearest latitude ring to the ring mean at the exact pole.

These rules prevent an arbitrary longitude choice from creating a discontinuous polar surface.

### Interpolation limits

Bilinear interpolation describes the supplied gridded shape model. It cannot recover sub-grid craters, boulders or relief removed by the source model's own gridding/interpolation. NASA notes that the global MOON_PA arrays themselves combine averaging and interpolation methods that vary with latitude.

At the native 64 pixels/degree resolution, the nominal longitudinal spacing is about `0.47 km` at the lunar equator and decreases with latitude. A prepared grid produced with `--stride 8` has about 8 pixels/degree and is intended only for preliminary analysis. Terrain-clearance decisions near the surface should use a resolution appropriate to the clearance margin.

No interpolation result should be interpreted as more accurate than the source grid and its effective spatial resolution.

## Loading the Goddard GMT grid

The original files are large and are intentionally not committed to Git.

Download the 64 pixels/degree gridline product:

```bash
python scripts/download_lola_pa_shape.py --registration gridline
```

The full grid can be loaded directly with the optional netCDF dependency:

```bash
python -m pip install -e .[terrain]
```

```python
terrain = load_lola_moon_pa_grd(
    "data/LDEM64_PA_gridline_202405.grd",
    registration="gridline",
)
```

For repeated preliminary analyses, prepare a smaller metadata-bearing NPZ file:

```bash
python scripts/prepare_lola_pa_grid.py \
  data/LDEM64_PA_gridline_202405.grd \
  --stride 8 \
  --output data/lola_moon_pa_8ppd.npz
```

Then load it with:

```python
terrain = load_terrain_npz("data/lola_moon_pa_8ppd.npz")
```

The NPZ file stores the frame, reference radius, registration and source metadata along with the coordinates and elevations.

## Standard LOLA PDS global grids

`load_lola_pds_global_gdr(...)` supports standard signed 16-bit global cylindrical LOLA PDS3 GDR elevation images such as `LDEM_4`, `LDEM_16` and `LDEM_64`.

The reader obtains dimensions, byte order, scaling, offset, resolution, positive-longitude direction and coordinate-system name from the PDS label rather than hard-coding them. For the standard products, physical elevation is reconstructed from the stored integer DN using the PDS label scaling.

The small official `LDEM_4` product is also used as an external validation source for the reader. It is a `4 pixels/degree` pixel-registered global grid in the `MEAN EARTH/POLAR AXIS OF DE421` coordinate system, with a `1737400 m` reference radius.

On 17 August 2026, `scripts/validate_lola_pds_reference.py` downloaded the official NASA/PDS `LDEM_4.IMG` and `LDEM_4.LBL` and obtained these selected grid values:

| Latitude deg | Longitude deg east | Elevation m |
|---:|---:|---:|
| 0.125 | 0.125 | -796.0 |
| 0.125 | 90.125 | -3814.5 |
| 0.125 | 180.125 | 2432.0 |
| 45.125 | 45.125 | -699.0 |
| -45.125 | 315.125 | -301.0 |
| 89.875 | 0.125 | -119.5 |
| -89.875 | 180.125 | 172.0 |

Those observed values are frozen into an offline regression test so routine CI does not depend on NASA network availability. The external validation script remains available to repeat the check against the archive.

The PDS `MOON_ME` grid is useful for validating file decoding, but it should not be silently combined with PA-frame gravity. Prefer the dedicated Goddard MOON_PA product for gravity/topography studies.

## Terrain-aware impact event

`make_terrain_impact_event(...)` returns a terminal descending zero-clearance event for `solve_ivp`. `propagate_with_terrain(...)` uses that event and retains dense output so the clearance history can be analysed after propagation.

The terrain-aware report contains:

- minimum terrain clearance;
- time of minimum clearance;
- latitude and east-positive longitude of minimum clearance;
- terrain elevation at minimum clearance;
- impact status;
- impact time;
- impact latitude and longitude;
- terrain elevation at impact.

`analyze_terrain_clearance(...)` first scans a configurable dense time grid and then refines an interior minimum with bounded scalar minimisation. For an impact trajectory, the zero-clearance event root is used as the impact geometry.

The surface is treated as a radial shape `r(phi, lambda)`. This is appropriate for a global shape-grid collision boundary, but it is not a facet-level ray intersection model and does not resolve sub-grid terrain.

## Spherical fallback

The previous collision model remains available explicitly as:

```python
make_mean_radius_surface_event(radius_m)
```

`make_surface_event(...)` remains as a backward-compatible alias.

The spherical event is useful for simple propagation and fast screening, but it can be misleading for very low lunar orbits because local LOLA relief can differ from the mean reference sphere by kilometres.

## Comparing the models

The self-contained example uses a synthetic 6 km mountain so it can run in CI without external data:

```bash
python examples/terrain_clearance.py
```

It compares the minimum clearance and impact outcome obtained from the mean-radius sphere with the terrain-aware surface.

A prepared LOLA grid can be supplied with:

```bash
python examples/terrain_clearance.py --terrain-npz data/lola_moon_pa_8ppd.npz
```

That mode deliberately labels its constant-rate orientation as a demonstration. A scientific run should use a SPICE rotation provider for the exact terrain frame and epoch.

## Current limitations

- The global terrain model is a radial height grid, not a triangular surface mesh.
- Bilinear interpolation cannot represent terrain below the supplied grid spacing.
- Downsampling with `stride` trades spatial fidelity for memory and speed and must be reflected in the interpretation of minimum clearance.
- The terrain module does not yet propagate topography uncertainty.
- The recommended 2024 global PA product is DE421, while GRGM1200A is DE430; separate frame transformations are required.
- Caller-supplied SPICE kernels and epoch remain part of the numerical model provenance.
- The code does not model finite spacecraft dimensions, attitude-dependent collision geometry or landing-footprint contact.
- The simple global clearance model should not replace specialised high-resolution landing-site DEMs for terminal descent or surface operations.

These limitations are intentionally explicit so a terrain-aware result is not mistaken for an operational flight-clearance product.
