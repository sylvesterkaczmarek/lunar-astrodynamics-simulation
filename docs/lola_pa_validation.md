# LOLA MOON_PA live validation

This document records validation of the production terrain loader against the actual NASA Goddard `LDEM64_PA_gridline_202405.grd` product rather than a synthetic netCDF fixture.

## Product

NASA Goddard PGDA describes the 2024 LOLA MOON_PA product as a global 64 pixels/degree lunar shape model in the `MOON_PA_DE421` principal-axes frame, referenced to a 1737.4 km sphere. The gridline version includes the 0/360 longitude boundary and the poles.

Validation on 17 August 2026 downloaded the actual gridline netCDF/GMT file. The observed file and native grid properties were:

```text
file size:              736545335 bytes
longitude variable:     lon
latitude variable:      lat
elevation variable:     z
elevation dimensions:   lat, lon
native shape:            11521 x 23041
native spacing:          0.015625 deg = 1/64 deg
longitude bounds:        0 to 360 deg
latitude bounds:         -90 to +90 deg
elevation dtype:         float32
z units attribute:       absent
terrain frame:           MOON_PA_DE421
reference radius:        1737400 m
```

## Unit finding

The live archive check exposed an important issue that the synthetic fixture had not detected. The real GMT/netCDF `z` samples are numerically kilometre-valued. For example, the native product returned approximately `-3.8173318` at 0 deg latitude, 90 deg east. Treating that value as metres would understate lunar relief by a factor of 1000.

`load_lola_moon_pa_grd(...)` therefore performs the product-specific kilometre-to-metre conversion. `RegularLatLonTerrain` and all public clearance APIs continue to use SI metres. The standard PDS IMG loader is unaffected because its PDS label scaling already reconstructs elevations in metres.

A regression test now exercises this conversion explicitly.

## Reference elevations

After the conversion fix, the production loader returned the following elevations from a direct 1 degree subsample of the actual 64 ppd grid:

| Latitude deg | Longitude deg east | Elevation m |
|---:|---:|---:|
| 0 | 0 | -716.4626121521 |
| 0 | 90 | -3817.3317909241 |
| 0 | 180 | 2744.9388504028 |
| 0 | 270 | 258.5588097572 |
| 45 | 45 | -543.4415936470 |
| -45 | 315 | -376.7144680023 |
| 89 | 0 | -1468.7265157700 |
| -89 | 180 | -772.3435163498 |

These values are encoded in `scripts/validate_lola_pa_reference.py`. A future live validation fails if the dated NASA product no longer matches the recorded file size, grid structure, or reference elevations.

The full observed result is stored in `results/lola_pa_validation.json`.

## Real terrain-clearance probes

The same live run placed an instantaneous spacecraft position 10 km radially above the 1737.4 km reference sphere at each reference latitude/longitude and evaluated clearance with the production terrain model.

For example:

| Location | Reference-sphere altitude | Terrain elevation | Terrain clearance |
|---|---:|---:|---:|
| 0 N, 90 E | 10.000 km | -3.817 km | 13.817 km |
| 0 N, 180 E | 10.000 km | +2.745 km | 7.255 km |

This calculation uses the actual MOON_PA grid and demonstrates the distinction between altitude above the reference sphere and clearance above the local lunar surface.

## Repeating the validation

Download the dated product and run:

```bash
python -m pip install -e .[terrain]
python scripts/download_lola_pa_shape.py --registration gridline
python scripts/validate_lola_pa_reference.py data/LDEM64_PA_gridline_202405.grd
```

The 703 MB-class source product remains external and is deliberately excluded from routine CI. Normal CI retains the synthetic unit-conversion regression and the smaller frozen reference tests; this live utility is the archive-level validation path.
