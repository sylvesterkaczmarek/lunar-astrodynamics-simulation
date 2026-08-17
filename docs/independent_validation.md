# Independent scientific validation

This document records an external validation campaign for the production lunar astrodynamics implementation. The objective is to separate ordinary unit testing from evidence based on independent software and public NASA/PDS/NAIF products.

The campaign was executed on 17 August 2026. The lightweight numerical record is stored in `results/independent_validation.json`. Large source products are not committed and are downloaded by `scripts/download_independent_validation_data.py`.

## Validation strategy

Two different questions are treated separately.

1. **Implementation correctness.** When the candidate and reference are given the same mathematical force model, frame, epoch and coefficients, an independently evaluated reference should agree to the level implied by floating-point and numerical-integration error.
2. **Physical-model adequacy.** A free propagation can be compared with a reconstructed real spacecraft trajectory. Differences then contain omitted forces, maneuvers and orbit-determination effects as well as numerical error. Such a comparison must not be forced through a conveniently selected pass threshold.

The hard acceptance checks apply only to the first category. The LRO reconstructed-orbit comparison is deliberately reported as a measured external residual.

## Authoritative public sources

The campaign uses only public products from NASA/JPL/PDS/NAIF for mission and lunar data.

### LRO reconstructed trajectory

The PDS LRO Radio Science archive describes its PDS4 SPK products as GEODYN trajectory files containing monthly orbit reconstructions in NAIF SPK format, derived from radio-science tracking and used for LOLA geolocation.

Source information:

- PDS LRO Radio Science: `https://pds-geosciences.wustl.edu/missions/lro/rss.htm`
- PDS LRO mission ephemeris note: `https://pds-geosciences.wustl.edu/missions/lro/default.htm`
- validation SPK: `LRO_ES_36_GRGM900C_L600.BSP`

The validation SPK contains LRO object ID `-85`. Its observed coverage was ET `489658568.18390733` to `490946768.1834973` seconds past J2000. The deterministic validation epoch was ET `490079198.1837638`, UTC `2015-07-13T17:05:30.000`.

### GRGM900C and GRGM1200B

GRGM900C is a degree/order-900 GRAIL gravity solution. The published GRGM900C paper states that the `3.6e-4/l^2` power-law constraint was applied only above degree 600.

Source:

- NASA NTRS 20160005754, *GRGM900C: A Degree 900 Lunar Gravity Model from GRAIL Primary and Extended Mission Data*: `https://ntrs.nasa.gov/citations/20160005754`

A critical naming point is preserved throughout this validation: the `L600` suffix in the LRO archive product name identifies the gravity solution's constraint threshold. It does **not** mean the reconstructed LRO SPK was generated with a 600x600 truncation. Our six-hour candidate propagation uses an explicit 600x600 truncation and therefore is not an identical reproduction of the GEODYN gravity setup.

GRGM1200B is independently checked at the body-fixed acceleration level through degree/order 1200. No DE421/DE430 orientation equivalence is assumed for this check because both implementations receive exactly the same body-fixed Cartesian points.

### Lunar orientation and ephemerides

The validation uses the public NAIF files:

- `naif0012.tls`
- `de421.bsp`
- `moon_pa_de421_1900-2050.bpc`
- `moon_080317.tf`

The trajectory work uses Moon-centred J2000 and the `MOON_PA_DE421` principal-axes frame.

### LOLA terrain

Terrain file decoding and coordinates are checked against the official PDS `LDEM_4.IMG` and `LDEM_4.LBL` products. This validates binary decoding, grid coordinates, reference radius and elevation scaling. It does not claim to validate the measurement uncertainty of LOLA itself.

## Independent implementations

The reference side is intentionally not another wrapper around the production implementation.

### Gravity

SHADR coefficients are parsed by a separate fixed-width parser in the validation script. Gravity is evaluated using SHTOOLS `MakeGravGridPoint`, which accepts real geodesy 4-pi-normalized spherical-harmonic coefficients and independently evaluates the spherical acceleration components.

SHTOOLS documentation:

- `https://shtools.github.io/SHTOOLS/pymakegravgridpoint.html`

The SHTOOLS spherical `(r, theta, phi)` result is independently transformed to Cartesian coordinates before comparison with `gravity_acceleration_body_fixed(...)`.

### Frames

Production `spice_rotation_provider(...)` and `ground_track_history(...)` are compared against direct CSPICE calls through SpiceyPy using `sxform`, `pxform` and `reclat`.

### Third-body perturbations

Earth and Sun differential third-body accelerations are recomputed with 80-decimal-digit `mpmath` arithmetic from DE421 positions obtained directly from SPICE. This is especially relevant for the Sun because the Moon-centred differential acceleration is obtained by subtracting two much larger, nearly equal terms.

### Terrain

The reference implementation memory-maps `LDEM_4.IMG` directly as signed 16-bit data, parses the required scale and grid metadata from the PDS label independently, reconstructs pixel-centre coordinates, and compares them with the production terrain loader.

### Propagation

The same-model trajectory reference uses:

- independently parsed coefficients;
- SHTOOLS gravity;
- direct SpiceyPy frame and body-position calls;
- a separate third-body implementation;
- a separate SciPy `solve_ivp` right-hand side.

Both sides use DOP853. This intentionally isolates force/model implementation rather than integrator-family differences. The reconstructed LRO SPK provides the fully external trajectory-level evidence.

## Tolerance policy

Tolerances were defined as functions of numerical conditioning or explicit integration settings before inspecting the final residuals.

### Harmonic acceleration

The relative envelope is

```text
max(5e-13, 200 * eps * (degree + 1)^(3/2))
```

where `eps` is IEEE-754 double precision machine epsilon. The degree dependence allows for recurrence and summation roundoff at high degree. It remains much tighter than mission-level force-model requirements.

### Frames

- rotation-matrix maximum absolute difference: `5e-13`
- altitude difference: `1e-6 m`
- angular coordinate difference: `1e-10 deg`

These are numerical consistency limits, not lunar geolocation-accuracy claims.

### Third bodies

The tolerance is based on a cancellation condition estimate:

```text
max(1e-13, 100 * eps * condition)
```

where `condition` is the sum of the norms of the two point-mass acceleration terms divided by the norm of their differential result.

### Terrain

The terrain decoder comparison uses `1e-6 m` for elevation and `1e-12 deg` for reconstructed grid coordinates. The stored elevations are quantized by the source product much more coarsely than those numerical decoding limits.

### Same-model propagation

The position and velocity envelopes are derived from the maximum independently sampled acceleration discrepancy over the reference trajectory, multiplied by a fixed 50x bounded-growth factor, plus explicit integration-error floors of `0.02 m` and `2e-5 m/s`.

The final measured residuals were not used to choose those tolerances.

## Gravity validation

Eight body-fixed positions were selected over low lunar altitudes, nearside/farside longitudes, mid-latitudes and near-polar regions. The same points are used for every truncation.

### GRGM900C versus SHTOOLS

| Degree/order | Maximum relative acceleration difference |
|---:|---:|
| 10x10 | `4.7552e-16` |
| 60x60 | `2.1816e-15` |
| 120x120 | `3.3648e-15` |
| 300x300 | `1.1337e-14` |
| 600x600 | `1.0827e-14` |

All values are many orders of magnitude below their a-priori numerical envelopes.

The independently parsed GRGM900C constants were:

- GM: `4.902799967088641e12 m^3/s^2`
- reference radius: `1,738,000 m`

### GRGM1200B versus SHTOOLS

| Degree/order | Maximum relative acceleration difference |
|---:|---:|
| 60x60 | `1.9037e-15` |
| 120x120 | `7.0355e-15` |
| 300x300 | `1.2877e-14` |
| 600x600 | `1.1820e-14` |
| 1200x1200 | `1.1820e-14` |

The degree-1200 a-priori relative envelope was `1.8484e-9`; the observed worst difference was about five orders of magnitude smaller.

The independently parsed GRGM1200B constants were:

- GM: `4.9028001224453e12 m^3/s^2`
- reference radius: `1,738,000 m`

**Result:** both independent gravity checks passed.

## Frame validation

At three epochs within the selected LRO arc:

- maximum difference between production and direct-CSPICE rotation matrices: `0.0`
- maximum altitude difference after Cartesian-to-latitude/longitude conversion: `0.0 m`
- maximum latitude difference: `3.5527e-15 deg`
- maximum longitude difference: `2.8422e-14 deg`

**Result:** frame and surface-coordinate transformations passed the numerical consistency limits.

## Third-body validation

At the LRO validation epoch:

| Body | Absolute acceleration difference | Relative difference | Cancellation condition estimate | A-priori relative tolerance |
|---|---:|---:|---:|---:|
| Earth | `2.048e-19 m/s^2` | `1.358e-14` | `356.9` | `7.925e-12` |
| Sun | `3.418e-19 m/s^2` | `3.152e-12` | `106295` | `2.360e-9` |

The larger solar relative residual is expected from much more severe cancellation in the Moon-centred differential acceleration. The absolute difference remains approximately `3.4e-19 m/s^2`.

**Result:** both third-body checks passed their conditioning-derived tolerances.

## Terrain validation

The independent raw PDS decoder and production `load_lola_pds_global_gdr(...)` agreed exactly for all selected grid samples.

Representative values include:

| Latitude | East longitude | Elevation |
|---:|---:|---:|
| `89.875 deg` | `0.125 deg` | `-119.5 m` |
| `89.875 deg` | `180.125 deg` | `-36.5 m` |
| `44.875 deg` | `45.125 deg` | `-1053.0 m` |
| `0.125 deg` | `0.125 deg` | `-796.0 m` |
| `0.125 deg` | `90.125 deg` | `-3814.5 m` |
| `-89.875 deg` | `180.125 deg` | `172.0 m` |

Maximum observed differences were `0.0 m`, `0.0 deg` latitude and `0.0 deg` longitude.

**Result:** PDS terrain decoding and coordinate reconstruction passed.

## Independent same-model propagation

An LRO state from the reconstructed SPK at the selected epoch was propagated for two hours with identical physical models on two independently implemented force paths:

- `120x120` GRGM900C;
- `MOON_PA_DE421` orientation;
- DE421 Earth and Sun point-mass perturbations;
- Moon-centred J2000 propagation.

Measured production-versus-independent differences were:

| Quantity | Difference |
|---|---:|
| final position | `0.000128635 m` |
| maximum position | `0.000129586 m` |
| RMS position | `8.3639e-5 m` |
| final velocity | `1.1253e-7 m/s` |
| maximum velocity | `1.1253e-7 m/s` |
| maximum sampled acceleration | `1.6801e-14 m/s^2` |

The a-priori envelopes were `0.02002 m` maximum position difference and `2.0006e-5 m/s` maximum velocity difference.

**Result:** the production high-fidelity force/propagation stack agreed with the independent implementation to approximately **0.13 mm maximum position difference over two hours**.

## Reconstructed LRO trajectory comparison

The strongest mission-level external evidence uses `LRO_ES_36_GRGM900C_L600.BSP` from the PDS LRO Radio Science archive.

The comparison starts exactly from the reconstructed LRO state at UTC `2015-07-13T17:05:30.000` and then runs open loop.

The candidate force model contains:

- GRGM900C spherical harmonics;
- DE421 planetary ephemeris;
- `MOON_PA_DE421` orientation;
- point-mass Earth and Sun perturbations.

It does not reproduce the complete GEODYN precision-orbit-determination model. NASA LRO orbit-determination documentation identifies additional operational terms such as lunar solid tides, solar-radiation pressure and Jupiter gravity, and precision reconstruction also incorporates tracking-data estimation and spacecraft operations.

### Six-hour 600x600 candidate

| Quantity | Difference from reconstructed LRO SPK |
|---|---:|
| final position | `12.4260 m` |
| maximum position | `12.5633 m` |
| RMS position | `6.9310 m` |
| final velocity | `0.0104803 m/s` |
| radial RMS | `0.5016 m` |
| along-track RMS | `6.7370 m` |
| cross-track RMS | `1.5491 m` |

A separate tighter numerical propagation of the **same candidate force model** differed from the nominal integration by only `0.00647 m` at the final epoch and `6.17e-6 m/s` in velocity.

This is an important separation: the roughly 12 m spacecraft residual is not an integration-accuracy artifact. The numerical convergence difference is millimetric, while the remaining discrepancy is dominated by physical/estimation differences between the open-loop candidate and the reconstructed orbit.

### Twenty-four-hour 120x120 candidate

| Quantity | Difference from reconstructed LRO SPK |
|---|---:|
| final position | `24.0340 m` |
| maximum position | `333.1338 m` |
| RMS position | `139.7591 m` |
| final velocity | `0.0831306 m/s` |
| maximum velocity | `0.208895 m/s` |
| radial RMS | `38.0003 m` |
| along-track RMS | `133.9698 m` |
| cross-track RMS | `11.8610 m` |

The much larger 24-hour error, particularly along track, is useful validation evidence in its own right: reducing gravity to 120x120 and omitting the full precision-OD force/estimation model accumulates observable trajectory error. This result must not be interpreted as a universal statement about 120x120 accuracy for every lunar orbit.

No pass/fail mission threshold is assigned to either LRO comparison. Selecting a threshold after seeing the residual, or pretending the open-loop force model exactly matches GEODYN, would weaken rather than strengthen the validation.

## Public LRO accuracy context

NASA LRO orbit-determination studies report that gravity and SRP modelling are major drivers of orbit/prediction performance. Published LRO work documents operational force models including high-degree lunar gravity, lunar tides, SRP, and Earth/Sun/Jupiter point masses, and reports improved reconstructed/overlap performance after adoption of GRAIL gravity.

Useful NASA references include:

- *Lunar Reconnaissance Orbiter Orbit Determination Accuracy Analysis*: `https://ntrs.nasa.gov/citations/20140008968`
- *Orbit Determination for the Lunar Reconnaissance Orbiter Using an Extended Kalman Filter*: `https://ntrs.nasa.gov/citations/20150019754`

The PDS reconstructed SPK therefore provides credible external spacecraft evidence, but it is not a perfect truth trajectory with zero uncertainty.

## Reproducibility

Install the optional independent-validation dependencies:

```bash
python -m pip install -e .[validation]
```

Download the exact public inputs and generate SHA-256 provenance:

```bash
python scripts/download_independent_validation_data.py \
  --output-dir data/validation
```

Run the full archived campaign:

```bash
python scripts/run_independent_validation.py \
  --data-dir data/validation \
  --output results/independent_validation.json
```

The full campaign includes the six-hour 600x600 LRO propagation and can take substantially longer than the ordinary test suite because the production degree-600 evaluator is pure Python.

A more practical repeat profile is also available:

```bash
python scripts/run_independent_validation_compact.py \
  --data-dir data/validation \
  --output results/independent_validation_repeat.json
```

The compact profile keeps the degree-600 spacecraft comparison near one low-lunar orbital period while retaining the separate 24-hour 120x120 evolution check.

Routine CI intentionally does **not** download the external products. It remains deterministic and network-independent. The archive-level campaign is rerun explicitly when external scientific revalidation is required.

## What could not be validated completely

### Exact GEODYN LRO reproduction

The public reconstructed SPK is available, but this repository does not contain the complete GEODYN estimation setup, tracking weights, spacecraft attitude/area history, maneuver history and all solved-for parameters for the selected monthly arc. The spacecraft comparison is therefore an open-loop validation against an externally reconstructed trajectory, not a bit-for-bit reproduction of the POD solution.

### GRGM1200B long-term trajectory with authoritative orientation

GRGM1200B was independently validated through degree/order 1200 at explicit body-fixed points. A long-duration GRGM1200B trajectory campaign was not claimed because the repository has not identified an authoritative public lunar principal-axes orientation context that should be silently treated as equivalent to the gravity solution's DE430-era frame. DE421 and DE440 lunar frames are not substituted merely because they are all lunar body-fixed frames.

### Independent integrator family

The same-model trajectory comparison uses DOP853 from SciPy on both sides. Everything upstream of the integrator is independently constructed, including coefficient parsing, gravity evaluation, frame/ephemeris access, third-body arithmetic and the RHS. The LRO reconstructed SPK supplies a fully independent mission-trajectory comparison. A future validation could add Tudat, GMAT or another trusted propagator with an exactly matched public setup.

### LOLA measurement uncertainty

The terrain check validates binary decoding, coordinate reconstruction and scaling against the public PDS product. It does not independently validate LOLA ranging/topographic measurement uncertainty.

## Conclusion

The campaign materially strengthens the scientific evidence for the implementation:

- two independent harmonic engines agree through degree/order 1200 to roughly `1e-14` relative acceleration at the tested points;
- frame transformations agree with direct CSPICE at floating-point level;
- differential third-body arithmetic agrees with 80-digit reference calculations within conditioning-derived tolerances;
- PDS terrain coordinates and elevations are reproduced exactly at tested pixels;
- independently implemented identical-force propagation stays within approximately `0.13 mm` over two hours;
- an open-loop lunar propagation initialized from a real reconstructed LRO orbit stays within approximately `12.6 m` maximum position difference over the tested six-hour 600x600 arc, with only millimetric numerical-integration sensitivity.

These results support research and preliminary mission-analysis use. They do not make the repository flight-certified, do not replace orbit determination, and do not demonstrate that omitted force models are negligible over arbitrary mission horizons.
