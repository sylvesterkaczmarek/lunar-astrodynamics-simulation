# DE421 ground-track and access validation

A live external-data validation was run on 17 August 2026 to exercise the production ground-track, named-site access, gridded coverage and Earth-occultation workflow with actual NAIF lunar orientation and planetary ephemeris kernels.

The machine-readable record is [`results/groundtrack_spice_validation.json`](../results/groundtrack_spice_validation.json).

## Kernel and frame context

The run downloaded and caller-loaded:

- `naif0012.tls`, 5,257 bytes;
- `de421.bsp`, 16,790,528 bytes;
- `moon_pa_de421_1900-2050.bpc`, 1,770,496 bytes;
- `moon_080317.tf`, 21,437 bytes.

The propagation epoch was `2026-08-17T00:00:00` UTC, corresponding to ET `840196869.1829052 s`. Spacecraft and Earth geometry used Moon-centred `J2000`; ground-track/site geometry used `MOON_ME_DE421`. SPICE Earth positions used geometric `NONE` aberration correction.

No kernels are committed to the repository. Routine CI remains offline.

## Test trajectory

The validation intentionally used a simple central-gravity, 100 km circular polar orbit so that the exercise isolates coordinate transformations, surface visibility, access-window extraction, coverage accounting and lunar occultation rather than claiming a high-fidelity orbit truth case.

The epoch polar plane crossed approximately `27 deg E`, between the two named Apollo sites. The run lasted `0.25 day` with `241` samples and used a `10 deg` minimum site elevation mask.

Observed ground-track latitude extended from `-89.769987 deg` to `+89.271032 deg`. Reference-radius altitude remained numerically at 100 km to integration tolerance.

## Named-site access

The worked sites use NASA-published east-positive coordinates:

| Site | Coordinates | Windows | Total access | Maximum elevation | Minimum slant range |
|---|---|---:|---:|---:|---:|
| Apollo 11 Tranquility Base | `0.67409 N, 23.47298 E` | 4 | `1582.466 s` | `67.644 deg` | `107.634 km` |
| Apollo 17 Taurus-Littrow | `20.1911 N, 30.7769 E` | 4 | `1421.599 s` | `36.220 deg` | `161.418 km` |

Apollo 11 start-to-start revisit intervals averaged `6995.858 s`; Apollo 17 averaged `7078.453 s` over this short test span.

The example leaves both site elevations at the 1737.4 km reference radius. For local terrain/horizon studies, construct sites from a frame-compatible terrain model and optionally enable terrain-aware line-of-sight sampling.

## Coverage and Earth visibility

The `13 x 12` regular latitude/longitude coverage grid produced a configured-grid-point covered fraction of `0.346154`. This is not an equal-area fraction of lunar surface. Mean dwell per configured grid point was `447.979 s`; maximum dwell was `1566.055 s`.

The same trajectory used DE421 Moon-centred Earth positions and finite apparent Earth/Moon disks. Earth was geometrically visible for `0.633333` of the sampled span under the selected visibility threshold. Fully occulted samples accounted for `0.377593` and partial-occultation samples for `0.008299`; four visible intervals were identified.

## Limits

This result validates the production geometry/data path, not operational mission performance. Access boundaries and revisit statistics depend on the trajectory sampling cadence. The run did not enable terrain-aware site LOS. Earth occultation uses a spherical Moon and nominal circular Earth disk. The central-gravity validation orbit omits high-degree lunar gravity and perturbation forces by design.
