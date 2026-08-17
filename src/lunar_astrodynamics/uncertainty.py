"""Gravity-field uncertainty sampling and lunar trajectory ensemble analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import GRGM1200A, MOON_MEAN_RADIUS_M
from .frames import RotationProvider
from .harmonics import SphericalHarmonicModel, gravity_acceleration_inertial
from .propagation import PropagationSettings, propagate_with_acceleration

FloatArray = NDArray[np.float64]

GRGM1200A_CLONE_BASE_URL = (
    "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/"
    "grail_1001/extras/clones"
)
GRGM1200A_CLONE_COUNT = 500
GRGM1200A_CLONE_EXPECTED_SIZE_BYTES = 44_029_817


@dataclass(frozen=True)
class OrbitUncertaintySample:
    """Scalar trajectory metrics for one gravity realization."""

    model_name: str
    minimum_altitude_m: float
    maximum_altitude_m: float
    minimum_osculating_periselene_altitude_m: float
    maximum_osculating_aposelene_altitude_m: float
    maximum_eccentricity: float
    final_eccentricity: float
    lifetime_s: float
    impacted: bool


@dataclass(frozen=True)
class EnsembleUncertaintyResult:
    """Metrics and percentile summaries across a gravity-field ensemble."""

    samples: tuple[OrbitUncertaintySample, ...]
    percentile_levels: tuple[float, ...]
    percentiles: dict[str, dict[float, float]]
    impact_fraction: float

    def metric_values(self, metric: str) -> FloatArray:
        if not self.samples:
            return np.empty(0, dtype=float)
        if not hasattr(self.samples[0], metric):
            raise ValueError(f"unknown ensemble metric: {metric}")
        values = [getattr(sample, metric) for sample in self.samples]
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("boolean metrics are not returned as numeric arrays")
        return np.asarray(values, dtype=float)


def grgm1200a_clone_url(index: int) -> str:
    """Return the official PDS URL for one archived GRGM1200A clone field."""
    clone = int(index)
    if clone < 1 or clone > GRGM1200A_CLONE_COUNT:
        raise ValueError("GRGM1200A clone index must be within [1, 500]")
    group_start = ((clone - 1) // 100) * 100 + 1
    group_end = group_start + 99
    group = f"gggrx_1200a_clones_{group_start:04d}_{group_end:04d}"
    filename = f"gggrx_1200a_clone{clone:04d}_sha.tab"
    return f"{GRGM1200A_CLONE_BASE_URL}/{group}/{filename}"


def _parse_clone_row(line: str, line_number: int) -> tuple[int, int, float, float]:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 4:
        raise ValueError(f"clone coefficient row {line_number} has fewer than four fields")
    try:
        n = int(fields[0])
        m = int(fields[1])
        c_nm = float(fields[2].replace("D", "E").replace("d", "e"))
        s_nm = float(fields[3].replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"invalid clone coefficient row {line_number}") from exc
    if (
        n < 0
        or m < 0
        or m > n
        or not np.isfinite(c_nm)
        or not np.isfinite(s_nm)
    ):
        raise ValueError(f"invalid clone coefficient row {line_number}")
    return n, m, c_nm, s_nm


def read_grgm1200a_clone(
    path_or_file: str | Path | TextIO,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    frame: str = GRGM1200A.body_fixed_frame,
) -> SphericalHarmonicModel:
    """Read a PDS GRGM1200A coefficient-only clone realization.

    The archived clone realizations are not treated as uncertainty-bearing
    nominal SHADR products. Their C/S coefficients define one correlated draw
    from the GRGM1200A least-squares covariance system. Nominal GRGM1200A GM,
    reference radius, normalization, and frame metadata are supplied from the
    archived product metadata in this package.

    A single non-coefficient preamble line is tolerated before the first
    coefficient row. Once coefficient parsing begins, malformed rows fail.
    """
    degree = GRGM1200A.max_degree if max_degree is None else min(
        int(max_degree), GRGM1200A.max_degree
    )
    if degree < 0:
        raise ValueError("max_degree must be non-negative")

    close = False
    if hasattr(path_or_file, "read"):
        handle = path_or_file  # type: ignore[assignment]
    else:
        handle = open(
            path_or_file,
            "r",
            encoding="ascii",
            errors="strict",
            newline="",
        )
        close = True

    c = np.zeros((degree + 1, degree + 1), dtype=float)
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    seen: set[tuple[int, int]] = set()
    coefficient_rows = 0
    skipped_preamble = False

    try:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                n, m, c_nm, s_nm = _parse_clone_row(line, line_number)
            except ValueError:
                if coefficient_rows == 0 and not skipped_preamble:
                    skipped_preamble = True
                    continue
                raise

            coefficient_rows += 1
            if n > GRGM1200A.max_degree or m > GRGM1200A.max_order:
                raise ValueError(
                    f"clone coefficient ({n}, {m}) exceeds GRGM1200A degree/order 1200"
                )
            if n <= degree:
                key = (n, m)
                if key in seen:
                    raise ValueError(f"duplicate clone coefficient ({n}, {m})")
                seen.add(key)
                c[n, m] = c_nm
                s[n, m] = s_nm
    finally:
        if close:
            handle.close()

    if degree > 0:
        expected = (degree + 1) * (degree + 2) // 2 - 1
        if len(seen) != expected:
            raise ValueError(
                f"clone field is incomplete through degree {degree}: "
                f"found {len(seen)} of {expected} coefficients"
            )

    return SphericalHarmonicModel(
        mu_m3_s2=GRGM1200A.mu_m3_s2,
        reference_radius_m=GRGM1200A.reference_radius_m,
        c=c,
        s=s,
        name=name or getattr(path_or_file, "name", "GRGM1200A clone field"),
        frame=frame,
        normalization=GRGM1200A.normalization,
    )


def load_grgm1200a_clone_ensemble(
    paths: Iterable[str | Path],
    *,
    max_degree: int | None = None,
) -> tuple[SphericalHarmonicModel, ...]:
    """Load multiple archived GRGM1200A clone realizations."""
    models = tuple(
        read_grgm1200a_clone(path, max_degree=max_degree, name=Path(path).name)
        for path in paths
    )
    if not models:
        raise ValueError("at least one clone path is required")
    return models


def sample_independent_coefficient_uncertainty(
    model: SphericalHarmonicModel,
    *,
    seed: int,
    count: int = 1,
    sigma_scale: float = 1.0,
    assume_independent: bool = False,
    include_mu: bool = False,
) -> tuple[SphericalHarmonicModel, ...]:
    """Draw reproducible diagonal-only coefficient perturbations.

    This is intentionally opt-in because the SHADR coefficient uncertainty
    fields do not encode the cross-covariances. For GRGM1200A science studies,
    archived covariance-derived clone fields are preferred when correlations
    matter.
    """
    if not assume_independent:
        raise ValueError(
            "independent coefficient sampling requires assume_independent=True; "
            "SHADR sigmas are not a full covariance model"
        )
    if not model.has_coefficient_uncertainty:
        raise ValueError("model does not contain coefficient uncertainties")
    if count < 1:
        raise ValueError("count must be at least one")
    if not np.isfinite(sigma_scale) or sigma_scale < 0.0:
        raise ValueError("sigma_scale must be finite and non-negative")

    assert model.sigma_c is not None
    assert model.sigma_s is not None
    rng = np.random.default_rng(int(seed))
    realizations: list[SphericalHarmonicModel] = []
    valid = np.tril(np.ones(model.c.shape, dtype=bool))
    valid[0, 0] = False

    for index in range(count):
        c = model.c.copy()
        s = model.s.copy()
        c_noise = rng.normal(size=model.c.shape) * model.sigma_c * sigma_scale
        s_noise = rng.normal(size=model.s.shape) * model.sigma_s * sigma_scale
        c[valid] += c_noise[valid]
        s[valid] += s_noise[valid]
        c[0, 0] = 1.0
        s[0, 0] = 0.0

        mu = model.mu_m3_s2
        if include_mu:
            if model.mu_sigma_m3_s2 is None:
                raise ValueError("include_mu=True requires a stored GM uncertainty")
            mu += float(rng.normal()) * model.mu_sigma_m3_s2 * sigma_scale
            if mu <= 0.0:
                raise ValueError("sampled GM is non-positive")

        realizations.append(
            SphericalHarmonicModel(
                mu_m3_s2=mu,
                reference_radius_m=model.reference_radius_m,
                c=c,
                s=s,
                name=f"{model.name} diagonal-sigma draw {index + 1} seed {int(seed)}",
                frame=model.frame,
                normalization=model.normalization,
                sigma_c=model.sigma_c,
                sigma_s=model.sigma_s,
                mu_sigma_m3_s2=model.mu_sigma_m3_s2,
            )
        )

    return tuple(realizations)


def _eccentricity_and_apsides(
    states: FloatArray,
    mu_m3_s2: float,
    reference_radius_m: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    eccentricity = np.empty(states.shape[1], dtype=float)
    periselene = np.full(states.shape[1], np.nan, dtype=float)
    aposelene = np.full(states.shape[1], np.nan, dtype=float)

    for index in range(states.shape[1]):
        r = states[:3, index]
        v = states[3:, index]
        r_norm = float(np.linalg.norm(r))
        h = np.cross(r, v)
        e_vec = np.cross(v, h) / mu_m3_s2 - r / r_norm
        e = float(np.linalg.norm(e_vec))
        eccentricity[index] = e
        energy = 0.5 * float(np.dot(v, v)) - mu_m3_s2 / r_norm
        if energy < 0.0:
            a = -mu_m3_s2 / (2.0 * energy)
            periselene[index] = a * (1.0 - e) - reference_radius_m
            aposelene[index] = a * (1.0 + e) - reference_radius_m

    return eccentricity, periselene, aposelene


def _trajectory_metrics(
    model_name: str,
    solution: object,
    *,
    mu_m3_s2: float,
    reference_radius_m: float,
    requested_duration_s: float,
) -> OrbitUncertaintySample:
    states = np.asarray(solution.y, dtype=float)  # type: ignore[attr-defined]
    radii = np.linalg.norm(states[:3], axis=0)
    altitude = radii - reference_radius_m
    eccentricity, periselene, aposelene = _eccentricity_and_apsides(
        states, mu_m3_s2, reference_radius_m
    )

    event_times = solution.t_events[0]  # type: ignore[attr-defined]
    impacted = bool(len(event_times))
    lifetime_s = float(event_times[0]) if impacted else float(requested_duration_s)

    finite_periselene = periselene[np.isfinite(periselene)]
    finite_aposelene = aposelene[np.isfinite(aposelene)]
    minimum_periselene = (
        float(np.min(finite_periselene)) if finite_periselene.size else float("nan")
    )
    maximum_aposelene = (
        float(np.max(finite_aposelene)) if finite_aposelene.size else float("nan")
    )

    return OrbitUncertaintySample(
        model_name=model_name,
        minimum_altitude_m=float(np.min(altitude)),
        maximum_altitude_m=float(np.max(altitude)),
        minimum_osculating_periselene_altitude_m=minimum_periselene,
        maximum_osculating_aposelene_altitude_m=maximum_aposelene,
        maximum_eccentricity=float(np.max(eccentricity)),
        final_eccentricity=float(eccentricity[-1]),
        lifetime_s=lifetime_s,
        impacted=impacted,
    )


def summarize_ensemble(
    samples: Sequence[OrbitUncertaintySample],
    *,
    percentile_levels: Sequence[float] = (5.0, 50.0, 95.0),
) -> EnsembleUncertaintyResult:
    """Calculate percentile summaries for trajectory ensemble metrics."""
    if not samples:
        raise ValueError("at least one uncertainty sample is required")
    levels = tuple(float(level) for level in percentile_levels)
    if not levels or any(
        not np.isfinite(level) or level < 0.0 or level > 100.0 for level in levels
    ):
        raise ValueError("percentile levels must be finite values within [0, 100]")

    metric_names = (
        "minimum_altitude_m",
        "maximum_altitude_m",
        "minimum_osculating_periselene_altitude_m",
        "maximum_osculating_aposelene_altitude_m",
        "maximum_eccentricity",
        "final_eccentricity",
        "lifetime_s",
    )
    percentiles: dict[str, dict[float, float]] = {}
    for metric in metric_names:
        values = np.asarray([getattr(sample, metric) for sample in samples], dtype=float)
        if np.all(np.isnan(values)):
            percentiles[metric] = {level: float("nan") for level in levels}
        else:
            computed = np.nanpercentile(values, levels)
            percentiles[metric] = {
                level: float(value) for level, value in zip(levels, computed, strict=True)
            }

    return EnsembleUncertaintyResult(
        samples=tuple(samples),
        percentile_levels=levels,
        percentiles=percentiles,
        impact_fraction=float(np.mean([sample.impacted for sample in samples])),
    )


def propagate_gravity_ensemble(
    initial_state: ArrayLike,
    duration_s: float,
    models: Sequence[SphericalHarmonicModel],
    body_fixed_from_inertial: RotationProvider,
    *,
    collision_radius_m: float = MOON_MEAN_RADIUS_M,
    reference_radius_m: float = MOON_MEAN_RADIUS_M,
    max_degree: int | None = None,
    max_order: int | None = None,
    sample_times_s: ArrayLike | None = None,
    sample_count: int = 257,
    settings: PropagationSettings = PropagationSettings(),
    percentile_levels: Sequence[float] = (5.0, 50.0, 95.0),
) -> EnsembleUncertaintyResult:
    """Propagate one initial state through several gravity realizations."""
    if not models:
        raise ValueError("at least one gravity realization is required")
    if sample_times_s is None:
        if sample_count < 2:
            raise ValueError("sample_count must be at least two")
        sample_times = np.linspace(0.0, float(duration_s), int(sample_count))
    else:
        sample_times = np.asarray(sample_times_s, dtype=float)

    samples: list[OrbitUncertaintySample] = []
    for model in models:
        acceleration = lambda time_s, position_m, _model=model: gravity_acceleration_inertial(
            time_s,
            position_m,
            _model,
            body_fixed_from_inertial,
            max_degree=max_degree,
            max_order=max_order,
        )
        solution = propagate_with_acceleration(
            initial_state,
            duration_s,
            acceleration,
            collision_radius_m=collision_radius_m,
            sample_times_s=sample_times,
            settings=settings,
        )
        samples.append(
            _trajectory_metrics(
                model.name,
                solution,
                mu_m3_s2=model.mu_m3_s2,
                reference_radius_m=reference_radius_m,
                requested_duration_s=duration_s,
            )
        )

    return summarize_ensemble(samples, percentile_levels=percentile_levels)
