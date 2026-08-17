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
class GravityCoefficientPerturbation:
    """A coefficient delta field to be added to a compatible nominal model."""

    c_delta: FloatArray
    s_delta: FloatArray
    name: str = "gravity coefficient perturbation"
    frame: str = "body-fixed"
    normalization: str = "4pi"

    def __post_init__(self) -> None:
        c_delta = np.array(self.c_delta, dtype=float, copy=True)
        s_delta = np.array(self.s_delta, dtype=float, copy=True)
        if (
            c_delta.ndim != 2
            or c_delta.shape[0] != c_delta.shape[1]
            or s_delta.shape != c_delta.shape
        ):
            raise ValueError("c_delta and s_delta must be matching square arrays")
        if not np.all(np.isfinite(c_delta)) or not np.all(np.isfinite(s_delta)):
            raise ValueError("coefficient perturbations must be finite")
        if not np.isclose(c_delta[0, 0], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("c_delta[0,0] must be zero")
        if not np.isclose(s_delta[0, 0], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("s_delta[0,0] must be zero")
        upper = np.triu_indices(c_delta.shape[0], k=1)
        if np.any(c_delta[upper] != 0.0) or np.any(s_delta[upper] != 0.0):
            raise ValueError("perturbations with order m > degree n must be zero")
        if self.normalization.lower() != "4pi":
            raise ValueError("this library requires geodesy 4pi-normalized perturbations")
        c_delta.setflags(write=False)
        s_delta.setflags(write=False)
        object.__setattr__(self, "c_delta", c_delta)
        object.__setattr__(self, "s_delta", s_delta)

    @property
    def max_degree(self) -> int:
        return self.c_delta.shape[0] - 1

    def truncated(self, max_degree: int, max_order: int | None = None) -> "GravityCoefficientPerturbation":
        degree = min(int(max_degree), self.max_degree)
        if degree < 0:
            raise ValueError("max_degree must be non-negative")
        order = degree if max_order is None else min(int(max_order), degree)
        if order < 0:
            raise ValueError("max_order must be non-negative")
        c_delta = self.c_delta[: degree + 1, : degree + 1].copy()
        s_delta = self.s_delta[: degree + 1, : degree + 1].copy()
        if order < degree:
            c_delta[:, order + 1 :] = 0.0
            s_delta[:, order + 1 :] = 0.0
        return GravityCoefficientPerturbation(
            c_delta,
            s_delta,
            name=f"{self.name} (n<={degree}, m<={order})",
            frame=self.frame,
            normalization=self.normalization,
        )


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
    """Return the official PDS URL for one archived GRGM1200A clone delta field."""
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
        c_delta = float(fields[2].replace("D", "E").replace("d", "e"))
        s_delta = float(fields[3].replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"invalid clone coefficient row {line_number}") from exc
    if (
        n < 0
        or m < 0
        or m > n
        or not np.isfinite(c_delta)
        or not np.isfinite(s_delta)
    ):
        raise ValueError(f"invalid clone coefficient row {line_number}")
    return n, m, c_delta, s_delta


def read_grgm1200a_clone(
    path_or_file: str | Path | TextIO,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    frame: str = GRGM1200A.body_fixed_frame,
) -> GravityCoefficientPerturbation:
    """Read an archived PDS GRGM1200A covariance-derived clone delta field.

    GRGM1200A clone coefficients represent deviations from the nominal field,
    not standalone lunar gravity models. This function therefore returns a
    ``GravityCoefficientPerturbation`` with C00=S00=0. Apply it to a compatible
    nominal model with ``apply_coefficient_perturbation`` before propagation.

    A single non-coefficient preamble line is tolerated before the first data
    row. Once coefficient parsing begins, malformed or duplicate rows fail.
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

    c_delta = np.zeros((degree + 1, degree + 1), dtype=float)
    s_delta = np.zeros_like(c_delta)
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
            if n == 0:
                if m != 0 or c_nm != 0.0 or s_nm != 0.0:
                    raise ValueError("clone degree-zero perturbation must be zero")
                continue
            if n <= degree:
                key = (n, m)
                if key in seen:
                    raise ValueError(f"duplicate clone coefficient ({n}, {m})")
                seen.add(key)
                c_delta[n, m] = c_nm
                s_delta[n, m] = s_nm
    finally:
        if close:
            handle.close()

    if degree > 0:
        expected = (degree + 1) * (degree + 2) // 2 - 1
        if len(seen) != expected:
            raise ValueError(
                f"clone field is incomplete through degree {degree}: "
                f"found {len(seen)} of {expected} coefficient perturbations"
            )

    return GravityCoefficientPerturbation(
        c_delta,
        s_delta,
        name=name or getattr(path_or_file, "name", "GRGM1200A clone perturbation"),
        frame=frame,
        normalization=GRGM1200A.normalization,
    )


def apply_coefficient_perturbation(
    nominal_model: SphericalHarmonicModel,
    perturbation: GravityCoefficientPerturbation,
    *,
    name: str | None = None,
) -> SphericalHarmonicModel:
    """Add a coefficient delta field to a compatible nominal gravity model."""
    if nominal_model.normalization.lower() != perturbation.normalization.lower():
        raise ValueError("nominal model and perturbation normalization must match")
    if nominal_model.frame != perturbation.frame:
        raise ValueError("nominal model and perturbation body-fixed frames must match")
    if perturbation.max_degree > nominal_model.max_degree:
        raise ValueError("perturbation degree exceeds the nominal model degree")

    c = nominal_model.c.copy()
    s = nominal_model.s.copy()
    degree = perturbation.max_degree
    c[: degree + 1, : degree + 1] += perturbation.c_delta
    s[: degree + 1, : degree + 1] += perturbation.s_delta
    c[0, 0] = 1.0
    s[0, 0] = 0.0
    return SphericalHarmonicModel(
        nominal_model.mu_m3_s2,
        nominal_model.reference_radius_m,
        c,
        s,
        name=name or f"{nominal_model.name} + {perturbation.name}",
        frame=nominal_model.frame,
        normalization=nominal_model.normalization,
    )


def _validate_grgm1200a_nominal(model: SphericalHarmonicModel) -> None:
    if abs(model.mu_m3_s2 - GRGM1200A.mu_m3_s2) > 1.0:
        raise ValueError("nominal model GM does not match archived GRGM1200A metadata")
    if abs(model.reference_radius_m - GRGM1200A.reference_radius_m) > 1e-6:
        raise ValueError("nominal model reference radius does not match GRGM1200A")
    if model.normalization.lower() != GRGM1200A.normalization.lower():
        raise ValueError("nominal model normalization does not match GRGM1200A")
    if model.frame != GRGM1200A.body_fixed_frame:
        raise ValueError(
            "nominal GRGM1200A clone analysis requires the archived DE430 principal-axes frame metadata"
        )


def load_grgm1200a_clone_ensemble(
    nominal_model: SphericalHarmonicModel,
    paths: Iterable[str | Path],
    *,
    max_degree: int | None = None,
) -> tuple[SphericalHarmonicModel, ...]:
    """Apply multiple archived GRGM1200A clone deltas to the nominal field."""
    _validate_grgm1200a_nominal(nominal_model)
    degree = nominal_model.max_degree if max_degree is None else min(
        int(max_degree), nominal_model.max_degree
    )
    if degree < 0:
        raise ValueError("max_degree must be non-negative")
    base = nominal_model.truncated(degree)

    models = tuple(
        apply_coefficient_perturbation(
            base,
            read_grgm1200a_clone(
                path,
                max_degree=degree,
                name=Path(path).name,
                frame=base.frame,
            ),
            name=f"{base.name} + {Path(path).name}",
        )
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

    This is intentionally opt-in because SHADR coefficient uncertainty fields
    do not encode cross-covariances. For GRGM1200A science studies, archived
    covariance-derived clone fields are preferred when correlations matter.
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
                mu,
                model.reference_radius_m,
                c,
                s,
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
    return OrbitUncertaintySample(
        model_name=model_name,
        minimum_altitude_m=float(np.min(altitude)),
        maximum_altitude_m=float(np.max(altitude)),
        minimum_osculating_periselene_altitude_m=(
            float(np.min(finite_periselene)) if finite_periselene.size else float("nan")
        ),
        maximum_osculating_aposelene_altitude_m=(
            float(np.max(finite_aposelene)) if finite_aposelene.size else float("nan")
        ),
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
    """Propagate one initial state through several compatible gravity realizations."""
    if not models:
        raise ValueError("at least one gravity realization is required")
    if len({model.frame for model in models}) != 1:
        raise ValueError("all gravity realizations must use the same body-fixed frame")
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
