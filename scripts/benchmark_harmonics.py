"""Report high-degree lunar spherical-harmonic acceleration timing.

This is a measurement harness, not a pass/fail benchmark. Shared CI runners are
not stable enough for scientifically meaningful wall-clock thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from lunar_astrodynamics import (
    SphericalHarmonicModel,
    gravity_acceleration_body_fixed,
    read_shadr,
)


def _synthetic_model(degree: int) -> SphericalHarmonicModel:
    if degree < 2:
        raise ValueError("synthetic benchmark degree must be at least 2")
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[degree, 1] = 1.0e-14
    if degree >= 13:
        c[degree // 2, 13] = 1.0e-11
    s[degree, degree] = 1.0e-12
    return SphericalHarmonicModel(
        4.90280011526323e12,
        1.738e6,
        c,
        s,
        name=f"synthetic degree-{degree} benchmark field",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="optional PDS SHADR gravity model")
    parser.add_argument("--degree", type=int, default=1200)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--position-km",
        type=float,
        nargs=3,
        default=(1900.0, 200.0, 300.0),
        metavar=("X", "Y", "Z"),
    )
    args = parser.parse_args()

    if args.degree < 0:
        parser.error("--degree must be non-negative")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    if args.model is None:
        model = _synthetic_model(args.degree)
    else:
        model = read_shadr(args.model, max_degree=args.degree, name=args.model.name)

    position_m = np.asarray(args.position_km, dtype=float) * 1e3

    # Warm the Python/NumPy execution path before recording wall time.
    gravity_acceleration_body_fixed(position_m, model, max_degree=args.degree)

    samples_s: list[float] = []
    acceleration = np.zeros(3)
    for _ in range(args.repetitions):
        start = perf_counter()
        acceleration = gravity_acceleration_body_fixed(
            position_m,
            model,
            max_degree=args.degree,
        )
        samples_s.append(perf_counter() - start)

    median_s = float(np.median(samples_s))
    report = {
        "model": model.name,
        "degree": min(args.degree, model.max_degree),
        "position_m": position_m.tolist(),
        "repetitions": args.repetitions,
        "minimum_seconds": min(samples_s),
        "median_seconds": median_s,
        "maximum_seconds": max(samples_s),
        "median_evaluations_per_second": 1.0 / median_s,
        "acceleration_m_s2": acceleration.tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
