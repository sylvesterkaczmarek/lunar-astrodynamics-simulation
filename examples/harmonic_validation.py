"""Validate normalization and tesseral gradients without external gravity data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    GRGM1200A_J2,
    SphericalHarmonicModel,
    central_acceleration,
    gravity_acceleration_body_fixed,
    gravity_potential_body_fixed,
    j2_acceleration,
)


def j2_as_harmonics() -> SphericalHarmonicModel:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -GRGM1200A_J2.j2 / np.sqrt(5.0)
    return SphericalHarmonicModel(
        GRGM1200A_J2.mu_m3_s2,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="J2 represented by normalized C20",
    )


def tesseral_model() -> SphericalHarmonicModel:
    c = np.zeros((4, 4))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[2, 2] = 2.2e-5
    s[2, 2] = -1.7e-5
    c[3, 1] = 8.0e-6
    s[3, 1] = 5.0e-6
    return SphericalHarmonicModel(4.9028e12, 1.738e6, c, s, name="synthetic tesseral validation")


def run() -> dict[str, float]:
    position = np.array([1.91e6, -0.42e6, 0.31e6])
    model = j2_as_harmonics()
    harmonic = gravity_acceleration_body_fixed(position, model)
    closed_form = central_acceleration(position, model.mu_m3_s2) + j2_acceleration(
        position,
        model.mu_m3_s2,
        model.reference_radius_m,
        GRGM1200A_J2.j2,
    )
    j2_relative = float(np.linalg.norm(harmonic - closed_form) / np.linalg.norm(closed_form))

    model_t = tesseral_model()
    point = np.array([1.82e6, 0.51e6, 0.37e6])
    analytic = gravity_acceleration_body_fixed(point, model_t)
    h = 0.25
    finite = np.empty(3)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = h
        finite[axis] = (
            gravity_potential_body_fixed(point + delta, model_t)
            - gravity_potential_body_fixed(point - delta, model_t)
        ) / (2.0 * h)
    gradient_relative = float(np.linalg.norm(analytic - finite) / np.linalg.norm(analytic))
    return {
        "j2_c20_relative_acceleration_difference": j2_relative,
        "tesseral_finite_difference_relative_gradient_difference": gradient_relative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/harmonic_validation.json"))
    args = parser.parse_args()
    metrics = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
