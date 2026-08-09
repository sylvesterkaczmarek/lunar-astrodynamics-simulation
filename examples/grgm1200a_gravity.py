"""Evaluate the official GRGM1200A field at a body-fixed Cartesian point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    GRGM1200A,
    gravity_acceleration_body_fixed,
    gravity_potential_body_fixed,
    read_shadr,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/gggrx_1200a_sha.tab"))
    parser.add_argument("--degree", type=int, default=1200)
    parser.add_argument("--order", type=int, default=None)
    parser.add_argument(
        "--position-km",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(1900.0, 200.0, 300.0),
        help="Cartesian point in the GRGM1200A body-fixed principal-axes frame",
    )
    args = parser.parse_args()

    model = read_shadr(
        args.model,
        max_degree=args.degree,
        name=GRGM1200A.name,
        frame=GRGM1200A.body_fixed_frame,
    )
    position_m = np.asarray(args.position_km, dtype=float) * 1000.0
    potential = gravity_potential_body_fixed(
        position_m, model, max_degree=args.degree, max_order=args.order
    )
    acceleration = gravity_acceleration_body_fixed(
        position_m, model, max_degree=args.degree, max_order=args.order
    )

    print(f"Model: {model.name}")
    print(f"Frame: {model.frame}")
    print(f"Degree/order: {args.degree}/{args.order if args.order is not None else args.degree}")
    print(f"Potential: {potential:.9e} m^2/s^2")
    print("Acceleration [m/s^2]:", np.array2string(acceleration, precision=12))


if __name__ == "__main__":
    main()
