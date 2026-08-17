"""Prepare a NASA LOLA MOON_PA GMT grid for repeated Python use."""

from __future__ import annotations

import argparse
from pathlib import Path

from lunar_astrodynamics import load_lola_moon_pa_grd, save_terrain_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the external NASA Goddard LOLA MOON_PA GMT/netCDF grid to "
            "the repository's metadata-bearing NPZ format. Use --stride 1 for "
            "native 64 ppd or a divisor of 64 such as 2/4/8 for a smaller "
            "preliminary-analysis grid."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/lola_moon_pa_8ppd.npz"))
    parser.add_argument("--registration", choices=("gridline", "pixel"), default="gridline")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    terrain = load_lola_moon_pa_grd(
        args.input,
        registration=args.registration,
        stride=args.stride,
    )
    save_terrain_npz(terrain, args.output)
    print(f"Prepared {terrain.name}")
    print(f"Frame: {terrain.frame}")
    print(f"Reference radius: {terrain.reference_radius_m:.1f} m")
    print(
        f"Grid: {terrain.latitude_deg.size} x {terrain.longitude_deg_east.size}; "
        f"spacing about {terrain.latitude_spacing_deg:.6f} x "
        f"{terrain.longitude_spacing_deg:.6f} deg"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
