"""Download a small official LOLA PDS DEM and print selected reference grid values."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    LOLA_PDS_LDEM4_IMG_URL,
    LOLA_PDS_LDEM4_LBL_URL,
    load_lola_pds_global_gdr,
)

REFERENCE_POINTS_DEG = (
    (0.125, 0.125),
    (0.125, 90.125),
    (0.125, 180.125),
    (45.125, 45.125),
    (-45.125, 315.125),
    (89.875, 0.125),
    (-89.875, 180.125),
)


def _download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-dir", type=Path)
    args = parser.parse_args()

    if args.keep_dir is None:
        context = tempfile.TemporaryDirectory()
        directory = Path(context.name)
    else:
        args.keep_dir.mkdir(parents=True, exist_ok=True)
        directory = args.keep_dir
        context = None

    try:
        image = directory / "LDEM_4.IMG"
        label = directory / "LDEM_4.LBL"
        _download(LOLA_PDS_LDEM4_IMG_URL, image)
        _download(LOLA_PDS_LDEM4_LBL_URL, label)
        terrain = load_lola_pds_global_gdr(image, label)
        print(f"frame={terrain.frame}")
        print(f"reference_radius_m={terrain.reference_radius_m:.1f}")
        print(f"shape={terrain.elevation_grid_m.shape}")
        for latitude_deg, longitude_deg in REFERENCE_POINTS_DEG:
            i = int(np.argmin(np.abs(terrain.latitude_deg - latitude_deg)))
            j = int(np.argmin(np.abs(terrain.longitude_deg_east - longitude_deg)))
            print(
                f"lat={terrain.latitude_deg[i]:.3f},lon={terrain.longitude_deg_east[j]:.3f},"
                f"elevation_m={terrain.elevation_grid_m[i, j]:.1f}"
            )
    finally:
        if context is not None:
            context.cleanup()


if __name__ == "__main__":
    main()
