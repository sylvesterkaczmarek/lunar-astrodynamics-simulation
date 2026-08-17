"""Download the official NASA Goddard LOLA MOON_PA global shape grid."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from lunar_astrodynamics import LOLA_MOON_PA_64_GRIDLINE_URL, LOLA_MOON_PA_64_PIXEL_URL


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size < 100_000_000:
            raise RuntimeError("downloaded file is unexpectedly small for the LOLA global grid")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download NASA Goddard's 2024 LOLA MOON_PA 64-pixels/degree global "
            "shape grid. The gridline file is about 703 MB and the pixel file "
            "about 702 MB, so no data are bundled in this repository."
        )
    )
    parser.add_argument(
        "--registration",
        choices=("gridline", "pixel"),
        default="gridline",
        help="Gridline registration includes the poles and 0/360 boundaries",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.registration == "gridline":
        url = LOLA_MOON_PA_64_GRIDLINE_URL
        default_name = "LDEM64_PA_gridline_202405.grd"
    else:
        url = LOLA_MOON_PA_64_PIXEL_URL
        default_name = "LDEM64_PA_pixel_202405.grd"
    destination = args.output or Path("data") / default_name
    print(f"Downloading {url}")
    download(url, destination)
    print(f"Saved {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
