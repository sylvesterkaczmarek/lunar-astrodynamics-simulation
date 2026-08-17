"""Download the NAIF kernels used by the MOON_ME_DE421 ground-track example."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

KERNELS = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de421.bsp": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
        "a_old_versions/de421.bsp"
    ),
    "moon_pa_de421_1900-2050.bpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
        "moon_pa_de421_1900-2050.bpc"
    ),
    "moon_080317.tf": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf"
    ),
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urlopen(url) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"Saved {destination} ({destination.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/spice/de421"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for filename, url in KERNELS.items():
        destination = args.output_dir / filename
        if destination.exists() and not args.overwrite:
            print(f"Keeping existing {destination} ({destination.stat().st_size} bytes)")
            continue
        download(url, destination)


if __name__ == "__main__":
    main()
