"""Download the official NASA PDS GRGM1200A SHADR product."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from lunar_astrodynamics import GRGM1200A, read_shadr


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gggrx_1200a_sha.tab"),
        help="Destination for the 88 MB coefficient table",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        default=Path("data/gggrx_1200a_sha.lbl"),
    )
    args = parser.parse_args()

    download(GRGM1200A.source_url, args.output)
    download(GRGM1200A.label_url, args.label_output)

    size = args.output.stat().st_size
    if size != GRGM1200A.expected_size_bytes:
        raise RuntimeError(
            f"unexpected GRGM1200A size: {size} bytes; expected "
            f"{GRGM1200A.expected_size_bytes}"
        )

    model = read_shadr(args.output, max_degree=2, name=GRGM1200A.name, frame=GRGM1200A.body_fixed_frame)
    if abs(model.mu_m3_s2 - GRGM1200A.mu_m3_s2) > 1.0:
        raise RuntimeError("downloaded file GM does not match archived GRGM1200A metadata")
    if abs(model.reference_radius_m - GRGM1200A.reference_radius_m) > 1e-6:
        raise RuntimeError("downloaded file reference radius does not match GRGM1200A metadata")

    print(f"Downloaded {GRGM1200A.product_id} to {args.output}")
    print(f"Verified size: {size} bytes")
    print(f"GM: {model.mu_m3_s2:.8f} m^3/s^2")
    print(f"Reference radius: {model.reference_radius_m:.1f} m")


if __name__ == "__main__":
    main()
