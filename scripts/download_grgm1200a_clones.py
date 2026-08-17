"""Download selected official NASA PDS GRGM1200A clone gravity fields."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from lunar_astrodynamics import (
    GRGM1200A_CLONE_EXPECTED_SIZE_BYTES,
    grgm1200a_clone_url,
    read_grgm1200a_clone,
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected covariance-derived GRGM1200A clone fields. "
            "The archive contains 500 large files, so indices must be explicit."
        )
    )
    parser.add_argument(
        "indices",
        type=int,
        nargs="+",
        help="Clone indices within 1..500, for example: 1 2 3 10 50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/grgm1200a_clones"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep an existing file only if its archived byte size is correct",
    )
    args = parser.parse_args()

    indices = list(dict.fromkeys(args.indices))
    for index in indices:
        url = grgm1200a_clone_url(index)
        filename = Path(urlparse(url).path).name
        destination = args.output_dir / filename

        if args.skip_existing and destination.exists():
            if destination.stat().st_size == GRGM1200A_CLONE_EXPECTED_SIZE_BYTES:
                print(f"Keeping verified-size existing clone {index}: {destination}")
                continue
            raise RuntimeError(
                f"existing clone {destination} has unexpected size; "
                "remove it or omit --skip-existing"
            )

        print(f"Downloading clone {index} from {url}")
        download(url, destination)
        size = destination.stat().st_size
        if size != GRGM1200A_CLONE_EXPECTED_SIZE_BYTES:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"unexpected clone size: {size} bytes; expected "
                f"{GRGM1200A_CLONE_EXPECTED_SIZE_BYTES}"
            )

        model = read_grgm1200a_clone(destination, max_degree=2, name=filename)
        print(
            f"Verified clone {index}: {size} bytes, "
            f"GM={model.mu_m3_s2:.8f} m^3/s^2, degree smoke={model.max_degree}"
        )


if __name__ == "__main__":
    main()
