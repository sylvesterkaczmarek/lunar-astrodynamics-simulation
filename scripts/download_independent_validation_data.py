"""Download public external data used by the independent validation campaign.

The repository intentionally does not commit the large NASA/PDS/NAIF products.
This script downloads them verbatim, records byte sizes and SHA-256 digests, and
writes a manifest beside the data so a validation result can identify the exact
inputs that were used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

FILES = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de421.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp",
    "moon_pa_de421_1900-2050.bpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf",
    "gggrx_0900c_sha.tab": "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_0900c_sha.tab",
    "gggrx_0900c_sha.lbl": "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_0900c_sha.lbl",
    "gggrx_1200b_sha.tab": "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200b_sha.tab",
    "gggrx_1200b_sha.lbl": "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200b_sha.lbl",
    "LRO_ES_36_GRGM900C_L600.BSP": "https://imbrium.mit.edu/LRORS/DATA/SPK/LRO_ES_36_GRGM900C_L600.BSP",
    "LRO_ES_36_GRGM900C_L600.LBL": "https://imbrium.mit.edu/LRORS/DATA/SPK/LRO_ES_36_GRGM900C_L600.LBL",
    "LDEM_4.IMG": "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.IMG",
    "LDEM_4.LBL": "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.LBL",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    print(f"Saved {destination} ({destination.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="omit GRGM1200B; keeps the LRO/GRGM900C and LOLA validation set",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = dict(FILES)
    if args.minimal:
        selected.pop("gggrx_1200b_sha.tab")
        selected.pop("gggrx_1200b_sha.lbl")

    records: list[dict[str, object]] = []
    for filename, url in selected.items():
        destination = args.output_dir / filename
        if not destination.exists() or args.overwrite:
            download(url, destination)
        else:
            print(f"Keeping existing {destination} ({destination.stat().st_size} bytes)")
        records.append(
            {
                "filename": filename,
                "source_url": url,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    manifest = {
        "purpose": "external data manifest for independent lunar astrodynamics validation",
        "files": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
