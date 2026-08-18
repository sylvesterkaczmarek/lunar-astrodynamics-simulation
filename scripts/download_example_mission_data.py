"""Download the official public data used by the complete example mission."""

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
    "LDEM_4.IMG": "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.IMG",
    "LDEM_4.LBL": "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.LBL",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/example_mission"), help="destination for official kernels, gravity and terrain products")
    parser.add_argument("--force", action="store_true", help="redownload files that already exist")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in FILES.items():
        destination = args.output_dir / filename
        if args.force or not destination.exists():
            print(f"Downloading {url}")
            with urlopen(url) as response, destination.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
        record = {"filename": filename, "source_url": url, "size_bytes": destination.stat().st_size, "sha256": _sha256(destination)}
        records.append(record)
        print(f"{filename}: {record['size_bytes']} bytes")

    manifest = {"purpose": "official public data for examples/missions/polar_high_fidelity.toml", "files": records}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
