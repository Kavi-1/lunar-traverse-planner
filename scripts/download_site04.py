"""Download the two milestone-1 PGDA rasters and retain local SHA-256 hashes.

Source: https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README
Run from the repository root: python scripts/download_site04.py
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04"
FILENAMES = (
    "Site04_final_adj_5mpp_surf.tif",
    "Site04_final_adj_5mpp_slp.tif",
)


def download_site04(output_dir: Path, timeout_seconds: float = 60.0) -> None:
    """Fetch original rasters; timeout_seconds is the network operation timeout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        destination = output_dir / filename
        checksum_path = output_dir / f"{filename}.sha256"
        if destination.exists():
            if not checksum_path.exists():
                raise RuntimeError(f"Existing file has no recorded hash: {destination}")
            with destination.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum_path.read_text().split()[0] != actual_hash:
                raise RuntimeError(f"Hash mismatch: {destination}; inspect before redownloading")
            print(f"Verified existing {filename}: {actual_hash}", flush=True)
            continue

        url = f"{BASE_URL}/{filename}"
        temporary_path = destination.with_suffix(".tif.part")
        digest = hashlib.sha256()
        print(f"Downloading {url}", flush=True)
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                expected_bytes = response.headers.get("Content-Length")
                received_bytes = 0
                with temporary_path.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                        received_bytes += len(chunk)
            if received_bytes == 0:
                raise RuntimeError(f"Empty download: {url}")
            if expected_bytes is not None and received_bytes != int(expected_bytes):
                raise RuntimeError(f"Incomplete download: {url}")
            with temporary_path.open("rb") as stream:
                magic = stream.read(4)
            if magic not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                raise RuntimeError(f"Response is not a TIFF: {url}")
            temporary_path.replace(destination)
            checksum_path.write_text(f"{digest.hexdigest()}  {filename}\n# {url}\n")
            print(f"Saved {filename}: {digest.hexdigest()}", flush=True)
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/Site04"))
    args = parser.parse_args()
    download_site04(args.output_dir)
