"""Download Site04 rasters, elevation clones, or kernels with local SHA-256 hashes.

Source: https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/README
Run from the repository root: python scripts/download_site04.py
"""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site04"
FILENAMES = (
    "Site04_final_adj_5mpp_surf.tif",
    "Site04_final_adj_5mpp_slp.tif",
)
NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
KERNEL_URLS = (
    f"{NAIF}/lsk/naif0012.tls",
    f"{NAIF}/spk/planets/a_old_versions/de421.bsp",
    f"{NAIF}/pck/moon_pa_de421_1900-2050.bpc",
    f"{NAIF}/fk/satellites/moon_080317.tf",
)


def download_site04(
    output_dir: Path, timeout_seconds: float = 60.0, *, kernels_only: bool = False,
    clones_only: bool = False, workers: int = 1,
) -> None:
    """Fetch rasters or pinned NAIF kernels; network timeout is in seconds."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if kernels_only and clones_only:
        raise ValueError("Choose kernels or clones, not both")
    if clones_only:
        output_dir = output_dir / "Clones"
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = KERNEL_URLS if kernels_only else tuple(f"{BASE_URL}/{f}" for f in FILENAMES)
    if clones_only:
        urls = tuple(f"{BASE_URL}/Clones/Site04_final_adj_5mpp_{i:04d}_err.tif"
                     for i in range(1, 101))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(partial(download_file, output_dir=output_dir,
                                  timeout_seconds=timeout_seconds, kernels_only=kernels_only), urls))


def download_file(url: str, output_dir: Path, timeout_seconds: float,
                  kernels_only: bool) -> None:
    """Download one URL atomically; timeout in seconds, hash verified on reuse."""
    filename = url.rsplit("/", 1)[1]
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
        return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
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
            magic = stream.read(8)
        if kernels_only:
            if not magic.startswith((b"KPL/LSK", b"KPL/FK", b"DAF/SPK", b"DAF/PCK")):
                raise RuntimeError(f"Response is not a SPICE kernel: {url}")
        elif magic[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise RuntimeError(f"Response is not a TIFF: {url}")
        temporary_path.replace(destination)
        checksum_path.write_text(f"{digest.hexdigest()}  {filename}\n# {url}\n")
        print(f"Saved {filename}: {digest.hexdigest()}", flush=True)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/Site04"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--kernels-only", action="store_true")
    selection.add_argument("--clones-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent downloads (default: 1)")
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/kernels"))
    args = parser.parse_args()
    download_site04(args.kernel_dir if args.kernels_only else args.output_dir,
                    kernels_only=args.kernels_only, clones_only=args.clones_only, workers=args.workers)
