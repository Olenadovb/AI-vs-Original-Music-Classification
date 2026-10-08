#!/usr/bin/env python
"""Download and verify the FakeMusicCaps dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ARCHIVE_PATH = RAW_DIR / "FakeMusicCaps.zip"
DATASET_URL = ("https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1")
EXPECTED_MD5 = "db418dc95ab7dc378a55f29d6021fd66"

def calculate_md5(path: Path) -> str:
    """Calculate the MD5 checksum without loading the whole file."""
    md5 = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()

def download_dataset() -> None:
    """Download FakeMusicCaps if a verified archive is not available."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        if calculate_md5(ARCHIVE_PATH) == EXPECTED_MD5:
            print("Dataset archive already downloaded and verified.")
            return

        raise ValueError("Existing archive has an invalid checksum.")

    temporary_path = ARCHIVE_PATH.with_suffix(".zip.part")
    if temporary_path.exists():
        raise FileExistsError(f"Incomplete download found: {temporary_path}. ")
    print(f"Downloading to: {ARCHIVE_PATH}")

    try:
        with urlopen(DATASET_URL, timeout=120) as response:
            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        print("Verifying checksum...")

        if calculate_md5(temporary_path) != EXPECTED_MD5:
            raise ValueError("Downloaded archive failed MD5 verification.")
        temporary_path.rename(ARCHIVE_PATH)

    except Exception:
        print(f"Download interrupted. Check: {temporary_path}")
        raise
    print("Download completed successfully.")


if __name__ == "__main__":
    download_dataset()

# def sha256_of(path: Path) -> str:
#     return hashlib.sha256(path.read_bytes()).hexdigest()


# def read_recorded_checksum(checksums_file: Path, filename: str) -> str | None:
#     if not checksums_file.exists():
#         return None
#     for line in checksums_file.read_text().splitlines():
#         line = line.strip()
#         if not line or line.startswith("#"):
#             continue
#         digest, recorded_name = line.split(maxsplit=1)
#         if recorded_name == filename:
#             return digest
#     return None


# def write_checksum(checksums_file: Path, filename: str, digest: str) -> None:
#     checksums_file.parent.mkdir(parents=True, exist_ok=True)
#     checksums_file.write_text(
#         "# sha256 checksums for files in data/raw/, so reproducibility can be checked\n"
#         "# without the data itself being committed. Regenerate by rerunning this "
#         "script.\n"
#         f"{digest}  {filename}\n"
#     )


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument(
#         "--force", action="store_true", help="Re-materialize even if checksums already match"
#     )
#     args = parser.parse_args()
#     logging.basicConfig(level=logging.INFO, format="%(message)s")

#     recorded = read_recorded_checksum(CHECKSUMS_FILE, RAW_CSV.name)
#     if not args.force and RAW_CSV.exists() and recorded is not None:
#         if sha256_of(RAW_CSV) == recorded:
#             logger.info("%s already up to date (sha256 matches CHECKSUMS.txt), skipping", RAW_CSV)
#             return
#         logger.warning("%s exists but checksum doesn't match — re-materializing", RAW_CSV)

#     materialize_raw_csv(RAW_CSV)
#     digest = sha256_of(RAW_CSV)
#     write_checksum(CHECKSUMS_FILE, RAW_CSV.name, digest)
#     logger.info("Wrote %s (sha256 %s)", RAW_CSV, digest)


# if __name__ == "__main__":
#     main()
