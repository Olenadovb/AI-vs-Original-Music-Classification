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

METADATA_PATH = RAW_DIR / "musiccaps-public.csv"
METADATA_URL = ("https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv")

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


def download_metadata() -> None:
    """Download MusicCaps metadata if it is not already available."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.exists():
        print("MusicCaps metadata already exists, skipping download.")
        return
    print(f"Downloading MusicCaps metadata to: {METADATA_PATH}")
    temporary_path = METADATA_PATH.with_suffix(".csv.part")

    try:
        with urlopen(METADATA_URL, timeout=120) as response:
            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        temporary_path.replace(METADATA_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print("MusicCaps metadata downloaded successfully.")


def main() -> None:
    download_dataset()
    download_metadata()


if __name__ == "__main__":
    main()
