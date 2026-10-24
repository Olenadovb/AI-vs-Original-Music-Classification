"""Prepare reproducible, prompt-aware FakeMusicCaps train/validation/test splits.

Place this script at scripts/prepare_splits.py in the project root.

Examples (from the project root):
    python scripts/prepare_splits.py --mode both
    python scripts/prepare_splits.py --mode full
    python scripts/prepare_splits.py --mode pilot
    python scripts/prepare_splits.py --mode both --resplit

The full dataset is always written to data/interim/split_metadata.csv.
Pilot metadata is additionally written for --mode pilot or --mode both.
An existing full split is reused rather than silently changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/audio_analysis.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim"
SPLIT_NAMES = ("train", "validation", "test")
PILOT_IDS_PER_SPLIT = {"train": 420, "validation": 90, "test": 90}


def normalize_caption(value: str) -> str:
    """Catch captions differing only in case or whitespace."""
    return " ".join(str(value).casefold().split())


def check_source(df: pd.DataFrame) -> pd.DataFrame:
    required = {"track_id", "generator", "caption", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    df = df.copy()
    if df[list(required)].isna().any().any():
        raise ValueError("Null values in required metadata columns")
    df["track_id"] = df["track_id"].astype(str)
    if df.duplicated(["track_id", "generator"]).any():
        raise ValueError("Duplicated (track_id, generator) entries")
    # All five versions of an ID should exist in this dataset.
    expected_generators = set(df["generator"].unique())
    coverage = df.groupby("track_id")["generator"].agg(lambda s: set(s))
    if not coverage.map(lambda s: s == expected_generators).all():
        raise ValueError("Some track IDs lack one or more generators. Inspect dataset completeness.")
    captions_per_track = df.groupby("track_id")["caption"].agg(
        lambda s: s.map(normalize_caption).nunique()
    )
    if not captions_per_track.eq(1).all():
        raise ValueError("A track_id has different prompts for its generator versions")
    return df


def create_group_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Group by track ID, or by normalized caption when multiple IDs share a caption.

    For datasets with unique captions per track ID this preserves the *same*
    split logic as the original script: shuffled 70%/15%/15% unique track IDs.
    """
    track_captions = df.drop_duplicates("track_id")[["track_id", "caption"]].copy()
    track_captions["caption_key"] = track_captions["caption"].map(normalize_caption)
    captions_reused = track_captions["caption_key"].duplicated().any()

    if captions_reused:
        # Exact/whitespace-normalized duplicate prompts must not cross splits.
        print("Some captions are shared across track IDs: grouping by caption to avoid leakage.")
        group_keys = sorted(track_captions["caption_key"].unique())
        id_to_group = track_captions.set_index("track_id")["caption_key"]
    else:
        group_keys = sorted(track_captions["track_id"].unique())
        id_to_group = pd.Series(
            track_captions["track_id"].to_numpy(),
            index=track_captions["track_id"],
        )

    train_groups, temp_groups = train_test_split(
        group_keys, test_size=0.30, random_state=seed
    )
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=0.50, random_state=seed
    )
    group_to_split = {
        **{k: "train" for k in train_groups},
        **{k: "validation" for k in val_groups},
        **{k: "test" for k in test_groups},
    }

    result = df.copy()
    result["split"] = result["track_id"].map(id_to_group).map(group_to_split)
    return result


def validate_splits(df: pd.DataFrame) -> None:
    """Verify class coverage, track grouping, caption grouping and counts."""
    if df["split"].isna().any() or set(df["split"].unique()) != set(SPLIT_NAMES):
        raise ValueError("Expected nonempty train, validation and test labels")
    if df.groupby("track_id")["split"].nunique().max() != 1:
        raise ValueError("The same track ID appears in different splits")
    normalized_captions = df["caption"].map(normalize_caption)
    if df.assign(caption_key=normalized_captions).groupby("caption_key")["split"].nunique().max() != 1:
        raise ValueError("Caption leakage across splits; run again with --resplit")
    counts = pd.crosstab(df["split"], df["generator"])
    if not counts.nunique(axis=1).eq(1).all():
        raise ValueError("Generator classes are not balanced within each split")
    print("Full split validation passed.")
    print("Audio files by split and generator:\n", counts.to_string())
    print("Unique prompts (track IDs) by split:\n", df.groupby("split")["track_id"].nunique().to_string())


def reuse_previous_splits(source: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Reuse fixed splits but take all other metadata from the current input."""
    old = pd.read_csv(output_path, dtype={"track_id": str})
    for required in ("track_id", "generator", "caption", "split"):
        if required not in old.columns:
            raise ValueError(f"Existing split CSV lacks {required}; use --resplit")
    keys = ["track_id", "generator"]
    if old.duplicated(keys).any():
        raise ValueError("Duplicate IDs in saved split; use --resplit")
    aligned = source.merge(
        old[keys + ["split", "caption"]].rename(columns={"caption": "saved_caption"}),
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Source and saved split have different audio IDs; use --resplit")
    if not (aligned["caption"].map(normalize_caption) == aligned["saved_caption"].map(normalize_caption)).all():
        raise ValueError("Captions changed since the saved split; review and use --resplit")
    result = aligned.drop(columns=["_merge", "saved_caption"])
    return result[source.columns.tolist() + ["split"]]


def create_pilot_subset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Balanced 600-ID pilot with labels inherited from the *full* split."""
    rng = np.random.default_rng(seed)
    selected_ids: list[str] = []
    for split, n in PILOT_IDS_PER_SPLIT.items():
        candidates = np.sort(df.loc[df["split"].eq(split), "track_id"].unique())
        if len(candidates) < n:
            raise ValueError(f"Split {split} contains only {len(candidates)} IDs; need {n}")
        selected_ids.extend(rng.choice(candidates, size=n, replace=False).tolist())
    return df.loc[df["track_id"].isin(selected_ids)].copy().reset_index(drop=True)


def validate_pilot(full: pd.DataFrame, pilot: pd.DataFrame) -> None:
    validate_splits(pilot)
    if pilot["track_id"].nunique() != 600 or len(pilot) != 600 * full["generator"].nunique():
        raise ValueError("Unexpected number of samples in pilot")
    expected = pd.Series(PILOT_IDS_PER_SPLIT)
    observed = pilot.groupby("split")["track_id"].nunique().reindex(expected.index)
    if not observed.equals(expected):
        raise ValueError("Wrong pilot split sizes")
    key = ["track_id", "generator"]
    joined = pilot[key + ["split"]].merge(
        full[key + ["split"]], on=key, suffixes=("_pilot", "_full"),
        validate="one_to_one"
    )
    if len(joined) != len(pilot) or not joined["split_pilot"].eq(joined["split_full"]).all():
        raise ValueError("Pilot labels are inconsistent with the full split")
    print("Pilot validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "pilot", "both"], default="both",
                        help="What to prepare: split_metadata is always the full dataset")
    parser.add_argument("--resplit", action="store_true",
                        help="Rebuild full train/val/test labels instead of reusing the saved split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}; run 01-data-exploration first")
    source = check_source(pd.read_csv(args.input, dtype={"track_id": str}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "split_metadata.csv"

    if full_path.is_file() and not args.resplit:
        print("Reusing saved full split:", full_path)
        full = reuse_previous_splits(source, full_path)
    else:
        print("Building new full split with random seed", args.seed)
        full = create_group_split(source, seed=args.seed)
    validate_splits(full)
    full.to_csv(full_path, index=False)
    print(f"Saved FULL dataset: {len(full):,} records → {full_path}")

    if args.mode in ("pilot", "both"):
        pilot = create_pilot_subset(full, seed=args.seed)
        validate_pilot(full, pilot)
        pilot_path = args.output_dir / "pilot_metadata.csv"
        pilot.to_csv(pilot_path, index=False)
        print(f"Saved PILOT dataset: {len(pilot):,} records → {pilot_path}")
    else:
        print("Full-only mode: existing pilot_metadata.csv (if any) was left unchanged.")


if __name__ == "__main__":
    main()
