
"""Prepare group-aware train, validation, and test splits."""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "audio_analysis.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "split_metadata.csv"


def create_group_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign splits so each track ID belongs to only one split."""
    track_ids = sorted(df["track_id"].unique())
    train_ids, temp_ids = train_test_split(track_ids, test_size=0.30, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)
    split_map = {
        **{track_id: "train" for track_id in train_ids},
        **{track_id: "validation" for track_id in val_ids},
        **{track_id: "test" for track_id in test_ids},
    }

    result = df.copy()
    result["split"] = result["track_id"].map(split_map)
    return result


def validate_splits(df: pd.DataFrame) -> None:
    """Check split assignments and class balance."""
    assert df["split"].notna().all()
    track_split_counts = df.groupby("track_id")["split"].nunique()
    assert (track_split_counts == 1).all(), ("Some track IDs appear in multiple splits.")
    class_counts = pd.crosstab(df["split"], df["generator"])
    assert class_counts.nunique(axis=1).eq(1).all(), ("Generator classes are not balanced.")
    caption_splits = df.groupby("caption")["split"].nunique()
    assert (caption_splits == 1).all(), ("Caption leakage detected between splits.")
    print("Split validation passed.")
    print("\nFiles per generator and split:")
    print(class_counts)


def create_pilot_subset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a balanced pilot subset while preserving split labels."""
    tracks_per_split = {"train": 420, "validation": 90, "test": 90}
    rng = np.random.default_rng(seed)
    selected_ids = []

    for split, n in tracks_per_split.items():
        available_ids = (df.loc[df["split"] == split, "track_id"].drop_duplicates().sort_values().to_numpy())
        chosen = rng.choice(available_ids, size=n, replace=False)
        selected_ids.extend(chosen)
    pilot_df = df[df["track_id"].isin(selected_ids)].copy()
    return pilot_df.reset_index(drop=True)


def validate_pilot(full_df: pd.DataFrame, pilot_df: pd.DataFrame) -> None:
    """Verify pilot balance and split consistency."""
    assert len(pilot_df) == 3000
    assert pilot_df["track_id"].nunique() == 600
    assert (pilot_df.groupby("track_id")["split"].nunique() == 1).all()

    counts = pd.crosstab(pilot_df["split"], pilot_df["generator"])
    assert counts.nunique(axis=1).eq(1).all()

    original_splits = full_df.set_index(["track_id", "generator"])["split"]
    pilot_splits = pilot_df.set_index(["track_id", "generator"])["split"]
    assert pilot_splits.equals(original_splits.loc[pilot_splits.index])
    assert not pilot_df.duplicated(subset=["track_id", "generator"]).any()
    n_generators = full_df["generator"].nunique()
    assert (pilot_df.groupby("track_id")["generator"].nunique() == n_generators).all()
    print("Pilot validation passed.")
    print("\nPilot files per generator and split:")
    print(counts)




def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input metadata not found: {INPUT_PATH}. Run the data exploration notebook first.")
    df = pd.read_csv(INPUT_PATH, dtype={"track_id": str})
    split_df = create_group_split(df)
    validate_splits(split_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(OUTPUT_PATH, index=False)
    pilot_df = create_pilot_subset(split_df)
    print(f"\nSaved {len(split_df)} records to {OUTPUT_PATH}")

    validate_pilot(split_df, pilot_df)
    pilot_path = PROJECT_ROOT / "data" / "interim" / "pilot_metadata.csv"
    pilot_df.to_csv(pilot_path, index=False)
    print(f"Saved {len(pilot_df)} pilot records to {pilot_path}")


if __name__ == "__main__":
    main()
