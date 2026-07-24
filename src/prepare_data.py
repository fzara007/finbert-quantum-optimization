import pandas as pd
from sklearn.model_selection import train_test_split
import os

SEED = 42
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def load_primary_csv(path):
    """Load all-data.csv: format is label,"text" per line."""
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Split on first comma only
            parts = line.split(",", 1)
            label = parts[0].strip().lower()
            text = parts[1].strip().strip('"')
            rows.append({"id": i, "text": text, "label": label})
    return pd.DataFrame(rows)


def load_allagree(path):
    """Load Sentences_AllAgree.txt: format is text@label per line."""
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Split on last @ since text might contain @
            idx = line.rfind("@")
            text = line[:idx].strip()
            label = line[idx + 1:].strip().lower()
            rows.append({"id": i, "text": text, "label": label})
    return pd.DataFrame(rows)


def split_and_save(df, output_dir, track_name):
    """Stratified 70/15/15 split, map labels, save to CSV."""
    # Map string labels to integers
    df["label_id"] = df["label"].map(LABEL_MAP)

    # Verify no unmapped labels
    unmapped = df["label_id"].isna().sum()
    if unmapped > 0:
        print(f"WARNING: {unmapped} rows with unmapped labels in {track_name}")
        print(df[df["label_id"].isna()]["label"].unique())
        return

    df["label_id"] = df["label_id"].astype(int)

    # First split: 70% train, 30% temp
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=SEED
    )

    # Second split: 50/50 of the 30% = 15%/15%
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label_id"], random_state=SEED
    )

    # Save
    track_dir = os.path.join(output_dir, track_name)
    os.makedirs(track_dir, exist_ok=True)

    train.to_csv(os.path.join(track_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(track_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(track_dir, "test.csv"), index=False)

    # Print summary
    print(f"\n=== {track_name} ===")
    print(f"Total: {len(df)}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"\nLabel distribution (full):")
    print(df["label"].value_counts())
    print(f"\nLabel distribution (train):")
    print(train["label"].value_counts())


if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    # Track 1: Primary CSV
    print("Loading primary CSV...")
    df_primary = load_primary_csv(os.path.join(raw_dir, "all-data.csv"))
    split_and_save(df_primary, processed_dir, "primary")

    # Track 2: Financial PhraseBank AllAgree
    print("\nLoading Financial PhraseBank AllAgree...")
    df_allagree = load_allagree(
        os.path.join(raw_dir, "FinancialPhraseBank-v1.0", "Sentences_AllAgree.txt")
    )
    split_and_save(df_allagree, processed_dir, "allagree")

    print("\nDone. Files saved to data/processed/")