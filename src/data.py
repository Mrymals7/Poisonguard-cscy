from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_raw(folder):
    """Read all day CSV files in `folder` and combine them into one DataFrame."""
    files = sorted(Path(folder).glob("*.csv"))

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df[df.select_dtypes("float64").columns] = df.select_dtypes("float64").astype("float32")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data.columns = data.columns.str.strip()

    return data

ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]

def clean(df):
        """Fix labels, drop identifiers, remove bad and duplicate rows."""
        

        # 1. Merge the two port-scan labels into one
        df["Label"] = df["Label"].replace("Port_scan", "PortScan")

        # 2. Keep the attack type, and add a binary label (0 = benign, 1 = attack)
        df["attack_type"] = df["Label"]
        df["label"] = (df["Label"] != "BENIGN").astype(int)

        # 3. Drop identifier columns and the original Label
        df = df.drop(columns=ID_COLUMNS + ["Label"])

        # 4. Turn infinite values into missing values, then drop those rows
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # 5. Remove duplicate rows
        df = df.drop_duplicates()

        return df

def sample(df, n=200_000, seed=42, min_count=3000):
    """Take about n rows: keep all rows of rare attack types, sample the rest evenly."""
    counts = df["attack_type"].value_counts()
    rare_types = counts[counts < min_count].index

    rare = df[df["attack_type"].isin(rare_types)]
    common = df[~df["attack_type"].isin(rare_types)]

    frac = (n - len(rare)) / len(common)
    common_sample = common.groupby("attack_type").sample(frac=frac, random_state=seed)

    result = pd.concat([common_sample, rare])
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    return result

def split(df, seed=42):
    """Split into 70% train, 15% validation, 15% test, keeping the same attack mix in each."""
    train, rest = train_test_split(
        df, test_size=0.30, stratify=df["attack_type"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest["attack_type"], random_state=seed
    )
    return train, val, test

def scale(train, val, test):
    """Scale features to 0-1 using the training set only, and clip val/test to that range."""
    features = [c for c in train.columns if c not in ("label", "attack_type")]

    scaler = MinMaxScaler()
    scaler.fit(train[features])

    scaled = []
    for part in (train, val, test):
        part = part.copy()
        part[features] = scaler.transform(part[features]).clip(0, 1)
        scaled.append(part)

    return scaled[0], scaled[1], scaled[2], scaler
def save(train, val, test, scaler, out_dir="data/processed"):
    """Save the three sets and the scaler so every teammate uses identical data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    joblib.dump(scaler, out / "scaler.joblib")

def prepare_data(raw_dir="data/raw/lanvin_exp_rp", out_dir="data/processed", n=200_000, seed=42):
    """Run the full pipeline: load, clean, sample, split, scale and save."""
    df = load_raw(raw_dir)
    df = clean(df)
    df = sample(df, n=n, seed=seed)
    train, val, test = split(df, seed=seed)
    train, val, test, scaler = scale(train, val, test)
    save(train, val, test, scaler, out_dir)
    print(f"Saved train {train.shape}, val {val.shape}, test {test.shape} to {out_dir}")


if __name__ == "__main__":
    prepare_data()