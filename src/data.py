from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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