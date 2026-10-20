from pathlib import Path
import pandas as pd
import numpy as np

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