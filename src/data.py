from pathlib import Path
import pandas as pd


def load_raw(folder):
    """Read all day CSV files in `folder` and combine them into one DataFrame."""
    files = sorted(Path(folder).glob("*.csv"))

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data.columns = data.columns.str.strip()

    return data