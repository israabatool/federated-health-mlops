# src/run_preprocessing.py
from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.preprocessing import load_and_preprocess

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
SHUFFLE = True

def main():
    csv_files = sorted(DATA_DIR.glob("node_*.csv"))
    if not csv_files:
        print("No node CSV files found in src/data/")
        return

    for file in csv_files:
        print(f"\nProcessing {file.name}...")
        ds, mean, std = load_and_preprocess(file, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        
        # Optional: save normalized dataset to CSV for verification
        df = pd.read_csv(file).dropna()
        features = df[["heart_rate","steps","pm25","temp","humidity"]].astype(float)
        features = (features - pd.Series(mean)) / pd.Series(std)
        df_normalized = features.copy()
        df_normalized["risk"] = df["risk"].astype(int)
        out_path = OUTPUT_DIR / file.name
        df_normalized.to_csv(out_path, index=False)
        print(f"Saved normalized data to {out_path}")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")

    print("\nAll node datasets processed successfully.")

if __name__ == "__main__":
    main()

