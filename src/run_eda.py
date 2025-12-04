import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent / "data" / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# List all node CSVs
csv_files = list(DATA_DIR.glob("node_*.csv"))

for csv_file in csv_files:
    node_name = csv_file.stem
    print(f"Processing {node_name}...")

    # Load data
    df = pd.read_csv(csv_file)

    # Summary statistics
    summary = df.describe()
    summary.to_csv(OUTPUT_DIR / f"{node_name}_summary.csv")
    print(f"Saved summary stats to {node_name}_summary.csv")

    # Histograms for all numeric features
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{node_name} - {col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{node_name}_{col}_hist.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"{node_name} - Feature Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{node_name}_correlation.png")
    plt.close()

    print(f"EDA plots saved for {node_name}")

print("All nodes processed successfully.")

