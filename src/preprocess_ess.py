import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_ess_period(period_path: str) -> pd.DataFrame:
    """
    Loads all time-series CSV files recursively from an ESS period folder,
    aligns them on the 'Timestamp' column, and renames the 'Value' column
    in each file to the filename (sensor ID).
    """
    csv_files = glob.glob(os.path.join(period_path, "**/*.csv"), recursive=True)
    # csv_files = glob.glob(os.path.join(period_path, "**/*.csv"), recursive=True)
    csv_files = csv_files[:50] # Limit to first 100 files for performance
    print(f"Found {len(csv_files)} CSV files in {period_path}")
    # If no CSV files found, raise an error 
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {period_path}")

    merged_df = None

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "Timestamp" not in df.columns:
                continue  # Skip invalid files

            sensor_name = os.path.splitext(os.path.basename(file))[0]

            # Rename 'Value' column to sensor name
            if "Value" in df.columns:
                df = df[["Timestamp", "Value"]].rename(columns={"Value": sensor_name})
            else:
                value_col = df.columns[1]
                df = df[["Timestamp", value_col]].rename(columns={value_col: sensor_name})

            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df = df.drop_duplicates(subset="Timestamp")
            df = df.set_index("Timestamp")

            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how="outer")

        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")

    return merged_df.reset_index()


def preprocess_ess(period: str = "Period_1"):
    base_dir = os.path.join("data", "ess", "accp_dataset")
    full_path = os.path.join(base_dir, period)

    print(f"🔍 Loading time-series from: {full_path}")
    df = load_ess_period(full_path)

    if df is None or df.empty:
        print("❌ No data found after merging.")
        return

    print(f"✅ Loaded {df.shape[0]} timestamps with {df.shape[1] - 1} sensors.")

    df_clean = df.dropna(how="all", subset=df.columns[1:])
    df_clean = df_clean.fillna(method="ffill").fillna(method="bfill")



    # Only keep the first 100,000 rows for testing
    df_clean = df_clean.head(100000)

    features = df_clean.columns[1:]
    scaler = StandardScaler()
    df_clean[features] = scaler.fit_transform(df_clean[features])

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ess_{period.lower()}.csv")
    df_clean.to_csv(output_path, index=False)

    print(f"📁 Saved preprocessed file to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess ESS dataset period.")
    parser.add_argument(
        "--period",
        type=str,
        default="Period_1",
        help="Which ESS period to process (e.g., Period_1, Period_2, etc.)"
    )
    args = parser.parse_args()
    preprocess_ess(args.period)
