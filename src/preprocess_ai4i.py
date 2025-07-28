import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_ai4i(input_path="data/ai4i/ai4i2020.csv", output_path="data/processed/ai4i_processed.csv"):
    df = pd.read_csv(input_path)

    # Drop non-feature columns
    drop_cols = [
        "Product ID", "Type", "Machine failure",
        "TWF", "HDF", "PWF", "OSF", "RNF",
        "FailureType", "FailureTypeLabel", "RUL"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(features_scaled, columns=df.columns)

    # Save processed file
    os.makedirs("data/processed", exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned AI4I dataset to: {output_path}")

if __name__ == "__main__":
    preprocess_ai4i()
