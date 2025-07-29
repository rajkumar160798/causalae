import pandas as pd
import shap
import torch
from train_autoencoder_ai4i import Autoencoder
import os

def explain_latents(
    data_path="data/processed/ai4i_processed.csv",
    model_path="outputs/models/autoencoder_ai4i.pt",
    latent_dim=5,
    max_samples=1000,
    shap_prefix="ai4i",
    hidden_dim=32,
):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Drop Timestamp only if it exists
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    feature_names = df.columns
    X = df.values.astype("float32")

    if len(X) > max_samples:
        X = X[:max_samples]

    print(f"Input shape: {X.shape}, Features: {len(feature_names)}")

    # Load model with correct hidden_dim
    model = Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # SHAP Explainer
    explainer = shap.Explainer(
        lambda x: model.encoder(torch.tensor(x).float()).detach().numpy(),
        X
    )

    print("Computing SHAP values...")
    shap_values = explainer(X)

    os.makedirs("outputs/shap", exist_ok=True)

    for i in range(latent_dim):
        sv = shap_values.values[:, :, i]
        df_sv = pd.DataFrame(sv, columns=feature_names)
        df_sv_mean = df_sv.abs().mean().sort_values(ascending=False)
        df_sv_mean.reset_index().to_csv(
            f"outputs/shap/latent_{i}_shap_{shap_prefix}.csv",
            index=False,
            header=["feature", "SHAP_Importance"]
        )
        print(
            f"Saved SHAP importances for latent {i} → outputs/shap/latent_{i}_shap_{shap_prefix}.csv"
        )

    print("✅ SHAP attribution complete for all latent dimensions.")

if __name__ == "__main__":
    explain_latents()
