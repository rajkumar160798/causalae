import pandas as pd
import shap
import torch
from train_autoencoder import Autoencoder
import os

def explain_latents(
    data_path="data/processed/ess_period_1.csv",
    model_path="outputs/models/autoencoder.pt",
    latent_dim=10,
    max_samples=1000
):
    print("Loading data...")
    df = pd.read_csv(data_path)
    feature_names = df.columns[1:]  # skip Timestamp
    X = df.drop(columns=["Timestamp"]).values.astype("float32")

    if len(X) > max_samples:
        X = X[:max_samples]

    print(f"Input shape: {X.shape}, Features: {len(feature_names)}")

    # Load model
    model = Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    explanations = []
    explainer = shap.Explainer(
        lambda x: model.encoder(torch.tensor(x).float()).detach().numpy(),
        X
    )

    print("Computing SHAP values...")
    shap_values = explainer(X)

    os.makedirs("outputs/shap", exist_ok=True)

    for i in range(latent_dim):
        # Get SHAP values for latent i: shape (samples, features)
        sv = shap_values.values[:, :, i]
        df_sv = pd.DataFrame(sv, columns=feature_names)
        df_sv_mean = df_sv.abs().mean().sort_values(ascending=False)
        df_sv_mean.reset_index().to_csv(
            f"outputs/shap/latent_{i}_shap.csv",
            index=False,
            header=["feature", "SHAP_Importance"]
        )
        print(f"Saved SHAP importances for latent {i} → outputs/shap/latent_{i}_shap.csv")

    print("✅ SHAP attribution complete for all latent dimensions.")

if __name__ == "__main__":
    explain_latents()
