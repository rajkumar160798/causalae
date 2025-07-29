import pandas as pd
import numpy as np
import torch
from statsmodels.tsa.stattools import grangercausalitytests
from dowhy import CausalModel
from src.train_autoencoder_ai4i import Autoencoder as AutoencoderAI4I
from src.train_autoencoder import Autoencoder as AutoencoderESS
import os


def compute_granger_causality(data_path="data/processed/ess_period_1.csv",
                              latents_path="outputs/latents/ess_latents.csv",
                              maxlag=3,
                              output_path="outputs/granger_results.csv"):
    """Compute Granger causality between each feature and latent dimension."""
    print("\n📈 Running Granger causality analysis...")
    df = pd.read_csv(data_path)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    latents = pd.read_csv(latents_path)

    results = []
    for feature in df.columns:
        feature_series = df[feature].values
        for latent_col in latents.columns:
            latent_series = latents[latent_col].values
            try:
                test_result = grangercausalitytests(
                    np.column_stack([latent_series, feature_series]),
                    maxlag=maxlag,
                    verbose=False
                )
                p_vals = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
                min_p = min(p_vals)
            except Exception as e:
                print(f"⚠️ Granger test failed for {feature}->{latent_col}: {e}")
                min_p = np.nan
            results.append({"feature": feature, "latent": latent_col, "p_value": min_p})
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"✅ Granger results saved to {output_path}")
    return res_df


def compute_dowhy_effects(data_path="data/processed/ai4i_processed.csv",
                          latents_path="outputs/latents/ai4i_latents.csv",
                          output_path="outputs/dowhy_effects.csv"):
    """Estimate causal effects of features on latents using DoWhy."""
    print("\n📈 Running DoWhy causal effect estimation...")
    df = pd.read_csv(data_path)
    latents = pd.read_csv(latents_path)
    df_combined = pd.concat([df, latents.add_prefix("latent_")], axis=1)

    effects = []
    for latent_col in latents.columns:
        outcome = f"latent_{latent_col}" if not latent_col.startswith("latent_") else latent_col
        for feature in df.columns:
            model = CausalModel(
                data=df_combined,
                treatment=feature,
                outcome=outcome,
                graph=f"{feature} -> {outcome};"
            )
            identified = model.identify_effect()
            try:
                estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
                effect_val = estimate.value
            except Exception as e:
                print(f"⚠️ DoWhy failed for {feature}->{latent_col}: {e}")
                effect_val = np.nan
            effects.append({"feature": feature, "latent": latent_col, "effect": effect_val})
    eff_df = pd.DataFrame(effects)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eff_df.to_csv(output_path, index=False)
    print(f"✅ DoWhy effects saved to {output_path}")
    return eff_df


def counterfactual_latent_shift(data_path="data/processed/ai4i_processed.csv",
                               model_path="outputs/models/autoencoder_ai4i.pt",
                               latent_dim=5,
                               output_path="outputs/counterfactual_latents.csv",
                               hidden_dim=32):
    """Simulate counterfactuals by perturbing each feature of the first sample."""
    print("\n📈 Running counterfactual simulation...")
    df = pd.read_csv(data_path)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    orig = torch.tensor(df.iloc[0].values.astype(np.float32))
    model = AutoencoderAI4I(input_dim=df.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        _, base_latent = model(orig)

    shifts = []
    for i, feature in enumerate(df.columns):
        perturbed = orig.clone()
        perturbed[i] += 1  # unit perturbation
        with torch.no_grad():
            _, latent_new = model(perturbed)
        diff = (latent_new - base_latent).numpy()
        for j in range(latent_dim):
            shifts.append({"feature": feature, "latent": j, "shift": diff[j]})

    shift_df = pd.DataFrame(shifts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shift_df.to_csv(output_path, index=False)
    print(f"✅ Counterfactual latent shifts saved to {output_path}")
    return shift_df

if __name__ == "__main__":
    compute_granger_causality()
    compute_dowhy_effects()
    counterfactual_latent_shift()
