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
            # Check if feature and outcome are in df_combined
            if feature not in df_combined.columns or outcome not in df_combined.columns:
                print(f"[WARN] Skipping: {feature} or {outcome} not in dataframe columns.")
                continue
            # Use valid DOT graph string with quoted variable names
            graph = f'digraph {{ "{feature}" -> "{outcome}"; }}'
            try:
                model = CausalModel(
                    data=df_combined,
                    treatment=feature,
                    outcome=outcome,
                    graph=graph
                )
                identified = model.identify_effect()
                estimate = model.estimate_effect(
                    identified, method_name="backdoor.linear_regression"
                )
                effect_val = estimate.value
                try:
                    ci = estimate.get_confidence_intervals()
                    ci_lower, ci_upper = ci[0][0], ci[0][1]
                except Exception:
                    ci_lower, ci_upper = np.nan, np.nan
            except Exception as e:
                print(f"⚠️ DoWhy failed for {feature}->{outcome}: {e}")
                effect_val = np.nan
                ci_lower, ci_upper = np.nan, np.nan
            effects.append(
                {
                    "feature": feature,
                    "latent": outcome,
                    "effect": effect_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )
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


def run_causal_discovery(
    data_path="data/processed/ai4i_processed.csv",
    method="pc",
    output_dir="outputs/discovery",
):
    """Run causal discovery on a dataset using PC, FCI, or NOTEARS."""
    print(f"\n📈 Running causal discovery ({method})...")
    df = pd.read_csv(data_path)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    data = df.values
    features = df.columns.tolist()
    os.makedirs(output_dir, exist_ok=True)

    try:
        if method.lower() == "pc":
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz

            cg = pc(data, indep_test=fisherz)
            adj = (cg.G.graph != 0).astype(int)
        elif method.lower() == "fci":
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz

            cg = fci(data, indep_test=fisherz)
            adj = (cg.G.graph != 0).astype(int)
        elif method.lower() == "notears":
            from causalnex.structure.notears import from_pandas

            sm = from_pandas(df)
            import networkx as nx

            adj = nx.to_numpy_array(sm, nodelist=features)
        else:
            raise ValueError("method must be pc, fci, or notears")
    except Exception as e:
        print(f"⚠️ Causal discovery failed: {e}")
        return None

    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    G.add_nodes_from(features)
    for i, src in enumerate(features):
        for j, tgt in enumerate(features):
            if adj[i, j] != 0:
                G.add_edge(src, tgt)

    adj_df = pd.DataFrame(adj, index=features, columns=features)
    adj_path = os.path.join(output_dir, f"adjacency_{method}.csv")
    adj_df.to_csv(adj_path)

    try:
        from networkx.drawing.nx_pydot import write_dot

        write_dot(G, os.path.join(output_dir, f"graph_{method}.dot"))
    except Exception as e:
        print(f"[WARN] Failed to write DOT file: {e}")

    try:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, arrows=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"graph_{method}.png"))
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to save PNG: {e}")

    print(f"✅ Discovery outputs saved to {output_dir}")
    return G, adj_df

if __name__ == "__main__":
    compute_granger_causality()
    compute_dowhy_effects()
    counterfactual_latent_shift()
    run_causal_discovery()
