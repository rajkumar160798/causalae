import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set paths
shap_dir = "outputs/shap"
output_dir = "outputs/plots"
os.makedirs(output_dir, exist_ok=True)

# Constants
LATENT_DIM = 10
shap_files = [f"{shap_dir}/latent_{i}_shap.csv" for i in range(LATENT_DIM)]

# 1. Latent-Feature SHAP Heatmap
def plot_latent_shap_heatmap():
    print("🔹 Generating SHAP heatmap...")

    heatmap_data = []
    features = None

    for i, file in enumerate(shap_files):
        df = pd.read_csv(file)
        if features is None:
            features = df['feature'].tolist()
        heatmap_data.append(df['SHAP_Importance'].tolist())

    df_heatmap = pd.DataFrame(heatmap_data, columns=features, index=[f"Latent_{i}" for i in range(LATENT_DIM)])

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_heatmap, cmap="Blues", annot=False, cbar_kws={'label': 'Mean |SHAP| Value'})
    plt.title("Latent-Feature SHAP Attribution Heatmap (ESS Dataset)")
    plt.xlabel("Features")
    plt.ylabel("Latent Dimensions")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_shap_heatmap_ess.png")
    plt.close()
    print("✅ SHAP heatmap saved.")

# 2. Cumulative SHAP Contribution per Latent
def plot_cumulative_shap_contribution():
    print("🔹 Generating cumulative SHAP contribution plot...")

    plt.figure(figsize=(8, 6))

    for i, file in enumerate(shap_files):
        df = pd.read_csv(file)
        sorted_vals = df['SHAP_Importance'].sort_values(ascending=False).values
        cum_vals = np.cumsum(sorted_vals) / np.sum(sorted_vals)
        plt.plot(range(1, len(cum_vals)+1), cum_vals, label=f"Latent {i}")

    plt.xlabel("Top N Features")
    plt.ylabel("Cumulative SHAP Contribution")
    plt.title("Cumulative Feature Attribution per Latent (ESS)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_shap_contribution_ESS.png")
    plt.close()
    print("✅ Cumulative contribution plot saved.")

# 3. Feature Dominance Map (Top 3 Count)
def plot_feature_top3_dominance():
    print("🔹 Generating top-3 feature dominance plot...")

    top3_features = []

    for file in shap_files:
        df = pd.read_csv(file)
        top3 = df.sort_values("SHAP_Importance", ascending=False).head(3)['feature'].tolist()
        top3_features.extend(top3)

    counts = Counter(top3_features)
    df_counts = pd.DataFrame(counts.items(), columns=["Feature", "Top3_Count"]).sort_values("Top3_Count", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_counts, x="Feature", y="Top3_Count", palette="viridis")
    plt.title("Feature Frequency in Top-3 Latent SHAP Attributions (ESS)")
    plt.ylabel("Count (out of 10 latents)")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_top3_dominance_ESS.png")
    plt.close()
    print("✅ Top-3 feature dominance plot saved.")

# Execute all plots
if __name__ == "__main__":
    plot_latent_shap_heatmap()
    plot_cumulative_shap_contribution()
    plot_feature_top3_dominance()
    print("🎉 All SHAP visualizations for ESS completed.")
