import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_shap_plots(shap_prefix="ess", shap_dir="outputs/shap", plot_dir="outputs/shap/plots", latent_dim=10, top_n=10):
    """
    Generate SHAP summary plots for all latent dimensions.
    """
    os.makedirs(plot_dir, exist_ok=True)
    for i in range(latent_dim):
        shap_file = os.path.join(shap_dir, f"latent_{i}_shap_{shap_prefix}.csv")
        if not os.path.exists(shap_file):
            print(f"[WARN] SHAP file not found: {shap_file}")
            continue
        df = pd.read_csv(shap_file)
        # Sort by SHAP importance if not already
        df = df.sort_values("SHAP_Importance", ascending=False)
        # Plot top N features
        plt.figure(figsize=(8, 5))
        sns.barplot(x="SHAP_Importance", y="feature", data=df.head(top_n), palette="viridis")
        plt.title(f"Top {top_n} SHAP Importances for Latent {i} ({shap_prefix})")
        plt.xlabel("Mean(|SHAP value|)")
        plt.ylabel("Feature")
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"latent_{i}_shap_{shap_prefix}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Saved plot: {plot_path}")
    print("✅ All SHAP plots generated.")
