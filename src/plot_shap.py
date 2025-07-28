import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_shap_for_latent(latent_id, shap_dir="outputs/shap", top_k=10):
    filepath = os.path.join(shap_dir, f"latent_{latent_id}_shap.csv")
    df = pd.read_csv(filepath)
    df_top = df.sort_values("SHAP_Importance", ascending=False).head(top_k)

    plt.figure(figsize=(8, 5))
    plt.barh(df_top["feature"], df_top["SHAP_Importance"], edgecolor="black")
    plt.xlabel("SHAP Importance")
    plt.title(f"Latent {latent_id} - Top {top_k} Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = os.path.join(shap_dir, f"latent_{latent_id}_shap_plot.png")
    plt.savefig(output_path)
    print(f"📊 Saved: {output_path}")

if __name__ == "__main__":
    for i in range(10):  # latent_0 to latent_9
        plot_shap_for_latent(i)
        print(f"✅ Plotted SHAP values for latent {i}")
    print("All SHAP plots generated successfully.")