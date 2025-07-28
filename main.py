import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.preprocess_ess import preprocess_ess
from src.preprocess_ai4i import preprocess_ai4i
from src.train_autoencoder import train_autoencoder
from src.train_autoencoder_ai4i import train_autoencoder_ai4i
from src.shap_explainer import explain_latents
from src.shap_visualizer import generate_all_shap_plots


def main():
    parser = argparse.ArgumentParser(description="CausalAE++ Full Pipeline Runner")
    parser.add_argument("--dataset", choices=["ess", "ai4i"], required=True, help="Which dataset to process")
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent dimension for Autoencoder")
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of samples to explain with SHAP")
    args = parser.parse_args()

    if args.dataset == "ess":
        print("\n🔧 Preprocessing ESS dataset...")
        preprocess_ess("Period_1")
        data_path = "data/processed/ess_period_1.csv"
        model_path = "outputs/models/autoencoder.pt"
        latents_path = "outputs/latents/ess_latents.csv"
        shap_prefix = "ess"

    elif args.dataset == "ai4i":
        print("\n🔧 Preprocessing AI4I dataset...")
        preprocess_ai4i()  # Assumes function writes to ai4i_processed.csv
        data_path = "data/processed/ai4i_processed.csv"
        model_path = "outputs/models/autoencoder_ai4i.pt"
        latents_path = "outputs/latents/ai4i_latents.csv"
        shap_prefix = "ai4i"

    print("\n🧠 Training Autoencoder...")
    train_autoencoder(
        data_path=data_path,
        model_path=model_path,
        latent_dim=args.latent_dim,
        latent_save_path=latents_path
    )

    print("\n🔍 Running SHAP Explainability...")
    explain_latents(
        data_path=data_path,
        model_path=model_path,
        latent_dim=args.latent_dim,
        max_samples=args.max_samples,
        shap_prefix=shap_prefix
    )

    print("\n📊 Generating SHAP Visualizations...")
    generate_all_shap_plots(shap_prefix=shap_prefix)

    print("\n✅ Pipeline completed for dataset:", args.dataset)


if __name__ == "__main__":
    main()
