import os
import pandas as pd
import streamlit as st

from src.train_autoencoder_ai4i import train_autoencoder_ai4i
from src.train_autoencoder import train_autoencoder
from src.shap_explainer import explain_latents
from src.shap_visualizer import generate_all_shap_plots
from causal_analysis import (
    compute_granger_causality,
    compute_dowhy_effects,
    counterfactual_latent_shift,
    run_causal_discovery,
)

st.set_page_config(page_title="CausalAE++ Explorer", layout="wide")

if "data_path" not in st.session_state:
    st.session_state.data_path = None
if "dataset_type" not in st.session_state:
    st.session_state.dataset_type = "ai4i"
if "latent_dim" not in st.session_state:
    st.session_state.latent_dim = 5

st.title("CausalAE++ Pipeline")

upload_tab, train_tab, shap_tab, causal_tab, discovery_tab, export_tab = st.tabs(
    [
        "Upload Data",
        "Train Autoencoder",
        "SHAP Explainability",
        "Causal Analysis",
        "Causal Discovery",
        "Export Results",
    ]
)

with upload_tab:
    st.header("Upload Dataset")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    dataset_type = st.selectbox("Dataset type", ["ai4i", "ess"], index=0)
    if uploaded is not None:
        os.makedirs("data/processed", exist_ok=True)
        path = os.path.join("data/processed", "uploaded.csv")
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.data_path = path
        st.session_state.dataset_type = dataset_type
        df = pd.read_csv(path)
        st.write("Data preview:", df.head())

with train_tab:
    st.header("Train Autoencoder")
    latent_dim = st.number_input("Latent dimension", 1, 100, st.session_state.latent_dim)
    epochs = st.number_input("Epochs", 1, 100, 5)
    if st.button("Train", key="train"):
        st.session_state.latent_dim = latent_dim
        if st.session_state.dataset_type == "ai4i":
            train_autoencoder_ai4i(
                file_path=st.session_state.data_path,
                latent_dim=latent_dim,
                num_epochs=int(epochs),
            )
        else:
            train_autoencoder(
                file_path=st.session_state.data_path,
                latent_dim=latent_dim,
                num_epochs=int(epochs),
            )
        st.success("Training complete")

with shap_tab:
    st.header("SHAP Explainability")
    if st.button("Generate SHAP", key="shap"):
        explain_latents(
            data_path=st.session_state.data_path,
            latent_dim=st.session_state.latent_dim,
            shap_prefix=st.session_state.dataset_type,
        )
        generate_all_shap_plots(shap_prefix=st.session_state.dataset_type, latent_dim=st.session_state.latent_dim)
        st.success("SHAP results saved to outputs/shap/")
        for i in range(st.session_state.latent_dim):
            img_path = f"outputs/shap/plots/latent_{i}_shap_{st.session_state.dataset_type}_plot.png"
            if os.path.exists(img_path):
                st.image(img_path)

with causal_tab:
    st.header("Causal Analysis")
    if st.button("Run Granger"):
        compute_granger_causality(data_path=st.session_state.data_path)
        st.success("Granger causality computed")
    if st.button("Run DoWhy"):
        compute_dowhy_effects(data_path=st.session_state.data_path)
        st.success("DoWhy estimates computed")
    if st.button("Counterfactual Latents"):
        counterfactual_latent_shift(data_path=st.session_state.data_path, latent_dim=st.session_state.latent_dim)
        st.success("Counterfactual simulation done")

with discovery_tab:
    st.header("Causal Discovery")
    method = st.selectbox("Method", ["pc", "fci", "notears"])
    if st.button("Run Discovery"):
        run_causal_discovery(data_path=st.session_state.data_path, method=method)
        img = os.path.join("outputs/discovery", f"graph_{method}.png")
        if os.path.exists(img):
            st.image(img)

with export_tab:
    st.header("Export Results")
    for root, _, files in os.walk("outputs"):
        for f in files:
            file_path = os.path.join(root, f)
            with open(file_path, "rb") as fp:
                st.download_button(f"Download {f}", fp, file_name=f)
