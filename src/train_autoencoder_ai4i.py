import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=5, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def train_autoencoder_ai4i(
    file_path="data/processed/ai4i_processed.csv",
    model_save_path="outputs/models/autoencoder_ai4i.pt",
    latent_dim=5,
    batch_size=64,
    num_epochs=20,
    lr=1e-3
):
    print(f"📦 Loading: {file_path}")
    df = pd.read_csv(file_path)

    X = df.values.astype("float32")
    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=batch_size)

    model = Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"🚀 Training Autoencoder: {X.shape[1]} features → {latent_dim} latent units")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon, _ = model(batch)
                val_loss += criterion(recon, batch).item()

        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save model and latent codes
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/latents", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    with torch.no_grad():
        _, latents = model(torch.tensor(X))
        pd.DataFrame(latents.numpy()).to_csv("outputs/latents/ai4i_latents.csv", index=False)

    print("✅ Training complete. Latents saved to outputs/latents/ai4i_latents.csv")

if __name__ == "__main__":
    train_autoencoder_ai4i()
