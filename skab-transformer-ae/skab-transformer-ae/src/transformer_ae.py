"""Transformer autoencoder for multivariate time-series anomaly detection.

Two decoder variants are provided because the initial notebook and the written
report describe different decoders:

* "original": the decoder target is the projected input window and the encoder
  output is the memory. This is what the initial notebook implements. The
  decoder can read the input directly, so there is no hard bottleneck.
* "latent": the decoder target is the encoder output itself, as in the
  equation of the report (TransformerDecoder(z, z)).

Positional encoding is disabled by default to match the initial notebook. Without
it, self-attention treats a window as an unordered set of time steps. It can be
enabled with pos_enc=True.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerAE(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, variant="original", pos_enc=False):
        super().__init__()
        if variant not in ("original", "latent"):
            raise ValueError("variant must be 'original' or 'latent'")
        self.variant = variant
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model) if pos_enc else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.pos(self.input_proj(x))
        memory = self.encoder(h)
        target = h if self.variant == "original" else memory
        return self.output_proj(self.decoder(target, memory))


def train_model(x_train, variant="original", pos_enc=False, epochs=50,
                batch_size=256, lr=1e-3, seed=0, device=None, log_every=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    model = TransformerAE(x_train.shape[-1], variant=variant, pos_enc=pos_enc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total / len(loader):.6f}")
    return model


@torch.no_grad()
def reconstruction_scores(model, x, score="max_dim", batch_size=1024, device=None):
    """Anomaly score per window.

    max_dim: maximum over features of the time-averaged squared error.
    mean:    mean squared error over all features and time steps.
    """
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[i : i + batch_size], dtype=torch.float32).to(device)
        err = (batch - model(batch)) ** 2
        if score == "max_dim":
            s = err.mean(dim=1).max(dim=1).values
        elif score == "mean":
            s = err.mean(dim=(1, 2))
        else:
            raise ValueError("score must be 'max_dim' or 'mean'")
        out.append(s.cpu().numpy())
    return np.concatenate(out)
