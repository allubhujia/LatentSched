"""
vae_model.py — BiLSTM Variational Autoencoder for concurrency execution traces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .trace_encoder import NUM_FEATURES, NUM_EVENT_TYPES


class TraceVAE(nn.Module):
    """
    Bidirectional LSTM VAE for execution-trace data.

    Encoder: Input → BiLSTM → concat final hidden states → FC → μ, log σ²
    Decoder: z → FC → initial hidden → repeat z → LSTM → two heads (categorical + continuous)
    """

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        encoder_output_dim = hidden_dim * 2
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, latent_dim)

        # Decoder
        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_z_to_cell = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Reconstruction heads
        self.fc_event_type = nn.Linear(hidden_dim, NUM_EVENT_TYPES)
        self.fc_continuous = nn.Linear(hidden_dim, input_dim - 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.encoder_lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        return self.fc_mu(h_cat), self.fc_log_var(h_cat)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        h_0 = self.fc_z_to_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = self.fc_z_to_cell(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(z_repeated, (h_0, c_0))
        return self.fc_event_type(decoder_out), self.fc_continuous(decoder_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        event_type_logits, continuous_recon = self.decode(z, x.size(1))
        return event_type_logits, continuous_recon, mu, log_var


# ── Loss function ─────────────────────────────────────────────────────────────

def vae_loss(
    event_type_logits: torch.Tensor,
    continuous_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    padding_mask: torch.Tensor = None,
    free_nats: float = 0.125,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Masked reconstruction (CrossEntropy + MSE) + β-weighted KL with free bits.
    """
    batch_size, seq_len, _ = x.shape

    # Categorical: event_type (column 0)
    event_type_target = x[:, :, 0].long()
    ce_loss_per_pos = F.cross_entropy(
        event_type_logits.permute(0, 2, 1), event_type_target, reduction="none",
    )

    # Continuous: columns 1–4
    continuous_target = x[:, :, 1:]
    mse_loss_per_pos = F.mse_loss(
        continuous_recon, continuous_target, reduction="none",
    ).mean(dim=-1)

    # Combined reconstruction
    recon_per_pos = ce_loss_per_pos + mse_loss_per_pos

    # Apply padding mask
    if padding_mask is not None:
        mask_float = padding_mask.float()
        recon_per_pos = recon_per_pos * mask_float
        recon_loss = recon_per_pos.sum() / mask_float.sum().clamp(min=1.0)
    else:
        recon_loss = recon_per_pos.mean()

    # KL divergence with free bits
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_nats)
    kl_loss = kl_per_dim.sum(dim=-1).mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
