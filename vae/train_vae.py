"""
train_vae.py
============
Training loop for the LatentSched Trace VAE.

Trains on ALL traces combined (safe + race_condition + deadlock) without
using variant labels — the VAE is fully unsupervised.

Features
--------
  • **Train / Validation / Test split** (70 / 15 / 15), stratified by variant.
  • **Cyclical β‑annealing**: β cycles from 0 → β_max multiple times to
    prevent posterior collapse on small datasets.
  • **Free bits (free_nats)**: per‑dimension KL floor ensures KL stays in a
    healthy 1–5 range even with few traces.
  • Masked reconstruction loss: padding positions are excluded.
  • MLflow integration: logs hyperparameters and per‑epoch metrics.
  • Saves model checkpoint + **loss curve plot** after training.

Usage
-----
  From the project root::

      python -m vae.train_vae

  Or with custom args::

      python -m vae.train_vae --epochs 200 --latent-dim 32 --lr 0.001
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Local imports ─────────────────────────────────────────────────────────────
from .trace_encoder import load_all_traces, SEQ_LEN, NUM_FEATURES
from .vae_model import TraceVAE, vae_loss

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import mlflow
    # pyrefly: ignore [missing-import]
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[WARN] mlflow not installed — metrics will only be printed to stdout.")

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for saving to disk
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not installed — loss curve plot will be skipped.")

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed — falling back to manual split.")


# ── Configuration defaults ────────────────────────────────────────────────────

DEFAULTS = {
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "latent_dim": 16,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.1,
    "num_cycles": 4,               # cyclical β-annealing cycles
    "beta_max": 1.0,               # maximum β value
    "free_nats": 0.125,            # free bits per latent dim (16 * 0.125 = 2.0 min KL)
    "seed": 42,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train LatentSched Trace VAE")
    p.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    p.add_argument("--latent-dim",    type=int,   default=DEFAULTS["latent_dim"])
    p.add_argument("--hidden-dim",    type=int,   default=DEFAULTS["hidden_dim"])
    p.add_argument("--num-layers",    type=int,   default=DEFAULTS["num_layers"])
    p.add_argument("--dropout",       type=float, default=DEFAULTS["dropout"])
    p.add_argument("--num-cycles",    type=int,   default=DEFAULTS["num_cycles"])
    p.add_argument("--beta-max",      type=float, default=DEFAULTS["beta_max"])
    p.add_argument("--free-nats",     type=float, default=DEFAULTS["free_nats"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--no-mlflow",     action="store_true", help="Disable MLflow logging")
    return p.parse_args()


# ── Data splitting ────────────────────────────────────────────────────────────

def stratified_split(features, masks, labels, seed=42):
    """
    Split data into Train (70%) / Validation (15%) / Test (15%),
    stratified by variant label so each split has a proportional mix
    of safe, race_condition, and deadlock traces.

    Returns
    -------
    (X_train, M_train), (X_val, M_val), (X_test, M_test), split_info
    """
    n = features.shape[0]
    indices = np.arange(n)
    label_arr = np.array(labels)

    if HAS_SKLEARN:
        # First split: 70% train, 30% temp
        idx_train, idx_temp = train_test_split(
            indices, test_size=0.30, stratify=label_arr, random_state=seed
        )
        # Second split: 50% of temp → val, 50% → test (= 15% each overall)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.50, stratify=label_arr[idx_temp], random_state=seed
        )
    else:
        # Manual fallback: shuffle then split
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        idx_train = indices[:n_train]
        idx_val   = indices[n_train:n_train + n_val]
        idx_test  = indices[n_train + n_val:]

    def _subset(idx):
        return features[idx], masks[idx]

    split_info = {
        "train": len(idx_train),
        "val":   len(idx_val),
        "test":  len(idx_test),
    }

    return _subset(idx_train), _subset(idx_val), _subset(idx_test), split_info


# ── Evaluation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, beta, free_nats):
    """Run a no-grad evaluation pass and return average losses."""
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    n_batches      = 0

    for batch_x, batch_mask in dataloader:
        batch_x    = batch_x.to(device)
        batch_mask = batch_mask.to(device)

        event_logits, cont_recon, mu, log_var = model(batch_x)
        total_loss, recon_loss, kl_loss = vae_loss(
            event_type_logits=event_logits,
            continuous_recon=cont_recon,
            x=batch_x,
            mu=mu,
            log_var=log_var,
            beta=beta,
            padding_mask=batch_mask,
            free_nats=free_nats,
        )

        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum    += kl_loss.item()
        n_batches      += 1

    return {
        "total": total_loss_sum / max(n_batches, 1),
        "recon": recon_loss_sum / max(n_batches, 1),
        "kl":    kl_loss_sum    / max(n_batches, 1),
    }


# ── Loss curve plotting ──────────────────────────────────────────────────────

def save_loss_curves(history: dict, save_path: str):
    """
    Generate and save a 3-panel loss curve figure:
      Top:    Total loss  (train vs val)
      Middle: Reconstruction loss
      Bottom: KL divergence
    """
    if not HAS_MATPLOTLIB:
        print("  [WARN] matplotlib not available — skipping loss curve plot.")
        return

    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("LatentSched Trace VAE — Training & Validation Loss Curves",
                 fontsize=14, fontweight="bold")

    # Total loss
    axes[0].plot(epochs, history["train_total"], label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_total"],   label="Val",   linewidth=1.5, linestyle="--")
    axes[0].set_ylabel("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, history["train_recon"], label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["val_recon"],   label="Val",   linewidth=1.5, linestyle="--")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(epochs, history["train_kl"], label="Train KL", linewidth=1.5)
    axes[2].plot(epochs, history["val_kl"],   label="Val KL",   linewidth=1.5, linestyle="--")
    axes[2].axhline(y=1.0, color="green", linestyle=":", alpha=0.6, label="Target range (1–5)")
    axes[2].axhline(y=5.0, color="green", linestyle=":", alpha=0.6)
    axes[2].axhspan(1.0, 5.0, color="green", alpha=0.05)
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved → {save_path}")


# ── Cyclical β schedule ──────────────────────────────────────────────────────

def cyclical_beta(epoch: int, total_epochs: int, num_cycles: int, beta_max: float) -> float:
    """
    Cyclical β-annealing schedule.

    Each cycle:
      • First half: ramp β from 0 → beta_max linearly
      • Second half: hold at beta_max

    This prevents posterior collapse on small datasets by repeatedly
    "resetting" the KL pressure, giving the encoder multiple chances to
    learn a meaningful latent representation.
    """
    cycle_length = total_epochs / num_cycles
    position = ((epoch - 1) % cycle_length) / cycle_length   # 0.0 → 1.0
    beta = beta_max * min(1.0, position * 2.0)               # ramp in first half
    return beta


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    # ── Reproducibility ───────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    traces_dir = os.path.join(project_root, "trace_harness", "traces")

    print("\n  Loading and encoding traces...")
    features, masks, labels = load_all_traces(traces_dir)
    # features: (N, SEQ_LEN, NUM_FEATURES),  masks: (N, SEQ_LEN)

    # ── Train / Validation / Test split ───────────────────────────────
    (feat_train, mask_train), (feat_val, mask_val), (feat_test, mask_test), split_info = \
        stratified_split(features, masks, labels, seed=args.seed)

    print(f"  Split: Train={split_info['train']}, Val={split_info['val']}, Test={split_info['test']}")

    # Convert to tensors and DataLoaders
    train_ds = TensorDataset(
        torch.tensor(feat_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.bool),
    )
    val_ds = TensorDataset(
        torch.tensor(feat_val, dtype=torch.float32),
        torch.tensor(mask_val, dtype=torch.bool),
    )
    test_ds = TensorDataset(
        torch.tensor(feat_test, dtype=torch.float32),
        torch.tensor(mask_test, dtype=torch.bool),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch:   {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = TraceVAE(
        input_dim=NUM_FEATURES,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── MLflow setup ──────────────────────────────────────────────────
    use_mlflow = HAS_MLFLOW and not args.no_mlflow
    if use_mlflow:
        mlflow.set_experiment("LatentSched-TraceVAE")
        mlflow.start_run()
        mlflow.log_params({
            "architecture": "BiLSTM-VAE",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "num_cycles": args.num_cycles,
            "beta_max": args.beta_max,
            "free_nats": args.free_nats,
            "seq_len": SEQ_LEN,
            "num_features": NUM_FEATURES,
            "num_train": split_info["train"],
            "num_val": split_info["val"],
            "num_test": split_info["test"],
            "device": str(device),
        })
        print("  MLflow run started.")
    else:
        print("  MLflow disabled — logging to stdout only.")

    # ── History tracking ──────────────────────────────────────────────
    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total":   [], "val_recon":   [], "val_kl":   [],
        "beta":        [],
    }

    # ── Training ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Training started")
    print(f"  Epochs: {args.epochs} | Cycles: {args.num_cycles} | "
          f"free_nats: {args.free_nats} | β_max: {args.beta_max}")
    print("=" * 80)

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Cyclical β-annealing
        beta = cyclical_beta(epoch, args.epochs, args.num_cycles, args.beta_max)

        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        for batch_x, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)

            # Forward pass
            event_logits, cont_recon, mu, log_var = model(batch_x)

            # Compute masked loss with free bits
            total_loss, recon_loss, kl_loss = vae_loss(
                event_type_logits=event_logits,
                continuous_recon=cont_recon,
                x=batch_x,
                mu=mu,
                log_var=log_var,
                beta=beta,
                padding_mask=batch_mask,
                free_nats=args.free_nats,
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping to prevent exploding gradients with LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        # Average training losses
        avg_train_total = epoch_total_loss / num_batches
        avg_train_recon = epoch_recon_loss / num_batches
        avg_train_kl    = epoch_kl_loss / num_batches

        # Validation pass
        val_metrics = evaluate(model, val_loader, device, beta, args.free_nats)

        # Record history
        history["train_total"].append(avg_train_total)
        history["train_recon"].append(avg_train_recon)
        history["train_kl"].append(avg_train_kl)
        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])
        history["beta"].append(beta)

        # Log to MLflow
        if use_mlflow:
            mlflow.log_metrics({
                "train_total_loss": avg_train_total,
                "train_recon_loss": avg_train_recon,
                "train_kl_loss": avg_train_kl,
                "val_total_loss": val_metrics["total"],
                "val_recon_loss": val_metrics["recon"],
                "val_kl_loss": val_metrics["kl"],
                "beta": beta,
            }, step=epoch)

        # Print every 10 epochs + first + last
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.perf_counter() - start_time
            print(
                f"  Epoch {epoch:>4d}/{args.epochs} │ "
                f"T-loss={avg_train_total:>8.4f} │ "
                f"T-recon={avg_train_recon:>8.4f} │ "
                f"T-kl={avg_train_kl:>6.4f} │ "
                f"V-loss={val_metrics['total']:>8.4f} │ "
                f"V-kl={val_metrics['kl']:>6.4f} │ "
                f"β={beta:.3f} │ "
                f"time={elapsed:>6.1f}s"
            )

    elapsed_total = time.perf_counter() - start_time

    # ── Test set evaluation ───────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  Evaluating on held-out TEST set...")
    test_metrics = evaluate(model, test_loader, device, beta=args.beta_max, free_nats=args.free_nats)
    print(
        f"  TEST │ "
        f"loss={test_metrics['total']:>8.4f} │ "
        f"recon={test_metrics['recon']:>8.4f} │ "
        f"kl={test_metrics['kl']:>6.4f}"
    )
    print("-" * 80)

    if use_mlflow:
        mlflow.log_metrics({
            "test_total_loss": test_metrics["total"],
            "test_recon_loss": test_metrics["recon"],
            "test_kl_loss": test_metrics["kl"],
        })

    # ── Save checkpoint ───────────────────────────────────────────────
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "trace_vae_checkpoint.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "input_dim": NUM_FEATURES,
        "free_nats": args.free_nats,
        "test_metrics": test_metrics,
    }, checkpoint_path)

    print(f"\n  Checkpoint saved → {checkpoint_path}")

    # ── Save loss curve plot ──────────────────────────────────────────
    vae_dir = os.path.dirname(__file__)
    plot_path = os.path.join(vae_dir, "loss_curve.png")
    save_loss_curves(history, plot_path)

    # ── Wrap up ───────────────────────────────────────────────────────
    if use_mlflow:
        mlflow.log_artifact(checkpoint_path)
        mlflow.log_artifact(plot_path)
        mlflow.end_run()
        print("  MLflow run ended.")

    print(f"\n  Training complete in {elapsed_total:.1f}s")
    print("=" * 80)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
