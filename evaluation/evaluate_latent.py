"""
evaluate_latent.py — Extract latent vectors and visualize with PCA + t-SNE.

Usage:  python -m evaluation.evaluate_latent
"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vae.trace_encoder import load_all_traces, NUM_FEATURES
from vae.vae_model import TraceVAE


# ── Config ────────────────────────────────────────────────────────────────────

VARIANT_COLORS = {"safe": "#2ecc71", "race_condition": "#e67e22", "deadlock": "#e74c3c"}
VARIANT_LABELS = {"safe": "Safe", "race_condition": "Race Condition", "deadlock": "Deadlock"}


# ── Extract latent vectors ────────────────────────────────────────────────────

@torch.no_grad()
def extract_latent_vectors(model, features, device):
    """Run all traces through the encoder and return μ vectors."""
    model.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    mu, _ = model.encode(X)
    return mu.cpu().numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_latent_2d(coords_2d, labels, title, save_path):
    """Create a color-coded scatter plot of 2D latent coordinates."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for variant in ["safe", "race_condition", "deadlock"]:
        mask = np.array([l == variant for l in labels])
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
            s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint_path = os.path.join(project_root, "checkpoints", "trace_vae_checkpoint.pt")
    traces_dir = os.path.join(project_root, "trace_harness", "traces")
    output_dir = os.path.dirname(__file__)

    # Load checkpoint
    print("\n  Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model from checkpoint params
    model = TraceVAE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Model loaded (latent_dim={checkpoint['latent_dim']})")

    # Load all traces
    print("\n  Loading traces...")
    features, masks, labels = load_all_traces(traces_dir)

    # Extract μ vectors
    print("  Extracting latent vectors...")
    mu_vectors = extract_latent_vectors(model, features, device)
    print(f"  Latent matrix shape: {mu_vectors.shape}")  # (300, 16)

    # ── PCA ────────────────────────────────────────────────────────────
    print("\n  Running PCA...")
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(mu_vectors)
    var_explained = pca.explained_variance_ratio_
    print(f"  Variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, "
          f"Total={sum(var_explained):.3f}")

    pca_path = os.path.join(output_dir, "latent_pca.png")
    plot_latent_2d(
        pca_2d, labels,
        f"Latent Space — PCA  (Var explained: {sum(var_explained):.1%})",
        pca_path,
    )

    # ── t-SNE ─────────────────────────────────────────────────────────
    print("\n  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_2d = tsne.fit_transform(mu_vectors)

    tsne_path = os.path.join(output_dir, "latent_tsne.png")
    plot_latent_2d(
        tsne_2d, labels,
        "Latent Space — t-SNE",
        tsne_path,
    )

    # ── Per-variant stats ─────────────────────────────────────────────
    print("\n  Per-variant latent statistics:")
    print(f"  {'Variant':<20s} {'μ mean':>10s} {'μ std':>10s} {'μ norm':>10s}")
    print(f"  {'-'*50}")
    for variant in ["safe", "race_condition", "deadlock"]:
        mask = np.array([l == variant for l in labels])
        vecs = mu_vectors[mask]
        print(f"  {variant:<20s} {vecs.mean():>10.4f} {vecs.std():>10.4f} "
              f"{np.linalg.norm(vecs, axis=1).mean():>10.4f}")

    print(f"\n  Done! Check {output_dir} for the plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
