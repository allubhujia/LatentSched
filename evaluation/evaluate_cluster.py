"""
evaluate_cluster.py — Quantitative latent-space evaluation for LatentSched.

Answers the question the PCA/t-SNE plots can only hint at:
  1. Do buggy and safe traces occupy *measurably* distinguishable latent regions?
     → silhouette score (by variant, and by binary buggy/safe)
  2. How should latent space be partitioned into discrete regions?
     → k-means with a k-sweep, plus per-cluster bug enrichment

Writes `latent_regions.npz` — the handoff artifact consumed by guided
generation and the bandit search (cluster centroids become bandit arms,
bug_rate becomes each arm's reward prior).

Usage:  python -m evaluation.evaluate_cluster
        python -m evaluation.evaluate_cluster --k-min 2 --k-max 12
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows consoles default to cp1252, which cannot encode the arrows/μ used in
# this project's log lines (including trace_encoder's own prints).

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

# ── Project location ──────────────────────────────────────────────────────────

PROJECT_ROOT = os.environ.get(
    "LATENTSCHED_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vae.trace_encoder import load_all_traces
from vae.vae_model import TraceVAE

try:
    #pyrefly: ignore [missing-import]
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[WARN] mlflow not installed - metrics printed to stdout only.")

VARIANTS = ("safe", "race_condition", "deadlock")
VARIANTS_COLORS = {"safe": "#2ecc71", "race_condition": "#e67e22", "deadlock": "#e74c3c"}
VARIANTS_LABELS = {"safe": "Safe", "race_condition": "Race condition", "deadlock": "Deadlock"}
BUGGY_VARIANTS = {"race_condition", "deadlock"}

CLUSTER_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f",
                   "#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#d35400"]


def parse_args():
    p = argparse.ArgumentParser(description="Latent-space cluster evaluation")
    # k must be >= the number of variants, otherwise the partition collapses
    # distinct bug classes into one arm and the bandit cannot distinguish them.
    p.add_argument("--k-min", type=int, default=len(VARIANTS))
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--k", type=int, default=None,
                   help="Force a specific k instead of picking the sweep's best")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


# ── Latent extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(model, features, device, batch_size=64):
    """Encode every trace and return its μ vector."""
    model.eval()
    out = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.tensor(features[start:start + batch_size],
                             dtype=torch.float32, device=device)
        mu, _ = model.encode(batch)
        out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def load_model(project_root, device):
    ckpt_path = os.path.join(project_root, "checkpoints", "trace_vae_checkpoint.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TraceVAE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


# ── Separability ──────────────────────────────────────────────────────────────

def separability_scores(latents, labels):
    """
    How cleanly the latent space separates traces, by variant and buggy/safe.

    Silhouette is computed against the *true* labels rather than a clustering,
    so it measures the embedding itself and not k-means' luck.
    """
    variant_ids = np.array([VARIANTS.index(l) for l in labels])
    buggy_ids = np.array([1 if l in BUGGY_VARIANTS else 0 for l in labels])

    scores = {
        "silhouette_by_variant": float(silhouette_score(latents, variant_ids)),
        "silhouette_by_buggy": float(silhouette_score(latents, buggy_ids)),
    }

    per_variant = {}
    samples = silhouette_samples(latents, variant_ids)
    for i, variant in enumerate(VARIANTS):
        mask = variant_ids == i
        if mask.any():
            per_variant[variant] = float(samples[mask].mean())
    scores["silhouette_per_variant"] = per_variant
    return scores


# ── k sweep ───────────────────────────────────────────────────────────────────

def sweep_k(latents, k_min, k_max, seed):
    """Silhouette and inertia across candidate k values."""
    rows = []
    for k in range(k_min, min(k_max, latents.shape[0] - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        assignments = km.fit_predict(latents)
        if len(set(assignments.tolist())) < 2:
            continue
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(latents, assignments)),
            "inertia": float(km.inertia_),
        })
    return rows


# ── Cluster characterisation ──────────────────────────────────────────────────

def characterise(assignments, labels, k):
    """
    Per-cluster size, variant composition and bug rate.

    ``bug_rate`` is the fraction of a cluster's traces that came from a buggy
    variant — the prior the bandit starts each arm with.
    """
    composition = np.zeros((k, len(VARIANTS)), dtype=np.int64)
    for cluster, label in zip(assignments, labels):
        composition[cluster, VARIANTS.index(label)] += 1

    sizes = composition.sum(axis=1)
    buggy_cols = [VARIANTS.index(v) for v in VARIANTS if v in BUGGY_VARIANTS]
    buggy_counts = composition[:, buggy_cols].sum(axis=1)
    bug_rate = np.divide(buggy_counts, sizes, out=np.zeros(k, dtype=np.float64),
                         where=sizes > 0)
    return composition, sizes, bug_rate


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_k_sweep(rows, path):
    ks = [r["k"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, [r["silhouette"] for r in rows], "o-", color="#3498db", label="Silhouette")
    ax1.set_xlabel("k (number of latent regions)")
    ax1.set_ylabel("Silhouette score", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(ks, [r["inertia"] for r in rows], "s--", color="#e74c3c", label="Inertia")
    ax2.set_ylabel("Inertia", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    ax1.set_title("k-means sweep over latent space", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_silhouette_profile(latents, assignments, k, path):
    samples = silhouette_samples(latents, assignments)
    overall = float(samples.mean())

    fig, ax = plt.subplots(figsize=(9, 6))
    y = 10
    for cluster in range(k):
        vals = np.sort(samples[assignments == cluster])
        if vals.size == 0:
            continue
        color = CLUSTER_PALETTE[cluster % len(CLUSTER_PALETTE)]
        ax.fill_betweenx(np.arange(y, y + vals.size), 0, vals, color=color, alpha=0.8)
        ax.text(-0.05, y + vals.size / 2, str(cluster), va="center", ha="right", fontsize=10)
        y += vals.size + 10

    ax.axvline(overall, color="#e74c3c", linestyle="--",
               label=f"mean = {overall:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.set_title(f"Silhouette profile (k={k})", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_latent_clusters(latents, assignments, labels, k, path):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(latents)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    for cluster in range(k):
        mask = assignments == cluster
        ax_left.scatter(coords[mask, 0], coords[mask, 1],
                        c=CLUSTER_PALETTE[cluster % len(CLUSTER_PALETTE)],
                        label=f"Region {cluster}", s=55, alpha=0.75,
                        edgecolors="white", linewidth=0.5)
    ax_left.set_title("k-means regions", fontsize=13, fontweight="bold")
    ax_left.legend(fontsize=9)
    ax_left.grid(True, alpha=0.2)

    for variant in VARIANTS:
        mask = np.array([l == variant for l in labels])
        ax_right.scatter(coords[mask, 0], coords[mask, 1],
                         c=VARIANTS_COLORS[variant], label=VARIANTS_LABELS[variant],
                         s=55, alpha=0.75, edgecolors="white", linewidth=0.5)
    ax_right.set_title("True variants", fontsize=13, fontweight="bold")
    ax_right.legend(fontsize=9)
    ax_right.grid(True, alpha=0.2)

    for ax in (ax_left, ax_right):
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_composition(composition, bug_rate, path):
    k = composition.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(k)
    x = np.arange(k)
    for i, variant in enumerate(VARIANTS):
        ax.bar(x, composition[:, i], bottom=bottom,
               color=VARIANTS_COLORS[variant], label=VARIANTS_LABELS[variant])
        bottom += composition[:, i]

    for cluster in range(k):
        ax.text(cluster, bottom[cluster] + 1, f"bug rate\n{bug_rate[cluster]:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Region {i}" for i in range(k)])
    ax.set_ylabel("Traces")
    ax.set_ylim(0, bottom.max() * 1.25)
    ax.set_title("Variant composition per latent region", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("  LatentSched - latent-space cluster evaluation")
    print("=" * 78)

    print("\n  [1/5] Loading model and traces...")
    model, checkpoint = load_model(PROJECT_ROOT, device)
    features, masks, labels = load_all_traces(os.path.join(PROJECT_ROOT, "trace_harness", "traces"))
    latents = extract_latents(model, features, device)
    print(f"        latent matrix: {latents.shape}  (latent_dim={checkpoint['latent_dim']})")

    print("\n  [2/5] Measuring separability...")
    scores = separability_scores(latents, labels)
    print(f"        silhouette by variant    : {scores['silhouette_by_variant']:.4f}")
    print(f"        silhouette by buggy/safe : {scores['silhouette_by_buggy']:.4f}")
    for variant, value in scores["silhouette_per_variant"].items():
        print(f"          {variant:<16s} {value:>7.4f}")

    print("\n  [3/5] Sweeping k...")
    sweep = sweep_k(latents, args.k_min, args.k_max, args.seed)
    for row in sweep:
        print(f"        k={row['k']:>2d}  silhouette={row['silhouette']:>7.4f}  "
              f"inertia={row['inertia']:>12.4f}")

    best = max(sweep, key=lambda r: r["silhouette"])
    k = args.k or best["k"]
    print(f"        best silhouette at k={best['k']}  →  using k={k}")

    print(f"\n  [4/5] Partitioning latent space into {k} regions...")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
    assignments = kmeans.fit_predict(latents)
    centroids = kmeans.cluster_centers_

    composition, sizes, bug_rate = characterise(assignments, labels, k)
    variant_ids = np.array([VARIANTS.index(l) for l in labels])
    ari = float(adjusted_rand_score(variant_ids, assignments))
    nmi = float(normalized_mutual_info_score(variant_ids, assignments))

    print(f"        adjusted Rand index      : {ari:.4f}")
    print(f"        normalised mutual info   : {nmi:.4f}")
    print(f"\n        {'region':>8s} {'size':>6s} {'bug rate':>10s}   composition "
          f"({', '.join(VARIANTS)})")
    for cluster in range(k):
        print(f"        {cluster:>8d} {sizes[cluster]:>6d} {bug_rate[cluster]:>10.3f}   "
              f"{composition[cluster].tolist()}")

    print("\n  [5/5] Writing plots and the handoff artifact...")
    plot_k_sweep(sweep, os.path.join(output_dir, "kmeans_sweep.png"))
    plot_silhouette_profile(latents, assignments, k, os.path.join(output_dir, "silhouette_profile.png"))
    plot_latent_clusters(latents, assignments, labels, k, os.path.join(output_dir, "latent_clusters.png"))
    plot_composition(composition, bug_rate, os.path.join(output_dir, "cluster_composition.png"))

    regions_path = os.path.join(output_dir, "latent_regions.npz")
    np.savez(
        regions_path,
        latent_vectors=latents.astype(np.float32),
        cluster_assignments=assignments.astype(np.int32),
        centroids=centroids.astype(np.float32),
        cluster_bug_rate=bug_rate,
        cluster_sizes=sizes,
        cluster_composition=composition,
        variant_names=np.array(VARIANTS),
        labels=np.array(labels),
        k=np.array(k),
        latent_dim=np.array(latents.shape[1]),
    )
    print(f"  Saved → {regions_path}")

    if HAS_MLFLOW and not args.no_mlflow:
        mlflow.set_experiment("latentsched-clusters")
        with mlflow.start_run(run_name=f"clusters_k{k}"):
            mlflow.log_params({"k": k, "seed": args.seed,
                               "latent_dim": int(latents.shape[1])})
            mlflow.log_metrics({
                "silhouette_by_variant": scores["silhouette_by_variant"],
                "silhouette_by_buggy": scores["silhouette_by_buggy"],
                "adjusted_rand_index": ari,
                "normalized_mutual_info": nmi,
            })
            mlflow.log_artifact(regions_path)
        print("  Logged to MLflow.")

    print("\n" + "=" * 78)
    print(f"  Next: python -m scheduler.search --target race --strategy latent-bandit")
    print("=" * 78)


if __name__ == "__main__":
    main()
