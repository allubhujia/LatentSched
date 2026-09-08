"""
evaluate_clusters.py — Quantitative latent-space evaluation for LatentSched.

Answers the question the PCA/t-SNE plots can only hint at:
  1. Do buggy and safe traces occupy *measurably* distinguishable latent regions?
     → silhouette score (by variant, and by binary buggy/safe)
  2. How should latent space be partitioned into discrete regions?
     → k-means with a k-sweep, plus per-cluster bug enrichment

Writes `latent_regions.npz` — the handoff artifact consumed by guided
generation and the bandit search (cluster centroids become bandit arms,
bug_rate becomes each arm's reward prior).

Usage:  python daa_save.py
        python daa_save.py --k-min 2 --k-max 12
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
# This script lives outside the repo, so the project root is resolved explicitly.

PROJECT_ROOT = os.environ.get(
    "LATENTSCHED_ROOT",
    r"C:\VIT_CHENNAI\Computer_science\Machine Learning\Projects\LatentSched",
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
VARIANTS_COLORS = {"safe": "#2ecc71", "race_condition": "e67e22"," deadlock": "#e74c3c"}
VARIANTS_LABELS = {"safe": "Safe", "race_condition": "Race condition", "deadlock": "Deadlock"}
BUGGY_VARIANTS = {"race_condition", "deadlock"}

def parse_args():
    p = argparse.ArgumentParser(description="LAtent-space cluster evaluation")
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
