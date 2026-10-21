"""
latent_policy.py — Turns latent regions into schedule proposals, and picks
between those regions with a bandit.

Two generators are provided.

``region`` (default) fits a first-order interleaving model to the *traces* that
fall in each latent region: how often the running thread changes, and how
attention is spread across threads. Those statistics differ sharply by region
— race traces switch thread on 49% of events, safe traces on 38%, deadlock
traces use only two threads — so sampling from a region reproduces that
region's characteristic interleaving style.

``decoder`` is the original design: push a latent vector through the VAE
decoder and read its thread-index column as the schedule. It is kept because
it is the obvious thing to try, and because measuring it is worthwhile — on
the current checkpoint it is degenerate. Varying z changes the decoded thread
column by a standard deviation of ~0.06 while the column varies by ~0.33 along
its own length, so every latent vector decodes to essentially one canonical
near-serial ramp. A single z repeated across 300 steps, trained under MSE
against three highly stereotyped trace shapes, has no way to retain the
event-level interleaving detail a schedule needs. Prefer ``region``.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Thread indices the encoder can emit: 0 is MainThread, 1..N the workers.
MAX_THREAD_INDEX = 8


@dataclass
class RegionModel:
    """A latent region, and how the traces inside it interleave."""

    cluster: int
    dominant_variant: str
    bug_rate: float
    size: int
    switch_prob: float                  # P(running thread changes at an event)
    thread_weights: np.ndarray          # distribution over thread indices
    centroid: np.ndarray

    def describe(self) -> str:
        return (f"region {self.cluster} [{self.dominant_variant:<14s}] "
                f"n={self.size:>3d} bug_rate={self.bug_rate:.2f} "
                f"switch={self.switch_prob:.3f}")


# ── Fitting regions from traces ───────────────────────────────────────────────

def fit_region_models(
    features: np.ndarray,
    masks: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    bug_rates: Sequence[float],
    labels: Sequence[str],
) -> List[RegionModel]:
    """
    Measure each latent region's interleaving style from its member traces.

    ``features`` column 1 is the encoder's relative thread index, so a trace's
    thread column read end to end *is* the interleaving that trace performed.
    """
    regions: List[RegionModel] = []

    for k in range(centroids.shape[0]):
        member_idx = np.where(assignments == k)[0]

        switch_rates = []
        weights = np.zeros(MAX_THREAD_INDEX, dtype=np.float64)
        for i in member_idx:
            threads = features[i][:, 1][masks[i]].astype(int)
            if threads.size < 2:
                continue
            switch_rates.append(float(np.mean(threads[1:] != threads[:-1])))
            for t in threads:
                if 0 <= t < MAX_THREAD_INDEX:
                    weights[t] += 1.0

        total = weights.sum()
        thread_weights = weights / total if total > 0 else np.ones(MAX_THREAD_INDEX) / MAX_THREAD_INDEX

        member_labels = [labels[i] for i in member_idx]
        dominant = max(set(member_labels), key=member_labels.count) if member_labels else "unknown"

        regions.append(RegionModel(
            cluster=k,
            dominant_variant=dominant,
            bug_rate=float(bug_rates[k]),
            size=int(member_idx.size),
            switch_prob=float(np.mean(switch_rates)) if switch_rates else 0.5,
            thread_weights=thread_weights,
            centroid=centroids[k].astype(np.float32),
        ))

    return regions


# ── Schedule generation ───────────────────────────────────────────────────────

def _thread_distribution(region: RegionModel, num_threads: int) -> np.ndarray:
    """Restrict a region's thread preference to the target's worker ids."""
    w = region.thread_weights[1:num_threads + 1].astype(np.float64).copy()
    if w.sum() <= 0:
        w = np.ones(num_threads, dtype=np.float64)
    return w / w.sum()


def sample_region_schedule(
    region: RegionModel,
    num_threads: int,
    length: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Draw a schedule in the interleaving style of one latent region.

    The switch probability is applied as a probability of moving to a
    *different* thread, which is what the fitted statistic actually measured —
    resampling from the full distribution instead would land on the current
    thread sometimes and quietly undershoot the target switch rate.
    """
    ids = np.arange(1, num_threads + 1)
    probs = _thread_distribution(region, num_threads)

    current = int(rng.choice(ids, p=probs))
    schedule = [current]
    for _ in range(length - 1):
        if rng.random() < region.switch_prob and num_threads > 1:
            others = ids != current
            p = probs[others]
            p = p / p.sum() if p.sum() > 0 else np.ones(p.size) / p.size
            current = int(rng.choice(ids[others], p=p))
        schedule.append(current)
    return schedule


def sample_random_schedule(num_threads: int, length: int, rng: np.random.Generator) -> List[int]:
    """Uniform independent choice at every decision point — the baseline."""
    return [int(x) for x in rng.integers(1, num_threads + 1, size=length)]


def decode_schedule(
    model,
    z: np.ndarray,
    num_threads: int,
    length: int,
    device=None,
) -> List[int]:
    """
    Read a schedule out of the VAE decoder's thread-index column.

    Degenerate on the current checkpoint — see the module docstring.
    """
    import torch

    with torch.no_grad():
        z_t = torch.tensor(np.asarray(z), dtype=torch.float32, device=device).unsqueeze(0)
        _, continuous = model.decode(z_t, length)
        column = continuous[0, :, 0].cpu().numpy()

    rounded = np.rint(column).astype(int)
    return [int(np.clip(v, 1, num_threads)) for v in rounded]


# ── Bandit over latent regions ────────────────────────────────────────────────

class LatentBandit:
    """
    Thompson sampling over latent regions.

    Each region is an arm with a Beta posterior over "a schedule drawn from
    this region exposes a bug". Rewards are severities in [0, 1] rather than
    coin flips, which a Beta absorbs as fractional evidence.

    The clustering's own bug enrichment seeds the priors, so the search starts
    out already believing what the latent space showed — which is the point of
    having clustered it.
    """

    def __init__(
        self,
        regions: List[RegionModel],
        prior_strength: float = 2.0,
        seed: int = 0,
    ):
        self.regions = regions
        self.rng = np.random.default_rng(seed)
        self.alpha = np.array([1.0 + prior_strength * r.bug_rate for r in regions])
        self.beta = np.array([1.0 + prior_strength * (1.0 - r.bug_rate) for r in regions])
        self.pulls = np.zeros(len(regions), dtype=int)
        self.total_reward = np.zeros(len(regions))

    def select(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
        self.pulls[arm] += 1
        self.total_reward[arm] += reward

    def stats(self) -> List[dict]:
        return [
            {
                "arm": i,
                "cluster": r.cluster,
                "variant": r.dominant_variant,
                "pulls": int(self.pulls[i]),
                "mean_reward": float(self.total_reward[i] / self.pulls[i]) if self.pulls[i] else 0.0,
                "posterior_mean": float(self.alpha[i] / (self.alpha[i] + self.beta[i])),
            }
            for i, r in enumerate(self.regions)
        ]


# ── Asset loading ─────────────────────────────────────────────────────────────

@dataclass
class PolicyAssets:
    regions: List[RegionModel]
    centroids: np.ndarray
    model: Optional[object] = None
    device: Optional[object] = None
    features: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None
    labels: Optional[List[str]] = field(default=None)


def load_policy_assets(project_root: str, need_model: bool = False) -> PolicyAssets:
    """Load the clustering handoff artifact, the traces, and optionally the VAE."""
    from vae.trace_encoder import load_all_traces

    regions_path = os.path.join(project_root, "evaluation", "latent_regions.npz")
    if not os.path.exists(regions_path):
        raise FileNotFoundError(
            f"{regions_path} not found — run `python -m evaluation.evaluate_cluster` first."
        )

    data = np.load(regions_path, allow_pickle=True)
    features, masks, labels = load_all_traces(os.path.join(project_root, "trace_harness", "traces"))

    regions = fit_region_models(
        features=features,
        masks=masks,
        assignments=data["cluster_assignments"],
        centroids=data["centroids"],
        bug_rates=data["cluster_bug_rate"],
        labels=labels,
    )

    model = None
    device = None
    if need_model:
        import torch
        from vae.vae_model import TraceVAE

        ckpt_path = os.path.join(project_root, "checkpoints", "trace_vae_checkpoint.pt")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TraceVAE(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            num_layers=checkpoint["num_layers"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    return PolicyAssets(
        regions=regions,
        centroids=data["centroids"],
        model=model,
        device=device,
        features=features,
        masks=masks,
        labels=labels,
    )


def nearest_region(mu: np.ndarray, centroids: np.ndarray) -> int:
    """Which latent region a trace's encoding landed in."""
    return int(np.argmin(np.linalg.norm(centroids - mu[None, :], axis=1)))
