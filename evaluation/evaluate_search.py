"""
evaluate_search.py — Compares schedule-search strategies against each other.

Reads every ``results/<target>_<strategy>_seed<n>.json`` written by
``scheduler.search``, averages across seeds, and reports whether steering the
search by latent region beats searching at random.

Which number matters depends on the target. On ``race`` almost every schedule
with any interleaving loses at least one update, so the bug *rate* saturates
near 100% and says nothing; the discriminating measure is mean reward, the
fraction of transacted value the lost updates destroyed. On ``deadlock`` a run
either deadlocks or does not, so the bug rate is the measure.

Usage:  python -m evaluation.evaluate_search --target race
        python -m evaluation.evaluate_search --target deadlock
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STRATEGY_ORDER = ["random", "latent-uniform", "latent-bandit", "decoder"]
STRATEGY_COLORS = {
    "random": "#7f8c8d",
    "latent-uniform": "#3498db",
    "latent-bandit": "#e74c3c",
    "decoder": "#9b59b6",
}
STRATEGY_LABELS = {
    "random": "Random schedule",
    "latent-uniform": "Latent region (uniform)",
    "latent-bandit": "Latent region (bandit)",
    "decoder": "VAE decoder",
}


def parse_args():
    p = argparse.ArgumentParser(description="Compare schedule-search strategies")
    p.add_argument("--target", default="race")
    p.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    return p.parse_args()


# ── Loading ───────────────────────────────────────────────────────────────────

def load_runs(results_dir, target):
    """Group result files by strategy: {strategy: [run, run, ...]}."""
    runs = defaultdict(list)
    pattern = os.path.join(results_dir, f"{target}_*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        runs[payload["summary"]["strategy"]].append(payload)
    return runs


def ordered_strategies(runs):
    known = [s for s in STRATEGY_ORDER if s in runs]
    return known + sorted(s for s in runs if s not in STRATEGY_ORDER)


# ── Curves ────────────────────────────────────────────────────────────────────

def curves(payloads, key):
    """
    Stack one per-trial quantity across seeds into a (seeds, trials) array.

    Seeds may differ in length if a campaign was cut short, so everything is
    truncated to the shortest.
    """
    series = [[t[key] for t in p["trials"]] for p in payloads]
    length = min(len(s) for s in series)
    return np.array([s[:length] for s in series], dtype=float)


def summarise(strategy, payloads):
    bug = curves(payloads, "bug")
    reward = curves(payloads, "reward")
    n_seeds, n_trials = bug.shape

    first_bug = []
    for row in bug:
        hits = np.nonzero(row)[0]
        first_bug.append(int(hits[0]) + 1 if hits.size else None)
    resolved = [f for f in first_bug if f is not None]

    return {
        "strategy": strategy,
        "seeds": n_seeds,
        "trials": n_trials,
        "bug_rate": float(bug.mean()),
        "mean_reward": float(reward.mean()),
        "reward_std": float(reward.mean(axis=1).std()),
        "best_reward": float(reward.max()),
        "median_first_bug": float(np.median(resolved)) if resolved else float("nan"),
        "invalid": int(sum(1 for p in payloads for t in p["trials"] if not t["valid"])),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_discovery(runs, strategies, target, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in strategies:
        bug = curves(runs[strategy], "bug")
        cumulative = np.cumsum(bug, axis=1)
        mean = cumulative.mean(axis=0)
        x = np.arange(1, mean.size + 1)
        color = STRATEGY_COLORS.get(strategy, "#34495e")
        ax.plot(x, mean, color=color, linewidth=2,
                label=f"{STRATEGY_LABELS.get(strategy, strategy)} ({bug.shape[0]} seeds)")
        if bug.shape[0] > 1:
            ax.fill_between(x, cumulative.min(axis=0), cumulative.max(axis=0),
                            color=color, alpha=0.15)

    ax.set_xlabel("Executions")
    ax.set_ylabel("Cumulative bugs found")
    ax.set_title(f"Bug discovery — {target}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_reward(runs, strategies, target, path):
    fig, (ax_curve, ax_box) = plt.subplots(1, 2, figsize=(15, 6))

    box_data, box_labels, box_colors = [], [], []
    for strategy in strategies:
        reward = curves(runs[strategy], "reward")
        running = np.cumsum(reward, axis=1) / np.arange(1, reward.shape[1] + 1)
        mean = running.mean(axis=0)
        x = np.arange(1, mean.size + 1)
        color = STRATEGY_COLORS.get(strategy, "#34495e")
        ax_curve.plot(x, mean, color=color, linewidth=2,
                      label=STRATEGY_LABELS.get(strategy, strategy))
        if reward.shape[0] > 1:
            ax_curve.fill_between(x, running.min(axis=0), running.max(axis=0),
                                  color=color, alpha=0.15)
        box_data.append(reward.ravel())
        box_labels.append(STRATEGY_LABELS.get(strategy, strategy))
        box_colors.append(color)

    ax_curve.set_xlabel("Executions")
    ax_curve.set_ylabel("Running mean reward")
    ax_curve.set_title(f"Bug severity — {target}", fontsize=13, fontweight="bold")
    ax_curve.legend(fontsize=10)
    ax_curve.grid(True, alpha=0.2)

    bp = ax_box.boxplot(box_data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax_box.set_xticks(range(1, len(box_labels) + 1))
    ax_box.set_xticklabels(box_labels, rotation=18, ha="right", fontsize=9)
    ax_box.set_ylabel("Reward per execution")
    ax_box.set_title("Reward distribution", fontsize=13, fontweight="bold")
    ax_box.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_arms(runs, target, path):
    """How the bandit's region preference shifted as evidence accumulated."""
    payloads = runs.get("latent-bandit")
    if not payloads:
        return

    arms = curves(payloads, "arm").astype(int)
    regions = payloads[0]["summary"]["regions"]
    n_arms = len(regions)

    share = np.zeros((n_arms, arms.shape[1]))
    for seed_row in arms:
        counts = np.zeros(n_arms)
        for t, arm in enumerate(seed_row):
            if 0 <= arm < n_arms:
                counts[arm] += 1
            share[:, t] += counts / (t + 1)
    share /= arms.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, share.shape[1] + 1)
    for i, region in enumerate(regions):
        ax.plot(x, share[i], linewidth=2,
                label=f"Region {region['cluster']} — {region['variant']} "
                      f"(switch={region['switch_prob']:.2f})")
    ax.set_xlabel("Executions")
    ax.set_ylabel("Share of pulls")
    ax.set_ylim(0, 1)
    ax.set_title(f"Bandit region preference — {target}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 84)
    print(f"  LatentSched - search strategy comparison  (target={args.target})")
    print("=" * 84)

    runs = load_runs(args.results_dir, args.target)
    if not runs:
        print(f"\n  No results found in {args.results_dir} for target '{args.target}'.")
        print("  Run scheduler.search first, e.g.:")
        print(f"    python -m scheduler.search --target {args.target} --strategy random --trials 200")
        return

    strategies = ordered_strategies(runs)
    rows = [summarise(s, runs[s]) for s in strategies]

    print(f"\n  {'strategy':<24s} {'seeds':>5s} {'trials':>7s} {'bug rate':>9s} "
          f"{'mean reward':>12s} {'best':>7s} {'1st bug':>8s} {'void':>5s}")
    print("  " + "-" * 82)
    for r in rows:
        first = "-" if np.isnan(r["median_first_bug"]) else f"{r['median_first_bug']:.0f}"
        print(f"  {r['strategy']:<24s} {r['seeds']:>5d} {r['trials']:>7d} "
              f"{100 * r['bug_rate']:>8.1f}% {r['mean_reward']:>12.4f} "
              f"{r['best_reward']:>7.4f} {first:>8s} {r['invalid']:>5d}")

    # Which measure actually separates the strategies on this target.
    key = "bug_rate" if args.target == "deadlock" else "mean_reward"
    baseline = next((r for r in rows if r["strategy"] == "random"), None)
    if baseline and baseline[key] > 0:
        print(f"\n  Relative to the random baseline ({key}):")
        for r in rows:
            if r["strategy"] == "random":
                continue
            lift = 100.0 * (r[key] - baseline[key]) / baseline[key]
            print(f"    {r['strategy']:<24s} {lift:>+8.1f}%")

    print("\n  Plots:")
    plot_discovery(runs, strategies, args.target,
                   os.path.join(args.out_dir, f"search_{args.target}_discovery.png"))
    plot_reward(runs, strategies, args.target,
                os.path.join(args.out_dir, f"search_{args.target}_reward.png"))
    plot_arms(runs, args.target,
              os.path.join(args.out_dir, f"search_{args.target}_arms.png"))

    print("\n" + "=" * 84)


if __name__ == "__main__":
    main()
