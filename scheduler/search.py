"""
search.py — Latent-guided schedule search.

Runs a search campaign against one target and writes a JSON record of every
trial, for ``evaluation.evaluate_search`` to compare.

Strategies
----------
random          Uniform schedule at every decision point. No latent space
                involved — the baseline everything else has to beat.
latent-uniform  Schedules drawn in a latent region's interleaving style, with
                the region picked uniformly. Isolates what the *generator* is
                worth, separately from the bandit.
latent-bandit   The same generator, with Thompson sampling choosing the region
                and the clustering's bug enrichment seeding the priors. The
                full method.
decoder         The original design: decode a latent vector's thread-index
                column into a schedule. Degenerate on the current checkpoint
                (see ``latent_policy``); kept so the claim stays measurable.

Usage
-----
    python -m scheduler.search --target race     --strategy latent-bandit --trials 200
    python -m scheduler.search --target deadlock --strategy random        --trials 200
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

from scheduler.latent_policy import (
    LatentBandit,
    decode_schedule,
    load_policy_assets,
    nearest_region,
    sample_random_schedule,
    sample_region_schedule,
)
from scheduler.targets import TARGETS, make_workload, num_threads_for, run_schedule

STRATEGIES = ("random", "latent-uniform", "latent-bandit", "decoder")

# Matches the VAE's sequence length, so one schedule spans a whole execution.
SCHEDULE_LEN = 300

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Latent-guided schedule search")
    p.add_argument("--target", choices=TARGETS, default="race")
    p.add_argument("--strategy", choices=STRATEGIES, default="latent-bandit")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule-len", type=int, default=SCHEDULE_LEN)
    p.add_argument("--prior-strength", type=float, default=2.0,
                   help="Weight given to each region's clustered bug rate as a bandit prior")
    p.add_argument("--decoder-sigma", type=float, default=1.0,
                   help="Spread of latent samples around a centroid (decoder strategy)")
    p.add_argument("--wall-timeout", type=float, default=15.0)
    p.add_argument("--report-every", type=int, default=20)
    p.add_argument("--no-encode", action="store_true",
                   help="Skip encoding each produced trace back into latent space")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def _encode_region(assets, events) -> Optional[int]:
    """Which latent region the trace this schedule produced actually landed in."""
    if assets.model is None or not events:
        return None
    import torch
    from vae.trace_encoder import encode_trace

    features, _ = encode_trace(events)
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=assets.device).unsqueeze(0)
        mu, _ = assets.model.encode(x)
    return nearest_region(mu[0].cpu().numpy(), assets.centroids)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    needs_model = (args.strategy == "decoder") or not args.no_encode

    print("=" * 78)
    print(f"  LatentSched - schedule search")
    print(f"  target={args.target}  strategy={args.strategy}  trials={args.trials}  seed={args.seed}")
    print("=" * 78)

    print("\n  [1/3] Loading latent regions and traces...")
    assets = load_policy_assets(PROJECT_ROOT, need_model=needs_model)
    for region in assets.regions:
        print(f"        {region.describe()}")

    num_threads = num_threads_for(args.target)
    workload = make_workload() if args.target != "deadlock" else None
    bandit = LatentBandit(assets.regions, prior_strength=args.prior_strength, seed=args.seed)

    print(f"\n  [2/3] Searching ({num_threads} threads, schedule length {args.schedule_len})...")
    header = (f"        {'trial':>6s} {'bugs':>6s} {'bug%':>7s} {'mean_r':>8s} "
              f"{'best_r':>8s} {'void':>5s} {'elapsed':>9s}  arm pulls")
    print(header)

    records: List[dict] = []
    bugs = 0
    voids = 0
    best_reward = 0.0
    reward_sum = 0.0
    t_start = time.perf_counter()

    for trial in range(1, args.trials + 1):
        # ── Propose a schedule ────────────────────────────────────────────
        if args.strategy == "random":
            arm = -1
            schedule = sample_random_schedule(num_threads, args.schedule_len, rng)
        elif args.strategy == "latent-uniform":
            arm = int(rng.integers(0, len(assets.regions)))
            schedule = sample_region_schedule(assets.regions[arm], num_threads, args.schedule_len, rng)
        elif args.strategy == "latent-bandit":
            arm = bandit.select()
            schedule = sample_region_schedule(assets.regions[arm], num_threads, args.schedule_len, rng)
        else:  # decoder
            arm = bandit.select()
            centroid = assets.regions[arm].centroid
            z = centroid + args.decoder_sigma * rng.standard_normal(centroid.shape[0])
            schedule = decode_schedule(assets.model, z, num_threads, args.schedule_len, assets.device)

        # ── Execute it ────────────────────────────────────────────────────
        result = run_schedule(args.target, schedule, workload, wall_timeout=args.wall_timeout)

        if args.strategy in ("latent-bandit", "decoder"):
            bandit.update(arm, result.reward)

        bugs += int(result.bug)
        voids += int(not result.valid)
        reward_sum += result.reward
        best_reward = max(best_reward, result.reward)

        records.append({
            "trial": trial,
            "arm": arm,
            "bug": bool(result.bug),
            "reward": round(float(result.reward), 6),
            "valid": bool(result.valid),
            "abort_reason": result.abort_reason,
            "steps": result.steps,
            "landed_region": _encode_region(assets, result.events) if not args.no_encode else None,
            "detail": result.detail,
        })

        if trial % args.report_every == 0 or trial == args.trials:
            elapsed = time.perf_counter() - t_start
            pulls = "-" if args.strategy in ("random", "latent-uniform") else [int(p) for p in bandit.pulls]
            print(f"        {trial:>6d} {bugs:>6d} {100.0 * bugs / trial:>6.1f}% "
                  f"{reward_sum / trial:>8.4f} {best_reward:>8.4f} {voids:>5d} "
                  f"{elapsed:>8.1f}s  {pulls}")

    elapsed = time.perf_counter() - t_start

    # ── Summarise ─────────────────────────────────────────────────────────
    valid_records = [r for r in records if r["valid"]]
    first_bug = next((r["trial"] for r in records if r["bug"]), None)

    summary = {
        "target": args.target,
        "strategy": args.strategy,
        "trials": args.trials,
        "seed": args.seed,
        "bugs": bugs,
        "bug_rate": bugs / args.trials if args.trials else 0.0,
        "mean_reward": reward_sum / args.trials if args.trials else 0.0,
        "best_reward": best_reward,
        "trials_to_first_bug": first_bug,
        "invalid_runs": voids,
        "elapsed_sec": round(elapsed, 2),
        "bandit": bandit.stats() if args.strategy in ("latent-bandit", "decoder") else None,
        "regions": [
            {"cluster": r.cluster, "variant": r.dominant_variant,
             "bug_rate": r.bug_rate, "switch_prob": round(r.switch_prob, 4)}
            for r in assets.regions
        ],
    }

    print(f"\n  [3/3] Done in {elapsed:.1f}s")
    print(f"        bugs found          : {bugs}/{args.trials}  ({100.0 * summary['bug_rate']:.1f}%)")
    print(f"        mean reward         : {summary['mean_reward']:.4f}")
    print(f"        best reward         : {best_reward:.4f}")
    print(f"        trials to first bug : {first_bug}")
    print(f"        invalid runs        : {voids}")
    if summary["bandit"]:
        print("\n        bandit arms:")
        for s in summary["bandit"]:
            print(f"          arm {s['arm']} [{s['variant']:<14s}] pulls={s['pulls']:>4d} "
                  f"mean_reward={s['mean_reward']:.4f} posterior={s['posterior_mean']:.3f}")
    if valid_records and valid_records[0]["landed_region"] is not None:
        landed = [r["landed_region"] for r in valid_records]
        counts = {k: landed.count(k) for k in sorted(set(landed))}
        print(f"\n        produced traces landed in regions: {counts}")

    out_path = args.out or os.path.join(
        PROJECT_ROOT, "results", f"{args.target}_{args.strategy}_seed{args.seed}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({"summary": summary, "trials": records}, fp, indent=2)

    print(f"\n        results -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
