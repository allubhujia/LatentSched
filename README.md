# LatentSched

Latent-space-guided search for concurrency bugs.

A VAE learns an unsupervised embedding of execution traces from buggy and
correct concurrent programs. That latent space is then partitioned into
regions, and those regions drive a search over *thread interleavings*: a
bandit picks a region, a schedule is generated in that region's style, and a
deterministic scheduler forces the program to follow it. Bugs that surface
become the bandit's reward.

The point is the loop — observed executions shape a latent space, and the
latent space shapes which executions get tried next.

```
concurrency_problems/   buggy + correct programs
        │
        ▼
trace_harness/          instrument, run, record 300 traces
        │
        ▼
vae/                    BiLSTM VAE  →  16-d latent space
        │
        ▼
evaluation/             cluster into latent regions, measure separability
        │
        ▼
scheduler/              region → schedule → forced interleaving → bug → reward
```

## Running the pipeline

Everything runs from the project root. Only stage 3 trains anything; stages 1,
2, 4 and 5 reuse the saved checkpoint.

```bash
# 1. Generate traces          (~2 min, writes 300 JSON files)
cd trace_harness && python runner.py && cd ..

# 2. Verify the scheduler     (~5 s, no model needed)
python -m scheduler.targets

# 3. Train the VAE            (the only training step; skip if the checkpoint exists)
python -m vae.train_vae

# 4. Partition the latent space into regions
python -m evaluation.evaluate_latent
python -m evaluation.evaluate_cluster

# 5. Search for bugs, then compare strategies
python -m scheduler.search --target race     --strategy random        --trials 200 --seed 0
python -m scheduler.search --target race     --strategy latent-bandit --trials 200 --seed 0
python -m evaluation.evaluate_search --target race
```

Stage 4 writes `evaluation/latent_regions.npz`, the handoff artifact the
scheduler consumes: cluster centroids become bandit arms and each cluster's
bug rate becomes that arm's reward prior.

## Inspecting traces in the browser

```bash
python frontend/serve.py      # opens localhost:8000/frontend/trace-bench.html
```

Trace Bench loads any run from `trace_harness/traces/` and shows its thread
timeline, per-thread lock health, and the exact feature rows the encoder
produces for it. It reimplements `vae/trace_encoder.py` in JavaScript, so the
encoding rules are duplicated and have to be kept in sync — `frontend/README.md`
lists which parts. Serving the project root is required; over `file://` the
browser cannot fetch the corpus and the page falls back to bundled samples.

## How the scheduler works

Reproducing a concurrency bug means controlling the interleaving, not hoping
for it. `scheduler/controller.py` makes every instrumented event a scheduling
point: a thread parks there and waits to be handed the turn.

A decision is only taken once *every unfinished thread is parked*. The
controller therefore always chooses with full knowledge of where each thread
stands, which makes an execution a pure function of its schedule — the same
schedule replays to the same final balance, every time.

Locks need care. A thread blocking on a real `threading.Lock` would vanish
from the scheduler and the barrier would never complete, so instrumented locks
acquire non-blockingly and re-park on failure. That keeps every thread visible
and turns deadlock into something decidable: when every unfinished thread is
parked on a lock it could not take, *and* each has retried since the last time
any thread made progress, nothing can ever change and the run is a deadlock.
Both halves matter — a blocked flag goes stale the moment its lock is freed,
so without the retry requirement five threads queued on one mutex are
indistinguishable from a deadlock.

## Targets

| Target | Bug | Reward |
|---|---|---|
| `race` | Final balance disagrees with what the threads transacted | Fraction of transacted value destroyed by lost updates |
| `safe` | *(control — should never fire)* | Always 0 |
| `deadlock` | Every thread blocked on a lock another holds | 1 if deadlocked |

`safe` runs the identical workload against the lock-protected account. It is
the false-positive check: any reward above zero there means the harness is
reporting bugs that do not exist.

On `race`, nearly any schedule with real interleaving loses at least one
update, so the *bug rate* saturates near 100% and carries no information. The
graded severity reward is what separates strategies there. On `deadlock` the
outcome is binary and the bug rate is the measure.

## Search strategies

| Strategy | Schedule source |
|---|---|
| `random` | Uniform choice at every decision point — the baseline |
| `latent-uniform` | A latent region's interleaving style, region chosen uniformly |
| `latent-bandit` | Same generator, region chosen by Thompson sampling |
| `decoder` | The VAE decoder's thread-index column |

`latent-uniform` exists to separate two questions that are easy to conflate:
whether latent regions generate *better schedules*, and whether the bandit
*picks between them* well. Comparing it against `random` answers the first;
against `latent-bandit`, the second.

## Findings

**The latent space separates variants perfectly.** k-means at k=3 recovers the
three variants exactly (ARI = 1.00, NMI = 1.00), and k=3 is also what the
silhouette sweep selects. Silhouette by variant is 0.894.

**But "buggy vs safe" is not the axis it separates on.** Silhouette by the
binary buggy/safe split is only 0.172. The embedding's dominant split is
race-vs-everything-else: the race centroid sits 3.65 away from both others,
while the deadlock and safe centroids are 0.048 apart. Deadlock and safe are
still separable — they are recovered perfectly — but by a very thin margin,
and nothing about the geometry groups the two *bug* classes together.

**The decoder cannot generate schedules.** This was the original design, and
it does not work on this checkpoint. Varying z moves the decoded thread-index
column by a standard deviation of ~0.06, while the column varies by ~0.33
along its own length — so every latent vector decodes to essentially the same
near-serial ramp, which provably produces no race at all. A single z repeated
across 300 steps and trained under MSE against three highly stereotyped trace
shapes has no capacity left for event-level interleaving detail. The `decoder`
strategy is kept so the claim stays measurable rather than assumed.

**What the regions do carry is interleaving style.** The traces inside each
region differ exactly where it matters: they switch running thread on 49% of
events in the race region, 38% in the safe region, and use only two threads in
the deadlock region. Generation is fitted to those statistics instead of the
decoder — which is what `latent-bandit` and `latent-uniform` sample from.

## Module map

| Path | Role |
|---|---|
| `concurrency_problems/` | The three bank-account programs: safe, racy, deadlocking |
| `trace_harness/instrumented_bank.py` | Wraps them to emit events; hosts the scheduling hook |
| `trace_harness/runner.py` | Generates the trace corpus |
| `vae/trace_encoder.py` | Trace JSON → `(300, 5)` feature tensor |
| `vae/vae_model.py` | BiLSTM VAE + masked loss with free bits |
| `vae/train_vae.py` | Training loop, cyclical β-annealing, MLflow |
| `evaluation/evaluate_latent.py` | PCA / t-SNE views of the latent space |
| `evaluation/evaluate_cluster.py` | k-sweep, region partition, writes `latent_regions.npz` |
| `evaluation/evaluate_search.py` | Compares search strategies |
| `scheduler/controller.py` | Deterministic schedule-driven thread scheduler |
| `scheduler/targets.py` | Runs the programs under a schedule; scores the outcome |
| `scheduler/latent_policy.py` | Region models, schedule generation, the bandit |
| `scheduler/search.py` | Search campaign CLI |
| `frontend/trace-bench.html` | Browser inspector for traces and their encoded feature rows |
| `frontend/serve.py` | Serves the project root so the page can read the corpus |

## Requirements

`torch`, `numpy`, `scikit-learn`, `matplotlib`. `mlflow` is optional — every
script falls back to stdout without it.
