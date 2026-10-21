<div align="center">

# LatentSched

**Latent-space-guided search for concurrency bugs**

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM%20VAE-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-k--means-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)

</div>

---

LatentSched learns what concurrent executions *look like*, then uses that
knowledge to decide which thread interleavings to try next.

A variational autoencoder embeds execution traces from buggy and correct
concurrent programs into a 16-dimensional latent space. That space is
partitioned into regions, and the regions drive a search over thread
schedules: a bandit picks a region, a schedule is generated in that region's
interleaving style, and a deterministic scheduler forces the program to follow
it. Any bug that surfaces becomes the bandit's reward.

> **The core idea is the loop:** observed executions shape the latent space,
> and the latent space shapes which executions get tried next.

## Table of contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Deterministic scheduling](#deterministic-scheduling)
- [Targets and search strategies](#targets-and-search-strategies)
- [Findings](#findings)
- [Trace Bench](#trace-bench)
- [Project structure](#project-structure)
- [Limitations](#limitations)

## How it works

```
 ┌─────────────────────────────────────────────┐
 │  1. CONCURRENCY PROGRAMS                    │
 │     safe  ·  race_condition  ·  deadlock    │
 └──────────────────────┬──────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────┐
 │  2. TRACE HARNESS                           │
 │     instrument locks & balance access       │
 │     record 300 execution traces (JSON)      │
 └──────────────────────┬──────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────┐
 │  3. VAE                                     │
 │     trace → (300 × 5) tensor                │
 │     BiLSTM encoder → 16-d latent space      │
 └──────────────────────┬──────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────┐
 │  4. LATENT REGIONS                          │
 │     k-means partition → latent_regions.npz  │
 │     centroids = arms · bug rate = prior     │
 └──────────────────────┬──────────────────────┘
                        │
     ╔══════════════════╪═══ SEARCH LOOP ═══════════╗
     ║                  ▼                           ║
     ║   ┌─────────────────────────────────┐        ║
     ║   │  5a. BANDIT                     │◄───┐   ║
     ║   │      Thompson sampling          │    │   ║
     ║   │      picks a latent region      │    │   ║
     ║   └────────────────┬────────────────┘    │   ║
     ║                    ▼                     │   ║
     ║   ┌─────────────────────────────────┐    │   ║
     ║   │  5b. SCHEDULE GENERATOR         │    │   ║
     ║   │      sample a thread schedule   │    │   ║
     ║   │      in the region's style      │    │   ║
     ║   └────────────────┬────────────────┘    │   ║
     ║                    ▼                     │   ║
     ║   ┌─────────────────────────────────┐    │   ║
     ║   │  5c. DETERMINISTIC CONTROLLER   │    │   ║
     ║   │      force the interleaving     │    │   ║
     ║   │      detect race / deadlock     │    │   ║
     ║   └────────────────┬────────────────┘    │   ║
     ║                    └──── reward ─────────┘   ║
     ╚══════════════════════════════════════════════╝
```

| Stage | What happens |
|---|---|
| **1. Programs** | Three bank-account programs: a lock-protected account (safe), an unsynchronised read-modify-write account (race), and two-account transfers that lock in argument order (deadlock). |
| **2. Trace harness** | The real classes are wrapped to emit `LOCK_*`, `READ/WRITE_BALANCE` and `DEPOSIT/WITHDRAW_*` events. 100 runs per variant are recorded as JSON. |
| **3. VAE** | Each trace becomes a `(300, 5)` tensor: event type, relative thread index, log-scaled value, has-value flag, normalised time gap. A BiLSTM encoder maps it to a 16-d latent vector, trained with cyclical β-annealing and free bits. |
| **4. Regions** | k-means over the latent vectors (k chosen by silhouette sweep). Centroids become bandit arms; each cluster's bug rate becomes that arm's prior. |
| **5. Search** | Thompson sampling chooses a region, a schedule is sampled in that region's interleaving style, and the controller replays it exactly. The outcome is scored and fed back. |

## Quick start

### Requirements

```bash
pip install torch numpy scikit-learn matplotlib
pip install mlflow   # optional: every script falls back to stdout without it
```

### Running the pipeline

All commands run from the project root. Only step 3 trains a model; the rest
reuse the saved checkpoint.

```bash
# 1. Generate the trace corpus          (~2 min, writes 300 JSON files)
cd trace_harness && python runner.py && cd ..

# 2. Verify the scheduler is deterministic   (~5 s, no model needed)
python -m scheduler.targets

# 3. Train the VAE                        (writes checkpoints/trace_vae_checkpoint.pt)
python -m vae.train_vae

# 4. Visualise and partition the latent space
python -m evaluation.evaluate_latent
python -m evaluation.evaluate_cluster

# 5. Search for bugs and compare strategies
python -m scheduler.search --target race --strategy random        --trials 200 --seed 0
python -m scheduler.search --target race --strategy latent-bandit --trials 200 --seed 0
python -m evaluation.evaluate_search --target race
```

> [!NOTE]
> Model checkpoints, MLflow runs and generated plots are git-ignored. After a
> fresh clone, run step 3 before steps 4 and 5.

Step 4 writes `evaluation/latent_regions.npz`, the handoff artifact the
scheduler consumes. Step 5 writes one JSON file per campaign to `results/`.

## Deterministic scheduling

Reproducing a concurrency bug means *controlling* the interleaving, not hoping
for it. `scheduler/controller.py` turns every instrumented event into a
scheduling point where a thread parks and waits to be handed the turn.

- **Full-barrier decisions.** The controller only chooses once *every*
  unfinished thread is parked, so it always decides with complete knowledge of
  where each thread stands. An execution is therefore a pure function of its
  schedule: the same schedule replays to the same final balance, every time.
- **Visible blocking.** A thread blocking on a real `threading.Lock` would
  vanish from the scheduler and the barrier would never complete. Instrumented
  locks acquire non-blockingly and re-park on failure instead.
- **Sound deadlock detection.** A deadlock is declared only when every
  unfinished thread is parked on a lock it could not take, *and* each has
  retried since the last time any thread made progress. The retry requirement
  matters: without it, five threads queued on one mutex are indistinguishable
  from a deadlock.

## Targets and search strategies

### Targets

| Target | Bug | Reward |
|---|---|---|
| `race` | Final balance disagrees with what the threads transacted | Fraction of transacted value destroyed by lost updates |
| `safe` | *Control: should never fire* | Always 0 |
| `deadlock` | Every thread blocked on a lock another holds | 1 if deadlocked |

`safe` runs the identical workload against the lock-protected account. It is
the false-positive check: any reward above zero there means the harness is
reporting bugs that do not exist.

On `race`, nearly any schedule with real interleaving loses at least one
update, so the bug *rate* saturates near 100% and carries no information. The
graded severity reward is what separates strategies there. On `deadlock` the
outcome is binary and the bug rate is the measure.

### Strategies

| Strategy | Schedule source |
|---|---|
| `random` | Uniform choice at every decision point (the baseline) |
| `latent-uniform` | A latent region's interleaving style, region chosen uniformly |
| `latent-bandit` | Same generator, region chosen by Thompson sampling |
| `decoder` | The VAE decoder's thread-index column |

`latent-uniform` separates two questions that are easy to conflate: whether
latent regions generate *better schedules* (compare against `random`), and
whether the bandit *picks between them* well (compare against
`latent-bandit`).

## Findings

| Metric | Value |
|---|---|
| Adjusted Rand index (k=3 vs. true variant) | **1.00** |
| Normalised mutual information | **1.00** |
| Silhouette by variant | 0.894 |
| Silhouette by buggy / safe | 0.172 |
| Race centroid distance to others | ~3.65 |
| Deadlock ↔ safe centroid distance | 0.048 |

**The latent space separates variants perfectly.** k-means at k=3 recovers
the three variants exactly, and k=3 is also what the silhouette sweep selects.

**But "buggy vs. safe" is not the axis it separates on.** The dominant split is
race versus everything else. Deadlock and safe are still recovered perfectly,
but by a very thin margin, and nothing in the geometry groups the two *bug*
classes together.

**The decoder cannot generate schedules.** This was the original design, and
it does not work on this checkpoint. Varying z moves the decoded thread-index
column by a standard deviation of ~0.06, while the column varies by ~0.33
along its own length, so every latent vector decodes to essentially the same
near-serial ramp. A single z repeated across 300 steps and trained under MSE
against three stereotyped trace shapes has no capacity left for event-level
interleaving detail. The `decoder` strategy is kept so this stays measurable
rather than assumed.

**What the regions do carry is interleaving style.** Traces inside each region
switch running thread on 49% of events in the race region and 38% in the safe
region, and use only two threads in the deadlock region. Schedule generation
is fitted to those statistics instead of the decoder, which is what
`latent-bandit` and `latent-uniform` sample from.

## Trace Bench

A browser inspector for the trace corpus.

```bash
python frontend/serve.py      # opens http://localhost:8000/frontend/trace-bench.html
```

Load any run from `trace_harness/traces/` to see its thread timeline,
per-thread lock health, and the exact feature rows the encoder produces for
it. The page reimplements `vae/trace_encoder.py` in JavaScript, so the two
must be kept in sync; [`frontend/README.md`](frontend/README.md) lists which
parts. Serve the project root rather than opening the file directly: over
`file://` the browser cannot fetch the corpus and the page falls back to
bundled samples.

## Project structure

```
LatentSched/
├── concurrency_problems/        # the programs under test
│   ├── safe/                    #   lock-protected account
│   ├── race_condition/          #   unsynchronised account
│   └── deadlock/                #   opposite-order two-lock transfer
├── trace_harness/
│   ├── instrumented_bank.py     # event logging + scheduling hook
│   ├── runner.py                # generates the trace corpus
│   └── traces/                  # 100 JSON traces per variant
├── vae/
│   ├── trace_encoder.py         # trace JSON → (300, 5) tensor
│   ├── vae_model.py             # BiLSTM VAE, masked loss with free bits
│   └── train_vae.py             # training loop, β-annealing, MLflow
├── evaluation/
│   ├── evaluate_latent.py       # PCA / t-SNE views of the latent space
│   ├── evaluate_cluster.py      # k-sweep, regions → latent_regions.npz
│   └── evaluate_search.py       # compares search strategies
├── scheduler/
│   ├── controller.py            # deterministic schedule-driven scheduler
│   ├── targets.py               # runs programs under a schedule, scores them
│   ├── latent_policy.py         # region models, schedule generation, bandit
│   └── search.py                # search campaign CLI
└── frontend/                    # Trace Bench inspector
```

## Limitations

- **Small, synthetic corpus.** 300 traces from three toy programs. The
  separation results say more about these programs than about concurrency bugs
  in general.
- **Labels enter after training.** The VAE is trained without labels, but each
  region's bug-rate prior is computed from variant labels.
- **Training/search distribution shift.** The corpus is recorded under OS
  scheduling with sleeps; search-time traces are produced under the controller
  without them.
- **Encoder is trace-length sensitive.** Deadlock traces are ~9 events padded
  to 300, so sequence length is a strong separating signal in its own right.
