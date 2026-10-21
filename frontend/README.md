# Trace Bench

Browser inspector for LatentSched execution traces. Pick a trace from the
corpus and it shows the thread timeline, the lock-acquire/acquired balance per
thread, and the exact feature rows `vae/trace_encoder.py` would hand the VAE.

## Running

From the project root:

```bash
python frontend/serve.py
```

That serves the project root at `http://localhost:8000` and opens
`/frontend/trace-bench.html`. Serving the root is what makes the page and the
trace corpus same-origin — opening the HTML directly over `file://` leaves the
browser unable to fetch the traces, and the page falls back to three small
bundled samples with a warning in the left rail.

The corpus must exist first:

```bash
cd trace_harness && python runner.py && cd ..
```

## What it reads

`../trace_harness/traces/<variant>/trace_run_<NNN>.json`, for the three variant
folders `safe`, `race_condition` and `deadlock`, runs 001–100. The run number
is set in the left rail. Any other trace in the same JSON shape can be uploaded
or pasted instead.

## Encoder parity

`app.js` reimplements `vae/trace_encoder.py` in JavaScript so the table shows
real encoder output rather than an approximation. The parts that must stay in
sync if the Python changes:

- `EVENT_TYPE_VOCAB` — the 12-token event-type vocabulary, ID 0 reserved for padding
- `SEQ_LEN = 300` — applied *before* the thread map and the delta-time
  normaliser are built, because Python truncates first and both are derived
  from that same prefix
- `buildThreadIndex` — appearance-order indices, `MainThread` pinned to 0
- `safeLogValue` — sign-preserving `log1p(abs(v))`
- delta times — `log1p(ns / 1e6)`, then normalised by the per-trace maximum

The truncation is load-bearing here: safe runs are ~529 events and race runs
~301, so both lose events at `SEQ_LEN`, and the page reports how many.
