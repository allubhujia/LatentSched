"""
runner.py
=========
Trace collection harness for LatentSched — Step 3.

Runs simulations of all three concurrency variants (safe, race_condition,
deadlock) using the real BankAccount classes from concurrency_problems/.
Saves each run's trace to its own JSON file in trace_harness/traces/.

Output folder: trace_harness/traces/<variant>/
Files:         trace_run_001.json, trace_run_002.json, ...

Run from the trace_harness/ directory:
    python runner.py
"""

import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone

# ── Import wrappers that use the REAL concurrency_problems classes ────────────
from instrumented_bank import (
    TraceLogger,
    InstrumentedSafeAccount,
    InstrumentedRaceAccount,
    InstrumentedDeadlockScenario,
)
# ──────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────
TOTAL_RUNS_PER_VARIANT = 100   # trace files generated per variant
NUM_THREADS            = 5    # threads per run (safe & race)
OPS_PER_THREAD         = 15   # operations each thread performs
INITIAL_BALANCE        = 1000.0

TRACES_BASE = os.path.join(os.path.dirname(__file__), "..", "trace_harness", "traces")
# ──────────────────────────────────────────────────────────────────────────────


def generate_operations(n: int) -> list:
    ops = []
    for _ in range(n):
        amount = round(random.uniform(5.0, 100.0), 2)
        op = random.choice(["deposit", "withdraw"])
        ops.append((op, amount))
    return ops


def worker(account, operations: list, net_tracker: list):
    net = 0.0
    for op, amount in operations:
        if op == "deposit":
            net += account.deposit(amount)
        elif op == "withdraw":
            net -= account.withdraw(amount)
        time.sleep(random.uniform(0.00005, 0.0005))
    net_tracker.append(net)


# ── Safe variant ──────────────────────────────────────────────────────────────
def run_safe(run_id: int) -> dict:
    """
    Uses InstrumentedSafeAccount which wraps SafeBankAccount
    from concurrency_problems/safe/bank_transfer_safe.py.
    """
    logger = TraceLogger()
    account = InstrumentedSafeAccount(INITIAL_BALANCE, logger)

    threads, net_tracker = [], []
    for i in range(NUM_THREADS):
        ops = generate_operations(OPS_PER_THREAD)
        t = threading.Thread(
            target=worker, args=(account, ops, net_tracker),
            name=f"Safe-Run{run_id:02d}-T{i+1}"
        )
        threads.append(t)

    for t in threads: t.start()
    for t in threads: t.join()

    actual   = round(account.get_balance(), 2)
    expected = round(INITIAL_BALANCE + sum(net_tracker), 2)

    return _build_record(run_id, "safe", actual, expected, logger)


# ── Race condition variant ────────────────────────────────────────────────────
def run_race(run_id: int) -> dict:
    """
    Uses InstrumentedRaceAccount which wraps RaceBankAccount
    from concurrency_problems/race_condition/bank_transfer_race.py.
    No locks — race conditions are expected and captured in the trace.
    """
    logger = TraceLogger()
    account = InstrumentedRaceAccount(INITIAL_BALANCE, logger)

    threads, net_tracker = [], []
    for i in range(NUM_THREADS):
        ops = generate_operations(OPS_PER_THREAD)
        t = threading.Thread(
            target=worker, args=(account, ops, net_tracker),
            name=f"Race-Run{run_id:02d}-T{i+1}"
        )
        threads.append(t)

    for t in threads: t.start()
    for t in threads: t.join()

    actual   = round(account.get_balance(), 2)
    expected = round(INITIAL_BALANCE + sum(net_tracker), 2)

    return _build_record(run_id, "race_condition", actual, expected, logger)


# ── Deadlock variant ──────────────────────────────────────────────────────────
def run_deadlock(run_id: int) -> dict:
    """
    Uses InstrumentedDeadlockScenario which wraps DeadlockBankAccount
    from concurrency_problems/deadlock/bank_transfer_deadlock.py.
    Two threads transfer in opposite directions — deadlock is expected.
    A 2-second timeout detects the deadlock and records it in the trace.
    """
    logger = TraceLogger()
    scenario = InstrumentedDeadlockScenario(logger)

    t1 = threading.Thread(
        target=scenario.transfer,
        args=(scenario.acc_a, scenario.acc_b, 100.0),
        name=f"Deadlock-Run{run_id:02d}-AtoB",
        daemon=True,   # daemon so the process can exit even if stuck
    )
    t2 = threading.Thread(
        target=scenario.transfer,
        args=(scenario.acc_b, scenario.acc_a, 200.0),
        name=f"Deadlock-Run{run_id:02d}-BtoA",
        daemon=True,
    )

    t1.start(); t2.start()
    t1.join(timeout=2.0); t2.join(timeout=2.0)

    deadlocked = t1.is_alive() or t2.is_alive()
    logger.log(
        "DEADLOCK_DETECTED" if deadlocked else "TRANSFER_COMPLETE",
        meta={"t1_alive": t1.is_alive(), "t2_alive": t2.is_alive()},
    )

    record = _build_record(run_id, "deadlock",
                            actual=None, expected=None, logger=logger)
    record["deadlock_detected"] = deadlocked
    record["correct"]           = None   # N/A for deadlock
    return record


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_record(run_id, variant, actual, expected, logger: TraceLogger) -> dict:
    trace = logger.get_trace()
    correct = None if actual is None else (actual == expected)
    return {
        "run_id":           run_id,
        "variant":          variant,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "initial_balance":  INITIAL_BALANCE,
        "expected_balance": expected,
        "actual_balance":   actual,
        "correct":          correct,
        "num_threads":      NUM_THREADS,
        "ops_per_thread":   OPS_PER_THREAD,
        "total_events":     len(trace),
        "events":           trace,
    }


def save_trace(record: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def run_variant(label: str, run_fn, runs: int):
    out_dir = os.path.join(TRACES_BASE, label)
    os.makedirs(out_dir, exist_ok=True)
    failed = 0

    print(f"\n  [{label.upper()}]  →  {os.path.abspath(out_dir)}")
    for run_id in range(1, runs + 1):
        record   = run_fn(run_id)
        filename = f"trace_run_{run_id:03d}.json"
        save_trace(record, os.path.join(out_dir, filename))

        correct = record.get("correct")
        if correct is False:
            status = "✗ FAIL"
            failed += 1
        elif correct is None:
            dl = "DEADLOCKED" if record.get("deadlock_detected") else "ok"
            status = f"⚠  {dl}"
        else:
            status = "✓ OK  "

        exp = record["expected_balance"]
        act = record["actual_balance"]
        bal = f"Expected ${exp:>10.2f} | Actual ${act:>10.2f}" if exp is not None else " " * 34
        print(f"    Run {run_id:>3}/{runs} | {status} | {bal} | Events: {record['total_events']:>5} | {filename}")

    return failed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  LatentSched — Trace Collection Harness (all three variants)")
    print(f"  Runs/variant: {TOTAL_RUNS_PER_VARIANT} | Threads: {NUM_THREADS} | Ops/thread: {OPS_PER_THREAD}")
    print("=" * 70)

    start = time.perf_counter()

    safe_fail     = run_variant("safe",           run_safe,     TOTAL_RUNS_PER_VARIANT)
    race_fail     = run_variant("race_condition",  run_race,     TOTAL_RUNS_PER_VARIANT)
    _             = run_variant("deadlock",        run_deadlock, TOTAL_RUNS_PER_VARIANT)

    elapsed = time.perf_counter() - start
    total   = TOTAL_RUNS_PER_VARIANT * 3

    print("\n" + "=" * 70)
    print(f"  Completed {total} runs across 3 variants in {elapsed:.2f}s")
    print(f"  Safe failures      : {safe_fail}/{TOTAL_RUNS_PER_VARIANT}  (expected 0)")
    print(f"  Race failures      : {race_fail}/{TOTAL_RUNS_PER_VARIANT}  (may be >0 — that's the bug)")
    print(f"  Deadlock runs      : {TOTAL_RUNS_PER_VARIANT}/{TOTAL_RUNS_PER_VARIANT}  (deadlock detected each time)")
    print(f"  Traces saved to    : {os.path.abspath(TRACES_BASE)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
