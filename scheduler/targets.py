"""
targets.py — Runs the concurrency programs under a controller-dictated schedule.

Each target executes a *fixed* workload, so the schedule is the only thing that
varies between trials and any difference in outcome is attributable to the
interleaving alone.

Targets
-------
race      Unsynchronised account. Bug = final balance disagrees with the sum of
          what the threads believe they transacted. Reward is graded by how
          much value the lost updates destroyed.
safe      Lock-protected account, same workload. A control: a correct search
          reports no bugs here, so any reward above zero is a false positive.
deadlock  Two threads transferring in opposite directions, taking locks in
          argument order. Bug = the controller observes every thread blocked.
"""

import random
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from trace_harness.instrumented_bank import (
    InstrumentedDeadlockScenario,
    InstrumentedRaceAccount,
    InstrumentedSafeAccount,
    TraceLogger,
)

from .controller import ScheduleController, ScheduleAborted

TARGETS = ("race", "safe", "deadlock")

# Sized so a run spans roughly 200 scheduling decisions — comfortably inside
# the VAE's 300-step sequence, so one decoded trace covers a whole execution.
NUM_THREADS = 5
OPS_PER_THREAD = 8
INITIAL_BALANCE = 1000.0

# Fixed so every trial faces the identical workload.
WORKLOAD_SEED = 20240801

Operation = Tuple[str, float]


@dataclass
class RunResult:
    """Outcome of executing one schedule against one target."""

    target: str
    bug: bool
    reward: float
    valid: bool                       # False if the run was abandoned
    deadlock: bool = False
    abort_reason: Optional[str] = None
    steps: int = 0
    detail: Dict = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        state = "BUG " if self.bug else ("----" if self.valid else "VOID")
        return (f"{state} reward={self.reward:.3f} steps={self.steps:>4d} "
                f"events={len(self.events):>4d}")


# ── Workload ──────────────────────────────────────────────────────────────────

def make_workload(
    num_threads: int = NUM_THREADS,
    ops_per_thread: int = OPS_PER_THREAD,
    seed: int = WORKLOAD_SEED,
) -> List[List[Operation]]:
    """Build the fixed per-thread operation lists shared by every trial."""
    rng = random.Random(seed)
    workload = []
    for _ in range(num_threads):
        ops = [
            (rng.choice(["deposit", "withdraw"]), round(rng.uniform(5.0, 100.0), 2))
            for _ in range(ops_per_thread)
        ]
        workload.append(ops)
    return workload


# ── Account targets (race / safe) ─────────────────────────────────────────────

def _account_worker(controller, thread_id, account, ops, nets, volumes, slot):
    controller.register(thread_id)
    net = 0.0
    volume = 0.0
    try:
        for op, amount in ops:
            moved = account.deposit(amount) if op == "deposit" else account.withdraw(amount)
            net += moved if op == "deposit" else -moved
            volume += moved
    except ScheduleAborted:
        pass
    finally:
        # Written even on abort: slot-indexed rather than appended, so the
        # bookkeeping itself never races.
        nets[slot] = net
        volumes[slot] = volume
        controller.unregister()


def _run_account_target(
    kind: str,
    schedule: Sequence[int],
    workload: List[List[Operation]],
    wall_timeout: float,
) -> RunResult:
    n = len(workload)
    controller = ScheduleController(schedule, num_threads=n, wall_timeout=wall_timeout)
    logger = TraceLogger(controller=controller)

    account_cls = InstrumentedRaceAccount if kind == "race" else InstrumentedSafeAccount
    account = account_cls(INITIAL_BALANCE, logger)

    nets = [0.0] * n
    volumes = [0.0] * n
    threads = []
    for slot, ops in enumerate(workload):
        thread_id = slot + 1          # 0 is the encoder's MainThread index
        threads.append(threading.Thread(
            target=_account_worker,
            args=(controller, thread_id, account, ops, nets, volumes, slot),
            name=f"{kind.capitalize()}-Sched-T{thread_id}",
            daemon=True,
        ))

    controller.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=wall_timeout + 2.0)

    leaked = any(t.is_alive() for t in threads)

    # Read the balance off the real account rather than through the
    # instrumented accessor: that path takes the account lock, which an
    # abandoned run may have left held by a thread that is no longer there.
    actual = round(account._real._balance, 2)
    expected = round(INITIAL_BALANCE + sum(nets), 2)
    volume = sum(volumes)

    lost = abs(expected - actual)
    bug = actual != expected
    reward = min(1.0, lost / volume) if volume > 0 else 0.0

    valid = (not controller.aborted) and not leaked
    if not valid:
        bug = False
        reward = 0.0

    return RunResult(
        target=kind,
        bug=bug,
        reward=reward,
        valid=valid,
        abort_reason="thread_leak" if leaked else controller.abort_reason,
        steps=controller.steps,
        detail={"expected": expected, "actual": actual, "lost": round(lost, 2),
                "volume": round(volume, 2)},
        events=logger.get_trace(),
    )


# ── Deadlock target ───────────────────────────────────────────────────────────

def _transfer_worker(controller, thread_id, scenario, acc_from, acc_to, amount):
    controller.register(thread_id)
    try:
        scenario.transfer(acc_from, acc_to, amount)
    except ScheduleAborted:
        pass
    finally:
        controller.unregister()


def _run_deadlock_target(schedule: Sequence[int], wall_timeout: float) -> RunResult:
    controller = ScheduleController(schedule, num_threads=2, wall_timeout=wall_timeout)
    logger = TraceLogger(controller=controller)
    scenario = InstrumentedDeadlockScenario(logger)

    threads = [
        threading.Thread(
            target=_transfer_worker,
            args=(controller, 1, scenario, scenario.acc_a, scenario.acc_b, 100.0),
            name="Deadlock-Sched-AtoB", daemon=True,
        ),
        threading.Thread(
            target=_transfer_worker,
            args=(controller, 2, scenario, scenario.acc_b, scenario.acc_a, 200.0),
            name="Deadlock-Sched-BtoA", daemon=True,
        ),
    ]

    controller.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=wall_timeout + 2.0)

    leaked = any(t.is_alive() for t in threads)
    deadlock = controller.deadlock_detected

    # A deadlock run *is* aborted — that abort is the discovery, not a failure.
    valid = deadlock or ((not controller.aborted) and not leaked)

    return RunResult(
        target="deadlock",
        bug=bool(deadlock and valid),
        reward=1.0 if (deadlock and valid) else 0.0,
        valid=valid,
        deadlock=deadlock,
        abort_reason="thread_leak" if leaked else controller.abort_reason,
        steps=controller.steps,
        detail={"balance_a": round(scenario.acc_a.balance, 2),
                "balance_b": round(scenario.acc_b.balance, 2)},
        events=logger.get_trace(),
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_schedule(
    target: str,
    schedule: Sequence[int],
    workload: Optional[List[List[Operation]]] = None,
    wall_timeout: float = 15.0,
) -> RunResult:
    """Execute one schedule against one target and report what happened."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")

    if target == "deadlock":
        return _run_deadlock_target(schedule, wall_timeout)

    if workload is None:
        workload = make_workload()
    return _run_account_target(target, schedule, workload, wall_timeout)


def num_threads_for(target: str) -> int:
    """How many worker threads a target runs — the schedule's alphabet size."""
    return 2 if target == "deadlock" else NUM_THREADS


# ── Self-check ────────────────────────────────────────────────────────────────

def _self_check():
    """Verify a schedule reproduces its outcome exactly. Run: python -m scheduler.targets"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rng = random.Random(7)
    print("=" * 72)
    print("  scheduler.targets - determinism self-check")
    print("=" * 72)

    for target in TARGETS:
        n_threads = num_threads_for(target)
        schedule = [rng.randrange(1, n_threads + 1) for _ in range(300)]
        results = [run_schedule(target, schedule) for _ in range(3)]

        fingerprints = {(r.bug, round(r.reward, 6), r.valid, len(r.events)) for r in results}
        verdict = "DETERMINISTIC" if len(fingerprints) == 1 else "NON-DETERMINISTIC"
        r0 = results[0]
        print(f"\n  [{target}]  {verdict}")
        print(f"    {r0.summary()}  detail={r0.detail}")
        if r0.abort_reason:
            print(f"    abort_reason={r0.abort_reason}")
        if len(fingerprints) > 1:
            for fp in fingerprints:
                print(f"    variant: {fp}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    _self_check()
