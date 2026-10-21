"""
controller.py — Deterministic, schedule-driven thread scheduler.

The trace harness observes whatever interleaving the OS happens to produce.
This module *dictates* the interleaving instead: a schedule (a sequence of
thread preferences) decides which thread runs at every instrumented event.

How it works
------------
Every instrumented event (see ``trace_harness.instrumented_bank``) calls
``yield_point()``. A thread parks there until the controller hands it the turn.
A decision is only made once *every live thread is parked* — so the controller
always chooses with full knowledge of where each thread stands, which makes a
run a pure function of its schedule.

Locks under control
-------------------
A thread that blocks on a real ``threading.Lock`` would never reach its next
yield point, and the barrier would never complete. So instrumented locks
acquire non-blockingly and call ``yield_point(blocked=True)`` on failure. A
thread that cannot make progress therefore keeps cycling through the scheduler
rather than disappearing into the kernel.

That turns deadlock into something the controller can decide soundly. A
deadlock is declared only when every unfinished thread is parked on a lock it
could not take *and* each has retried since the last time any thread made
progress. Nothing but a running thread can free a lock, so a state in which
every thread has freshly re-tried and failed can never change on its own.

The retry requirement is what makes this exact rather than a guess. A thread's
blocked flag goes stale the moment the lock it wanted is released, so "all
threads are flagged blocked" on its own would condemn perfectly healthy runs —
five threads queued on one mutex look identical to a deadlock until they are
asked to try again.
"""

import threading
import time
from typing import Dict, List, Optional, Sequence

# A run is abandoned once this many scheduling decisions have been made.
# Well above the ~300 a normal target run needs, but low enough that a
# schedule which starves the one thread able to progress is cut short quickly.
DEFAULT_MAX_STEPS = 5_000

# Hard wall-clock ceiling per run, as a backstop for anything the spin/step
# limits fail to catch.
DEFAULT_WALL_TIMEOUT = 15.0


class ScheduleAborted(Exception):
    """
    Raised inside a worker thread to unwind it when a run is abandoned.

    Instrumented lock acquisition happens inside ``with`` blocks, so letting
    this propagate releases every lock the thread holds on the way out.
    """


class ScheduleController:
    """
    Runs registered threads one at a time, in an order the schedule chooses.

    Parameters
    ----------
    schedule : sequence of int
        Thread preference at each decision point. Consumed cyclically, so a
        schedule shorter than the run still defines the whole execution.
    num_threads : int
        How many threads will take part. The barrier needs this upfront:
        counting only the threads that have *registered so far* would let the
        first one run to completion before the rest have started, and every
        schedule would produce the same serial execution.
    max_steps : int
        Abandon the run after this many scheduling decisions.
    wall_timeout : float
        Abandon the run after this many seconds.
    """

    def __init__(
        self,
        schedule: Sequence[int],
        num_threads: int,
        max_steps: int = DEFAULT_MAX_STEPS,
        wall_timeout: float = DEFAULT_WALL_TIMEOUT,
    ):
        self._schedule: List[int] = [max(0, int(s)) for s in schedule] or [0]
        self._pos = 0
        self._expected = int(num_threads)
        self._finished = 0

        self._cv = threading.Condition()
        self._ids: Dict[str, int] = {}
        self._live: set = set()
        self._parked: set = set()      # at a yield point, awaiting the turn
        self._spinning: set = set()    # parked *and* blocked on a real lock
        self._retried: set = set()     # re-tried its lock since the last progress
        self._notified: set = set()    # already raised ScheduleAborted
        self._current: Optional[int] = None

        self._steps = 0
        self._deadline = time.monotonic() + wall_timeout

        self.max_steps = max_steps
        self.wall_timeout = wall_timeout

        # Outcome flags, read after the run finishes.
        self.deadlock_detected = False
        self.aborted = False
        self.abort_reason: Optional[str] = None
        self.decisions: List[int] = []

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, thread_id: int) -> None:
        """Declare a thread as scheduled. Call from the thread itself."""
        name = threading.current_thread().name
        with self._cv:
            self._ids[name] = thread_id
            self._live.add(thread_id)

    def unregister(self) -> None:
        """Declare the calling thread finished, releasing its turn."""
        name = threading.current_thread().name
        with self._cv:
            tid = self._ids.pop(name, None)
            if tid is None:
                return
            self._live.discard(tid)
            self._parked.discard(tid)
            self._spinning.discard(tid)
            self._retried.clear()      # a departing thread releases what it held
            self._finished += 1
            if self._current == tid:
                self._current = None
            self._pick_locked()
            self._cv.notify_all()

    def start(self) -> None:
        """Reset the wall-clock deadline. Call just before starting threads."""
        with self._cv:
            self._deadline = time.monotonic() + self.wall_timeout

    # ── The scheduling point ─────────────────────────────────────────────────

    def yield_point(self, blocked: bool = False) -> None:
        """
        Hand the turn back and wait to be picked again.

        Parameters
        ----------
        blocked : bool
            True when the caller is spinning on a lock it could not take, so
            the controller knows this thread cannot progress if given a turn.
        """
        name = threading.current_thread().name
        with self._cv:
            tid = self._ids.get(name)
            if tid is not None and not self.aborted:
                self._wait_for_turn_locked(tid, blocked)

        # Always outside the lock: unwinding runs ``with`` exits that release
        # locks, and those fire further instrumented events.
        self.check_abort()

    def _wait_for_turn_locked(self, tid: int, blocked: bool) -> None:
        if blocked:
            self._spinning.add(tid)
            self._retried.add(tid)
        else:
            # Any unblocked yield is progress, and progress can free a lock —
            # every thread must prove itself stuck again from here.
            self._spinning.discard(tid)
            self._retried.clear()

        self._steps += 1
        if self._steps > self.max_steps:
            self._abort_locked("step_limit")
            return

        if self._current == tid:
            self._current = None
        self._parked.add(tid)
        self._pick_locked()
        self._cv.notify_all()

        while self._current != tid and not self.aborted:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self._abort_locked("timeout")
                break
            self._cv.wait(min(remaining, 0.05))
            # A thread may have finished while we slept, leaving the turn
            # vacant with nobody holding it.
            if self._current is None:
                self._pick_locked()
                self._cv.notify_all()

        self._parked.discard(tid)

    def check_abort(self) -> None:
        """
        Raise ``ScheduleAborted`` if the run has been abandoned.

        Each thread is raised at most once. Unwinding a thread runs ``with``
        exits, which log further events and so re-enter this check; raising
        again there would skip the ``lock.release()`` those exits exist to
        perform, stranding the lock.
        """
        if not self.aborted:
            return
        name = threading.current_thread().name
        with self._cv:
            tid = self._ids.get(name)
            # Unregistered callers (the main thread, a finished worker) are not
            # under the controller's authority and are left alone.
            if tid is None or tid in self._notified:
                return
            self._notified.add(tid)
        raise ScheduleAborted(self.abort_reason or "aborted")

    # ── Internals (caller must hold the condition) ───────────────────────────

    def _pick_locked(self) -> None:
        """
        Choose the next thread, but only once every unfinished thread is parked.

        Waiting for the full barrier is what makes a run reproducible: the
        choice never depends on which threads happen to have arrived.
        """
        if self._current is not None or not self._parked:
            return
        if len(self._parked) < self._expected - self._finished:
            return

        # Barrier reached, so _parked is every unfinished thread. If each is
        # blocked on a lock and each has re-tried since the last progress,
        # nothing is left that could ever change the situation.
        if self._parked <= self._spinning and self._parked <= self._retried:
            self.deadlock_detected = True
            self._abort_locked("deadlock")
            return

        candidates = sorted(self._parked)
        preference = self._schedule[self._pos % len(self._schedule)]
        self._pos += 1

        # A preference naming a thread that is not currently parked (it has
        # finished, say) folds onto the parked set rather than stalling.
        chosen = preference if preference in candidates else candidates[preference % len(candidates)]

        self._current = chosen
        self.decisions.append(chosen)

    def _abort_locked(self, reason: str) -> None:
        self.aborted = True
        self.abort_reason = reason
        self._current = None
        self._cv.notify_all()

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def steps(self) -> int:
        return self._steps
