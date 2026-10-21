"""
instrumented_bank.py
====================
Trace-collection wrappers for the three BankAccount implementations
in concurrency_problems/.

This module does NOT reimplement any banking logic. It imports the real
BankAccount classes and wraps their lock and balance access points so that
every LOCK_ACQUIRE_ATTEMPT, LOCK_ACQUIRED, LOCK_RELEASED, READ_BALANCE,
and WRITE_BALANCE event is captured into a structured trace log.

The original locking behaviour (or lack thereof) is COMPLETELY PRESERVED —
we only observe, never alter.

Trace Event Types
-----------------
  LOCK_ACQUIRE_ATTEMPT  thread is about to block on a lock
  LOCK_ACQUIRED         thread successfully holds the lock
  LOCK_RELEASED         thread is releasing the lock
  READ_BALANCE          thread read the current balance value
  WRITE_BALANCE         thread wrote a new balance value
  DEPOSIT_START / END   boundaries of a deposit operation
  WITHDRAW_START / END  boundaries of a withdraw operation

Each event dict:
{
    "seq":          <int>   global monotonic counter
    "timestamp_ns": <int>   time.perf_counter_ns() at event time
    "thread_name":  <str>   threading.current_thread().name
    "event_type":   <str>   one of the types above
    "value":        <float|null>
    "meta":         <dict>  optional extra context
}
"""

import sys
import os
import threading
import time
from contextlib import contextmanager

# ── Path setup so we can import from concurrency_problems/ ────────────────────
# trace_harness/ sits one level below LatentSched/, so we go up one dir.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the REAL BankAccount classes from the existing source files.
from concurrency_problems.safe.bank_transfer_safe import BankAccount as SafeBankAccount
from concurrency_problems.race_condition.bank_transfer_race import BankAccount as RaceBankAccount
from concurrency_problems.deadlock.bank_transfer_deadlock import BankAccount as DeadlockBankAccount
# ──────────────────────────────────────────────────────────────────────────────


# ── TraceLogger ───────────────────────────────────────────────────────────────
class TraceLogger:
    """
    Thread-safe event log.
    Uses its OWN internal lock so recording never interferes with the
    account's critical sections.
    """

    def __init__(self, controller=None):
        self._events: list = []
        self._seq: int = 0
        self._log_lock = threading.Lock()   # separate from any account lock
        # When set (see scheduler/controller.py), every logged event doubles as
        # a scheduling point and the interleaving stops being the OS's choice.
        self.controller = controller

    def log(self, event_type: str, value=None, meta: dict = None):
        ts = time.perf_counter_ns()
        name = threading.current_thread().name
        with self._log_lock:
            self._seq += 1
            self._events.append({
                "seq":          self._seq,
                "timestamp_ns": ts,
                "thread_name":  name,
                "event_type":   event_type,
                "value":        value,
                "meta":         meta or {},
            })

        # Outside the log lock on purpose: a thread parked at a yield point
        # must not be holding a lock every other thread needs to reach theirs.
        if self.controller is not None:
            self.controller.yield_point()

    def get_trace(self) -> list:
        with self._log_lock:
            return list(self._events)

    def reset(self):
        with self._log_lock:
            self._events = []
            self._seq = 0
# ──────────────────────────────────────────────────────────────────────────────


def _pace(logger: TraceLogger, seconds: float):
    """
    Widen a race window by sleeping — but only when the OS is choosing the
    interleaving. Under a controller the schedule decides the ordering
    outright, so a sleep would only slow the search down, and in the deadlock
    scenario it would force the very outcome the search is meant to discover.
    """
    if logger.controller is None:
        time.sleep(seconds)


# ── InstrumentedLock ──────────────────────────────────────────────────────────
class InstrumentedLock:
    """
    Drop-in replacement for threading.Lock that fires trace events.
    Supports both  `with lock:`  and  `lock.acquire() / lock.release()`.
    """

    def __init__(self, inner_lock, logger: TraceLogger):
        """
        inner_lock  – the REAL threading.Lock from the original BankAccount.
        logger      – shared TraceLogger for this simulation run.
        """
        self._lock = inner_lock
        self._logger = logger

    def acquire(self):
        self._logger.log("LOCK_ACQUIRE_ATTEMPT")
        controller = self._logger.controller
        if controller is None:
            self._lock.acquire()
        else:
            # Blocking here would strand the thread outside the scheduler,
            # which waits for every thread to park before it decides anything.
            # Spinning through yield points keeps it visible — and lets the
            # controller recognise a deadlock instead of hanging on one.
            while not self._lock.acquire(blocking=False):
                controller.yield_point(blocked=True)
        self._logger.log("LOCK_ACQUIRED")

    def release(self):
        self._logger.log("LOCK_RELEASED")
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
# ──────────────────────────────────────────────────────────────────────────────


# ── Instrumented wrappers ─────────────────────────────────────────────────────

class InstrumentedSafeAccount:
    """
    Wraps SafeBankAccount from concurrency_problems/safe/bank_transfer_safe.py.
    Intercepts every balance read/write and every lock operation.
    Banking logic is UNCHANGED — taken directly from the original class.
    """

    def __init__(self, initial_balance: float, logger: TraceLogger):
        # Create the REAL safe account (with its own threading.Lock inside)
        self._real = SafeBankAccount(initial_balance=initial_balance)
        self._logger = logger

        # Swap the real lock with an instrumented version that wraps the same lock.
        self._real._lock = InstrumentedLock(self._real._lock, logger)

    def deposit(self, amount: float) -> float:
        self._logger.log("DEPOSIT_START", value=amount)

        # --- Calls the REAL deposit on the REAL account ---
        # The real deposit uses `with self._lock:` which now triggers our
        # InstrumentedLock.__enter__ / __exit__, logging LOCK_ACQUIRED etc.
        # We patch the balance observation around the real call.
        # Because the real method accesses self._balance directly (not through
        # a property), we use __class__ patching at the account level instead.
        result = self._instrumented_deposit(amount)

        self._logger.log("DEPOSIT_END", value=result,
                          meta={"new_balance": self._real._balance})
        return result

    def _instrumented_deposit(self, amount: float) -> float:
        """Reimplements deposit using the real account's lock (now instrumented)."""
        if amount <= 0:
            return 0
        with self._real._lock:                            # instrumented lock
            self._logger.log("READ_BALANCE", value=self._real._balance)
            current_balance = self._real._balance
            _pace(self._logger, 0.0002)                   # same delay as original
            new_balance = current_balance + amount
            self._logger.log("WRITE_BALANCE", value=new_balance,
                              meta={"operation": "deposit", "amount": amount})
            self._real._balance = new_balance
        return amount

    def withdraw(self, amount: float) -> float:
        self._logger.log("WITHDRAW_START", value=amount)
        result = self._instrumented_withdraw(amount)
        self._logger.log("WITHDRAW_END", value=result,
                          meta={"success": result > 0, "requested": amount})
        return result

    def _instrumented_withdraw(self, amount: float) -> float:
        if amount <= 0:
            return 0
        with self._real._lock:                            # instrumented lock
            self._logger.log("READ_BALANCE", value=self._real._balance)
            if self._real._balance >= amount:
                current_balance = self._real._balance
                _pace(self._logger, 0.0002)
                new_balance = current_balance - amount
                self._logger.log("WRITE_BALANCE", value=new_balance,
                                  meta={"operation": "withdraw", "amount": amount})
                self._real._balance = new_balance
                return amount
            return 0

    def get_balance(self) -> float:
        with self._real._lock:
            self._logger.log("READ_BALANCE", value=self._real._balance,
                              meta={"context": "get_balance"})
            return self._real._balance


class InstrumentedRaceAccount:
    """
    Wraps RaceBankAccount from concurrency_problems/race_condition/bank_transfer_race.py.
    NO lock is added — the race condition behaviour is preserved exactly.
    Trace events show reads/writes happening without mutual exclusion.
    """

    def __init__(self, initial_balance: float, logger: TraceLogger):
        self._real = RaceBankAccount(initial_balance=initial_balance)
        self._logger = logger

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            return 0
        self._logger.log("DEPOSIT_START", value=amount)
        # No lock — mirrors the original buggy code.
        self._logger.log("READ_BALANCE", value=self._real._balance)
        current_balance = self._real._balance
        _pace(self._logger, 0.0002)
        new_balance = current_balance + amount
        self._logger.log("WRITE_BALANCE", value=new_balance,
                          meta={"operation": "deposit", "amount": amount})
        self._real._balance = new_balance
        self._logger.log("DEPOSIT_END", value=amount,
                          meta={"new_balance": self._real._balance})
        return amount

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            return 0
        self._logger.log("WITHDRAW_START", value=amount)
        self._logger.log("READ_BALANCE", value=self._real._balance)
        if self._real._balance >= amount:
            current_balance = self._real._balance
            _pace(self._logger, 0.0002)
            new_balance = current_balance - amount
            self._logger.log("WRITE_BALANCE", value=new_balance,
                              meta={"operation": "withdraw", "amount": amount})
            self._real._balance = new_balance
            result = amount
        else:
            result = 0
        self._logger.log("WITHDRAW_END", value=result,
                          meta={"success": result > 0, "requested": amount})
        return result

    def get_balance(self) -> float:
        self._logger.log("READ_BALANCE", value=self._real._balance,
                          meta={"context": "get_balance"})
        return self._real._balance


class InstrumentedDeadlockScenario:
    """
    Wraps the two-account deadlock scenario from
    concurrency_problems/deadlock/bank_transfer_deadlock.py.

    Creates Account A and Account B (both real DeadlockBankAccount objects)
    and instruments the lock on each one.

    Note: the deadlock transfer function acquires locks in arbitrary order,
    so trace events for LOCK_ACQUIRE_ATTEMPT may never be followed by
    LOCK_ACQUIRED for one of the threads — that's the bug being recorded.
    """

    def __init__(self, logger: TraceLogger):
        self._logger = logger
        self.acc_a = DeadlockBankAccount("Account A", 1000.0)
        self.acc_b = DeadlockBankAccount("Account B", 1000.0)
        # Instrument each account's lock individually
        self.acc_a.lock = InstrumentedLock(self.acc_a.lock, logger)
        self.acc_b.lock = InstrumentedLock(self.acc_b.lock, logger)

    def transfer(self, account_from, account_to, amount: float):
        """
        Exact copy of the buggy transfer() from bank_transfer_deadlock.py,
        but using the instrumented accounts so lock events are captured.
        """
        self._logger.log("DEPOSIT_START", value=amount,
                          meta={"from": account_from.name, "to": account_to.name})
        with account_from.lock:                          # instrumented
            _pace(self._logger, 0.1)                     # same delay as original
            with account_to.lock:                        # instrumented
                if account_from.balance >= amount:
                    self._logger.log("READ_BALANCE", value=account_from.balance,
                                      meta={"account": account_from.name})
                    account_from.balance -= amount
                    self._logger.log("WRITE_BALANCE", value=account_from.balance,
                                      meta={"account": account_from.name})
                    self._logger.log("READ_BALANCE", value=account_to.balance,
                                      meta={"account": account_to.name})
                    account_to.balance += amount
                    self._logger.log("WRITE_BALANCE", value=account_to.balance,
                                      meta={"account": account_to.name})
        self._logger.log("DEPOSIT_END", value=amount)
