import threading
import time
import random
import logging

# Configure logging to show timestamp, thread name, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s',
    datefmt='%H:%M:%S'
)

class BankAccount:
    """
    A thread-safe bank account simulator.
    This class encapsulates the balance and the lock used to protect it.
    """
    def __init__(self, initial_balance=0.0):
        self._balance = initial_balance
        # We use a standard Lock (mutex) to protect access to the shared state (_balance).
        # A lock ensures that only one thread can execute a critical section at a time.
        self._lock = threading.Lock()

    def deposit(self, amount):
        """
        Safely deposits an amount into the account.
        Returns the amount deposited.
        """
        if amount <= 0:
            return 0

        # --- CRITICAL SECTION START ---
        # Acquiring the lock ensures mutual exclusion. 
        # No other thread can read or write self._balance while we hold this lock.
        # This protects against race conditions where multiple threads try to update the balance concurrently.
        with self._lock:
            # We simulate some processing time to increase the likelihood of a race condition
            # if the lock were to be removed in a buggy version.
            current_balance = self._balance
            time.sleep(random.uniform(0.0001, 0.001)) 
            
            # Update the balance
            self._balance = current_balance + amount
        # --- CRITICAL SECTION END ---
        # The lock is automatically released here when exiting the 'with' block.
        
        return amount

    def withdraw(self, amount):
        """
        Safely withdraws an amount from the account if sufficient funds exist.
        Returns the actual amount withdrawn (either 'amount' or 0).
        """
        if amount <= 0:
            return 0

        # --- CRITICAL SECTION START ---
        # We must lock the entire read-check-update sequence. If we didn't lock,
        # two threads could read the balance simultaneously, both see enough funds,
        # and both withdraw, potentially leading to a negative balance. This is known as a check-then-act race condition.
        with self._lock:
            if self._balance >= amount:
                # Simulate processing time to exacerbate potential race conditions
                current_balance = self._balance
                time.sleep(random.uniform(0.0001, 0.001))
                
                self._balance = current_balance - amount
                return amount
            else:
                return 0
        # --- CRITICAL SECTION END ---

    def get_balance(self):
        """
        Safely returns the current balance.
        """
        # Even reads must be locked to ensure we don't read a partially updated state,
        # and to guarantee memory visibility across threads.
        with self._lock:
            return self._balance


def worker(account, operations, local_tracker):
    """
    Worker function executed by each thread.
    Performs a list of (operation_type, amount) tuples.
    Records the net change made by this thread to verify correctness later.
    """
    net_change = 0
    for op, amount in operations:
        if op == 'deposit':
            deposited = account.deposit(amount)
            net_change += deposited
            logging.info(f"Deposited ${deposited:.2f} | Net thread change: ${net_change:.2f}")
        elif op == 'withdraw':
            withdrawn = account.withdraw(amount)
            if withdrawn > 0:
                net_change -= withdrawn
                logging.info(f"Withdrew ${withdrawn:.2f} | Net thread change: ${net_change:.2f}")
            else:
                logging.warning(f"Failed withdrawal of ${amount:.2f} (Insufficient funds)")
                
        # Small sleep between operations to interleave thread execution
        time.sleep(random.uniform(0.0001, 0.001))
        
    local_tracker.append(net_change)

def run_simulation(run_id, num_threads=5, ops_per_thread=20):
    """
    Runs a single simulation of concurrent bank account operations.
    Returns True if the final balance matches the expected mathematical result.
    """
    initial_balance = 1000.0
    account = BankAccount(initial_balance=initial_balance)
    
    threads = []
    # Using a list to collect the net change contributed by each thread.
    # Lists are thread-safe for appending in Python.
    thread_net_changes = [] 
    
    for i in range(num_threads):
        operations = []
        for _ in range(ops_per_thread):
            amount = round(random.uniform(5, 100), 2)
            # Randomly pick deposit or withdraw
            op = random.choice(['deposit', 'withdraw'])
            operations.append((op, amount))
            
        t = threading.Thread(
            target=worker, 
            args=(account, operations, thread_net_changes), 
            name=f"Thread-{run_id}-{i+1}"
        )
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Verify the results
    actual_final_balance = account.get_balance()
    expected_final_balance = initial_balance + sum(thread_net_changes)
    
    # Due to floating point arithmetic, we round to 2 decimal places before comparing
    actual_rounded = round(actual_final_balance, 2)
    expected_rounded = round(expected_final_balance, 2)
    
    is_correct = (actual_rounded == expected_rounded)
    return is_correct, expected_rounded, actual_rounded


if __name__ == "__main__":
    print("=====================================================")
    print("   Starting Bank Account Concurrency Test Harness    ")
    print("=====================================================")
    
    # We will run multiple simulation loops to heavily test the locking.
    # To avoid overwhelming stdout with 100 runs of logs, we'll temporarily set logging to WARNING
    # for the heavy testing, and just do 1 verbose run at the beginning.
    
    # First, let's do 1 highly visible run with logs so the user can see thread interleaving.
    print("\n--- Running 1 verbose demonstration run ---")
    is_correct, expected, actual = run_simulation(run_id=0, num_threads=3, ops_per_thread=5)
    print(f"Demonstration Run Result: Expected=${expected:.2f}, Actual=${actual:.2f}. Safe: {is_correct}\n")
    
    # Now run the test harness multiple times quietly
    print("--- Running Test Harness (100 runs) ---")
    logging.getLogger().setLevel(logging.ERROR) # Suppress normal logs for the test loop
    
    TOTAL_RUNS = 100
    successful_runs = 0
    
    start_time = time.time()
    for run_idx in range(1, TOTAL_RUNS + 1):
        # We increase the load slightly for the silent tests
        is_correct, expected, actual = run_simulation(run_id=run_idx, num_threads=10, ops_per_thread=50)
        
        if is_correct:
            successful_runs += 1
        else:
            print(f"FAILED on run {run_idx}! Expected=${expected:.2f}, Actual=${actual:.2f}")

    elapsed = time.time() - start_time
    
    print("=====================================================")
    print(f"Test Harness Completed in {elapsed:.2f} seconds.")
    print(f"Total Runs: {TOTAL_RUNS}")
    print(f"Successful Runs (No Data Corruption): {successful_runs}/{TOTAL_RUNS}")
    print("=====================================================")
    
    if successful_runs == TOTAL_RUNS:
        print("SUCCESS! The locking mechanism completely protected the shared state.")
    else:
        print("ERROR! Race conditions detected. Check the locking logic.")
