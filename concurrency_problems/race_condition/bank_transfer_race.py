import threading
import time
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s',
    datefmt='%H:%M:%S'
)

class BankAccount:
    """
    A BUGGY bank account simulator with race conditions.
    Locking has been removed intentionally.
    """
    def __init__(self, initial_balance=0.0):
        self._balance = initial_balance

    def deposit(self, amount):
        if amount <= 0:
            return 0
        current_balance = self._balance
        time.sleep(random.uniform(0.0001, 0.001)) 
        self._balance = current_balance + amount
        return amount

    def withdraw(self, amount):
        if amount <= 0:
            return 0
        if self._balance >= amount:
            current_balance = self._balance
            time.sleep(random.uniform(0.0001, 0.001))
            self._balance = current_balance - amount
            return amount
        else:
            return 0

    def get_balance(self):
        return self._balance

def worker(account, operations, local_tracker):
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
        time.sleep(random.uniform(0.0001, 0.001))
    local_tracker.append(net_change)

def run_simulation(run_id, num_threads=5, ops_per_thread=20):
    initial_balance = 1000.0
    account = BankAccount(initial_balance=initial_balance)
    
    threads = []
    thread_net_changes = [] 
    
    for i in range(num_threads):
        operations = []
        for _ in range(ops_per_thread):
            amount = round(random.uniform(5, 100), 2)
            op = random.choice(['deposit', 'withdraw'])
            operations.append((op, amount))
            
        t = threading.Thread(
            target=worker, 
            args=(account, operations, thread_net_changes), 
            name=f"Thread-{run_id}-{i+1}"
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    actual_final_balance = account.get_balance()
    expected_final_balance = initial_balance + sum(thread_net_changes)
    
    actual_rounded = round(actual_final_balance, 2)
    expected_rounded = round(expected_final_balance, 2)
    
    is_correct = (actual_rounded == expected_rounded)
    return is_correct, expected_rounded, actual_rounded

if __name__ == "__main__":
    print("=====================================================")
    print("   Starting Bank Account RACE CONDITION Test Harness ")
    print("=====================================================")
    
    logging.getLogger().setLevel(logging.ERROR)
    
    TOTAL_RUNS = 10
    failed_runs = 0
    
    for run_idx in range(1, TOTAL_RUNS + 1):
        is_correct, expected, actual = run_simulation(run_id=run_idx, num_threads=10, ops_per_thread=50)
        if not is_correct:
            failed_runs += 1
            print(f"FAILED on run {run_idx}! Expected=${expected:.2f}, Actual=${actual:.2f} (Lost ${expected-actual:.2f})")
        else:
            print(f"Run {run_idx} passed (rare, but possible).")

    print("=====================================================")
    print(f"RACE CONDITION Results: Corrupted Data in {failed_runs}/{TOTAL_RUNS} runs.")
