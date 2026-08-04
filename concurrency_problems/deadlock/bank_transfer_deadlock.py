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
    def __init__(self, name, initial_balance=1000.0):
        self.name = name
        self.balance = initial_balance
        self.lock = threading.Lock()
        
    def __str__(self):
        return f"{self.name} (${self.balance:.2f})"

def transfer(account_from, account_to, amount):
    """
    BUGGY TRANSFER: Acquires locks in arbitrary order based on arguments.
    This creates a classic Deadlock scenario when threads try to transfer
    between the same two accounts in opposite directions simultaneously.
    """
    logging.info(f"Attempting to lock {account_from.name}...")
    with account_from.lock:
        logging.info(f"Locked {account_from.name}. Simulating work before locking {account_to.name}...")
        
        # Artificial delay to ensure both threads lock their first account
        # before trying to lock the second one (guarantees deadlock).
        time.sleep(0.1)
        
        logging.info(f"Attempting to lock {account_to.name}...")
        with account_to.lock:
            logging.info(f"Locked {account_to.name}. Proceeding with transfer.")
            
            if account_from.balance >= amount:
                account_from.balance -= amount
                account_to.balance += amount
                logging.info(f"Successfully transferred ${amount:.2f} from {account_from.name} to {account_to.name}")
            else:
                logging.warning(f"Insufficient funds in {account_from.name}")

def worker(account_from, account_to, amount):
    transfer(account_from, account_to, amount)

if __name__ == "__main__":
    print("=====================================================")
    print("   Starting Bank Account DEADLOCK Demonstration      ")
    print("=====================================================")
    
    acc1 = BankAccount("Account A", 1000.0)
    acc2 = BankAccount("Account B", 1000.0)
    
    print(f"Initial State: {acc1} | {acc2}")
    
    # Thread 1 transfers A -> B
    t1 = threading.Thread(target=worker, args=(acc1, acc2, 100.0), name="Thread-A-to-B")
    # Thread 2 transfers B -> A
    t2 = threading.Thread(target=worker, args=(acc2, acc1, 200.0), name="Thread-B-to-A")
    
    t1.start()
    t2.start()
    
    # In a deadlock, these joins will block indefinitely.
    # We will use a timeout to detect the deadlock.
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    
    if t1.is_alive() or t2.is_alive():
        print("\n[!] DEADLOCK DETECTED! Both threads are stuck waiting for each other's locks indefinitely.")
        print("    Thread-A-to-B holds Account A's lock, waiting for Account B.")
        print("    Thread-B-to-A holds Account B's lock, waiting for Account A.")
        print("\nExiting program (leaving threads hanging)...")
    else:
        print("\nFinished successfully (no deadlock).")
        print(f"Final State: {acc1} | {acc2}")
