import threading
import time

def cpu_bound_task(iterations):
    result = 0
    for _ in range(iterations):
        result += sum(i * i for i in range(10000))
    return result

def run_cpu_mode():
    start_time = time.time()
    result = cpu_bound_task(1000)  # Adjust the number of iterations as needed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"CPU Mode Result: {result}")
    print(f"CPU Mode Elapsed Time: {elapsed_time} seconds")

def run_thread_mode():
    start_time = time.time()

    # Create two threads and a list to store the results
    results = []
    threads = []
    for i in range(10):
        t = threading.Thread(target=lambda: results.append(cpu_bound_task(100)), daemon=True)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print results from each thread
    total = 0.0
    for i, result in enumerate(results):
        print(f"Thread {i + 1} Result: {result}")
        total += result
    print(f"Thread Mode Result: {total}")
    print(f"Thread Mode Elapsed Time: {elapsed_time} seconds")

if __name__ == "__main__":
    print("Running in CPU mode:")
    run_cpu_mode()

    print("\nRunning in Thread mode:")
    run_thread_mode()
