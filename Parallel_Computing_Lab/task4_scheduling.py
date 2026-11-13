
import multiprocessing as mp
import math
import time

from harness import emit_verify_tag


STUDENT_ID = 230107142

P = 8
N = 8000

CHUNK_SIZES = [1, 8, 64, 512]


# ==================================================
# WORKLOAD
# ==================================================

def calculate_item(i):

    iterations = 1 + int((i / 40) ** 2)

    x = (i % 97 + 1) * 0.01

    for _ in range(iterations):

        x = (
            math.sin(x)
            + math.cos(x)
            + math.sqrt(abs(x) + 1.0)
        )

    return x


# ==================================================
# WORKER
# ==================================================

def scheduling_worker(
    worker_id,
    strategy,
    next_index,
    worker_times,
    checksums,
    chunk_size
):

    start_time = time.perf_counter()

    checksum = 0.0

    # Strategy 1: Contiguous Block Static
    if strategy == "block":

        start = worker_id * N // P
        end = (worker_id + 1) * N // P

        for i in range(start, end):
            checksum += calculate_item(i)

    # Strategy 2: Cyclic Round-Robin Static
    elif strategy == "cyclic":

        for i in range(worker_id, N, P):
            checksum += calculate_item(i)

    # Strategy 3: Shared-Memory Dynamic Queue
    elif strategy == "dynamic":

        while True:

            # Claim the next chunk safely
            with next_index.get_lock():

                start = next_index.value

                if start >= N:
                    break

                end = min(start + chunk_size, N)

                next_index.value = end

            # Process the claimed chunk
            for i in range(start, end):
                checksum += calculate_item(i)

    else:
        raise ValueError("Unknown scheduling strategy")

    elapsed = time.perf_counter() - start_time

    # Each worker writes to its own array position
    worker_times[worker_id] = elapsed
    checksums[worker_id] = checksum


# ==================================================
# RUN ONE SCHEDULING STRATEGY
# ==================================================

def run_strategy(strategy, chunk_size=1):

    worker_times = mp.Array("d", P, lock=False)
    checksums = mp.Array("d", P, lock=False)

    next_index = mp.Value("i", 0)

    processes = []

    start_time = time.perf_counter()

    for worker_id in range(P):

        process = mp.Process(
            target=scheduling_worker,
            args=(
                worker_id,
                strategy,
                next_index,
                worker_times,
                checksums,
                chunk_size
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    wall_time = time.perf_counter() - start_time

    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError(
            f"A worker failed during {strategy} scheduling."
        )

    times = list(worker_times)

    average_time = sum(times) / P

    imbalance = max(times) / average_time

    total_checksum = sum(checksums)

    return {
        "wall_time": wall_time,
        "worker_times": times,
        "imbalance": imbalance,
        "checksum": total_checksum
    }


# ==================================================
# PRINT RESULTS
# ==================================================

def print_result(name, result):

    print(f"\n{name}")

    print(f"Wall time: {result['wall_time']:.4f} seconds")

    print(f"Load imbalance factor: {result['imbalance']:.4f}")

    print("Worker active times:")

    for worker_id, elapsed in enumerate(result["worker_times"]):

        print(
            f"  Worker {worker_id}: "
            f"{elapsed:.4f} seconds"
        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("PHASE 4 - SKEWED WORKLOADS AND DYNAMIC QUEUE")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Processes: {P}")
    print(f"Number of items: {N}")
    print(f"Dynamic chunk sizes: {CHUNK_SIZES}")

    # ----------------------------------------------
    # STRATEGY 1: BLOCK STATIC
    # ----------------------------------------------

    print("\n1. CONTIGUOUS BLOCK STATIC")
    print("-" * 65, flush=True)

    block_result = run_strategy("block")

    print_result("Block Static Results", block_result)

    # ----------------------------------------------
    # STRATEGY 2: CYCLIC STATIC
    # ----------------------------------------------

    print("\n2. CYCLIC ROUND-ROBIN STATIC")
    print("-" * 65, flush=True)

    cyclic_result = run_strategy("cyclic")

    print_result("Cyclic Static Results", cyclic_result)

    # ----------------------------------------------
    # STRATEGY 3: DYNAMIC QUEUE
    # ----------------------------------------------

    print("\n3. SHARED-MEMORY DYNAMIC QUEUE")
    print("-" * 65, flush=True)

    dynamic_results = {}

    for chunk in CHUNK_SIZES:

        print(
            f"\nTesting dynamic chunk size C = {chunk} ...",
            flush=True
        )

        result = run_strategy(
            "dynamic",
            chunk_size=chunk
        )

        dynamic_results[chunk] = result

        print_result(
            f"Dynamic Queue C = {chunk}",
            result
        )

    # ----------------------------------------------
    # COMPARISON
    # ----------------------------------------------

    print("\n4. FINAL COMPARISON")
    print("-" * 65)

    print(
        f"{'Strategy':<25}"
        f"{'Wall Time (s)':<18}"
        f"{'Imbalance':<15}"
    )

    print(
        f"{'Block Static':<25}"
        f"{block_result['wall_time']:<18.4f}"
        f"{block_result['imbalance']:<15.4f}"
    )

    print(
        f"{'Cyclic Static':<25}"
        f"{cyclic_result['wall_time']:<18.4f}"
        f"{cyclic_result['imbalance']:<15.4f}"
    )

    for chunk in CHUNK_SIZES:

        result = dynamic_results[chunk]

        print(
            f"{'Dynamic C=' + str(chunk):<25}"
            f"{result['wall_time']:<18.4f}"
            f"{result['imbalance']:<15.4f}"
        )

    # ----------------------------------------------
    # SELECT OPTIMAL CHUNK
    # ----------------------------------------------

    best_chunk = min(
        CHUNK_SIZES,
        key=lambda c: dynamic_results[c]["wall_time"]
    )

    best_dynamic = dynamic_results[best_chunk]

    print("\n5. OPTIMAL DYNAMIC CHUNK")
    print("-" * 65)

    print(f"Best chunk size: {best_chunk}")
    print(
        f"Dynamic wall time: "
        f"{best_dynamic['wall_time']:.4f} seconds"
    )
    print(
        f"Dynamic imbalance: "
        f"{best_dynamic['imbalance']:.4f}"
    )

    # ----------------------------------------------
    # VERIFICATION
    # ----------------------------------------------

    print("\n6. VERIFICATION")
    print("-" * 65)

    verification = emit_verify_tag(
        "PHASE4",
        STUDENT_ID,
        block_result["wall_time"],
        best_dynamic["wall_time"]
    )

    print(verification)


if __name__ == "__main__":

    mp.freeze_support()
    main()

