
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230107142
M = 24_000_000

MULT = 6364136223846793005
MASK = 0xFFFFFFFFFFFFFFFF


def fs_worker(shm_name, slot_idx, iters):

    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        acc = arr[slot_idx]

        for i in range(iters):
            acc = (acc * MULT + (i | 1)) & MASK
            arr[slot_idx] = acc

        del arr

    finally:
        shm.close()


def run_false_sharing(p, stride):

    shm = shared_memory.SharedMemory(create=True, size=4096)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(512):
            arr[i] = 1

        del arr

        iterations = M // p

        processes = []

        for t in range(p):
            slot_idx = t * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iterations)
            )

            processes.append(process)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

    finally:
        shm.close()
        shm.unlink()


def main():

    print("=" * 60)
    print("PHASE 1: FALSE SHARING EXPERIMENT")
    print("=" * 60)

    physical = 8
    logical = mp.cpu_count()

    print("Physical cores:", physical)
    print("Logical CPUs:", logical)

    # Experiment 1: Stride sweep
    print("\n--- STRIDE SWEEP ---")

    stride_results = {}

    for stride in [1, 2, 4, 8, 16, 32]:

        print(f"\nTesting stride = {stride}")

        stats = run_benchmark(
            lambda: run_false_sharing(physical, stride),
            warmup_runs=2,
            timed_runs=7
        )

        throughput = M / (stats.median_ms / 1000) / 1_000_000

        stride_results[stride] = stats

        print(f"Median: {stats.median_ms:.4f} ms")
        print(f"IQR: {stats.iqr_ms:.4f} ms")
        print(f"Throughput: {throughput:.4f} Mops/s")

    # Experiment 2: Process scaling
    print("\n--- PROCESS SCALING SWEEP ---")

    scaling_results = {}

    process_counts = [1, 2, 4, 8]

    for stride in [1, 16]:

        print(f"\nTesting stride = {stride}")

        scaling_results[stride] = {}

        for p in process_counts:

            stats = run_benchmark(
                lambda p=p, stride=stride:
                    run_false_sharing(p, stride),
                warmup_runs=2,
                timed_runs=7
            )

            scaling_results[stride][p] = stats

            print(
                f"p={p}: "
                f"Median={stats.median_ms:.4f} ms, "
                f"IQR={stats.iqr_ms:.4f} ms"
            )

    # Karp-Flatt calculations
    print("\n--- KAR P-FLATT ANALYSIS ---")

    for stride in [1, 16]:

        t1 = scaling_results[stride][1].median_ms

        print(f"\nStride = {stride}")
        print("p | Time(ms) | Speedup | Efficiency | Karp-Flatt")

        for p in process_counts:

            tp = scaling_results[stride][p].median_ms

            speedup = t1 / tp
            efficiency = speedup / p

            if p == 1:
                e = 0.0
            else:
                e = (1 / speedup - 1 / p) / (1 - 1 / p)

            print(
                f"{p} | {tp:.4f} | "
                f"{speedup:.4f} | "
                f"{efficiency:.4f} | "
                f"{e:.4f}"
            )

    # Verification hash
    t_stride1 = stride_results[1].median_ms
    t_stride16 = stride_results[16].median_ms

    print("\n--- VERIFICATION ---")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            t_stride1,
            t_stride16
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
