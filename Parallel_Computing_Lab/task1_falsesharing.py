
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import struct
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230107142

M = 16_000_000
SHM_SIZE = 4096

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]

WARMUP = 1
RUNS = 5


# Worker function
def fs_worker(shm_name, slot_idx, iters):

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * 6364136223846793005 + (i | 1)
        ) & 0xFFFFFFFFFFFFFFFF

        arr[slot_idx] = acc

    del arr
    shm.close()


# Execute one parallel trial
def run_trial(p, stride):

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    try:
        # Reset each worker's memory slot
        for worker in range(p):
            slot_idx = worker * stride
            struct.pack_into(
                "Q", shm.buf, slot_idx * 8, 0
            )

        iters = M // p

        processes = []

        start = time.perf_counter_ns()

        for worker in range(p):

            slot_idx = worker * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed with exit code {process.exitcode}"
                )

        end = time.perf_counter_ns()

        return (end - start) / 1e6

    finally:
        shm.close()
        shm.unlink()


# Calculate Karp-Flatt serial fraction
def karp_flatt(speedup, p):

    return (
        (1 / speedup - 1 / p)
        / (1 - 1 / p)
    )


def main():

    physical_cores = 8

    print("PHASE 1 - FALSE SHARING")
    print("=" * 55)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Shared memory: {SHM_SIZE} bytes")
    print(f"Total operations: {M:,}")
    print(f"Physical cores: {physical_cores}")
    print()

    # ==========================================
    # TASK 1: STRIDE SWEEP
    # ==========================================

    print("TASK 1: STRIDE SWEEP")
    print("-" * 55)

    stride_results = {}

    for stride in STRIDES:

        print(
            f"Testing stride = {stride} ...",
            flush=True
        )

        stats = run_benchmark(
            lambda s=stride: run_trial(physical_cores, s),
            warmup=WARMUP,
            runs=RUNS
        )

        stride_results[stride] = stats

        throughput = M / (stats.median_ms * 1000)

        print(
            f"Stride: {stride:2d} | "
            f"Bytes: {stride * 8:3d} | "
            f"Median: {stats.median_ms:.3f} ms | "
            f"IQR: {stats.iqr_ms:.3f} ms | "
            f"Throughput: {throughput:.3f} Mops/s"
        )

    # ==========================================
    # TASK 2: PROCESS SCALING
    # ==========================================

    print()
    print("TASK 2: PROCESS SCALING")
    print("-" * 55)

    scaling_results = {}

    for stride in [1, 16]:

        print(f"\nTesting stride = {stride}")

        results = {}

        # Reuse the p=8 results from Task 1
        results[8] = stride_results[stride]

        for p in [1, 2, 4]:

            print(
                f"Testing processes = {p} ...",
                flush=True
            )

            stats = run_benchmark(
                lambda n=p, s=stride: run_trial(n, s),
                warmup=WARMUP,
                runs=RUNS
            )

            results[p] = stats

        scaling_results[stride] = results

        t1 = results[1].median_ms

        print()
        print(
            f"{'P':<8}{'Median ms':<15}"
            f"{'Speedup':<12}{'Efficiency':<12}"
            f"{'Karp-Flatt':<12}"
        )

        for p in PROCESS_COUNTS:

            tp = results[p].median_ms
            speedup = t1 / tp
            efficiency = speedup / p

            if p == 1:
                serial_fraction = 0.0
            else:
                serial_fraction = karp_flatt(
                    speedup, p
                )

            print(
                f"{p:<8}{tp:<15.3f}"
                f"{speedup:<12.3f}"
                f"{efficiency:<12.3f}"
                f"{serial_fraction:<12.5f}"
            )

    # ==========================================
    # TASK 3: VERIFICATION HASH
    # ==========================================

    print()
    print("TASK 3: VERIFICATION")
    print("-" * 55)

    median_stride1 = stride_results[1].median_ms
    median_stride16 = stride_results[16].median_ms

    verification = emit_verify_tag(
        "PHASE1",
        STUDENT_ID,
        median_stride1,
        median_stride16
    )

    print(f"Median stride 1: {median_stride1:.4f} ms")
    print(f"Median stride 16: {median_stride16:.4f} ms")
    print()
    print(verification)


if __name__ == "__main__":
    mp.freeze_support()
    main()

