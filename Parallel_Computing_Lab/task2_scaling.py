
import multiprocessing as mp
import math
import time
from statistics import median

from harness import emit_verify_tag


STUDENT_ID = 230107142

PHYSICAL_CORES = 8
LOGICAL_CORES = mp.cpu_count()

STRONG_N = 1_200_000
WEAK_N0 = 200_000

RUNS = 3


# ==========================================
# STAGE A: SERIAL WORKLOAD
# ==========================================

def stage_a_work(n):

    result = 0.0

    for i in range(n):

        x = (i + 1) * 0.000001

        for j in range(10):

            if j % 2 == 0:
                x = math.sin(x) + 1.0
            else:
                x = math.sqrt(x + 1.0)

        result += x

    return result


# ==========================================
# STAGE B: PARALLEL WORKLOAD
# ==========================================

def stage_b_worker(bounds):

    start, end = bounds

    result = 0.0

    for i in range(start, end):

        x = (i + 1) * 0.000001

        for _ in range(10):

            x = x * 1.0000001 + math.sin(x)

        result += x

    return result


# ==========================================
# DIVIDE WORKLOAD INTO PROCESS RANGES
# ==========================================

def make_ranges(n, p):

    base, remainder = divmod(n, p)

    ranges = []
    start = 0

    for worker in range(p):

        length = base + (1 if worker < remainder else 0)

        end = start + length

        ranges.append((start, end))

        start = end

    return ranges


# ==========================================
# ONE PIPELINE TRIAL
# ==========================================

def run_trial(pool, n, p):

    stage_a_n = int(0.05 * n)

    total_start = time.perf_counter_ns()

    # Stage A: serial execution in parent
    stage_a_start = time.perf_counter_ns()

    a_result = stage_a_work(stage_a_n)

    stage_a_end = time.perf_counter_ns()

    stage_a_ms = (
        stage_a_end - stage_a_start
    ) / 1e6

    # Stage B: parallel execution using persistent Pool
    ranges = make_ranges(n, p)

    b_results = pool.map(
        stage_b_worker,
        ranges
    )

    total_end = time.perf_counter_ns()

    total_ms = (
        total_end - total_start
    ) / 1e6

    # Preserve the calculated results
    checksum = a_result + sum(b_results)

    return total_ms, stage_a_ms, checksum


# ==========================================
# RUN MULTIPLE TRIALS
# ==========================================

def benchmark_pipeline(n, p):

    total_samples = []
    stage_a_samples = []

    # Pool is created outside the timed trials
    with mp.Pool(processes=p) as pool:

        # One warm-up trial
        run_trial(pool, n, p)

        # Measured trials
        for _ in range(RUNS):

            total_ms, stage_a_ms, checksum = run_trial(
                pool, n, p
            )

            total_samples.append(total_ms)
            stage_a_samples.append(stage_a_ms)

    return {
        "total_ms": median(total_samples),
        "stage_a_ms": median(stage_a_samples)
    }


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("PHASE 2 - STRONG AND WEAK SCALING")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Logical cores: {LOGICAL_CORES}")
    print(f"Strong workload: {STRONG_N:,}")
    print(f"Weak workload per process: {WEAK_N0:,}")
    print(f"Measured runs: {RUNS}")
    print()

    # ======================================
    # TASK 1: STRONG SCALING
    # ======================================

    print("TASK 1: STRONG SCALING")
    print("-" * 65)

    strong_results = {}

    for p in range(1, LOGICAL_CORES + 1):

        print(
            f"Testing p = {p} ...",
            flush=True
        )

        result = benchmark_pipeline(
            STRONG_N, p
        )

        strong_results[p] = result

    t1_strong = strong_results[1]["total_ms"]

    serial_fraction = (
        strong_results[1]["stage_a_ms"]
        / t1_strong
    )

    amdahl_cap = 1 / serial_fraction

    print()
    print(f"Serial fraction: {serial_fraction:.6f}")
    print(f"Amdahl theoretical limit: {amdahl_cap:.4f}")
    print()

    print(
        f"{'P':<6}{'Time ms':<14}"
        f"{'Speedup':<12}{'Efficiency':<13}"
        f"{'Stage A ms':<13}"
    )

    strong_speedups = {}

    for p in range(1, LOGICAL_CORES + 1):

        tp = strong_results[p]["total_ms"]

        speedup = t1_strong / tp

        efficiency = speedup / p

        strong_speedups[p] = speedup

        print(
            f"{p:<6}{tp:<14.3f}"
            f"{speedup:<12.3f}"
            f"{efficiency:<13.3f}"
            f"{strong_results[p]['stage_a_ms']:<13.3f}"
        )

    # ======================================
    # TASK 2: WEAK SCALING
    # ======================================

    print()
    print("TASK 2: WEAK SCALING")
    print("-" * 65)

    weak_results = {}

    for p in range(1, LOGICAL_CORES + 1):

        n = p * WEAK_N0

        print(
            f"Testing p = {p}, N = {n:,} ...",
            flush=True
        )

        result = benchmark_pipeline(n, p)

        weak_results[p] = result

    t1_weak = weak_results[1]["total_ms"]

    print()

    print(
        f"{'P':<6}{'N':<13}{'Time ms':<14}"
        f"{'Weak Eff.':<13}{'Alpha':<12}"
        f"{'Gustafson':<12}"
    )

    gustafson_speedups = {}

    for p in range(1, LOGICAL_CORES + 1):

        n = p * WEAK_N0

        tp = weak_results[p]["total_ms"]

        stage_a_ms = weak_results[p]["stage_a_ms"]

        weak_efficiency = t1_weak / tp

        alpha = stage_a_ms / tp

        gustafson = p - alpha * (p - 1)

        gustafson_speedups[p] = gustafson

        print(
            f"{p:<6}{n:<13,}{tp:<14.3f}"
            f"{weak_efficiency:<13.3f}"
            f"{alpha:<12.5f}"
            f"{gustafson:<12.3f}"
        )

    # ======================================
    # TASK 3: MARGINAL SPEEDUP
    # ======================================

    print()
    print("TASK 3: MARGINAL SPEEDUP")
    print("-" * 65)

    for p in range(2, LOGICAL_CORES + 1):

        previous = strong_speedups[p - 1]
        current = strong_speedups[p]

        marginal = current - previous

        print(
            f"{p - 1} -> {p} processes: "
            f"{marginal:.4f} additional speedup"
        )

    # ======================================
    # TASK 4: VERIFICATION HASH
    # ======================================

    print()
    print("TASK 4: VERIFICATION")
    print("-" * 65)

    strong_log = strong_speedups[LOGICAL_CORES]

    gustafson_log = gustafson_speedups[LOGICAL_CORES]

    print(f"Strong scaling speedup: {strong_log:.4f}")
    print(f"Gustafson scaled speedup: {gustafson_log:.4f}")
    print(f"Amdahl theoretical limit: {amdahl_cap:.4f}")

    verification = emit_verify_tag(
        "PHASE2",
        STUDENT_ID,
        strong_log,
        gustafson_log
    )

    print()
    print(verification)


if __name__ == "__main__":
    mp.freeze_support()
    main()

