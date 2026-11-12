
import multiprocessing as mp
import math
import statistics

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230107142

N_STRONG = 1_600_000
N0 = 250_000

PHYSICAL_CORES = 8
LOGICAL_CORES = mp.cpu_count()

WARMUP_RUNS = 1
TIMED_RUNS = 3


# Stage A: Serial reduction and prefix filter
def stage_a(N):

    count = int(N * 0.05)
    x = 1.0

    for i in range(count):
        for _ in range(6):
            x = math.sin(x + i * 0.000001)
            x = math.sqrt(abs(x) + 1.0)

    return x


# Stage B: Parallel computational kernel
def compute_chunk(args):

    start, end, seed = args
    total = 0.0

    for i in range(start, end):

        x = (i % 1000) * 0.001 + seed

        for _ in range(35):
            x = x * 1.0000001 + 0.000001
            x = math.sin(x) + math.cos(x)

        total += x

    return total


# Execute one complete trial
def run_trial(N, pool, p, stage_a_samples):

    start_a = __import__("time").perf_counter_ns()

    seed = stage_a(N)

    end_a = __import__("time").perf_counter_ns()

    stage_a_ms = (end_a - start_a) / 1e6
    stage_a_samples.append(stage_a_ms)

    # Divide work evenly among processes
    jobs = []

    for worker in range(p):

        start = worker * N // p
        end = (worker + 1) * N // p

        jobs.append((start, end, seed))

    # Persistent pool: no pool creation inside timed region
    pool.map(compute_chunk, jobs)


def benchmark_configuration(N, p):

    stage_a_samples = []

    # Create pool outside benchmark timing
    with mp.Pool(processes=p) as pool:

        stats = run_benchmark(
            lambda: run_trial(
                N, pool, p, stage_a_samples
            ),
            warmup_runs=WARMUP_RUNS,
            timed_runs=TIMED_RUNS
        )

    median_stage_a = statistics.median(
        stage_a_samples[-TIMED_RUNS:]
    )

    return stats, median_stage_a


def main():

    print("=" * 65)
    print("PHASE 2: STRONG AND WEAK SCALING")
    print("=" * 65)

    print("Physical cores:", PHYSICAL_CORES)
    print("Logical CPUs:", LOGICAL_CORES)

    process_counts = list(range(1, LOGICAL_CORES + 1))

    # -------------------------------------------------
    # EXPERIMENT 2A: STRONG SCALING
    # -------------------------------------------------

    print("\n========== STRONG SCALING ==========")

    strong_results = {}

    for p in process_counts:

        print(f"\nTesting p={p}")

        stats, stage_a_ms = benchmark_configuration(
            N_STRONG, p
        )

        strong_results[p] = {
            "stats": stats,
            "stage_a": stage_a_ms
        }

        print(f"Median time: {stats.median_ms:.4f} ms")
        print(f"IQR: {stats.iqr_ms:.4f} ms")
        print(f"Stage A: {stage_a_ms:.4f} ms")

    T1_strong = strong_results[1]["stats"].median_ms

    print("\nSTRONG SCALING RESULTS")
    print("p | Time(ms) | Speedup | Efficiency")

    for p in process_counts:

        tp = strong_results[p]["stats"].median_ms

        speedup = T1_strong / tp
        efficiency = speedup / p

        strong_results[p]["speedup"] = speedup
        strong_results[p]["efficiency"] = efficiency

        print(
            f"{p} | {tp:.4f} | "
            f"{speedup:.4f} | {efficiency:.4f}"
        )

    # -------------------------------------------------
    # AMDahl's LAW
    # -------------------------------------------------

    T_stage_a_1 = strong_results[1]["stage_a"]

    serial_fraction = T_stage_a_1 / T1_strong

    theoretical_max = 1 / serial_fraction

    theoretical_8 = 1 / (
        serial_fraction +
        (1 - serial_fraction) / LOGICAL_CORES
    )

    observed_8 = strong_results[LOGICAL_CORES]["speedup"]

    print("\n========== AMDahl's LAW ==========")

    print(f"Serial fraction: {serial_fraction:.6f}")
    print(f"Theoretical maximum speedup: {theoretical_max:.4f}")
    print(f"Theoretical speedup at 8: {theoretical_8:.4f}")
    print(f"Observed speedup at 8: {observed_8:.4f}")

    # Marginal speedup analysis
    print("\nMARGINAL SPEEDUP ANALYSIS")

    marginal = {}

    for p in range(2, LOGICAL_CORES + 1):

        gain = (
            strong_results[p]["speedup"]
            - strong_results[p - 1]["speedup"]
        )

        marginal[p] = gain

        print(f"Workers {p-1} to {p}: {gain:.4f}")

    if LOGICAL_CORES >= 3:

        previous_gain = marginal[LOGICAL_CORES - 1]
        current_gain = marginal[LOGICAL_CORES]

        if previous_gain > 0:

            drop = (
                (previous_gain - current_gain)
                / previous_gain
            ) * 100

            print(
                "Marginal speedup drop from the "
                f"previous step: {drop:.2f}%"
            )

    # -------------------------------------------------
    # EXPERIMENT 2B: WEAK SCALING
    # -------------------------------------------------

    print("\n========== WEAK SCALING ==========")

    weak_results = {}

    for p in process_counts:

        N = p * N0

        print(f"\nTesting p={p}, N={N:,}")

        stats, stage_a_ms = benchmark_configuration(N, p)

        weak_results[p] = {
            "stats": stats,
            "stage_a": stage_a_ms
        }

        print(f"Median time: {stats.median_ms:.4f} ms")
        print(f"IQR: {stats.iqr_ms:.4f} ms")
        print(f"Stage A: {stage_a_ms:.4f} ms")

    T1_weak = weak_results[1]["stats"].median_ms

    print("\nWEAK SCALING RESULTS")
    print("p | N | Time(ms) | Efficiency | Gustafson Speedup")

    for p in process_counts:

        tp = weak_results[p]["stats"].median_ms

        stage_a_ms = weak_results[p]["stage_a"]

        efficiency = T1_weak / tp

        omega = stage_a_ms / tp

        gustafson = p - omega * (p - 1)

        weak_results[p]["efficiency"] = efficiency
        weak_results[p]["gustafson"] = gustafson

        print(
            f"{p} | {p * N0} | {tp:.4f} | "
            f"{efficiency:.4f} | {gustafson:.4f}"
        )

    # -------------------------------------------------
    # VERIFICATION HASH
    # -------------------------------------------------

    strong_speedup_8 = strong_results[LOGICAL_CORES]["speedup"]

    gustafson_speedup_8 = weak_results[LOGICAL_CORES]["gustafson"]

    print("\n========== VERIFICATION ==========")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup_8,
            gustafson_speedup_8
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
