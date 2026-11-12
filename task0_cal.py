
import time
import multiprocessing as mp

from harness import emit_verify_tag


STUDENT_ID = 230107142
CLOCK_READS = 2_000_000


def main():

    print("PHASE 0: HARDWARE CALIBRATION")

    print("Logical CPU count:", mp.cpu_count())

    # Measure clock-reading overhead
    start = time.perf_counter_ns()

    for _ in range(CLOCK_READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / CLOCK_READS
    total_ms = total_ns / 1_000_000

    print(f"Clock reads: {CLOCK_READS:,}")
    print(f"Total time: {total_ms:.4f} ms")
    print(f"Average clock latency: {average_ns:.4f} ns")

    # Generate verification hash
    tag = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_ns,
        total_ms
    )

    print(tag)


if __name__ == "__main__":
    main()
