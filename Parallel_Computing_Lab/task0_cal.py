
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230107142
ITERATIONS = 1_000_000


def main():

    # Check logical CPU count
    cpu_count = mp.cpu_count()

    # Measure timer call latency
    start = time.perf_counter_ns()

    for _ in range(ITERATIONS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time = end - start

    average_latency = total_time / ITERATIONS

    print("PHASE 0 - HARDWARE CALIBRATION")
    print("--------------------------------")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU count: {cpu_count}")
    print(f"Number of timer calls: {ITERATIONS}")
    print(f"Total time: {total_time} ns")
    print(f"Average timer call latency: {average_latency:.4f} ns")

    # The PDF specifies one measured calibration value.
    # The second hash parameter is therefore set to zero.
    verification = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_latency,
        0.0
    )

    print()
    print(verification)


if __name__ == "__main__":
    main()

