
import multiprocessing as mp
from multiprocessing import shared_memory
import math
import time
from statistics import median
from array import array

from harness import emit_verify_tag


STUDENT_ID = 230107142
P = 8

RUNS = 3
BUFFER_SIZE = 64 * 1024 * 1024
N = 256
FLOPS_PER_WORKER = 60_000_000

TILE_SIZES = [8, 16, 32, 64, 96, 128]


# ==================================================
# PART 3A - DRAM BANDWIDTH
# ==================================================

def copy_worker(args):
    source_name, target_name, start, end = args

    source = shared_memory.SharedMemory(name=source_name)
    target = shared_memory.SharedMemory(name=target_name)

    src = source.buf[start:end]
    dst = target.buf[start:end]

    dst[:] = src

    del src
    del dst

    source.close()
    target.close()


def measure_bandwidth():

    source = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )
    target = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )

    source.buf[:] = b"\x01" * BUFFER_SIZE

    chunk = BUFFER_SIZE // P
    jobs = []

    for worker in range(P):
        start = worker * chunk
        end = BUFFER_SIZE if worker == P - 1 else start + chunk

        jobs.append((
            source.name,
            target.name,
            start,
            end
        ))

    samples = []

    try:
        for _ in range(RUNS):

            processes = [
                mp.Process(target=copy_worker, args=(job,))
                for job in jobs
            ]

            start_time = time.perf_counter_ns()

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            elapsed = (
                time.perf_counter_ns() - start_time
            ) / 1e9

            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("A memory-copy worker failed.")

            samples.append(elapsed)

    finally:
        source.close()
        source.unlink()
        target.close()
        target.unlink()

    t = median(samples)

    bandwidth = (2 * BUFFER_SIZE) / t / 1e9

    return t, bandwidth


# ==================================================
# PART 3A - COMPUTE CEILING
# ==================================================

def compute_worker():

    loops = FLOPS_PER_WORKER // 32

    a = [1.000001 + i * 0.000001 for i in range(16)]
    b = [1.000002 + i * 0.000001 for i in range(16)]

    x = a[:]

    start = time.perf_counter_ns()

    for _ in range(loops):
        x[0] = x[0] * a[0] + b[0]
        x[1] = x[1] * a[1] + b[1]
        x[2] = x[2] * a[2] + b[2]
        x[3] = x[3] * a[3] + b[3]
        x[4] = x[4] * a[4] + b[4]
        x[5] = x[5] * a[5] + b[5]
        x[6] = x[6] * a[6] + b[6]
        x[7] = x[7] * a[7] + b[7]
        x[8] = x[8] * a[8] + b[8]
        x[9] = x[9] * a[9] + b[9]
        x[10] = x[10] * a[10] + b[10]
        x[11] = x[11] * a[11] + b[11]
        x[12] = x[12] * a[12] + b[12]
        x[13] = x[13] * a[13] + b[13]
        x[14] = x[14] * a[14] + b[14]
        x[15] = x[15] * a[15] + b[15]

    elapsed = (time.perf_counter_ns() - start) / 1e9

    return elapsed, sum(x)


def compute_process(result_queue):

    result_queue.put(compute_worker())


def measure_compute():

    samples = []

    for _ in range(RUNS):

        result_queue = mp.Queue()

        processes = [
            mp.Process(
                target=compute_process,
                args=(result_queue,)
            )
            for _ in range(P)
        ]

        start_time = time.perf_counter_ns()

        for process in processes:
            process.start()

        results = [
            result_queue.get()
            for _ in range(P)
        ]

        for process in processes:
            process.join()

        elapsed = (
            time.perf_counter_ns() - start_time
        ) / 1e9

        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("A compute worker failed.")

        samples.append(elapsed)

        result_queue.close()

    t = median(samples)

    total_flops = P * FLOPS_PER_WORKER

    performance = total_flops / t / 1e9

    return t, performance


# ==================================================
# PART 3B - GEMM WORKERS
# ==================================================

def gemm_worker(args):

    variant, a_name, b_name, c_name, row_start, row_end, tile = args

    a_shm = shared_memory.SharedMemory(name=a_name)
    b_shm = shared_memory.SharedMemory(name=b_name)
    c_shm = shared_memory.SharedMemory(name=c_name)

    A = memoryview(a_shm.buf).cast("d")
    B = memoryview(b_shm.buf).cast("d")
    C = memoryview(c_shm.buf).cast("d")

    if variant == 1:

        # Variant 1: i-j-k, strided access to B

        for i in range(row_start, row_end):
            for j in range(N):

                total = 0.0

                for k in range(N):
                    total += A[i * N + k] * B[k * N + j]

                C[i * N + j] = total

    elif variant == 2:

        # Variant 2: i-k-j, unit-stride access to B

        for i in range(row_start, row_end):
            for k in range(N):

                a_ik = A[i * N + k]
                b_row = k * N
                c_row = i * N

                for j in range(N):
                    C[c_row + j] += a_ik * B[b_row + j]

    elif variant == 3:

        # Variant 3: cache-blocked i-j-k
        # with i-k-j micro-kernel and scalar hoisting

        for ii in range(row_start, row_end, tile):

            i_end = min(ii + tile, row_end)

            for jj in range(0, N, tile):

                j_end = min(jj + tile, N)

                for kk in range(0, N, tile):

                    k_end = min(kk + tile, N)

                    for i in range(ii, i_end):

                        c_row = i * N

                        for k in range(kk, k_end):

                            a_ik = A[i * N + k]
                            b_row = k * N

                            for j in range(jj, j_end):
                                C[c_row + j] += (
                                    a_ik * B[b_row + j]
                                )

    del A
    del B
    del C

    a_shm.close()
    b_shm.close()
    c_shm.close()


# ==================================================
# GEMM BENCHMARK
# ==================================================


def run_gemm(variant, A_values, B_values, tile=32):

    size = N * N

    a_shm = shared_memory.SharedMemory(
        create=True, size=size * 8
    )
    b_shm = shared_memory.SharedMemory(
        create=True, size=size * 8
    )
    c_shm = shared_memory.SharedMemory(
        create=True, size=size * 8
    )

    # Convert Python lists into binary double-precision data
    a_shm.buf[:] = array("d", A_values).tobytes()
    b_shm.buf[:] = array("d", B_values).tobytes()

    rows_per_worker = N // P
    jobs = []

    for worker in range(P):

        row_start = worker * rows_per_worker
        row_end = (
            N if worker == P - 1
            else row_start + rows_per_worker
        )

        jobs.append((
            variant,
            a_shm.name,
            b_shm.name,
            c_shm.name,
            row_start,
            row_end,
            tile
        ))

    samples = []
    output = None

    try:
        for _ in range(RUNS):

            c_shm.buf[:] = b"\x00" * (size * 8)

            processes = [
                mp.Process(
                    target=gemm_worker,
                    args=(job,)
                )
                for job in jobs
            ]

            start_time = time.perf_counter_ns()

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            elapsed = (
                time.perf_counter_ns() - start_time
            ) / 1e6

            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("A GEMM worker failed.")

            samples.append(elapsed)

        c_view = memoryview(c_shm.buf).cast("d")
        output = list(c_view)
        c_view.release()

    finally:
        a_shm.close()
        a_shm.unlink()

        b_shm.close()
        b_shm.unlink()

        c_shm.close()
        c_shm.unlink()

    return median(samples), output


def calculate_gflops(time_ms):

    return (2 * N ** 3) / (time_ms * 1e6)


# ==================================================
# MAIN
# ==================================================

def main():

    print("PHASE 3 - EMPIRICAL ROOFLINE AND GEMM")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Processes: {P}")
    print(f"Measured runs: {RUNS}")

    # ----------------------------------------------
    # PART 3A: ROOFLINE
    # ----------------------------------------------

    print("\nPART 3A - ROOFLINE CEILINGS")
    print("-" * 65)

    bw_time, bandwidth = measure_bandwidth()

    print(f"DRAM copy median: {bw_time:.4f} seconds")
    print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")

    compute_time, performance = measure_compute()

    ridge_point = performance / bandwidth

    print(f"Compute median: {compute_time:.4f} seconds")
    print(f"Sustained compute: {performance:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")

    # ----------------------------------------------
    # PART 3B: GEMM
    # ----------------------------------------------

    print("\nPART 3B - GEMM")
    print("-" * 65)

    A_values = [
        ((i % 17) + 1) * 0.01
        for i in range(N * N)
    ]

    B_values = [
        ((i % 13) + 1) * 0.01
        for i in range(N * N)
    ]

    # Variant 1

    print("\nRunning Variant 1: Naive i-j-k ...", flush=True)

    t1, C1 = run_gemm(1, A_values, B_values)

    gflops1 = calculate_gflops(t1)

    print(f"Variant 1 median: {t1:.3f} ms")
    print(f"Variant 1 GFLOP/s: {gflops1:.4f}")

    # Variant 2

    print("\nRunning Variant 2: Reordered i-k-j ...", flush=True)

    t2, C2 = run_gemm(2, A_values, B_values)

    gflops2 = calculate_gflops(t2)

    error2 = max(
        abs(x - y) for x, y in zip(C1, C2)
    )

    print(f"Variant 2 median: {t2:.3f} ms")
    print(f"Variant 2 GFLOP/s: {gflops2:.4f}")
    print(f"Variant 2 maximum error: {error2:.12e}")

    # Variant 3 tile sweep

    print("\nRunning Variant 3: Cache-blocked tile sweep ...")

    best_tile = None
    best_time = None
    best_gflops = None
    best_output = None

    print(
        f"{'Tile':<10}{'Time ms':<15}"
        f"{'GFLOP/s':<15}{'Footprint KB':<15}"
        f"{'Intensity':<15}"
    )

    for tile in TILE_SIZES:

        t3, C3 = run_gemm(
            3, A_values, B_values, tile
        )

        gflops3 = calculate_gflops(t3)

        error3 = max(
            abs(x - y) for x, y in zip(C2, C3)
        )

        footprint = 3 * tile * tile * 8 / 1024

        intensity = tile / 12

        print(
            f"{tile:<10}{t3:<15.3f}"
            f"{gflops3:<15.4f}"
            f"{footprint:<15.2f}"
            f"{intensity:<15.4f}"
        )

        if error3 >= 1e-9:
            raise RuntimeError(
                f"Variant 3 correctness failed at tile {tile}: "
                f"{error3}"
            )

        if best_gflops is None or gflops3 > best_gflops:

            best_tile = tile
            best_time = t3
            best_gflops = gflops3
            best_output = C3

    print("\nGEMM CORRECTNESS")
    print("-" * 65)

    print(f"Variant 2 max error: {error2:.12e}")

    final_error = max(
        abs(x - y) for x, y in zip(C2, best_output)
    )

    print(f"Best Variant 3 max error: {final_error:.12e}")

    print("\nBEST CACHE-BLOCKED RESULT")
    print("-" * 65)

    print(f"Best tile size: {best_tile}")
    print(f"Best tile median: {best_time:.3f} ms")
    print(f"Best Variant 3 GFLOP/s: {best_gflops:.4f}")

    print("\nROOFLINE COMPARISON")
    print("-" * 65)

    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")
    print(f"Variant 1: {gflops1:.4f} GFLOP/s")
    print(f"Variant 2: {gflops2:.4f} GFLOP/s")
    print(f"Variant 3: {best_gflops:.4f} GFLOP/s")

    print("\nVERIFICATION")
    print("-" * 65)

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            gflops1,
            best_gflops
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

