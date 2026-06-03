"""
GPU vs CPU Profiling for multi_order_measures
=============================================

This script profiles why GPU might not be faster than CPU for THOI's
multi_order_measures function. It uses several complementary profiling
techniques:

1. WALL-CLOCK TIMING — Measures end-to-end and per-stage time.
2. PYTORCH PROFILER — Records GPU/CPU kernel-level traces (viewable in
   Chrome's chrome://tracing or https://ui.perfetto.dev).
3. MEMORY PROFILING — Tracks peak GPU memory and CPU→GPU transfer volume.
4. SCALING ANALYSIS — Varies N (number of variables) and batch_size to
   find the crossover point where GPU becomes worthwhile.

TYPICAL BOTTLENECKS FOR SMALL-TO-MEDIUM PROBLEMS ON GPU:
- Kernel launch overhead: each PyTorch op dispatches a CUDA kernel. For
  small tensors the launch cost (~5–10 µs) dominates actual compute.
- CPU↔GPU data transfers: moving tensors to/from GPU (e.g., nplets from
  DataLoader, results back to CPU for DataFrame) is expensive.
- DataLoader overhead: CovarianceDataset creates tensors directly on device,
  but itertools.combinations is Python-side and torch.tensor() per element
  adds overhead. With num_workers>0, workers can only produce CPU tensors.
- Collector overhead: the default batch_to_csv builds a pandas DataFrame
  per batch, forcing a GPU→CPU sync + Python object creation.
- torch.logdet on small matrices: for order=3–5, the matrices are tiny
  (3×3 to 5×5); GPU gains nothing over CPU for these sizes.
- _get_single_exclusion_covmats: heavy gather/expand ops that create
  large intermediate tensors, causing memory pressure on GPU.

HOW TO READ THE RESULTS:
- The "Stage Breakdown" table shows where time is spent per-stage.
- The .json trace files can be loaded in chrome://tracing for flame graphs.
- The "Scaling Analysis" shows how GPU/CPU ratio changes with problem size.

Usage:
    python scripts/profile_gpu_vs_cpu.py

Requirements:
    pip install torch numpy scipy pandas tqdm
    (CUDA-capable GPU + matching PyTorch for GPU profiling)
"""

import time
import sys
import os
import warnings
from contextlib import contextmanager
from functools import partial
from itertools import combinations

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thoi.measures.gaussian_copula import (
    multi_order_measures,
    _generate_nplets_covmats,
    _generate_nplets_marginal_entropies,
    _get_tc_dtc_from_batched_covmat,
    _get_bias_correctors,
)
from thoi.measures.utils import (
    _all_min_1_ids,
    _marginal_gaussian_entropies,
)
from thoi.commons import gaussian_copula_covmat


# ─── Utility ──────────────────────────────────────────────────────────────────

HAS_CUDA = torch.cuda.is_available()

@contextmanager
def timer(label):
    """Simple wall-clock timer context manager."""
    if HAS_CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if HAS_CUDA:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    timer.last = elapsed
    print(f"  {label}: {elapsed:.4f}s")

timer.last = 0.0


def make_synthetic_data(N, T=500, D=1, seed=42):
    """Generate correlated multivariate time series."""
    rng = np.random.default_rng(seed)
    # Random correlation structure so logdet is well-conditioned
    A = rng.standard_normal((N, N))
    cov = A @ A.T / N + np.eye(N) * 0.1
    L = np.linalg.cholesky(cov)
    datasets = []
    for _ in range(D):
        Z = rng.standard_normal((T, N))
        datasets.append((Z @ L.T).astype(np.float32))
    if D == 1:
        return datasets[0]
    return np.stack(datasets)


# ─── 1. WALL-CLOCK COMPARISON ────────────────────────────────────────────────

def wallclock_comparison(N=10, T=500, min_order=3, max_order=5, batch_size=50000):
    """Compare end-to-end wall-clock time for CPU vs GPU."""
    print("=" * 70)
    print(f"1. WALL-CLOCK COMPARISON  (N={N}, T={T}, orders={min_order}–{max_order})")
    print("=" * 70)

    X = make_synthetic_data(N, T)

    # Warm up GPU (first CUDA call has ~1s overhead for context creation)
    if HAS_CUDA:
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    # --- CPU ---
    print("\n[CPU]")
    with timer("Total"):
        result_cpu = multi_order_measures(
            X, min_order=min_order, max_order=max_order,
            batch_size=batch_size, device=torch.device('cpu')
        )
    cpu_time = timer.last

    # --- GPU ---
    if HAS_CUDA:
        print("\n[GPU]")
        with timer("Total"):
            result_gpu = multi_order_measures(
                X, min_order=min_order, max_order=max_order,
                batch_size=batch_size, device=torch.device('cuda')
            )
        gpu_time = timer.last
        print(f"\n  Speedup (CPU/GPU): {cpu_time/gpu_time:.2f}x")
        if gpu_time > cpu_time:
            print("  ⚠ GPU is SLOWER — see stage breakdown below for why.")
    else:
        print("\n  [No CUDA available — skipping GPU run]")

    return result_cpu


# ─── 2. PER-STAGE BREAKDOWN ─────────────────────────────────────────────────

def stage_breakdown(N=10, T=500, min_order=3, max_order=None, batch_size=50000):
    """Profile each computational stage independently."""
    max_order = max_order or min(N, 6)
    print("\n" + "=" * 70)
    print(f"2. PER-STAGE BREAKDOWN  (N={N}, T={T}, order={min_order})")
    print("=" * 70)

    X_np = make_synthetic_data(N, T)
    order = min_order

    for device_name in ['cpu'] + (['cuda'] if HAS_CUDA else []):
        device = torch.device(device_name)
        print(f"\n--- Device: {device_name.upper()} ---")

        # Stage A: Gaussian copula + covariance computation
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)  # (1, T, N)
        with timer("A. Gaussian copula → covmat"):
            _, covmats = gaussian_copula_covmat(X_tensor)
            covmats = covmats.to(device)

        # Stage B: Marginal entropies
        with timer("B. Marginal entropies"):
            marginal_ents = _marginal_gaussian_entropies(covmats)

        # Stage C: Generate n-plet indices via CovarianceDataset (as the real code does)
        # Note: CovarianceDataset creates tensors directly on the target device.
        # The overhead here is Python-side itertools.combinations + per-element
        # torch.tensor() calls, NOT a CPU→GPU bulk transfer.
        from thoi.dataset import CovarianceDataset
        from torch.utils.data import DataLoader
        with timer("C. DataLoader nplet generation (on device)"):
            dataset = CovarianceDataset(N, order, device=device)
            loader = DataLoader(dataset, batch_size=len(dataset))
            nplets = next(iter(loader))

        # Stage D: Extract n-plet covariance submatrices
        with timer("D. Gather nplet covmats"):
            nplets_covmats = _generate_nplets_covmats(covmats, nplets)
            B = nplets_covmats.shape[0]
            D_dim = nplets_covmats.shape[1]
            nplets_covmats = nplets_covmats.view(B * D_dim, order, order)

        # Stage E: Gather marginal entropies
        with timer("E. Gather marginal entropies"):
            nplets_marginal = _generate_nplets_marginal_entropies(marginal_ents, nplets)
            nplets_marginal = nplets_marginal.view(B * D_dim, order)

        # Stage F: Compute TC/DTC (logdet + exclusion gather)
        allmin1 = _all_min_1_ids(order, device=device)
        bc1, bcN, bcNmin1 = _get_bias_correctors(None, order, B, 1, device, covmats.dtype)
        with timer("F. TC/DTC computation (logdet + exclusion)"):
            tc, dtc, o, s = _get_tc_dtc_from_batched_covmat(
                nplets_covmats, allmin1,
                bc1[:B], bcN[:B], bcNmin1[:B],
                nplets_marginal,
            )

        # Stage G: Transfer results back to CPU
        with timer("G. Results → CPU"):
            tc_cpu = tc.cpu().numpy()
            dtc_cpu = dtc.cpu().numpy()

        # Stage H: DataFrame construction (always CPU-bound)
        import pandas as pd
        with timer("H. DataFrame construction"):
            df = pd.DataFrame({
                'tc': tc_cpu, 'dtc': dtc_cpu,
            })


# ─── 3. PYTORCH PROFILER TRACE ──────────────────────────────────────────────

def pytorch_profiler_trace(N=10, T=500, min_order=3, max_order=5, batch_size=50000):
    """
    Record a PyTorch profiler trace for both CPU and GPU.

    The output .json files can be opened in:
      - Chrome: navigate to chrome://tracing and load the file
      - Perfetto: https://ui.perfetto.dev (drag & drop)

    The trace shows:
      - Every PyTorch operator and its duration
      - CUDA kernel launches and GPU execution time
      - CPU↔GPU synchronization points
      - Memory allocation events
    """
    print("\n" + "=" * 70)
    print("3. PYTORCH PROFILER TRACES")
    print("=" * 70)

    X = make_synthetic_data(N, T)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'profiling_output')
    os.makedirs(output_dir, exist_ok=True)

    for device_name in ['cpu'] + (['cuda'] if HAS_CUDA else []):
        device = torch.device(device_name)

        # Warm-up run (JIT compilation, CUDA context, etc.)
        multi_order_measures(
            X, min_order=min_order, max_order=max_order,
            batch_size=batch_size, device=device
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_name == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            multi_order_measures(
                X, min_order=min_order, max_order=max_order,
                batch_size=batch_size, device=device
            )

        # Save trace
        trace_path = os.path.join(output_dir, f'trace_{device_name}.json')
        prof.export_chrome_trace(trace_path)
        print(f"\n  Saved trace: {trace_path}")

        # Print top operators by time
        sort_key = "cuda_time_total" if device_name == 'cuda' else "cpu_time_total"
        print(f"\n  Top 20 operators ({device_name.upper()}):")
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

        # Print memory summary for GPU
        if device_name == 'cuda':
            print(f"\n  GPU memory summary:")
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


# ─── 4. GPU MEMORY ANALYSIS ─────────────────────────────────────────────────

def gpu_memory_analysis(N=10, T=500, min_order=3, max_order=5, batch_size=50000):
    """Track GPU memory allocation during computation."""
    if not HAS_CUDA:
        print("\n  [No CUDA — skipping memory analysis]")
        return

    print("\n" + "=" * 70)
    print("4. GPU MEMORY ANALYSIS")
    print("=" * 70)

    X = make_synthetic_data(N, T)
    device = torch.device('cuda')

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_before = torch.cuda.memory_allocated()
    print(f"  Memory before: {mem_before / 1024**2:.1f} MB")

    multi_order_measures(
        X, min_order=min_order, max_order=max_order,
        batch_size=batch_size, device=device
    )
    torch.cuda.synchronize()

    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_reserved = torch.cuda.max_memory_reserved()

    print(f"  Memory after:    {mem_after / 1024**2:.1f} MB")
    print(f"  Peak allocated:  {mem_peak / 1024**2:.1f} MB")
    print(f"  Peak reserved:   {mem_reserved / 1024**2:.1f} MB")
    print(f"  Peak used:       {(mem_peak - mem_before) / 1024**2:.1f} MB")

    # Estimate data transfer volume
    n_nplets = sum(
        len(list(combinations(range(N), k)))
        for k in range(min_order, max_order + 1)
    )
    print(f"\n  Total n-plets:   {n_nplets}")
    print(f"  Estimated CPU→GPU transfer per batch: nplet indices + result tensors")
    print(f"  This sync overhead can dominate for small problems.")


# ─── 5. SCALING ANALYSIS ────────────────────────────────────────────────────

def scaling_analysis(T=500, batch_size=100000):
    """
    Vary N to find the crossover point where GPU becomes faster.

    For small N, the number of n-plets C(N,k) is tiny and GPU overhead
    dominates. As N grows, the computational volume increases and GPU
    parallelism pays off.
    """
    print("\n" + "=" * 70)
    print("5. SCALING ANALYSIS — Finding the GPU crossover point")
    print("=" * 70)

    if not HAS_CUDA:
        print("  [No CUDA — skipping scaling analysis]")
        return

    # Warm up
    _ = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()

    results = []
    for N in [6, 8, 10, 12, 14, 16, 18, 20]:
        X = make_synthetic_data(N, T)
        min_order = 3
        max_order = min(N, 5)

        # CPU
        t0 = time.perf_counter()
        multi_order_measures(X, min_order=min_order, max_order=max_order,
                             batch_size=batch_size, device=torch.device('cpu'))
        cpu_t = time.perf_counter() - t0

        # GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        multi_order_measures(X, min_order=min_order, max_order=max_order,
                             batch_size=batch_size, device=torch.device('cuda'))
        torch.cuda.synchronize()
        gpu_t = time.perf_counter() - t0

        n_nplets = sum(
            len(list(combinations(range(N), k)))
            for k in range(min_order, max_order + 1)
        )
        speedup = cpu_t / gpu_t
        results.append((N, n_nplets, cpu_t, gpu_t, speedup))

        marker = "✓ GPU wins" if speedup > 1.0 else "✗ CPU wins"
        print(f"  N={N:>2}, nplets={n_nplets:>7}, "
              f"CPU={cpu_t:.3f}s, GPU={gpu_t:.3f}s, "
              f"speedup={speedup:.2f}x  {marker}")

    print("\n  Interpretation:")
    print("  - If GPU never wins, the problem is too small for GPU parallelism.")
    print("  - The crossover typically happens when there are >100K n-plets")
    print("    AND the matrices are large enough (order >= 5) for logdet to")
    print("    benefit from GPU.")


# ─── 6. BATCH SIZE SENSITIVITY ──────────────────────────────────────────────

def batch_size_sensitivity(N=12, T=500, min_order=3, max_order=5):
    """
    Vary batch_size to see its effect on GPU performance.

    Small batches → more kernel launches, more CPU↔GPU syncs.
    Large batches → better GPU utilization, but more memory.
    """
    print("\n" + "=" * 70)
    print(f"6. BATCH SIZE SENSITIVITY  (N={N})")
    print("=" * 70)

    X = make_synthetic_data(N, T)

    for device_name in ['cpu'] + (['cuda'] if HAS_CUDA else []):
        device = torch.device(device_name)
        print(f"\n  --- {device_name.upper()} ---")

        for bs in [100, 1000, 10000, 100000, 500000, 1000000]:
            if HAS_CUDA:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            multi_order_measures(X, min_order=min_order, max_order=max_order,
                                 batch_size=bs, device=device)
            if HAS_CUDA:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"    batch_size={bs:>9}: {elapsed:.4f}s")


# ─── 7. COLLECTOR OVERHEAD ──────────────────────────────────────────────────

def collector_overhead(N=12, T=500, min_order=3, max_order=5, batch_size=100000):
    """
    Compare default DataFrame collector vs raw tensor collector.

    The default collector (batch_to_csv) creates a pandas DataFrame per
    batch, which forces GPU→CPU sync and is pure Python overhead.
    A tensor-only collector avoids this.
    """
    print("\n" + "=" * 70)
    print(f"7. COLLECTOR OVERHEAD  (N={N})")
    print("=" * 70)

    X = make_synthetic_data(N, T)

    def tensor_collector(nplets, tc, dtc, o, s, bn):
        """Minimal collector: just stack tensors, no DataFrame."""
        return torch.stack([tc, dtc, o, s], dim=-1)

    def tensor_aggregator(items):
        return torch.cat(items, dim=0)

    for device_name in ['cpu'] + (['cuda'] if HAS_CUDA else []):
        device = torch.device(device_name)
        print(f"\n  --- {device_name.upper()} ---")

        # Default (DataFrame) collector
        if HAS_CUDA:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        multi_order_measures(X, min_order=min_order, max_order=max_order,
                             batch_size=batch_size, device=device)
        if HAS_CUDA:
            torch.cuda.synchronize()
        df_time = time.perf_counter() - t0
        print(f"    Default (DataFrame) collector: {df_time:.4f}s")

        # Tensor-only collector
        if HAS_CUDA:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        multi_order_measures(X, min_order=min_order, max_order=max_order,
                             batch_size=batch_size, device=device,
                             batch_data_collector=tensor_collector,
                             batch_aggregation=tensor_aggregator)
        if HAS_CUDA:
            torch.cuda.synchronize()
        tensor_time = time.perf_counter() - t0
        print(f"    Tensor-only collector:         {tensor_time:.4f}s")
        print(f"    Collector overhead:            {df_time - tensor_time:.4f}s "
              f"({(df_time - tensor_time) / df_time * 100:.0f}% of total)")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("THOI multi_order_measures — GPU vs CPU Profiling")
    print("=" * 70)

    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("⚠ No CUDA GPU detected. Only CPU profiling will run.")
        print("  GPU-related sections will be skipped.")

    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print()

    # Run all profiling sections
    # You can comment out sections you don't need.

    # 1. Quick end-to-end comparison
    wallclock_comparison(N=10, T=500, min_order=3, max_order=5)

    # 2. Per-stage breakdown (most useful for finding bottlenecks)
    stage_breakdown(N=10, T=500, min_order=3, max_order=5)

    # 3. PyTorch profiler traces (generates .json files for flame graphs)
    pytorch_profiler_trace(N=10, T=500, min_order=3, max_order=5)

    # 4. GPU memory usage
    gpu_memory_analysis(N=10, T=500, min_order=3, max_order=5)

    # 5. Scaling: find when GPU becomes worthwhile
    scaling_analysis(T=500)

    # 6. Batch size tuning
    batch_size_sensitivity(N=12, T=500, min_order=3, max_order=5)

    # 7. Collector overhead (DataFrame vs raw tensors)
    collector_overhead(N=12, T=500, min_order=3, max_order=5)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print("""
Next steps based on common findings:

1. If Stage D/F dominates → the gather/logdet ops on small matrices
   don't benefit from GPU. Consider keeping computation on CPU for
   small orders, or fusing operations with custom CUDA kernels.

2. If Stage G/H dominates → the DataFrame construction is the bottleneck.
   Use a tensor-only collector (see section 7) and build the DataFrame
   only at the very end.

3. If GPU never wins in scaling analysis → your problem size doesn't
   justify GPU overhead. GPU shines when N > ~15 and orders > 5, where
   C(N,k) produces millions of n-plets with non-trivial matrix sizes.

4. If batch_size matters a lot on GPU → you're paying per-batch kernel
   launch overhead. Use the largest batch_size that fits in GPU memory.

5. Check the .json traces in chrome://tracing for the most detailed
   view of where time is spent kernel-by-kernel.
""")


if __name__ == '__main__':
    main()
