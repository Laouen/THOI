import os
import numpy as np
import pandas as pd
import torch

from thoi.measures.gaussian_copula_local import (
    time_averaged_local_measures,
)


def test_time_averaged_local_measures_match_traditional_on_disk_data():
    """Load the test timeseries from tests/data and compare traditional
    multi-order measures with the time-averaged local n-plet measures.

    This mirrors the data-loading used by the repository's existing tests
    so the comparison uses the same on-disk dataset.
    """

    # locate test data file (same pattern used by existing tests)
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "data", "X_random.tsv")
    X = pd.read_csv(file_path, sep="\t", header=None).values

    # dimensions of the full dataset

    # Work on a small subset: first 3 variables .
    N = 3
    X_sub = X[:, :N]

    # Load precomputed ground-truth measures from disk so the test does not
    # depend on the internal implementation of `multi_order_measures`.
    truth_file = os.path.join(current_dir, "data", "X_random__multi_order_measures.tsv")
    df_true = pd.read_csv(truth_file, sep="\t")

    # Find the row corresponding to the full n-plet of variables [0,1,2]
    # (order == N and var_0,var_1,var_2 == 1)
    mask = (df_true["order"] == N)
    mask &= (df_true[[f"var_{i}" for i in range(N)]] == 1).all(axis=1)
    # ensure we pick dataset 0
    mask &= (df_true["dataset"] == 0)
    row = df_true[mask].iloc[0]
    tc_tr = float(row["tc"])
    dtc_tr = float(row["dtc"])
    o_tr = float(row["o"])
    s_tr = float(row["s"])

    # Use the memory-friendly wrapper which gaussianizes, computes local
    # time-resolved measures, averages them and applies bias correction.
    # Tune parameters to limit RAM usage: small batch_size and moderate time_chunk;
    # use float32 to further reduce peak memory.
    # Note: the wrapper returns a dict; the measures for order N are accessed with key N.
    # For N=3, wrapper_res[3] gives the measures for the full 3-plet.
    # Run the wrapper on the same subsampled timeseries with conservative
    # memory settings. Decreasing time_chunk reduces peak memory usage.
    wrapper_res = time_averaged_local_measures(
        X_sub.astype(np.float32),
        min_order=N,
        max_order=N,
        device="cpu",
        batch_size=1,
        time_chunk=50,
        eps=1e-6,
        bias_correction=True,
    )

    # extract measures tensor and pick the single (full) n-plet
    # wrapper_res[3] shape: [n_combinations, D, 4] (or similar); follow previous usage
    meas = wrapper_res[3].squeeze(0).squeeze(0).cpu().numpy()
    tc_loc, dtc_loc, o_loc, s_loc = meas.tolist()

    # tolerance consistent with validation runs
    tol = 1e-6

    assert abs(tc_tr - tc_loc) < tol, f"TC mismatch: {tc_tr} vs {tc_loc}"
    assert abs(dtc_tr - dtc_loc) < tol, f"DTC mismatch: {dtc_tr} vs {dtc_loc}"
    assert abs(o_tr - o_loc) < tol, f"O mismatch: {o_tr} vs {o_loc}"
    assert abs(s_tr - s_loc) < tol, f"S mismatch: {s_tr} vs {s_loc}"
