# Differentiable higher-order interaction optimization

`differentiable_hoi` searches for high- or low-scoring Gaussian TC, DTC, O-information, and S-information n-plets with a projected-gradient relaxation.

```python
from thoi.heuristics import differentiable_hoi

best_nplets, best_scores = differentiable_hoi(
    X,
    initial_order=3,
    order=10,
    metric='o',  # 'tc', 'dtc', 'o', or 's'
    largest=False,
)
```

Each interaction order is optimized independently, while orders and random restarts are processed together in mixed-order batches. The optimizer projects every update onto the fixed-cardinality hypersimplex and uses top-k rounding followed by THOI's exact discrete scorer.

For multiple datasets, the optimized objective is the mean selected metric across datasets, matching THOI's standard string-metric semantics. Custom metric callables are not supported because the relaxed optimizer requires an analytic gradient.