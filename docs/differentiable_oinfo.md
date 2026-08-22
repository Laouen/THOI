# Differentiable O-information optimization

`differentiable_oinfo` searches for high- or low-O-information n-plets using a differentiable fixed-cardinality relaxation and warm starts across interaction orders.

```python
from thoi.heuristics import differentiable_oinfo

best_nplets, best_scores = differentiable_oinfo(
    X,
    initial_order=3,
    order=10,
    largest=False,
)
```

For multiple datasets, the optimized objective is the mean Gaussian O-information across datasets, matching THOI's standard O-information scoring. The optimizer is specific to Gaussian O-information and does not accept a custom `metric` callable.
