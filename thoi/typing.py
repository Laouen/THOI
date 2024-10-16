from typing import Union, Sequence, Any
import torch
import numpy as np

TensorLikeArray = Union[
    torch.Tensor,
    np.ndarray,
    Sequence[Union[np.ndarray, Sequence[Any]]],
]