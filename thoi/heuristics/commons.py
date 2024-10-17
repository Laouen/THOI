import torch


def _get_valid_candidates(initial_solution: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
    """
    Efficiently generates valid candidates by excluding elements present in the initial solution.

    Parameters:
    - initial_solution (torch.Tensor): Tensor of shape (batch_size, o) containing selected indices. Note: o < N and all elements must be unique.
    - N (int): Total number of possible elements.
    - device (torch.device): Device where tensors are located.

    Returns:
    - valid_candidates (torch.Tensor): Tensor of shape (batch_size, N - o) containing valid candidate indices.
    """

    batch_size, o = initial_solution.shape

    # Create a tensor of all elements, expanded across the batch
    # |batch_size| x |N|
    all_elements = torch.arange(N, device=device).unsqueeze(0).expand(batch_size, N)

    # Expand initial_solution for comparison
    # |batch_size| x |1| x |o|
    initial_solution_expanded = initial_solution.unsqueeze(1)

    # Expand all_elements for comparison
    # |batch_size| x |N| x |1|
    all_elements_expanded = all_elements.unsqueeze(2)

    # Compare all_elements with initial_solution
    # This creates a mask where True indicates the element is in initial_solution
    # |batch_size| x |N| x |o|
    is_in = (all_elements_expanded == initial_solution_expanded)

    # Determine if any of the o elements match for each (batch, N)
    # |batch_size| x |N|
    is_in_any = is_in.any(dim=2)

    # Create a mask for valid candidates (not in initial_solution)
    # |batch_size| x |N|
    valid_mask = ~is_in_any

    # Select valid candidates using the mask
    # |batch_size| x |N - o|
    valid_candidates = torch.masked_select(all_elements, valid_mask).view(batch_size, N - o)

    return valid_candidates