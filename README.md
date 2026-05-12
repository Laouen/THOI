# THOI: Torch - Higher Order Interactions

<img src="https://raw.githubusercontent.com/Laouen/THOI/main/img/logo.png" alt="THOI Logo" width="200">

## Description

THOI is a Python package designed to compute O information in Higher Order Interactions using batch processing. This package leverages PyTorch for efficient tensor operations.

## Installation

### Prerequisites

Ensure you have Python 3.6 or higher installed.

### Installing THOI with your prefered Versions of PyTorch

Because PyTorch installation can depend on the user environment and requirements (GPU or CPU support or a specific version of PyTorch), you need to install PyTorch separately before installing THOI. Follow these steps:

1. **Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/):**
    - Go to the PyTorch website and navigate to the "Get Started" page.
    - Select your preferences for the following options:
        - **PyTorch Build:** Stable or LTS (long-term support)
        - **Your Operating System:** Linux, Mac, or Windows
        - **Package:** Pip (recommended)
        - **Language:** Python
        - **Compute Platform:** CPU, CUDA 10.2, CUDA 11.1, etc.

2. **Get the Installation Command:**
    - Based on your selections, the PyTorch website will provide the appropriate installation command.
    - For example, for the CPU-only version, the command will look like this:

        ```bash
        pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
        ```

    - For the GPU version with CUDA 11.1, the command will look like this:

        ```bash
        pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        ```

3. **Install PyTorch:**
    - Copy and run the command provided by the PyTorch website in your terminal.

4. **Install THOI:**
    - Once PyTorch is installed, install THOI using:

        ```bash
        pip install thoi
        ```

## Usage

After installation, you can start using THOI in your projects. Here is a simple example:

```python
from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
from thoi.heuristics import simulated_annealing, greedy
import numpy as np

X = np.random.normal(0,1, (1000, 10))

# Computation of O information for the nplet that consider all the variables of X
measures = nplets_measures(X)

# Computation of O info for a single nplet (it must be a list of nplets even if it is a single nplet)
measures = nplets_measures(X, [[0,1,3]])

# Computation of O info for multiple nplets
measures = nplets_measures(X, [[0,1,3],[3,7,4],[2,6,3]])

# Extensive computation of O information measures over all combinations of features in X
measures = multi_order_measures(X)

# Compute the best 10 combinations of features (nplet) using greedy, starting by exaustive search in 
# lower order and building from there. Result shows best O information for 
# each built optimal orders
best_nplets, best_scores = greedy(X, 3, 5, repeat=10)

# Compute the best 10 combinations of features (nplet) using simulated annealing: There are two initialization options
# 1. Starting by a custom initial solution with shape (repeat, order) explicitely provided by the user.
# 2. Selecting random samples from the order.
# Result shows best O information for each built optimal orders
best_nplets, best_scores = simulated_annealing(X, 5, repeat=10)
```

For detailed usage and examples, please refer to the [documentation](https://laouen.github.io/THOI/).

## Contributing

We welcome contributions from the community. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Laouen/THOI/blob/10243c2d465dc81ee2180b652d08d3117381a01f/LICENSE) file for details.

## Citation

If you use the `thoi` library in a scientific project, please cite:

Belloli L, Mediano PAM, Cofré R, Slezak DF, Herzog R (2026) THOI: An efficient and accessible library for computing higher-order interactions enhanced by batch-processing. PLOS ONE 21(5): e0348005. https://doi.org/10.1371/journal.pone.0348005.

### BibTeX

```bibtex
@article{10.1371/journal.pone.0348005,
    doi = {10.1371/journal.pone.0348005},
    author = {Belloli, Laouen AND Mediano, Pedro A. M. AND Cofré, Rodrigo AND Slezak, Diego Fernandez AND Herzog, Rubén},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {THOI: An efficient and accessible library for computing higher-order interactions enhanced by batch-processing},
    year = {2026},
    month = {05},
    volume = {21},
    url = {https://doi.org/10.1371/journal.pone.0348005},
    pages = {1-23},
    abstract = {Complex systems are characterized by nonlinear dynamics, multi-level interactions, and emergent collective behaviors. Traditional analyses that focus solely on pairwise interactions often oversimplify these systems, neglecting the higher-order interactions critical for understanding their full collective dynamics. Recent advances in multivariate information theory provide a principled framework for quantifying these higher-order interactions, capturing key properties such as redundancy, synergy, shared randomness, and collective constraints. However, two major challenges persist: accurately estimating joint entropies and addressing the combinatorial explosion of interacting terms. To overcome these challenges, we introduce THOI (Torch-based High-Order Interactions), a novel, accessible, and efficient Python library for computing high-order interactions in continuous-valued systems. THOI leverages the well-established Gaussian copula method for joint entropy estimation, combined with state-of-the-art batch and parallel processing techniques to optimize performance across CPU, GPU, and TPU environments. Our results demonstrate that THOI significantly outperforms existing tools in terms of speed and scalability. Specifically, THOI reduces the time required to exhaustively analyze all interactions in small systems (≤ 30 variables). For larger systems, where exhaustive analysis is computationally impractical, THOI integrates optimization strategies that make higher-order interaction analysis feasible. We validate THOI’s accuracy using synthetic datasets with parametrically controlled interactions and further illustrate its utility by analyzing fMRI data from human subjects in wakeful resting states and under deep anesthesia. Finally, we analyzed over 900 real-world and synthetic datasets, establishing a comprehensive framework for applying higher-order interaction (HOI) analysis in complex systems. THOI opens new perspectives for testing both established and novel hypotheses about the multi-level, nonlinear, and multidimensional nature of complex systems.},
    number = {5},

}
```

**RIS**
```ris
TY  - JOUR
T1  - THOI: An efficient and accessible library for computing higher-order interactions enhanced by batch-processing
A1  - Belloli, Laouen
A1  - Mediano, Pedro A. M.
A1  - Cofré, Rodrigo
A1  - Slezak, Diego Fernandez
A1  - Herzog, Rubén
Y1  - 2026/05/11
N2  - Complex systems are characterized by nonlinear dynamics, multi-level interactions, and emergent collective behaviors. Traditional analyses that focus solely on pairwise interactions often oversimplify these systems, neglecting the higher-order interactions critical for understanding their full collective dynamics. Recent advances in multivariate information theory provide a principled framework for quantifying these higher-order interactions, capturing key properties such as redundancy, synergy, shared randomness, and collective constraints. However, two major challenges persist: accurately estimating joint entropies and addressing the combinatorial explosion of interacting terms. To overcome these challenges, we introduce THOI (Torch-based High-Order Interactions), a novel, accessible, and efficient Python library for computing high-order interactions in continuous-valued systems. THOI leverages the well-established Gaussian copula method for joint entropy estimation, combined with state-of-the-art batch and parallel processing techniques to optimize performance across CPU, GPU, and TPU environments. Our results demonstrate that THOI significantly outperforms existing tools in terms of speed and scalability. Specifically, THOI reduces the time required to exhaustively analyze all interactions in small systems (≤ 30 variables). For larger systems, where exhaustive analysis is computationally impractical, THOI integrates optimization strategies that make higher-order interaction analysis feasible. We validate THOI’s accuracy using synthetic datasets with parametrically controlled interactions and further illustrate its utility by analyzing fMRI data from human subjects in wakeful resting states and under deep anesthesia. Finally, we analyzed over 900 real-world and synthetic datasets, establishing a comprehensive framework for applying higher-order interaction (HOI) analysis in complex systems. THOI opens new perspectives for testing both established and novel hypotheses about the multi-level, nonlinear, and multidimensional nature of complex systems.
JF  - PLOS ONE
JA  - PLOS ONE
VL  - 21
IS  - 5
UR  - https://doi.org/10.1371/journal.pone.0348005
SP  - e0348005
EP  - 
PB  - Public Library of Science
M3  - doi:10.1371/journal.pone.0348005
ER  - 
```


## Authors

- [Laouen Mayal Louan Belloli](https://www.linkedin.com/in/laouen-belloli/)
- [Ruben Herzog](https://www.linkedin.com/in/rherzoga/)

For more details, visit the [GitHub repository](https://github.com/Laouen/THOI).
