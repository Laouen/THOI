# THOI: Torch - Higher Order Interactions

## Description

THOI is a Python package designed to compute O information in Higher Order Interactions using batch processing. This package leverages PyTorch for efficient tensor operations.

## Installation

### Prerequisites

Ensure you have Python 3.6 or higher installed.

### Installing THOI with the CPU Version of PyTorch

For users who do not require GPU support, you can install THOI with the CPU version of PyTorch by running the following command:

```bash
pip install thoi[cpu]
```

### Installing THOI with Other Versions of PyTorch

For users who need GPU support or a specific version of PyTorch, you need to install PyTorch separately before installing THOI. Follow these steps:

1. Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate options for your system.
2. Install PyTorch using the provided command. For example, for GPU support with CUDA 11.1:

    ```bash
    pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```

3. Once PyTorch is installed, install THOI:

    ```bash
    pip install thoi
    ```

## Usage

After installation, you can start using THOI in your projects. Here is a simple example:

```python
from thoi.measures.gaussian_copula import 
import numpy as np

X = np.random.normal(0,1, (1000, 10))

measures = multi_order_measures(X)
```

For detailed usage and examples, please refer to the [documentation](https://github.com/Laouen/THOI).

## Contributing

We welcome contributions from the community. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- [Laouen Mayal Louan Belloli](https://www.linkedin.com/in/laouen-belloli/)
- [Ruben Herzog](https://www.linkedin.com/in/rherzoga/)

For more details, visit the [GitHub repository](https://github.com/Laouen/THOI).