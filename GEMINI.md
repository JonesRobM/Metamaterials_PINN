# GEMINI.md

## Project Overview

This project implements a Physics-Informed Neural Network (PINN) to model and simulate Surface Plasmon Polaritons (SPPs) on metamaterial interfaces. The core of the project is a PyTorch-based neural network that learns to solve Maxwell's equations for electromagnetic wave propagation in these complex materials. The project aims to provide a fast and accurate alternative to traditional numerical simulation methods.

The project is structured as a Python application with the following key components:

*   **`src/`**: Contains the core source code for the project, including:
    *   **`physics/`**: Modules defining the physical laws (Maxwell's equations) and material properties.
    *   **`models/`**: The PINN architecture, loss functions, and other model-related components.
    *   **`data/`**: Data sampling and processing utilities.
    *   **`utils/`**: Helper functions for plotting, metrics, etc.
*   **`scripts/`**: Contains scripts for training and evaluating the models.
*   **`config/`**: Configuration files for the model, training, and material parameters.
*   **`notebooks/`**: Jupyter notebooks for analysis and visualization.
*   **`tests/`**: Unit and integration tests for the project.

## Building and Running

### 1. Installation

The project's dependencies are managed by the `pyproject.toml` file. To install the project and its dependencies, use pip:

```bash
# For users:
pip install .

# For developers (editable install):
pip install -e .
```

### 2. Training

The main training script is `scripts/train.py`. It can be run from the command line:

```bash
python scripts/train.py
```

The training process can be configured via command-line arguments and YAML configuration files in the `config/` directory.

**Key training script arguments:**

*   `--config`: Path to a custom configuration file.
*   `--resume`: Path to a checkpoint file to resume training from.
*   `--device`: The device to use for training (e.g., `cpu` or `cuda`).
*   `--debug`: Enable debug mode for a short training run.

### 3. Testing

The project includes a test suite in the `tests/` directory. The tests can be run using `pytest`:

```bash
pytest
```

## Development Conventions

*   **Code Style:** The code follows the PEP 8 style guide.
*   **Typing:** The code uses type hints for improved readability and static analysis.
*   **Testing:** The project has a dedicated `tests/` directory and uses `pytest` for testing.
*   **Configuration:** The project uses YAML files for configuration, which are loaded by the training script.
*   **Logging:** The training script uses the `logging` module and TensorBoard for logging and monitoring the training process.
*   **Modularity:** The code is organized into modules with clear responsibilities, following the structure outlined in the `README.md` file.
