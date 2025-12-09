"""
Plane Wave Validation for PINN Fundamentals

Trains a PINN to discover a plane wave solution in free space using only
Maxwell's equations (no data supervision). This validates that the physics
loss implementation is correct.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# Physical constants
OMEGA = 2 * np.pi * 1e9  # Angular frequency: 1 GHz
C = 299792458.0  # Speed of light (m/s)
MU0 = 4e-7 * np.pi  # Permeability of free space (H/m)
EPS0 = 8.854187817e-12  # Permittivity of free space (F/m)
K0 = OMEGA / C  # Wave number (rad/m)
WAVELENGTH = 2 * np.pi / K0  # Wavelength (m)

# Training configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2048
N_EPOCHS = 10000
LEARNING_RATE = 1e-3
