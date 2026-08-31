"""
Utilities module for SPP metamaterial PINN implementation.

This module provides metrics, plotting tools, and analysis utilities
specifically designed for electromagnetic Physics-Informed Neural Networks.
"""

from .metrics import (
    BoundaryConditionMetrics,
    EnergyConservationMetrics,
    FieldAccuracyMetrics,
    MaxwellResidualMetrics,
    MetricsCollector,
    SPPPhysicsMetrics,
    TrainingMetrics,
)
from .plotting import (
    ComplexFieldVisualizer,
    DispersionPlotter,
    EMFieldPlotter,
    InteractivePlotter,
    SPPAnalysisPlotter,
    TrainingPlotter,
)

__all__ = [
    # Metrics
    'MaxwellResidualMetrics',
    'SPPPhysicsMetrics',
    'BoundaryConditionMetrics',
    'FieldAccuracyMetrics',
    'EnergyConservationMetrics',
    'TrainingMetrics',
    'MetricsCollector',

    # Plotting
    'EMFieldPlotter',
    'TrainingPlotter',
    'SPPAnalysisPlotter',
    'ComplexFieldVisualizer',
    'DispersionPlotter',
    'InteractivePlotter'
]

__version__ = '1.0.0'
