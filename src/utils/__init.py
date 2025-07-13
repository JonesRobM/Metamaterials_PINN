"""
Utilities module for SPP metamaterial PINN implementation.

This module provides metrics, plotting tools, and analysis utilities
specifically designed for electromagnetic Physics-Informed Neural Networks.
"""

from .metrics import (
    MaxwellResidualMetrics,
    SPPPhysicsMetrics,
    BoundaryConditionMetrics,
    FieldAccuracyMetrics,
    EnergyConservationMetrics,
    TrainingMetrics,
    MetricsCollector
)

from .plotting import (
    EMFieldPlotter,
    TrainingPlotter,
    SPPAnalysisPlotter,
    ComplexFieldVisualizer,
    DispersionPlotter,
    InteractivePlotter
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