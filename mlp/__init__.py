"""
MLP Neural Network Package for NBA Team Selection

This package provides:
- MultilayerPerceptron: Base MLP implementation with forward/backprop
- TeamSelectionMLP: Specialized MLP for team selection
- NBADataProcessor: Data processing and feature engineering
- TeamBuilder: Team composition and selection logic
"""

from .neural_network import (
    MultilayerPerceptron,
    TeamSelectionMLP,
    ActivationFunctions,
    CostFunctions
)

from .data_processing import (
    NBADataProcessor,
    TeamBuilder,
    TeamComposition,
    load_sample_data,
    load_nba_data
)

__all__ = [
    'MultilayerPerceptron',
    'TeamSelectionMLP',
    'ActivationFunctions',
    'CostFunctions',
    'NBADataProcessor',
    'TeamBuilder',
    'TeamComposition',
    'load_sample_data',
    'load_nba_data'
]

__version__ = '1.0.0'
