"""
Factor Analysis Module

Provides tools for analyzing portfolio factor exposures and return attribution.
"""

from .fama_french import FamaFrenchAnalyzer, get_factor_data
from .attribution import SectorAttribution, BrinsonAttribution
from .style import StyleFactorAnalyzer, calculate_momentum, calculate_value_score
from .decomposition import FactorRiskDecomposition, calculate_factor_risk

__all__ = [
    'FamaFrenchAnalyzer',
    'get_factor_data',
    'SectorAttribution',
    'BrinsonAttribution',
    'StyleFactorAnalyzer',
    'calculate_momentum',
    'calculate_value_score',
    'FactorRiskDecomposition',
    'calculate_factor_risk'
]
