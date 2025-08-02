"""
Asian Options Pricing Library

A comprehensive Python library for pricing Asian call options using various 
numerical methods including Monte Carlo simulation, control variates, and 
antithetic variance reduction techniques.

This library supports:
- Geometric Brownian Motion with jumps and regime switching
- Monte Carlo simulations with variance reduction techniques
- Control variates using European and geometric Asian options
- Advanced stochastic processes for realistic option pricing
"""

from .pricing import AsianOptionPricer
from .models import JumpDiffusionModel, RegimeSwitchingModel
from .monte_carlo import MonteCarloEngine
from .variance_reduction import ControlVariates, AntitheticVariates
from .utils import BlackScholesFormulas

__version__ = "1.0.0"
__author__ = "Asian Options Pricing Team"

__all__ = [
    "AsianOptionPricer",
    "JumpDiffusionModel", 
    "RegimeSwitchingModel",
    "MonteCarloEngine",
    "ControlVariates",
    "AntitheticVariates", 
    "BlackScholesFormulas"
]