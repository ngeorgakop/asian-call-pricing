"""
Stochastic models for underlying asset price dynamics.

This module contains implementations of various stochastic models used
for simulating asset price paths in Asian option pricing.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MarketParameters:
    """Container for market parameters."""
    spot_price: float
    risk_free_rate: float
    volatilities: List[float]
    jump_intensities: List[float]
    time_to_maturity: float
    initial_regime: int
    jump_size_params: Tuple[float, float]  # (a, b) for jump size distribution


class JumpDiffusionModel:
    """
    Geometric Brownian Motion with jumps and regime switching.
    
    Models asset price dynamics with:
    - Multiple volatility regimes
    - Poisson jump processes
    - Regime switching based on jump occurrences
    """
    
    def __init__(self, params: MarketParameters):
        self.params = params
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """Validate model parameters."""
        if len(self.params.volatilities) != len(self.params.jump_intensities):
            raise ValueError("Volatilities and jump intensities must have same length")
        
        if self.params.initial_regime >= len(self.params.volatilities):
            raise ValueError("Initial regime index out of bounds")
        
        if self.params.spot_price <= 0:
            raise ValueError("Spot price must be positive")
    
    def simulate_path(self, n_steps: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a single asset price path.
        
        Args:
            n_steps: Number of time steps
            random_seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (price_path, regime_path)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        dt = self.params.time_to_maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        prices = np.zeros(n_steps + 1)
        regimes = np.zeros(n_steps + 1, dtype=int)
        
        prices[0] = self.params.spot_price
        regimes[0] = self.params.initial_regime
        
        current_regime = self.params.initial_regime
        
        for i in range(n_steps):
            # Current regime parameters
            vol = self.params.volatilities[current_regime]
            lam = self.params.jump_intensities[current_regime]
            
            # Generate random components
            dW = np.random.normal(0, sqrt_dt)
            
            # Check for jumps (Poisson process)
            n_jumps = np.random.poisson(lam * dt)
            
            # Jump size if jump occurs
            jump_size = 0
            if n_jumps > 0:
                # Log-normal jump sizes
                a, b = self.params.jump_size_params
                jump_size = np.sum(np.random.normal(a, b, n_jumps))
                
                # Regime switching on jump
                current_regime = 1 - current_regime  # Simple 2-regime switching
            
            # GBM evolution with jumps
            drift = (self.params.risk_free_rate - 0.5 * vol**2) * dt
            diffusion = vol * dW
            
            log_return = drift + diffusion + jump_size
            prices[i + 1] = prices[i] * np.exp(log_return)
            regimes[i + 1] = current_regime
        
        return prices, regimes
    
    def simulate_multiple_paths(self, n_paths: int, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple asset price paths.
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path
            
        Returns:
            Tuple of (price_paths, regime_paths) with shape (n_paths, n_steps+1)
        """
        price_paths = np.zeros((n_paths, n_steps + 1))
        regime_paths = np.zeros((n_paths, n_steps + 1), dtype=int)
        
        for i in range(n_paths):
            prices, regimes = self.simulate_path(n_steps)
            price_paths[i] = prices
            regime_paths[i] = regimes
        
        return price_paths, regime_paths


class RegimeSwitchingModel:
    """
    Advanced regime switching model with transition probabilities.
    
    Extends the basic jump diffusion model with more sophisticated
    regime switching dynamics based on transition matrices.
    """
    
    def __init__(self, params: MarketParameters, transition_matrix: np.ndarray):
        self.params = params
        self.transition_matrix = transition_matrix
        self.n_regimes = len(params.volatilities)
        self.validate_transition_matrix()
    
    def validate_transition_matrix(self) -> None:
        """Validate transition matrix properties."""
        if self.transition_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("Transition matrix dimensions don't match number of regimes")
        
        # Check if rows sum to 1 (stochastic matrix)
        row_sums = np.sum(self.transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Transition matrix rows must sum to 1")
    
    def simulate_path_with_transitions(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate path with Markov chain regime switching.
        
        Args:
            n_steps: Number of time steps
            
        Returns:
            Tuple of (price_path, regime_path)
        """
        dt = self.params.time_to_maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        
        prices = np.zeros(n_steps + 1)
        regimes = np.zeros(n_steps + 1, dtype=int)
        
        prices[0] = self.params.spot_price
        regimes[0] = self.params.initial_regime
        
        current_regime = self.params.initial_regime
        
        for i in range(n_steps):
            # Regime transition
            transition_probs = self.transition_matrix[current_regime]
            current_regime = np.random.choice(self.n_regimes, p=transition_probs)
            
            # Current regime parameters
            vol = self.params.volatilities[current_regime]
            lam = self.params.jump_intensities[current_regime]
            
            # Generate random components
            dW = np.random.normal(0, sqrt_dt)
            
            # Jump component
            n_jumps = np.random.poisson(lam * dt)
            jump_size = 0
            if n_jumps > 0:
                a, b = self.params.jump_size_params
                jump_size = np.sum(np.random.normal(a, b, n_jumps))
            
            # Price evolution
            drift = (self.params.risk_free_rate - 0.5 * vol**2) * dt
            diffusion = vol * dW
            
            log_return = drift + diffusion + jump_size
            prices[i + 1] = prices[i] * np.exp(log_return)
            regimes[i + 1] = current_regime
        
        return prices, regimes