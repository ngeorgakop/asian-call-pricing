"""
Monte Carlo simulation engine for Asian option pricing.

This module provides efficient Monte Carlo simulation capabilities
with various variance reduction techniques for pricing Asian options.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from .models import JumpDiffusionModel, MarketParameters


@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation results."""
    option_price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    convergence_path: np.ndarray
    n_simulations: int


class MonteCarloEngine:
    """
    High-performance Monte Carlo engine for Asian option pricing.
    
    Supports various simulation modes including standard Monte Carlo,
    antithetic variates, and control variate methods.
    """
    
    def __init__(self, model: JumpDiffusionModel, n_time_steps: int = 252):
        self.model = model
        self.n_time_steps = n_time_steps
        self.dt = model.params.time_to_maturity / n_time_steps
    
    def price_asian_call_vanilla(
        self,
        strike: float,
        n_simulations: int,
        option_type: str = "arithmetic",
        confidence_level: float = 0.95
    ) -> SimulationResults:
        """
        Price Asian call option using standard Monte Carlo.
        
        Args:
            strike: Option strike price
            n_simulations: Number of Monte Carlo simulations
            option_type: "arithmetic" or "geometric" averaging
            confidence_level: Confidence level for interval estimation
            
        Returns:
            SimulationResults with pricing information
        """
        payoffs = np.zeros(n_simulations)
        convergence_path = []
        
        for i in range(n_simulations):
            # Simulate asset price path
            price_path, _ = self.model.simulate_path(self.n_time_steps)
            
            # Calculate average price
            if option_type == "arithmetic":
                avg_price = np.mean(price_path[1:])  # Exclude initial price
            elif option_type == "geometric":
                avg_price = self._geometric_mean(price_path[1:])
            else:
                raise ValueError("option_type must be 'arithmetic' or 'geometric'")
            
            # Calculate payoff
            payoffs[i] = max(avg_price - strike, 0)
            
            # Track convergence every 1000 simulations
            if (i + 1) % 1000 == 0:
                current_price = np.exp(-self.model.params.risk_free_rate * 
                                     self.model.params.time_to_maturity) * np.mean(payoffs[:i+1])
                convergence_path.append(current_price)
        
        # Discount payoffs to present value
        discount_factor = np.exp(-self.model.params.risk_free_rate * 
                                self.model.params.time_to_maturity)
        discounted_payoffs = discount_factor * payoffs
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # Confidence interval
        z_score = self._get_z_score(confidence_level)
        ci_lower = option_price - z_score * std_error
        ci_upper = option_price + z_score * std_error
        
        return SimulationResults(
            option_price=option_price,
            standard_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            convergence_path=np.array(convergence_path),
            n_simulations=n_simulations
        )
    
    def price_asian_call_antithetic(
        self,
        strike: float,
        n_simulations: int,
        option_type: str = "arithmetic"
    ) -> SimulationResults:
        """
        Price Asian call option using antithetic variates for variance reduction.
        
        Args:
            strike: Option strike price
            n_simulations: Number of simulation pairs (total paths = 2 * n_simulations)
            option_type: "arithmetic" or "geometric" averaging
            
        Returns:
            SimulationResults with pricing information
        """
        payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Generate correlated paths using antithetic variates
            payoff_1, payoff_2 = self._simulate_antithetic_pair(strike, option_type)
            payoffs[i] = (payoff_1 + payoff_2) / 2
        
        # Discount to present value
        discount_factor = np.exp(-self.model.params.risk_free_rate * 
                                self.model.params.time_to_maturity)
        discounted_payoffs = discount_factor * payoffs
        
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return SimulationResults(
            option_price=option_price,
            standard_error=std_error,
            confidence_interval=(0, 0),  # Simplified for this implementation
            convergence_path=np.array([]),
            n_simulations=n_simulations * 2  # Total number of paths
        )
    
    def _simulate_antithetic_pair(self, strike: float, option_type: str) -> Tuple[float, float]:
        """
        Simulate a pair of antithetic paths for variance reduction.
        
        Args:
            strike: Option strike price
            option_type: "arithmetic" or "geometric"
            
        Returns:
            Tuple of (payoff_1, payoff_2) for the antithetic pair
        """
        # Generate standard normal random variables
        Z = np.random.standard_normal(self.n_time_steps)
        
        # Simulate first path
        price_path_1 = self._simulate_path_with_randoms(Z)
        
        # Simulate antithetic path (negated random variables)
        price_path_2 = self._simulate_path_with_randoms(-Z)
        
        # Calculate averages and payoffs
        if option_type == "arithmetic":
            avg_1 = np.mean(price_path_1[1:])
            avg_2 = np.mean(price_path_2[1:])
        else:  # geometric
            avg_1 = self._geometric_mean(price_path_1[1:])
            avg_2 = self._geometric_mean(price_path_2[1:])
        
        payoff_1 = max(avg_1 - strike, 0)
        payoff_2 = max(avg_2 - strike, 0)
        
        return payoff_1, payoff_2
    
    def _simulate_path_with_randoms(self, random_normals: np.ndarray) -> np.ndarray:
        """
        Simulate asset price path using provided random numbers.
        
        Args:
            random_normals: Array of standard normal random variables
            
        Returns:
            Asset price path
        """
        prices = np.zeros(self.n_time_steps + 1)
        prices[0] = self.model.params.spot_price
        
        # Simplified simulation without regime switching for antithetic variates
        vol = self.model.params.volatilities[0]  # Use first regime volatility
        sqrt_dt = np.sqrt(self.dt)
        
        for i in range(self.n_time_steps):
            drift = (self.model.params.risk_free_rate - 0.5 * vol**2) * self.dt
            diffusion = vol * sqrt_dt * random_normals[i]
            
            # Add simplified jump component
            lam = self.model.params.jump_intensities[0]
            n_jumps = np.random.poisson(lam * self.dt)
            jump_size = 0
            if n_jumps > 0:
                a, b = self.model.params.jump_size_params
                jump_size = np.sum(np.random.normal(a, b, n_jumps))
            
            log_return = drift + diffusion + jump_size
            prices[i + 1] = prices[i] * np.exp(log_return)
        
        return prices
    
    def _geometric_mean(self, prices: np.ndarray) -> float:
        """Calculate geometric mean of prices."""
        return np.exp(np.mean(np.log(prices)))
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level."""
        from scipy.stats import norm
        return norm.ppf((1 + confidence_level) / 2)
    
    def simulate_convergence_analysis(
        self,
        strike: float,
        max_simulations: int,
        step_size: int = 1000,
        option_type: str = "arithmetic"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze Monte Carlo convergence behavior.
        
        Args:
            strike: Option strike price
            max_simulations: Maximum number of simulations
            step_size: Step size for convergence tracking
            option_type: "arithmetic" or "geometric"
            
        Returns:
            Tuple of (simulation_counts, option_prices)
        """
        simulation_counts = np.arange(step_size, max_simulations + 1, step_size)
        option_prices = []
        
        cumulative_payoffs = []
        discount_factor = np.exp(-self.model.params.risk_free_rate * 
                                self.model.params.time_to_maturity)
        
        for i in range(max_simulations):
            # Simulate one path
            price_path, _ = self.model.simulate_path(self.n_time_steps)
            
            if option_type == "arithmetic":
                avg_price = np.mean(price_path[1:])
            else:
                avg_price = self._geometric_mean(price_path[1:])
            
            payoff = max(avg_price - strike, 0)
            cumulative_payoffs.append(payoff)
            
            # Record price at intervals
            if (i + 1) % step_size == 0:
                current_price = discount_factor * np.mean(cumulative_payoffs)
                option_prices.append(current_price)
        
        return simulation_counts, np.array(option_prices)