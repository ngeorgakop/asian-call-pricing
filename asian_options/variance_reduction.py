"""
Variance reduction techniques for Monte Carlo simulation.

This module implements control variates and other variance reduction
methods to improve the efficiency of Asian option pricing.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from .models import JumpDiffusionModel
from .utils import BlackScholesFormulas


@dataclass
class ControlVariateResults:
    """Results from control variate Monte Carlo simulation."""
    option_price: float
    variance_reduction_ratio: float
    optimal_beta: float
    standard_error: float
    control_variate_price: float


class ControlVariates:
    """
    Control variate methods for variance reduction in Asian option pricing.
    
    Uses European options and geometric Asian options as control variates
    to reduce the variance of arithmetic Asian option price estimates.
    """
    
    def __init__(self, model: JumpDiffusionModel, n_time_steps: int = 252):
        self.model = model
        self.n_time_steps = n_time_steps
        self.dt = model.params.time_to_maturity / n_time_steps
        self.bs_formulas = BlackScholesFormulas()
    
    def price_with_european_control(
        self,
        strike: float,
        n_simulations: int,
        beta: Optional[float] = None
    ) -> ControlVariateResults:
        """
        Price arithmetic Asian call using European call as control variate.
        
        Args:
            strike: Option strike price
            n_simulations: Number of Monte Carlo simulations
            beta: Control variate coefficient (if None, estimated optimally)
            
        Returns:
            ControlVariateResults with pricing and variance reduction info
        """
        # Calculate theoretical European call price
        vol_avg = np.mean(self.model.params.volatilities)  # Simplified
        european_price = self.bs_formulas.european_call_price(
            S=self.model.params.spot_price,
            K=strike,
            T=self.model.params.time_to_maturity,
            r=self.model.params.risk_free_rate,
            sigma=vol_avg
        )
        
        # Monte Carlo simulation
        asian_payoffs = np.zeros(n_simulations)
        european_payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            price_path, _ = self.model.simulate_path(self.n_time_steps)
            
            # Asian option payoff
            asian_avg = np.mean(price_path[1:])
            asian_payoffs[i] = max(asian_avg - strike, 0)
            
            # European option payoff (terminal price)
            european_payoffs[i] = max(price_path[-1] - strike, 0)
        
        # Discount factors
        discount_factor = np.exp(-self.model.params.risk_free_rate * 
                                self.model.params.time_to_maturity)
        
        discounted_asian = discount_factor * asian_payoffs
        discounted_european = discount_factor * european_payoffs
        
        # Estimate optimal beta if not provided
        if beta is None:
            covariance = np.cov(discounted_asian, discounted_european)[0, 1]
            variance_european = np.var(discounted_european)
            beta = covariance / variance_european if variance_european > 0 else 0
        
        # Apply control variate
        controlled_payoffs = (discounted_asian - 
                             beta * (discounted_european - european_price))
        
        # Calculate results
        option_price = np.mean(controlled_payoffs)
        variance_original = np.var(discounted_asian)
        variance_controlled = np.var(controlled_payoffs)
        variance_reduction = variance_original / variance_controlled if variance_controlled > 0 else 1
        
        standard_error = np.sqrt(variance_controlled / n_simulations)
        
        return ControlVariateResults(
            option_price=option_price,
            variance_reduction_ratio=variance_reduction,
            optimal_beta=beta,
            standard_error=standard_error,
            control_variate_price=european_price
        )
    
    def price_with_geometric_control(
        self,
        strike: float,
        n_simulations: int,
        beta: Optional[float] = None
    ) -> ControlVariateResults:
        """
        Price arithmetic Asian call using geometric Asian call as control variate.
        
        Args:
            strike: Option strike price
            n_simulations: Number of Monte Carlo simulations
            beta: Control variate coefficient (if None, estimated optimally)
            
        Returns:
            ControlVariateResults with pricing and variance reduction info
        """
        # Calculate theoretical geometric Asian call price
        vol_avg = np.mean(self.model.params.volatilities)
        geometric_price = self.bs_formulas.geometric_asian_call_price(
            S=self.model.params.spot_price,
            K=strike,
            T=self.model.params.time_to_maturity,
            r=self.model.params.risk_free_rate,
            sigma=vol_avg,
            n_steps=self.n_time_steps
        )
        
        # Monte Carlo simulation
        arithmetic_payoffs = np.zeros(n_simulations)
        geometric_payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            price_path, _ = self.model.simulate_path(self.n_time_steps)
            path_without_initial = price_path[1:]
            
            # Arithmetic Asian payoff
            arithmetic_avg = np.mean(path_without_initial)
            arithmetic_payoffs[i] = max(arithmetic_avg - strike, 0)
            
            # Geometric Asian payoff
            geometric_avg = np.exp(np.mean(np.log(path_without_initial)))
            geometric_payoffs[i] = max(geometric_avg - strike, 0)
        
        # Discount payoffs
        discount_factor = np.exp(-self.model.params.risk_free_rate * 
                                self.model.params.time_to_maturity)
        
        discounted_arithmetic = discount_factor * arithmetic_payoffs
        discounted_geometric = discount_factor * geometric_payoffs
        
        # Estimate optimal beta if not provided
        if beta is None:
            covariance = np.cov(discounted_arithmetic, discounted_geometric)[0, 1]
            variance_geometric = np.var(discounted_geometric)
            beta = covariance / variance_geometric if variance_geometric > 0 else 0
        
        # Apply control variate
        controlled_payoffs = (discounted_arithmetic - 
                             beta * (discounted_geometric - geometric_price))
        
        # Calculate results
        option_price = np.mean(controlled_payoffs)
        variance_original = np.var(discounted_arithmetic)
        variance_controlled = np.var(controlled_payoffs)
        variance_reduction = variance_original / variance_controlled if variance_controlled > 0 else 1
        
        standard_error = np.sqrt(variance_controlled / n_simulations)
        
        return ControlVariateResults(
            option_price=option_price,
            variance_reduction_ratio=variance_reduction,
            optimal_beta=beta,
            standard_error=standard_error,
            control_variate_price=geometric_price
        )


class AntitheticVariates:
    """
    Antithetic variates implementation for variance reduction.
    
    Generates negatively correlated random paths to reduce simulation variance
    while maintaining unbiased estimates.
    """
    
    def __init__(self, model: JumpDiffusionModel, n_time_steps: int = 252):
        self.model = model
        self.n_time_steps = n_time_steps
        self.dt = model.params.time_to_maturity / n_time_steps
    
    def generate_antithetic_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a pair of antithetic asset price paths.
        
        Returns:
            Tuple of (path_1, path_2) where path_2 uses negated random variables
        """
        # Generate random numbers for first path
        random_normals = np.random.standard_normal(self.n_time_steps)
        
        # Simulate both paths
        path_1 = self._simulate_path_with_randoms(random_normals)
        path_2 = self._simulate_path_with_randoms(-random_normals)  # Antithetic
        
        return path_1, path_2
    
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
        
        # Use average volatility for simplified antithetic implementation
        vol = np.mean(self.model.params.volatilities)
        sqrt_dt = np.sqrt(self.dt)
        
        for i in range(self.n_time_steps):
            # Geometric Brownian Motion component
            drift = (self.model.params.risk_free_rate - 0.5 * vol**2) * self.dt
            diffusion = vol * sqrt_dt * random_normals[i]
            
            # Simplified jump component (optional, can be made more sophisticated)
            lam_avg = np.mean(self.model.params.jump_intensities)
            n_jumps = np.random.poisson(lam_avg * self.dt)
            jump_size = 0
            if n_jumps > 0:
                a, b = self.model.params.jump_size_params
                jump_size = np.sum(np.random.normal(a, b, n_jumps))
            
            log_return = drift + diffusion + jump_size
            prices[i + 1] = prices[i] * np.exp(log_return)
        
        return prices
    
    def price_asian_call_antithetic(
        self,
        strike: float,
        n_pairs: int,
        option_type: str = "arithmetic"
    ) -> Tuple[float, float, float]:
        """
        Price Asian call option using antithetic variates.
        
        Args:
            strike: Option strike price
            n_pairs: Number of antithetic pairs to simulate
            option_type: "arithmetic" or "geometric" averaging
            
        Returns:
            Tuple of (option_price, standard_error, variance_reduction_ratio)
        """
        antithetic_payoffs = np.zeros(n_pairs)
        regular_payoffs = np.zeros(n_pairs * 2)  # For comparison
        
        for i in range(n_pairs):
            # Generate antithetic pair
            path_1, path_2 = self.generate_antithetic_paths()
            
            # Calculate averages
            if option_type == "arithmetic":
                avg_1 = np.mean(path_1[1:])
                avg_2 = np.mean(path_2[1:])
            else:  # geometric
                avg_1 = np.exp(np.mean(np.log(path_1[1:])))
                avg_2 = np.exp(np.mean(np.log(path_2[1:])))
            
            # Calculate payoffs
            payoff_1 = max(avg_1 - strike, 0)
            payoff_2 = max(avg_2 - strike, 0)
            
            # Antithetic estimate (average of pair)
            antithetic_payoffs[i] = (payoff_1 + payoff_2) / 2
            
            # Store individual payoffs for variance comparison
            regular_payoffs[2*i] = payoff_1
            regular_payoffs[2*i + 1] = payoff_2
        
        # Discount to present value
        discount_factor = np.exp(-self.model.params.risk_free_rate * 
                                self.model.params.time_to_maturity)
        
        # Calculate option price
        option_price = discount_factor * np.mean(antithetic_payoffs)
        
        # Calculate variance reduction
        variance_antithetic = np.var(antithetic_payoffs)
        variance_regular = np.var(regular_payoffs[:n_pairs])  # Compare with same sample size
        variance_reduction = variance_regular / variance_antithetic if variance_antithetic > 0 else 1
        
        # Standard error
        standard_error = discount_factor * np.sqrt(variance_antithetic / n_pairs)
        
        return option_price, standard_error, variance_reduction