"""
Utility functions and analytical formulas for option pricing.

This module contains Black-Scholes formulas and other analytical
solutions used in Asian option pricing and as control variates.
"""

import numpy as np
from typing import Optional
from scipy.stats import norm


class BlackScholesFormulas:
    """
    Collection of analytical Black-Scholes formulas for various option types.
    
    Provides exact solutions for European options and geometric Asian options
    that are used as control variates in Monte Carlo simulations.
    """
    
    @staticmethod
    def european_call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European call option price using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            European call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    
    @staticmethod
    def european_put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate European put option price using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            European put option price
        """
        call_price = BlackScholesFormulas.european_call_price(S, K, T, r, sigma)
        # Put-call parity: P = C - S + K * exp(-r * T)
        put_price = call_price - S + K * np.exp(-r * T)
        return max(put_price, 0)
    
    @staticmethod
    def geometric_asian_call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int
    ) -> float:
        """
        Calculate geometric Asian call option price analytically.
        
        The geometric Asian option has a closed-form solution that can be
        expressed in terms of the Black-Scholes formula with adjusted parameters.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            n_steps: Number of averaging points
            
        Returns:
            Geometric Asian call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        # Adjusted parameters for geometric Asian option
        dt = T / n_steps
        
        # Adjusted volatility
        sigma_adj = sigma * np.sqrt((n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2))
        
        # Adjusted risk-free rate
        r_adj = 0.5 * (r - 0.5 * sigma**2 + sigma_adj**2)
        
        # Adjusted spot price (accounts for continuous averaging)
        S_adj = S * np.exp((r_adj - r) * T)
        
        # Use Black-Scholes formula with adjusted parameters
        return BlackScholesFormulas.european_call_price(S_adj, K, T, r_adj, sigma_adj)
    
    @staticmethod
    def geometric_asian_put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int
    ) -> float:
        """
        Calculate geometric Asian put option price analytically.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            n_steps: Number of averaging points
            
        Returns:
            Geometric Asian put option price
        """
        call_price = BlackScholesFormulas.geometric_asian_call_price(
            S, K, T, r, sigma, n_steps
        )
        
        # Adjusted parameters for put-call parity
        sigma_adj = sigma * np.sqrt((n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2))
        r_adj = 0.5 * (r - 0.5 * sigma**2 + sigma_adj**2)
        S_adj = S * np.exp((r_adj - r) * T)
        
        # Put-call parity with adjusted parameters
        put_price = call_price - S_adj + K * np.exp(-r_adj * T)
        return max(put_price, 0)
    
    @staticmethod
    def black_scholes_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> float:
        """
        Calculate Black-Scholes delta (price sensitivity to underlying).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            
        Returns:
            Option delta
        """
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:  # put
                return -1.0 if S < K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == "call":
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    @staticmethod
    def black_scholes_gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Black-Scholes gamma (delta sensitivity to underlying).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Option gamma
        """
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def black_scholes_vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate Black-Scholes vega (price sensitivity to volatility).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Option vega
        """
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return S * norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def black_scholes_theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> float:
        """
        Calculate Black-Scholes theta (price sensitivity to time).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            
        Returns:
            Option theta (negative value indicates time decay)
        """
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            return term1 + term2
        else:  # put
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            return term1 + term2


class NumericalMethods:
    """
    Numerical methods and utilities for option pricing.
    
    Contains helper functions for numerical integration, optimization,
    and other computational tasks in option pricing.
    """
    
    @staticmethod
    def confidence_interval(
        data: np.ndarray,
        confidence_level: float = 0.95
    ) -> tuple:
        """
        Calculate confidence interval for sample data.
        
        Args:
            data: Sample data array
            confidence_level: Confidence level (default 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_error = np.std(data, ddof=1) / np.sqrt(n)
        
        # Use t-distribution for small samples, normal for large samples
        if n < 30:
            from scipy.stats import t
            t_value = t.ppf((1 + confidence_level) / 2, df=n-1)
        else:
            t_value = norm.ppf((1 + confidence_level) / 2)
        
        margin_error = t_value * std_error
        return mean - margin_error, mean + margin_error
    
    @staticmethod
    def relative_error(true_value: float, estimated_value: float) -> float:
        """
        Calculate relative error between true and estimated values.
        
        Args:
            true_value: True/reference value
            estimated_value: Estimated value
            
        Returns:
            Relative error as percentage
        """
        if true_value == 0:
            return float('inf') if estimated_value != 0 else 0.0
        
        return abs(estimated_value - true_value) / abs(true_value) * 100
    
    @staticmethod
    def variance_reduction_efficiency(
        var_original: float,
        var_reduced: float,
        computational_cost_ratio: float = 1.0
    ) -> float:
        """
        Calculate efficiency of variance reduction technique.
        
        Args:
            var_original: Variance of original estimator
            var_reduced: Variance of variance-reduced estimator
            computational_cost_ratio: Ratio of computational costs
            
        Returns:
            Efficiency ratio (higher is better)
        """
        if var_reduced <= 0:
            return float('inf')
        
        variance_ratio = var_original / var_reduced
        return variance_ratio / computational_cost_ratio
    
    @staticmethod
    def optimal_sample_size(
        target_error: float,
        estimated_variance: float,
        confidence_level: float = 0.95
    ) -> int:
        """
        Estimate optimal sample size for desired accuracy.
        
        Args:
            target_error: Desired standard error
            estimated_variance: Estimated variance of estimator
            confidence_level: Confidence level
            
        Returns:
            Recommended sample size
        """
        z_score = norm.ppf((1 + confidence_level) / 2)
        required_n = (z_score * np.sqrt(estimated_variance) / target_error) ** 2
        return int(np.ceil(required_n))