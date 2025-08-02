"""
Main Asian option pricing interface.

This module provides a high-level interface for pricing Asian options
using various methods including Monte Carlo simulation with variance
reduction techniques.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .models import JumpDiffusionModel, MarketParameters
from .monte_carlo import MonteCarloEngine, SimulationResults
from .variance_reduction import ControlVariates, AntitheticVariates, ControlVariateResults
from .utils import BlackScholesFormulas, NumericalMethods


@dataclass
class PricingResults:
    """Comprehensive results from Asian option pricing."""
    option_price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    method_used: str
    parameters: Dict[str, Any]
    convergence_data: Optional[np.ndarray] = None
    variance_reduction_ratio: Optional[float] = None
    computational_time: Optional[float] = None


class AsianOptionPricer:
    """
    Comprehensive Asian option pricing engine.
    
    Provides a unified interface for pricing Asian options using various
    methods including standard Monte Carlo, control variates, and
    antithetic variates for variance reduction.
    """
    
    def __init__(self, market_params: MarketParameters, n_time_steps: int = 252):
        """
        Initialize the Asian option pricer.
        
        Args:
            market_params: Market parameters for the underlying model
            n_time_steps: Number of time steps for simulation (default: 252)
        """
        self.market_params = market_params
        self.n_time_steps = n_time_steps
        
        # Initialize model and engines
        self.model = JumpDiffusionModel(market_params)
        self.mc_engine = MonteCarloEngine(self.model, n_time_steps)
        self.control_variates = ControlVariates(self.model, n_time_steps)
        self.antithetic_variates = AntitheticVariates(self.model, n_time_steps)
        
        # Initialize analytical formulas
        self.bs_formulas = BlackScholesFormulas()
        self.numerical_methods = NumericalMethods()
    
    def price_asian_call(
        self,
        strike: float,
        n_simulations: int,
        method: str = "monte_carlo",
        option_type: str = "arithmetic",
        confidence_level: float = 0.95,
        **kwargs
    ) -> PricingResults:
        """
        Price Asian call option using specified method.
        
        Args:
            strike: Option strike price
            n_simulations: Number of Monte Carlo simulations
            method: Pricing method ("monte_carlo", "control_variate_european", 
                   "control_variate_geometric", "antithetic")
            option_type: "arithmetic" or "geometric" averaging
            confidence_level: Confidence level for error estimation
            **kwargs: Additional method-specific parameters
            
        Returns:
            PricingResults with comprehensive pricing information
        """
        import time
        start_time = time.time()
        
        if method == "monte_carlo":
            result = self._price_monte_carlo(strike, n_simulations, option_type, confidence_level)
        elif method == "control_variate_european":
            result = self._price_control_variate_european(strike, n_simulations, **kwargs)
        elif method == "control_variate_geometric":
            result = self._price_control_variate_geometric(strike, n_simulations, **kwargs)
        elif method == "antithetic":
            result = self._price_antithetic(strike, n_simulations, option_type)
        else:
            raise ValueError(f"Unknown pricing method: {method}")
        
        computational_time = time.time() - start_time
        result.computational_time = computational_time
        
        return result
    
    def _price_monte_carlo(
        self,
        strike: float,
        n_simulations: int,
        option_type: str,
        confidence_level: float
    ) -> PricingResults:
        """Price using standard Monte Carlo simulation."""
        sim_results = self.mc_engine.price_asian_call_vanilla(
            strike, n_simulations, option_type, confidence_level
        )
        
        return PricingResults(
            option_price=sim_results.option_price,
            standard_error=sim_results.standard_error,
            confidence_interval=sim_results.confidence_interval,
            method_used="Standard Monte Carlo",
            parameters={
                "strike": strike,
                "n_simulations": n_simulations,
                "option_type": option_type,
                "n_time_steps": self.n_time_steps
            },
            convergence_data=sim_results.convergence_path
        )
    
    def _price_control_variate_european(
        self,
        strike: float,
        n_simulations: int,
        **kwargs
    ) -> PricingResults:
        """Price using European option as control variate."""
        beta = kwargs.get("beta", None)
        
        cv_results = self.control_variates.price_with_european_control(
            strike, n_simulations, beta
        )
        
        # Calculate confidence interval
        ci_lower = cv_results.option_price - 1.96 * cv_results.standard_error
        ci_upper = cv_results.option_price + 1.96 * cv_results.standard_error
        
        return PricingResults(
            option_price=cv_results.option_price,
            standard_error=cv_results.standard_error,
            confidence_interval=(ci_lower, ci_upper),
            method_used="Control Variate (European)",
            parameters={
                "strike": strike,
                "n_simulations": n_simulations,
                "beta": cv_results.optimal_beta,
                "control_price": cv_results.control_variate_price
            },
            variance_reduction_ratio=cv_results.variance_reduction_ratio
        )
    
    def _price_control_variate_geometric(
        self,
        strike: float,
        n_simulations: int,
        **kwargs
    ) -> PricingResults:
        """Price using geometric Asian option as control variate."""
        beta = kwargs.get("beta", None)
        
        cv_results = self.control_variates.price_with_geometric_control(
            strike, n_simulations, beta
        )
        
        # Calculate confidence interval
        ci_lower = cv_results.option_price - 1.96 * cv_results.standard_error
        ci_upper = cv_results.option_price + 1.96 * cv_results.standard_error
        
        return PricingResults(
            option_price=cv_results.option_price,
            standard_error=cv_results.standard_error,
            confidence_interval=(ci_lower, ci_upper),
            method_used="Control Variate (Geometric Asian)",
            parameters={
                "strike": strike,
                "n_simulations": n_simulations,
                "beta": cv_results.optimal_beta,
                "control_price": cv_results.control_variate_price
            },
            variance_reduction_ratio=cv_results.variance_reduction_ratio
        )
    
    def _price_antithetic(
        self,
        strike: float,
        n_simulations: int,
        option_type: str
    ) -> PricingResults:
        """Price using antithetic variates."""
        option_price, std_error, variance_reduction = (
            self.antithetic_variates.price_asian_call_antithetic(
                strike, n_simulations, option_type
            )
        )
        
        # Calculate confidence interval
        ci_lower = option_price - 1.96 * std_error
        ci_upper = option_price + 1.96 * std_error
        
        return PricingResults(
            option_price=option_price,
            standard_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            method_used="Antithetic Variates",
            parameters={
                "strike": strike,
                "n_pairs": n_simulations,
                "total_paths": n_simulations * 2,
                "option_type": option_type
            },
            variance_reduction_ratio=variance_reduction
        )
    
    def compare_methods(
        self,
        strike: float,
        n_simulations: int,
        option_type: str = "arithmetic"
    ) -> Dict[str, PricingResults]:
        """
        Compare all pricing methods for the same option.
        
        Args:
            strike: Option strike price
            n_simulations: Number of simulations for each method
            option_type: "arithmetic" or "geometric" averaging
            
        Returns:
            Dictionary mapping method names to PricingResults
        """
        methods = [
            "monte_carlo",
            "control_variate_european",
            "control_variate_geometric",
            "antithetic"
        ]
        
        results = {}
        
        for method in methods:
            try:
                results[method] = self.price_asian_call(
                    strike, n_simulations, method, option_type
                )
            except Exception as e:
                print(f"Error in method {method}: {e}")
                continue
        
        return results
    
    def analyze_convergence(
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
            step_size: Step size for convergence analysis
            option_type: "arithmetic" or "geometric" averaging
            
        Returns:
            Tuple of (simulation_counts, option_prices)
        """
        return self.mc_engine.simulate_convergence_analysis(
            strike, max_simulations, step_size, option_type
        )
    
    def sensitivity_analysis(
        self,
        base_strike: float,
        n_simulations: int,
        parameter_ranges: Dict[str, np.ndarray],
        method: str = "monte_carlo"
    ) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis by varying model parameters.
        
        Args:
            base_strike: Base strike price
            n_simulations: Number of simulations per calculation
            parameter_ranges: Dictionary mapping parameter names to value arrays
            method: Pricing method to use
            
        Returns:
            Dictionary mapping parameter names to option price arrays
        """
        results = {}
        
        for param_name, param_values in parameter_ranges.items():
            prices = []
            
            for value in param_values:
                # Create modified parameters
                modified_params = self._modify_parameter(param_name, value)
                
                # Create temporary pricer with modified parameters
                temp_pricer = AsianOptionPricer(modified_params, self.n_time_steps)
                
                # Price the option
                result = temp_pricer.price_asian_call(
                    base_strike, n_simulations, method
                )
                prices.append(result.option_price)
            
            results[param_name] = np.array(prices)
        
        return results
    
    def _modify_parameter(self, param_name: str, value: float) -> MarketParameters:
        """Create modified market parameters for sensitivity analysis."""
        params_dict = {
            "spot_price": self.market_params.spot_price,
            "risk_free_rate": self.market_params.risk_free_rate,
            "volatilities": self.market_params.volatilities.copy(),
            "jump_intensities": self.market_params.jump_intensities.copy(),
            "time_to_maturity": self.market_params.time_to_maturity,
            "initial_regime": self.market_params.initial_regime,
            "jump_size_params": self.market_params.jump_size_params
        }
        
        if param_name == "spot_price":
            params_dict["spot_price"] = value
        elif param_name == "risk_free_rate":
            params_dict["risk_free_rate"] = value
        elif param_name == "volatility":
            params_dict["volatilities"] = [value] * len(params_dict["volatilities"])
        elif param_name == "time_to_maturity":
            params_dict["time_to_maturity"] = value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        return MarketParameters(**params_dict)
    
    def get_analytical_benchmark(self, strike: float) -> Optional[float]:
        """
        Get analytical price for geometric Asian option (when applicable).
        
        Args:
            strike: Option strike price
            
        Returns:
            Analytical price if available, None otherwise
        """
        try:
            # Use average volatility for simplified analytical solution
            avg_vol = np.mean(self.market_params.volatilities)
            
            return self.bs_formulas.geometric_asian_call_price(
                S=self.market_params.spot_price,
                K=strike,
                T=self.market_params.time_to_maturity,
                r=self.market_params.risk_free_rate,
                sigma=avg_vol,
                n_steps=self.n_time_steps
            )
        except Exception:
            return None