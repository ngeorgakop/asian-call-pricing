"""
Basic usage example for the Asian Options Pricing Library.

This script demonstrates how to use the library to price Asian call options
using various methods including Monte Carlo simulation and variance reduction techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from asian_options import AsianOptionPricer
from asian_options.models import MarketParameters


def main():
    """Demonstrate basic usage of the Asian options pricing library."""
    
    # Define market parameters
    market_params = MarketParameters(
        spot_price=100.0,           # Current stock price
        risk_free_rate=0.05,        # 5% risk-free rate
        volatilities=[0.2, 0.3],    # 20% and 30% volatility regimes
        jump_intensities=[0.1, 0.2], # Jump intensities for each regime
        time_to_maturity=1.0,       # 1 year to maturity
        initial_regime=0,           # Start in regime 0
        jump_size_params=(0.0, 0.1) # Jump size parameters (mean, std)
    )
    
    # Initialize the pricer
    pricer = AsianOptionPricer(market_params, n_time_steps=252)
    
    # Option parameters
    strike_price = 105.0
    n_simulations = 50000
    
    print("Asian Call Option Pricing Example")
    print("=" * 50)
    print(f"Spot Price: ${market_params.spot_price}")
    print(f"Strike Price: ${strike_price}")
    print(f"Time to Maturity: {market_params.time_to_maturity} years")
    print(f"Risk-free Rate: {market_params.risk_free_rate:.1%}")
    print(f"Volatility Regimes: {[f'{v:.1%}' for v in market_params.volatilities]}")
    print(f"Simulations: {n_simulations:,}")
    print()
    
    # Price using different methods
    methods = [
        ("monte_carlo", "Standard Monte Carlo"),
        ("control_variate_european", "Control Variate (European)"),
        ("control_variate_geometric", "Control Variate (Geometric Asian)"),
        ("antithetic", "Antithetic Variates")
    ]
    
    results = {}
    
    for method_key, method_name in methods:
        print(f"Pricing with {method_name}...")
        
        try:
            result = pricer.price_asian_call(
                strike=strike_price,
                n_simulations=n_simulations,
                method=method_key,
                option_type="arithmetic"
            )
            
            results[method_key] = result
            
            print(f"  Option Price: ${result.option_price:.4f}")
            print(f"  Standard Error: ${result.standard_error:.4f}")
            print(f"  95% CI: [${result.confidence_interval[0]:.4f}, ${result.confidence_interval[1]:.4f}]")
            
            if result.variance_reduction_ratio:
                print(f"  Variance Reduction: {result.variance_reduction_ratio:.2f}x")
            
            if result.computational_time:
                print(f"  Computation Time: {result.computational_time:.2f} seconds")
            
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    # Compare with analytical benchmark (geometric Asian)
    analytical_price = pricer.get_analytical_benchmark(strike_price)
    if analytical_price:
        print(f"Analytical Geometric Asian Price: ${analytical_price:.4f}")
        print()
    
    # Demonstrate convergence analysis
    print("Convergence Analysis")
    print("-" * 20)
    
    sim_counts, prices = pricer.analyze_convergence(
        strike=strike_price,
        max_simulations=20000,
        step_size=1000
    )
    
    # Plot convergence
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(sim_counts, prices)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price ($)')
    plt.title('Monte Carlo Convergence')
    plt.grid(True)
    
    # Plot method comparison
    plt.subplot(2, 2, 2)
    method_names = []
    option_prices = []
    standard_errors = []
    
    for method_key, method_name in methods:
        if method_key in results:
            method_names.append(method_name.replace(" ", "\n"))
            option_prices.append(results[method_key].option_price)
            standard_errors.append(results[method_key].standard_error)
    
    bars = plt.bar(range(len(method_names)), option_prices, yerr=standard_errors, 
                   capsize=5, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Option Price ($)')
    plt.title('Method Comparison')
    plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Sensitivity analysis
    plt.subplot(2, 2, 3)
    spot_prices = np.linspace(80, 120, 10)
    sensitivity_results = pricer.sensitivity_analysis(
        base_strike=strike_price,
        n_simulations=10000,
        parameter_ranges={"spot_price": spot_prices},
        method="monte_carlo"
    )
    
    plt.plot(spot_prices, sensitivity_results["spot_price"], 'o-')
    plt.xlabel('Spot Price ($)')
    plt.ylabel('Option Price ($)')
    plt.title('Sensitivity to Spot Price')
    plt.grid(True)
    
    # Volatility sensitivity
    plt.subplot(2, 2, 4)
    volatilities = np.linspace(0.1, 0.5, 10)
    vol_sensitivity = pricer.sensitivity_analysis(
        base_strike=strike_price,
        n_simulations=10000,
        parameter_ranges={"volatility": volatilities},
        method="monte_carlo"
    )
    
    plt.plot(volatilities, vol_sensitivity["volatility"], 'o-', color='red')
    plt.xlabel('Volatility')
    plt.ylabel('Option Price ($)')
    plt.title('Sensitivity to Volatility')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('asian_option_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis complete! Charts saved as 'asian_option_analysis.png'")


if __name__ == "__main__":
    main()