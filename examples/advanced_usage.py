"""
Advanced usage example for the Asian Options Pricing Library.

This script demonstrates advanced features including regime switching models,
custom parameter configurations, and comprehensive method comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from asian_options import AsianOptionPricer, RegimeSwitchingModel
from asian_options.models import MarketParameters


def demonstrate_regime_switching():
    """Demonstrate regime switching model capabilities."""
    
    print("Regime Switching Model Demonstration")
    print("=" * 50)
    
    # Multi-regime market parameters
    market_params = MarketParameters(
        spot_price=100.0,
        risk_free_rate=0.05,
        volatilities=[0.15, 0.25, 0.35],  # Three volatility regimes
        jump_intensities=[0.05, 0.15, 0.25],  # Corresponding jump intensities
        time_to_maturity=1.0,
        initial_regime=0,
        jump_size_params=(-0.02, 0.08)  # Slightly negative mean jumps
    )
    
    # Define transition matrix for regime switching
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],  # From regime 0 to regimes 0, 1, 2
        [0.3, 0.5, 0.2],  # From regime 1 to regimes 0, 1, 2
        [0.1, 0.3, 0.6]   # From regime 2 to regimes 0, 1, 2
    ])
    
    # Create regime switching model
    regime_model = RegimeSwitchingModel(market_params, transition_matrix)
    
    # Simulate sample paths to show regime dynamics
    n_paths = 5
    n_steps = 252
    
    plt.figure(figsize=(15, 10))
    
    # Plot sample paths with regime coloring
    plt.subplot(2, 3, 1)
    colors = ['blue', 'orange', 'red']
    
    for i in range(n_paths):
        prices, regimes = regime_model.simulate_path_with_transitions(n_steps)
        time_grid = np.linspace(0, 1, len(prices))
        
        # Color code by regime
        for t in range(len(prices) - 1):
            regime = regimes[t]
            plt.plot([time_grid[t], time_grid[t+1]], 
                    [prices[t], prices[t+1]], 
                    color=colors[regime], alpha=0.7)
    
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price ($)')
    plt.title('Sample Paths with Regime Switching')
    plt.grid(True, alpha=0.3)
    
    # Legend for regimes
    for i, vol in enumerate(market_params.volatilities):
        plt.plot([], [], color=colors[i], label=f'Regime {i} (σ={vol:.0%})')
    plt.legend()
    
    # Compare different volatility scenarios
    strike_price = 105.0
    n_simulations = 20000
    
    scenarios = [
        ("Low Vol", MarketParameters(100.0, 0.05, [0.15], [0.05], 1.0, 0, (0.0, 0.05))),
        ("Medium Vol", MarketParameters(100.0, 0.05, [0.25], [0.15], 1.0, 0, (0.0, 0.08))),
        ("High Vol", MarketParameters(100.0, 0.05, [0.35], [0.25], 1.0, 0, (0.0, 0.12))),
        ("Regime Switch", market_params)
    ]
    
    scenario_results = []
    
    for scenario_name, params in scenarios:
        pricer = AsianOptionPricer(params)
        result = pricer.price_asian_call(
            strike=strike_price,
            n_simulations=n_simulations,
            method="monte_carlo"
        )
        
        scenario_results.append({
            'Scenario': scenario_name,
            'Option Price': result.option_price,
            'Std Error': result.standard_error,
            'CI Lower': result.confidence_interval[0],
            'CI Upper': result.confidence_interval[1]
        })
    
    # Plot scenario comparison
    plt.subplot(2, 3, 2)
    df = pd.DataFrame(scenario_results)
    
    bars = plt.bar(df['Scenario'], df['Option Price'], 
                   yerr=df['Std Error'], capsize=5, alpha=0.7)
    plt.xlabel('Scenario')
    plt.ylabel('Option Price ($)')
    plt.title('Price Comparison Across Scenarios')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Variance reduction comparison
    plt.subplot(2, 3, 3)
    pricer = AsianOptionPricer(market_params)
    comparison_results = pricer.compare_methods(strike_price, n_simulations)
    
    methods = list(comparison_results.keys())
    prices = [comparison_results[m].option_price for m in methods]
    errors = [comparison_results[m].standard_error for m in methods]
    
    bars = plt.bar(range(len(methods)), prices, yerr=errors, capsize=5, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Option Price ($)')
    plt.title('Variance Reduction Methods')
    plt.xticks(range(len(methods)), [m.replace('_', '\n') for m in methods], 
               rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Strike sensitivity analysis
    plt.subplot(2, 3, 4)
    strikes = np.linspace(90, 120, 15)
    strike_prices = []
    
    for strike in strikes:
        result = pricer.price_asian_call(
            strike=strike,
            n_simulations=10000,
            method="control_variate_geometric"
        )
        strike_prices.append(result.option_price)
    
    plt.plot(strikes, strike_prices, 'o-', linewidth=2)
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Option Price ($)')
    plt.title('Option Price vs Strike')
    plt.grid(True)
    
    # Time to maturity sensitivity
    plt.subplot(2, 3, 5)
    maturities = np.linspace(0.1, 2.0, 10)
    maturity_prices = []
    
    for T in maturities:
        temp_params = MarketParameters(
            spot_price=100.0,
            risk_free_rate=0.05,
            volatilities=[0.2, 0.3],
            jump_intensities=[0.1, 0.2],
            time_to_maturity=T,
            initial_regime=0,
            jump_size_params=(0.0, 0.1)
        )
        temp_pricer = AsianOptionPricer(temp_params)
        result = temp_pricer.price_asian_call(
            strike=strike_price,
            n_simulations=10000,
            method="monte_carlo"
        )
        maturity_prices.append(result.option_price)
    
    plt.plot(maturities, maturity_prices, 'o-', color='green', linewidth=2)
    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('Option Price ($)')
    plt.title('Option Price vs Maturity')
    plt.grid(True)
    
    # Convergence comparison between methods
    plt.subplot(2, 3, 6)
    max_sims = 15000
    step = 1000
    
    sim_counts, mc_prices = pricer.analyze_convergence(
        strike_price, max_sims, step, "arithmetic"
    )
    
    # Compare with control variate method convergence
    cv_prices = []
    for n_sim in range(step, max_sims + 1, step):
        result = pricer.price_asian_call(
            strike=strike_price,
            n_simulations=n_sim,
            method="control_variate_geometric"
        )
        cv_prices.append(result.option_price)
    
    plt.plot(sim_counts, mc_prices, 'o-', label='Monte Carlo', alpha=0.7)
    plt.plot(sim_counts, cv_prices, 's-', label='Control Variate', alpha=0.7)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price ($)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('advanced_asian_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nScenario Analysis Results:")
    print(pd.DataFrame(scenario_results).to_string(index=False, float_format='%.4f'))
    
    print("\nMethod Comparison Results:")
    comparison_df = pd.DataFrame([
        {
            'Method': method,
            'Price': result.option_price,
            'Std Error': result.standard_error,
            'Variance Reduction': result.variance_reduction_ratio or 1.0,
            'Comp Time (s)': result.computational_time or 0.0
        }
        for method, result in comparison_results.items()
    ])
    print(comparison_df.to_string(index=False, float_format='%.4f'))


def demonstrate_portfolio_pricing():
    """Demonstrate pricing multiple options in a portfolio."""
    
    print("\n" + "=" * 50)
    print("Portfolio Pricing Demonstration")
    print("=" * 50)
    
    # Common market parameters
    base_params = MarketParameters(
        spot_price=100.0,
        risk_free_rate=0.05,
        volatilities=[0.2, 0.3],
        jump_intensities=[0.1, 0.2],
        time_to_maturity=1.0,
        initial_regime=0,
        jump_size_params=(0.0, 0.1)
    )
    
    pricer = AsianOptionPricer(base_params)
    
    # Define a portfolio of Asian options
    portfolio = [
        {'strike': 95, 'quantity': 10, 'type': 'call'},
        {'strike': 100, 'quantity': -5, 'type': 'call'},  # Short position
        {'strike': 105, 'quantity': 10, 'type': 'call'},
        {'strike': 110, 'quantity': -5, 'type': 'call'},  # Short position
    ]
    
    portfolio_value = 0
    portfolio_details = []
    
    for option in portfolio:
        result = pricer.price_asian_call(
            strike=option['strike'],
            n_simulations=20000,
            method="control_variate_geometric"
        )
        
        position_value = option['quantity'] * result.option_price
        portfolio_value += position_value
        
        portfolio_details.append({
            'Strike': option['strike'],
            'Quantity': option['quantity'],
            'Option Price': result.option_price,
            'Position Value': position_value,
            'Position Type': 'Long' if option['quantity'] > 0 else 'Short'
        })
    
    print(f"\nPortfolio Analysis:")
    print(pd.DataFrame(portfolio_details).to_string(index=False, float_format='%.4f'))
    print(f"\nTotal Portfolio Value: ${portfolio_value:.4f}")


if __name__ == "__main__":
    demonstrate_regime_switching()
    demonstrate_portfolio_pricing()
    print("\nAdvanced analysis complete! Charts saved as 'advanced_asian_analysis.png'")