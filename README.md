# Asian Options Pricing Library

A comprehensive Python library for pricing Asian call options using advanced Monte Carlo simulation techniques with variance reduction methods.

## Features

🚀 **Advanced Stochastic Models**
- Geometric Brownian Motion with jumps and regime switching
- Multi-regime volatility models with Markov chain transitions
- Customizable jump size distributions

📊 **Multiple Pricing Methods**
- Standard Monte Carlo simulation
- Control variates using European and geometric Asian options
- Antithetic variates for variance reduction
- Analytical solutions for geometric Asian options

⚡ **High Performance**
- Optimized NumPy-based computations
- Parallel simulation capabilities
- Efficient variance reduction techniques

📈 **Comprehensive Analysis Tools**
- Convergence analysis and visualization
- Sensitivity analysis for model parameters
- Method comparison and performance metrics
- Portfolio-level pricing capabilities

## Installation

```bash
pip install asian-options
```

Or install from source:

```bash
git clone https://github.com/asianoptions/asian-options-pricing.git
cd asian-options-pricing
pip install -e .
```

## Quick Start

```python
import numpy as np
from asian_options import AsianOptionPricer
from asian_options.models import MarketParameters

# Define market parameters
market_params = MarketParameters(
    spot_price=100.0,           # Current stock price
    risk_free_rate=0.05,        # 5% risk-free rate
    volatilities=[0.2, 0.3],    # Multiple volatility regimes
    jump_intensities=[0.1, 0.2], # Jump intensities
    time_to_maturity=1.0,       # 1 year to expiration
    initial_regime=0,           # Starting regime
    jump_size_params=(0.0, 0.1) # Jump size distribution
)

# Initialize pricer
pricer = AsianOptionPricer(market_params)

# Price an Asian call option
result = pricer.price_asian_call(
    strike=105.0,
    n_simulations=50000,
    method="control_variate_geometric",
    option_type="arithmetic"
)

print(f"Option Price: ${result.option_price:.4f}")
print(f"Standard Error: ${result.standard_error:.4f}")
print(f"95% Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
if result.variance_reduction_ratio:
    print(f"Variance Reduction: {result.variance_reduction_ratio:.2f}x")
```

## Available Pricing Methods

### 1. Standard Monte Carlo
```python
result = pricer.price_asian_call(strike=105, n_simulations=50000, method="monte_carlo")
```

### 2. Control Variate (European Option)
```python
result = pricer.price_asian_call(strike=105, n_simulations=50000, method="control_variate_european")
```

### 3. Control Variate (Geometric Asian)
```python
result = pricer.price_asian_call(strike=105, n_simulations=50000, method="control_variate_geometric")
```

### 4. Antithetic Variates
```python
result = pricer.price_asian_call(strike=105, n_simulations=25000, method="antithetic")
```

## Advanced Usage

### Regime Switching Models

```python
from asian_options import RegimeSwitchingModel

# Define transition matrix for regime switching
transition_matrix = np.array([
    [0.7, 0.3],  # Probabilities from regime 0 to regimes 0, 1
    [0.2, 0.8]   # Probabilities from regime 1 to regimes 0, 1
])

# Create regime switching model
regime_model = RegimeSwitchingModel(market_params, transition_matrix)

# Simulate paths with regime transitions
prices, regimes = regime_model.simulate_path_with_transitions(n_steps=252)
```

### Method Comparison

```python
# Compare all methods for the same option
comparison = pricer.compare_methods(
    strike=105.0,
    n_simulations=25000,
    option_type="arithmetic"
)

for method, result in comparison.items():
    print(f"{method}: ${result.option_price:.4f} ± ${result.standard_error:.4f}")
```

### Convergence Analysis

```python
# Analyze Monte Carlo convergence
sim_counts, prices = pricer.analyze_convergence(
    strike=105.0,
    max_simulations=50000,
    step_size=1000
)

import matplotlib.pyplot as plt
plt.plot(sim_counts, prices)
plt.xlabel('Number of Simulations')
plt.ylabel('Option Price')
plt.title('Monte Carlo Convergence')
plt.show()
```

### Sensitivity Analysis

```python
# Analyze sensitivity to spot price
spot_prices = np.linspace(80, 120, 10)
sensitivity_results = pricer.sensitivity_analysis(
    base_strike=105.0,
    n_simulations=10000,
    parameter_ranges={"spot_price": spot_prices}
)

plt.plot(spot_prices, sensitivity_results["spot_price"])
plt.xlabel('Spot Price')
plt.ylabel('Option Price')
plt.title('Sensitivity to Spot Price')
plt.show()
```

## Model Parameters

### MarketParameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `spot_price` | float | Current stock price |
| `risk_free_rate` | float | Risk-free interest rate |
| `volatilities` | List[float] | Volatility for each regime |
| `jump_intensities` | List[float] | Jump intensity for each regime |
| `time_to_maturity` | float | Time to option expiration |
| `initial_regime` | int | Starting regime index |
| `jump_size_params` | Tuple[float, float] | Jump size distribution (mean, std) |

## Variance Reduction Techniques

### Control Variates

The library implements two types of control variates:

1. **European Option Control Variate**: Uses the correlation between Asian and European options to reduce variance
2. **Geometric Asian Control Variate**: Uses the analytical solution for geometric Asian options

### Antithetic Variates

Generates pairs of negatively correlated paths using the same random numbers with opposite signs, reducing simulation variance while maintaining unbiased estimates.

## Performance Considerations

- Use `control_variate_geometric` for best accuracy-to-speed ratio
- `antithetic` method provides good variance reduction with minimal computational overhead
- For quick estimates, reduce `n_simulations` and use control variates
- For highest accuracy, use `monte_carlo` with large `n_simulations`

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_usage.py`: Introduction to library features
- `advanced_usage.py`: Regime switching models and portfolio pricing

Run examples:
```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## Mathematical Background

### Asian Options

Asian options are path-dependent derivatives where the payoff depends on the average price of the underlying asset over a specified period:

**Arithmetic Asian Call**: max(A - K, 0)
**Geometric Asian Call**: max(G - K, 0)

Where:
- A = (1/n) × Σ S(ti) (arithmetic average)
- G = (Π S(ti))^(1/n) (geometric average)
- K = strike price

### Stochastic Model

The underlying asset follows a jump-diffusion process with regime switching:

dS(t) = μ S(t) dt + σ(Xt) S(t) dW(t) + S(t-) ∫ (e^Y - 1) Ñ(dt, dy)

Where:
- Xt is the regime process
- σ(Xt) is regime-dependent volatility
- Ñ is a compensated Poisson random measure

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in academic research, please cite:

```bibtex
@software{asian_options_pricing,
  title={Asian Options Pricing Library},
  author={Asian Options Development Team},
  url={https://github.com/asianoptions/asian-options-pricing},
  version={1.0.0},
  year={2024}
}
```

## Support

- 📚 [Documentation](https://asian-options.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/asianoptions/asian-options-pricing/issues)
- 💬 [Discussions](https://github.com/asianoptions/asian-options-pricing/discussions)

---

**Disclaimer**: This library is for educational and research purposes. Always validate results and consult with financial professionals before making investment decisions.