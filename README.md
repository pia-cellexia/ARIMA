# Causal Marketing Mix Model (MMM)

A Python implementation of a causal marketing mix model for predicting conversions based on ad budget, ROAS, and other marketing variables. This model includes advanced features like adstock transformation (carryover effects), saturation effects (diminishing returns), and causal inference capabilities.

## Features

- **Adstock Transformation**: Models the carryover effect of marketing spend over time
- **Saturation Effects**: Captures diminishing returns as budget increases
- **Causal Inference**: Estimate the true causal effect of budget changes on conversions
- **Multi-Channel Support**: Optimize budget allocation across multiple marketing channels
- **Hyperparameter Optimization**: Automatically tune adstock decay and saturation parameters
- **What-If Analysis**: Answer counterfactual questions about budget changes
- **Budget Optimization**: Find optimal allocation of budget across channels
- **ROAS-Constrained Optimization**: Find ideal budget to maximize conversions while achieving target ROAS (NEW!)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Single-Channel Example

```python
from marketing_mix_model import MarketingMixModel
from data_generator import MarketingDataGenerator
import pandas as pd

# Generate sample data
generator = MarketingDataGenerator(seed=42)
df = generator.generate_data(n_periods=365)

# Prepare features and target
X = df[['ad_budget', 'ROAS']]
y = df['conversions'].values

# Split train/test
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model
model = MarketingMixModel(
    adstock_decay=0.5,
    saturation_lambda=1.0,
    use_saturation=True,
    use_adstock=True
)
model.fit(X_train, y_train, optimize_params=True)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
metrics = model.score(X_test, y_test)
print(f"R²: {metrics['R2']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
```

### Causal Effect Estimation

```python
# What happens if we increase budget by $1000?
effect = model.estimate_causal_effect(
    X_test,
    treatment_col='ad_budget',
    treatment_change=1000
)

print(f"Mean effect on conversions: {effect['mean_effect']:.2f}")
print(f"Total effect: {effect['total_effect']:.2f}")
```

### Multi-Channel Budget Optimization

```python
# Generate multi-channel data
df_multi = generator.generate_multi_channel_data(n_periods=365)

# Prepare data
budget_cols = ['search_budget', 'social_budget', 'display_budget']
X = df_multi[budget_cols + ['ROAS']]
y = df_multi['conversions'].values

# Train model
model.fit(X, y, optimize_params=True)

# Optimize budget allocation
optimal = model.optimize_budget_allocation(
    total_budget=10000,
    baseline_X=X.mean().to_frame().T,
    budget_columns=budget_cols
)

print("Optimal allocation:", optimal)
```

## Understanding the Model

### Adstock Effect

The adstock transformation captures the idea that marketing has a lasting impact beyond the immediate period:

```
x_adstock[t] = x[t] + decay × x_adstock[t-1]
```

Where `decay` (0-1) controls how long the effect lasts. Higher values mean longer carryover effects.

### Saturation Effect

The saturation curve models diminishing returns:

```
f(x) = x / (λ + x)
```

Where `λ` controls the rate of saturation. As spending increases, each additional dollar produces fewer conversions.

### Why This Matters

Traditional linear models assume:
1. Each dollar spent has the same effect (no diminishing returns)
2. Effects happen immediately (no carryover)

This causal MMM corrects for both, giving you more accurate predictions and better budget decisions.

## Use Cases

### 1. **Predicting Conversions**
Given historical ad budget and ROAS data, predict future conversions.

### 2. **What-If Analysis**
"What happens to conversions if we increase our search budget by 20%?"

### 3. **Budget Optimization**
"How should we split $50,000 across search, social, and display to maximize conversions?"

### 4. **Channel Effectiveness**
"Which marketing channel gives us the best return on investment?"

### 5. **Causal Impact**
"What is the true causal effect of our marketing spend on conversions?"

### 6. **ROAS-Constrained Budget Optimization** (NEW!)
"What's the ideal budget to maximize conversions while achieving a target ROAS of 3.5?"

**Quick Example:**
```python
from roas_optimizer import ROASConstrainedOptimizer

# Initialize optimizer
optimizer = ROASConstrainedOptimizer(model, conversion_value=100.0)

# Find optimal budget for ROAS = 3.5
result = optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_test,
    budget_column='ad_budget',
    budget_range=(1000, 20000)
)

print(f"Optimal Budget: ${result['optimal_budget']:,.2f}")
print(f"Expected Conversions: {result['predicted_conversions']:.0f}")
print(f"Expected ROAS: {result['predicted_roas']:.2f}")
```

**See full guide**: [ROAS_OPTIMIZATION_GUIDE.md](ROAS_OPTIMIZATION_GUIDE.md)

## API Reference

### MarketingMixModel

#### Constructor
```python
MarketingMixModel(
    adstock_decay=0.5,          # Decay rate (0-1)
    saturation_lambda=1.0,       # Saturation parameter
    use_saturation=True,         # Apply saturation transform
    use_adstock=True             # Apply adstock transform
)
```

#### Methods

**fit(X, y, optimize_params=False)**
- Fit the model to data
- `X`: DataFrame with features (ad_budget, ROAS, etc.)
- `y`: Target variable (conversions)
- `optimize_params`: Auto-tune hyperparameters

**predict(X)**
- Predict conversions for new data
- Returns: numpy array of predictions

**score(X, y)**
- Calculate model performance metrics
- Returns: dict with R², RMSE, MAE, MAPE

**estimate_causal_effect(X, treatment_col, treatment_change)**
- Estimate causal effect of changing a variable
- `treatment_col`: Column to change (e.g., 'ad_budget')
- `treatment_change`: Amount to change it by
- Returns: dict with mean_effect, total_effect, etc.

**optimize_budget_allocation(total_budget, baseline_X, budget_columns, constraints=None)**
- Find optimal budget allocation
- `total_budget`: Total budget to allocate
- `baseline_X`: Baseline feature values
- `budget_columns`: List of budget column names
- `constraints`: Optional min/max per channel
- Returns: dict with optimal allocation

**get_feature_importance()**
- Get feature importance scores
- Returns: DataFrame with features and coefficients

## Examples

### Standard Examples

Run the comprehensive examples:

```bash
python example_usage.py
```

This will:
1. Generate synthetic data
2. Train single-channel and multi-channel models
3. Make predictions and evaluate performance
4. Estimate causal effects
5. Optimize budget allocation
6. Generate visualization plots

### ROAS-Constrained Optimization (NEW!)

Find optimal budget to maximize conversions while achieving target ROAS:

```bash
python example_roas_optimization.py
```

This will:
1. Train a marketing mix model on your data
2. Find the ideal budget for target ROAS = 3.5
3. Generate budget recommendations for multiple ROAS targets
4. Create comprehensive visualizations
5. Provide actionable recommendations

**See the complete guide**: [ROAS_OPTIMIZATION_GUIDE.md](ROAS_OPTIMIZATION_GUIDE.md)

## Data Requirements

Your data should have:
- **ad_budget**: Marketing spend (the variable you control)
- **conversions**: Number of conversions (target to predict)
- **ROAS**: Return on ad spend (optional but helpful)
- Time series format with regular intervals (daily, weekly, etc.)

Example data format:

```csv
date,ad_budget,conversions,ROAS
2023-01-01,5000,120,2.4
2023-01-02,5500,135,2.5
2023-01-03,4800,115,2.4
...
```

## Generate Sample Data

Use the included data generator:

```bash
python data_generator.py
```

This creates:
- `marketing_data_single_channel.csv`: Single-channel data
- `marketing_data_multi_channel.csv`: Multi-channel data

## Model Assumptions

1. **Stationarity**: The relationship between budget and conversions is stable over time
2. **No Confounders**: Other factors affecting conversions are either controlled or uncorrelated with budget
3. **Correct Functional Form**: Adstock and saturation transformations capture the true dynamics

## Advanced Usage

### Custom Adstock Decay by Channel

```python
# For multi-channel data, you might want different decay rates
# This requires modifying the model or training separate models per channel
```

### Bayesian Inference

The package includes a `BayesianMarketingMixModel` class (currently a placeholder) for full Bayesian inference with uncertainty quantification.

## Troubleshooting

**Model performance is poor:**
- Try `optimize_params=True` to auto-tune hyperparameters
- Check for data quality issues (missing values, outliers)
- Ensure sufficient data (at least 100+ observations)
- Consider adding more features (seasonality, competitors, etc.)

**Predictions are too smooth:**
- The model captures long-term trends, not day-to-day noise
- This is expected behavior for a causal model
- Add more features if you need to capture short-term variations

**Optimization fails:**
- Check that your constraints are feasible
- Ensure baseline_X has reasonable values
- Try different initial values for adstock_decay and saturation_lambda

## Contributing

Contributions welcome! Areas for improvement:
- Full Bayesian implementation with PyMC
- Prophet-style seasonality decomposition
- Hierarchical models for multiple markets
- Advanced constraint handling for optimization
- Time-varying coefficients

## License

MIT License

## Citation

If you use this model in your research or business, please cite:

```
Causal Marketing Mix Model
https://github.com/yourusername/ARIMA
```

## References

1. Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.
2. Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.
3. Robyn: Facebook's Open Source Marketing Mix Modeling

## Contact

For questions or support, please open an issue on GitHub.
