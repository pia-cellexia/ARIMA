# ROAS-Constrained Marketing Budget Optimization Guide

## Overview

This guide explains how to use the **ROAS-Constrained Optimizer** to find the optimal marketing budget that maximizes conversions while achieving a target ROAS (Return on Ad Spend).

## Use Case

You have:
- ✅ **Complete control** over ad budget
- ✅ **Historical data** on ad budget, conversions, and ROAS
- ✅ **A target NC ROAS** (e.g., 3.5) that you need to achieve

You want to:
- 📈 **Maximize conversions** (purchases)
- 🎯 **Achieve a specific ROAS target** (e.g., 3.5 means $3.50 revenue per $1 spent)
- 💰 **Find the ideal budget** for your campaign

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

### 2. Prepare Your Data

Your data should have:
- `ad_budget`: Your marketing spend (the variable you control)
- `conversions`: Number of purchases/conversions (target to predict)
- `ROAS`: Return on ad spend (optional but helpful)

Example data format:

```csv
date,ad_budget,conversions,ROAS
2024-01-01,5000,120,2.4
2024-01-02,5500,135,2.5
2024-01-03,4800,115,2.4
...
```

### 3. Run the Complete Example

```bash
python example_roas_optimization.py
```

This will:
- Generate sample data (or use your own CSV)
- Train a marketing mix model
- Find optimal budget for ROAS = 3.5
- Generate visualizations and recommendations

---

## Step-by-Step Tutorial

### Step 1: Load and Prepare Data

```python
import pandas as pd
from marketing_mix_model import MarketingMixModel
from roas_optimizer import ROASConstrainedOptimizer

# Load your data
df = pd.read_csv('your_marketing_data.csv')

# Split train/test
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Prepare features
X_train = train_df[['ad_budget', 'ROAS']]
y_train = train_df['conversions'].values
X_test = test_df[['ad_budget', 'ROAS']]
y_test = test_df['conversions'].values
```

### Step 2: Train Marketing Mix Model

```python
# Initialize model with adstock and saturation effects
model = MarketingMixModel(
    adstock_decay=0.5,        # Carryover effect (0-1)
    saturation_lambda=1.0,    # Diminishing returns parameter
    use_saturation=True,      # Enable saturation
    use_adstock=True          # Enable adstock
)

# Train with hyperparameter optimization
model.fit(X_train, y_train, optimize_params=True)

# Evaluate
metrics = model.score(X_test, y_test)
print(f"Model R²: {metrics['R2']:.4f}")
print(f"RMSE: {metrics['RMSE']:.2f}")
```

### Step 3: Initialize ROAS Optimizer

```python
# Set your conversion value (revenue per conversion)
conversion_value = 100.0  # e.g., $100 per purchase

optimizer = ROASConstrainedOptimizer(
    model=model,
    conversion_value=conversion_value
)
```

### Step 4: Find Optimal Budget for Target ROAS

```python
# Find budget that maximizes conversions with ROAS = 3.5
result = optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_test,
    budget_column='ad_budget',
    budget_range=(1000, 20000),  # Search range
    tolerance=0.1                # ROAS tolerance (±0.1)
)

print(f"\n🎯 OPTIMAL BUDGET: ${result['optimal_budget']:,.2f}")
print(f"📊 Predicted Conversions: {result['predicted_conversions']:.0f}")
print(f"💵 Predicted ROAS: {result['predicted_roas']:.2f}")
print(f"💰 Expected Revenue: ${result['revenue']:,.2f}")
print(f"📈 Expected Profit: ${result['profit']:,.2f}")
```

### Step 5: Visualize Optimization Landscape

```python
optimizer.plot_optimization_landscape(
    baseline_X=X_test,
    budget_column='ad_budget',
    budget_range=(1000, 20000),
    target_roas=3.5,
    optimal_budget=result['optimal_budget'],
    save_path='optimization_landscape.png'
)
```

This generates a comprehensive visualization showing:
- **Budget vs Conversions** (with saturation effect)
- **Budget vs ROAS** (diminishing returns)
- **Budget vs Profit**
- **Efficiency Frontier** (conversions vs ROAS)

### Step 6: Get Recommendations for Multiple ROAS Targets

```python
# Compare different ROAS targets
recommendations = optimizer.get_budget_recommendations(
    baseline_X=X_test,
    budget_column='ad_budget',
    budget_range=(1000, 20000),
    roas_targets=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
)

print(recommendations)
```

### Step 7: Analyze Trade-offs

```python
# Analyze budget-ROAS-conversions trade-offs
tradeoff_df = optimizer.analyze_budget_roas_tradeoff(
    baseline_X=X_test,
    budget_column='ad_budget',
    budget_range=(1000, 20000),
    n_points=50
)

# View the trade-off data
print(tradeoff_df[['budget', 'conversions', 'roas', 'profit']].head(10))
```

---

## Understanding the Results

### Optimal Budget

The optimizer finds the **minimum budget** that:
1. Achieves your target ROAS (e.g., 3.5 ± tolerance)
2. **Maximizes conversions** given that constraint

### Example Result

```
Target ROAS: 3.5
Optimal Budget: $5,753.04
Predicted Conversions: 196
Predicted ROAS: 3.40
Expected Revenue: $19,561.35
Expected Profit: $13,808.31
```

**Interpretation:**
- Spend **$5,753** to get **196 conversions**
- Each $1 spent returns **$3.40** in revenue (meets your 3.5 target within tolerance)
- Total revenue: **$19,561** (196 × $100)
- Profit: **$13,808** (revenue - budget)

### Why Not Higher Budget?

You might wonder: "If I spend $10,000, I get 269 conversions and $16,912 profit. Why not do that?"

**Answer:** Because your ROAS would drop to **2.69**, which is below your target of 3.5.

The optimizer ensures you **meet your ROAS constraint** while still maximizing conversions.

---

## Multi-Channel Optimization

If you have multiple marketing channels (search, social, display), you can optimize budget allocation across all channels:

```python
from roas_optimizer import optimize_multi_channel_with_roas

# Prepare multi-channel data
budget_cols = ['search_budget', 'social_budget', 'display_budget']
X = df[budget_cols + ['ROAS']]
y = df['conversions'].values

# Train model
model.fit(X, y, optimize_params=True)

# Optimize allocation
optimal_allocation = optimize_multi_channel_with_roas(
    model=model,
    total_budget=10000,
    target_roas=3.5,
    baseline_X=X,
    budget_columns=budget_cols,
    conversion_value=100.0,
    tolerance=0.1
)

print("\n📊 Optimal Budget Allocation:")
for channel, budget in optimal_allocation.items():
    pct = (budget / 10000) * 100
    print(f"  {channel}: ${budget:,.2f} ({pct:.1f}%)")
```

**Example Output:**
```
Optimal Budget Allocation:
  search_budget: $4,200.00 (42.0%)
  social_budget: $3,500.00 (35.0%)
  display_budget: $2,300.00 (23.0%)
```

---

## Advanced Features

### 1. Find Budget for Conversion Target

If you have a conversion target instead of ROAS target:

```python
result = optimizer.find_budget_for_conversion_target(
    target_conversions=200,
    baseline_X=X_test,
    budget_column='ad_budget',
    budget_range=(1000, 20000)
)

print(f"Budget needed for 200 conversions: ${result['optimal_budget']:,.2f}")
```

### 2. What-If Scenarios

Test different budget scenarios:

```python
scenarios = [
    ("Conservative", 3000),
    ("Moderate", 6000),
    ("Aggressive", 10000)
]

for name, budget in scenarios:
    X_scenario = X_test.mean().to_frame().T
    X_scenario['ad_budget'] = budget

    conversions = model.predict(X_scenario)[0]
    revenue = conversions * conversion_value
    roas = revenue / budget

    print(f"\n{name} Budget (${budget}):")
    print(f"  Conversions: {conversions:.0f}")
    print(f"  ROAS: {roas:.2f}")
    print(f"  Revenue: ${revenue:,.2f}")
```

### 3. Custom Conversion Value

Adjust `conversion_value` based on your business:

```python
# E-commerce: Average order value
conversion_value = 75.0

# SaaS: Customer lifetime value
conversion_value = 500.0

# Lead generation: Value per lead
conversion_value = 25.0
```

---

## Key Parameters

### ROASConstrainedOptimizer

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `model` | Trained MarketingMixModel | Required | `model` |
| `conversion_value` | Revenue per conversion ($) | Required | `100.0` |

### optimize_for_target_roas()

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `target_roas` | Target ROAS to achieve | Required | `3.5` |
| `baseline_X` | Baseline feature values | Required | `X_test` |
| `budget_column` | Name of budget column | `'ad_budget'` | `'ad_budget'` |
| `budget_range` | (min, max) budget search range | `(100, 50000)` | `(1000, 20000)` |
| `tolerance` | ROAS tolerance (±) | `0.1` | `0.1` |

---

## Troubleshooting

### "Optimization not successful"

**Problem:** Optimizer can't find a feasible solution.

**Solutions:**
1. **Widen budget_range**: Try `(500, 30000)` instead of `(1000, 20000)`
2. **Increase tolerance**: Try `tolerance=0.2` instead of `0.1`
3. **Lower target_roas**: Your target might be too high for the budget range
4. **Check model quality**: Ensure R² > 0.7

### Predictions seem unrealistic

**Problem:** Predicted conversions are too high/low.

**Solutions:**
1. **Check conversion_value**: Make sure it matches your actual revenue per conversion
2. **Verify data quality**: Remove outliers, check for missing values
3. **Train with more data**: Ensure 100+ samples for reliable predictions
4. **Re-optimize hyperparameters**: Use `optimize_params=True`

### ROAS constraint not met

**Problem:** Predicted ROAS is below target.

**Solutions:**
1. **Increase tolerance**: The optimizer allows `target_roas ± tolerance`
2. **Check if target is feasible**: Some ROAS targets may be impossible given your data
3. **Review budget_range**: Make sure it includes feasible budgets

---

## Best Practices

### 1. Data Quality

- ✅ Use at least **180 days** of historical data (1 year is better)
- ✅ Ensure consistent time intervals (daily or weekly)
- ✅ Remove extreme outliers that distort the model
- ✅ Check for missing values and handle appropriately

### 2. Model Validation

- ✅ Always use **train/test split** (80/20)
- ✅ Check **R² > 0.7** for reliable predictions
- ✅ Monitor **MAPE < 10%** for good accuracy
- ✅ Use **optimize_params=True** for better results

### 3. Business Context

- ✅ Set realistic **conversion_value** based on your business
- ✅ Choose **target_roas** based on your profitability goals
- ✅ Consider **seasonality** when interpreting results
- ✅ Monitor actual performance and **adjust dynamically**

### 4. Optimization

- ✅ Start with a **wide budget_range** then narrow down
- ✅ Use **tolerance=0.1** for strict ROAS adherence
- ✅ Test **multiple ROAS targets** to understand trade-offs
- ✅ Compare with **current performance** to validate recommendations

---

## Real-World Example

**Scenario:** E-commerce company selling products with $85 average order value

```python
# 1. Load historical data
df = pd.read_csv('marketing_data.csv')  # 365 days of data

# 2. Prepare data
X_train = df[:292][['ad_budget', 'ROAS']]
y_train = df[:292]['conversions'].values
X_test = df[292:][['ad_budget', 'ROAS']]

# 3. Train model
model = MarketingMixModel(use_saturation=True, use_adstock=True)
model.fit(X_train, y_train, optimize_params=True)
# Result: R² = 0.85, RMSE = 6.2

# 4. Initialize optimizer
optimizer = ROASConstrainedOptimizer(model, conversion_value=85.0)

# 5. Find optimal budget for ROAS = 3.5
result = optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_test,
    budget_range=(2000, 15000)
)

# 6. Results
"""
Optimal Budget: $5,200.00
Predicted Conversions: 168
Predicted ROAS: 3.47
Expected Revenue: $14,280.00
Expected Profit: $9,080.00

Current Budget: $7,500.00
Current Conversions: 195
Current ROAS: 2.21
Current Profit: $8,987.50

Recommendation: REDUCE budget by $2,300 (-31%)
Why? You'll sacrifice 27 conversions BUT increase ROAS from 2.21 to 3.47,
maintaining similar profit ($9,080 vs $8,988) with better efficiency.
"""
```

---

## API Reference

### ROASConstrainedOptimizer

#### Methods

**`optimize_for_target_roas(...)`**
Find optimal budget for target ROAS.

**`find_budget_for_conversion_target(...)`**
Find minimum budget for conversion target.

**`analyze_budget_roas_tradeoff(...)`**
Analyze trade-offs between budget, conversions, and ROAS.

**`plot_optimization_landscape(...)`**
Visualize optimization landscape.

**`get_budget_recommendations(...)`**
Get recommendations for multiple ROAS targets.

### optimize_multi_channel_with_roas(...)

Optimize budget allocation across multiple channels with ROAS constraint.

---

## Visualizations Generated

### 1. Optimization Landscape (4 plots)

- **Budget vs Conversions**: Shows saturation effect
- **Budget vs ROAS**: Shows diminishing returns
- **Budget vs Profit**: Shows profit maximization point
- **Efficiency Frontier**: Shows conversions vs ROAS colored by budget

### 2. Trade-off Analysis (2 plots)

- **Budget vs ROAS with target line**: Shows where budget meets ROAS target
- **Conversions vs ROAS**: Shows efficiency frontier with optimal point

---

## Next Steps

1. ✅ Run `python example_roas_optimization.py` to see a complete example
2. ✅ Replace sample data with your own CSV file
3. ✅ Adjust `conversion_value` to match your business
4. ✅ Set your target ROAS (e.g., 3.5, 4.0, etc.)
5. ✅ Review the generated visualizations
6. ✅ Implement the recommended budget
7. ✅ Monitor actual performance and adjust

---

## Support

For questions or issues:
- Review the example code in `example_roas_optimization.py`
- Check the main README at `README.md`
- Review the core model documentation at `marketing_mix_model.py`

---

## License

MIT License
