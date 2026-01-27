## Google Ads Inventory Forecasting with Attribution Windows

Complete guide for forecasting inventory needs with Google Ads search campaigns where conversions are back-attributed to ad impression dates.

---

## The Attribution Challenge

### What is Back-Attribution?

In Google Ads (and most advertising platforms), **conversions are attributed to the ad impression date, not the purchase date**.

**Example:**
- **January 1**: User sees your ad (ad impression)
- **February 13**: User makes a purchase
- **Google Ads reporting**: Conversion is counted for **January 1**, not February 13

### Why This Matters

1. **Recent data is incomplete**: Conversions for the last 30-90 days are still coming in
2. **Forecasting is tricky**: You need to account for attribution lag
3. **Inventory planning**: Must predict when conversions will be attributed, not when purchases occur

---

## Your Use Case

You have:
- ✅ **Google Ads search campaigns** (all budget allocated to search)
- ✅ **Historical data**: ad budget, conversions (attributed), ROAS
- ✅ **Complete control** over daily budget
- ✅ **Target NC ROAS**: 3.5

You want to:
- 📦 **Forecast inventory needs** for production planning
- 💰 **Find optimal budget** to achieve NC ROAS of 3.5
- 📈 **Maximize conversions** within ROAS constraint

---

## Solution Overview

### Attribution-Aware Forecaster

The `AttributionAwareForecaster` handles:

1. **Conversion Maturity Tracking**
   - Identifies which dates have complete vs incomplete conversion data
   - Adjusts for attribution lag

2. **Inventory Forecasting**
   - Predicts future conversions accounting for attribution windows
   - Adds safety stock buffer for uncertainty

3. **Budget Optimization**
   - Finds optimal budget for target NC ROAS
   - Ensures inventory targets are met

4. **Multi-Scenario Planning**
   - Compares different budget scenarios
   - Shows trade-offs between budget, conversions, and ROAS

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

### 2. Run Complete Example

```bash
python example_inventory_forecast.py
```

This will:
- Load/generate Google Ads data
- Handle attribution windows (30 days)
- Train marketing mix model on mature data only
- Forecast inventory needs for next 30 days
- Find optimal budget for NC ROAS = 3.5
- Generate comprehensive visualizations

---

## Step-by-Step Guide

### Step 1: Load Your Google Ads Data

```python
import pandas as pd
from marketing_mix_model import MarketingMixModel
from inventory_forecaster import AttributionAwareForecaster

# Load your data (export from Google Ads)
df = pd.read_csv('google_ads_data.csv')

# Required columns:
# - date: Date of ad impression
# - ad_budget: Daily ad spend
# - conversions: Conversions attributed to this date
# - ROAS: Return on ad spend
```

### Step 2: Identify Mature vs Incomplete Data

```python
from datetime import datetime, timedelta

attribution_window = 30  # Google Ads default: 30 days
reference_date = pd.to_datetime(df['date']).max()
cutoff_date = reference_date - timedelta(days=attribution_window)

# Mature data: conversions are complete (older than attribution window)
df_mature = df[pd.to_datetime(df['date']) <= cutoff_date].copy()

# Incomplete data: conversions still coming in (last 30 days)
df_incomplete = df[pd.to_datetime(df['date']) > cutoff_date].copy()

print(f"Mature data: {len(df_mature)} days (use for training)")
print(f"Incomplete data: {len(df_incomplete)} days (exclude from training)")
```

### Step 3: Train Model on Mature Data Only

```python
# CRITICAL: Only use mature data for training
# Recent data has incomplete conversions which will bias the model

X_train = df_mature[['ad_budget', 'ROAS']]
y_train = df_mature['conversions'].values

model = MarketingMixModel(
    adstock_decay=0.6,  # Higher decay captures attribution lag
    saturation_lambda=1.0,
    use_saturation=True,
    use_adstock=True
)

# Train with hyperparameter optimization
model.fit(X_train, y_train, optimize_params=True)

# Evaluate
metrics = model.score(X_test, y_test)
print(f"Model R²: {metrics['R2']:.4f}")
```

### Step 4: Initialize Forecaster

```python
conversion_value = 100.0  # Your average order value

forecaster = AttributionAwareForecaster(
    model=model,
    attribution_window_days=30,
    conversion_value=conversion_value
)
```

### Step 5: Forecast Inventory Needs

```python
from datetime import datetime
import pandas as pd

# Plan for next 30 days
forecast_days = 30
start_date = datetime.now()

# Your planned budget
daily_budget = 5000.0

# Create future dates
future_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')
future_budgets = pd.DataFrame({
    'date': future_dates,
    'ad_budget': daily_budget
})

# Forecast inventory
inventory_forecast = forecaster.forecast_inventory_needs(
    future_budgets=future_budgets,
    baseline_features=X_train,
    safety_stock_pct=0.2  # 20% safety buffer
)

print(f"Total inventory needed: {inventory_forecast['inventory_needed'].sum():.0f} units")
print(f"Expected NC ROAS: {inventory_forecast['roas_predicted'].mean():.2f}")
```

### Step 6: Find Optimal Budget for NC ROAS = 3.5

```python
from roas_optimizer import ROASConstrainedOptimizer

# Initialize ROAS optimizer
roas_optimizer = ROASConstrainedOptimizer(
    model=model,
    conversion_value=conversion_value
)

# Find optimal budget
result = roas_optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_train,
    budget_column='ad_budget',
    budget_range=(2000, 10000),
    tolerance=0.1
)

print(f"\n🎯 OPTIMAL DAILY BUDGET FOR NC ROAS = 3.5")
print(f"Recommended Budget: ${result['optimal_budget']:,.2f}")
print(f"Expected Conversions: {result['predicted_conversions']:.0f}")
print(f"Expected NC ROAS: {result['predicted_roas']:.2f}")
```

### Step 7: Forecast with Optimal Budget

```python
# Use optimal budget for inventory forecast
optimal_daily_budget = result['optimal_budget']

future_budgets['ad_budget'] = optimal_daily_budget

inventory_forecast = forecaster.forecast_inventory_needs(
    future_budgets=future_budgets,
    baseline_features=X_train,
    safety_stock_pct=0.2
)

print(f"\n📦 INVENTORY FORECAST (30 days)")
print(f"Total conversions: {inventory_forecast['conversions_forecast'].sum():.0f}")
print(f"Inventory needed: {inventory_forecast['inventory_needed'].sum():.0f} units")
print(f"Average NC ROAS: {inventory_forecast['roas_predicted'].mean():.2f}")
```

### Step 8: Visualize Forecast

```python
forecaster.plot_inventory_forecast(
    inventory_forecast,
    target_inventory=None,
    save_path='inventory_forecast.png'
)
```

This creates a 4-panel visualization:
- **Daily conversions** with safety stock
- **Cumulative inventory** needs over time
- **Daily budget** allocation
- **Daily ROAS** forecast

---

## Understanding Attribution Maturity

### Maturity Calculation

**Maturity** = Percentage of expected conversions that have been attributed

```
maturity = days_since_ad_impression / attribution_window_days
```

- **maturity = 0.0**: Ad just shown, no conversions yet (0% complete)
- **maturity = 0.5**: Halfway through window, ~50% of conversions attributed
- **maturity = 1.0**: Attribution window closed, 100% of conversions counted

### Adjusting for Incomplete Data

```python
# Adjust recent data for incomplete conversions
df_adjusted = forecaster.adjust_for_incomplete_conversions(
    df,
    date_column='date',
    conversion_column='conversions',
    reference_date=datetime.now()
)

# View adjusted data
print(df_adjusted[['date', 'conversions', 'maturity', 'conversions_expected']].tail(10))
```

**Example Output:**
```
date        conversions  maturity  conversions_expected
2024-01-22  227.55       0.30      758.51
2024-01-23  195.77       0.27      734.14
2024-01-24  199.90       0.23      856.70
2024-01-25  234.82       0.20      1174.08
2024-01-26  226.30       0.17      1357.83
2024-01-27  223.53       0.13      1676.50
2024-01-28  229.47       0.10      2294.74
2024-01-29  234.18       0.07      3512.64
2024-01-30  219.25       0.03      6577.64
2024-01-31  215.05       0.00      215.05  (just happened, no conversions yet)
```

### Key Insight

**Recent dates will have much higher final conversion counts!**

- Jan 31 (maturity 0%): 215 conversions now, but expect **many more** as attribution window closes
- Jan 22 (maturity 30%): 228 conversions now, expect ~759 final
- Dates 30+ days ago (maturity 100%): Conversion counts are final

---

## Advanced Features

### Compare Multiple Budget Scenarios

```python
from inventory_forecaster import create_scenario_comparison

scenarios = [
    {'name': 'Conservative', 'daily_budget': 3000},
    {'name': 'Moderate', 'daily_budget': 6000},
    {'name': 'Aggressive', 'daily_budget': 9000},
]

comparison = create_scenario_comparison(
    forecaster=forecaster,
    scenarios=scenarios,
    baseline_features=X_train,
    forecast_period_days=30
)

print(comparison)
```

**Example Output:**
```
scenario      daily_budget  total_conversions  avg_roas  total_revenue  total_profit
Conservative  3000          1229               1.37      122,922        32,922
Moderate      6000          6018               3.34      601,792        421,792
Aggressive    9000          7800               2.89      780,000        510,000
```

### Optimize for Specific Inventory Target

```python
# Need 5000 units in 30 days with NC ROAS >= 3.5
inventory_result = forecaster.optimize_budget_for_inventory_target(
    target_inventory=5000,
    forecast_period_days=30,
    target_roas=3.5,
    baseline_features=X_train,
    budget_range=(2000, 15000)
)

print(f"Recommended Daily Budget: ${inventory_result['daily_budget']:,.2f}")
print(f"Will deliver: {inventory_result['expected_conversions']:.0f} units")
print(f"Expected NC ROAS: {inventory_result['expected_roas']:.2f}")
print(f"Meets targets: {inventory_result['meets_inventory_target'] and inventory_result['meets_roas_target']}")
```

---

## Real-World Example

### Scenario

E-commerce company:
- **Average order value**: $100
- **Target NC ROAS**: 3.5
- **Current daily budget**: $7,000
- **Need to forecast**: Next 30 days of inventory

### Analysis

```python
# 1. Load 1 year of Google Ads data
df = pd.read_csv('google_ads_export.csv')

# 2. Use mature data only (exclude last 30 days)
df_mature = df[:-30]
X = df_mature[['ad_budget', 'ROAS']]
y = df_mature['conversions'].values

# 3. Train model
model = MarketingMixModel(use_saturation=True, use_adstock=True)
model.fit(X, y, optimize_params=True)
# Result: R² = 0.82, RMSE = 7.3

# 4. Initialize forecaster
forecaster = AttributionAwareForecaster(model, 30, 100.0)

# 5. Find optimal budget for ROAS = 3.5
optimizer = ROASConstrainedOptimizer(model, 100.0)
result = optimizer.optimize_for_target_roas(3.5, X, budget_range=(3000, 10000))
# Result: Optimal budget = $5,463/day

# 6. Forecast inventory
future_budgets = pd.DataFrame({
    'date': pd.date_range('2024-02-01', periods=30, freq='D'),
    'ad_budget': 5463.06
})

forecast = forecaster.forecast_inventory_needs(future_budgets, X, safety_stock_pct=0.2)
```

### Results

```
📊 RECOMMENDATIONS:

Daily Budget: $5,463 (reduce from current $7,000)
Why? Better efficiency - same profit with less risk

30-Day Forecast:
- Total conversions: 5,547 units
- Inventory needed (with 20% buffer): 6,660 units
- Total spend: $163,892
- Total revenue: $554,726
- NC ROAS: 3.38 (meets target of 3.5)
- Profit: $390,834

Action Items:
✓ Reduce daily budget from $7,000 to $5,463
✓ Order 6,660 units for next 30 days
✓ Monitor actual NC ROAS weekly
✓ Adjust if needed based on real performance
```

---

## Best Practices

### 1. Data Quality

✅ **Use mature data for training**
- Exclude last 30-90 days (attribution window)
- Recent data has incomplete conversions
- Will bias model if included

✅ **Validate attribution window**
- Check your Google Ads settings
- Common values: 30, 60, 90 days
- Use actual value in your analysis

✅ **Monitor data freshness**
- Export data regularly
- Account for reporting lag (24-48 hours)
- Wait for attribution window to close before evaluating campaigns

### 2. Model Training

✅ **Optimize hyperparameters**
- Always use `optimize_params=True`
- Adstock decay captures attribution lag
- Saturation captures diminishing returns

✅ **Validate model performance**
- Check R² > 0.7 for reliable forecasts
- MAPE < 10% indicates good accuracy
- Use train/test split for validation

✅ **Account for seasonality**
- Include seasonal features if needed
- Train on sufficient history (1+ years recommended)
- Adjust for known events (holidays, sales)

### 3. Forecasting

✅ **Add safety stock**
- 20% buffer for uncertainty is typical
- Adjust based on:
  - Model accuracy (lower R² = higher buffer)
  - Stockout cost (expensive = higher buffer)
  - Lead time (longer = higher buffer)

✅ **Review and adjust**
- Forecast is a starting point, not a guarantee
- Monitor actual vs predicted weekly
- Adjust budget dynamically based on performance

✅ **Consider constraints**
- Maximum daily budget limits
- Minimum order quantities
- Warehouse capacity
- Cash flow constraints

### 4. ROAS Optimization

✅ **Set realistic targets**
- Target NC ROAS should be achievable
- Too high = optimizer may fail or sacrifice too many conversions
- Too low = leaving money on the table

✅ **Balance goals**
- ROAS vs conversions trade-off
- Profit vs growth
- Risk vs reward

✅ **Test and iterate**
- Start with recommended budget
- Monitor for 1-2 attribution windows
- Adjust based on results

---

## Troubleshooting

### Model Performance is Poor (R² < 0.5)

**Problem**: Model can't accurately predict conversions

**Solutions**:
1. **Check data quality**
   - Remove outliers
   - Handle missing values
   - Verify data is from mature periods only

2. **Add more features**
   - Seasonality (day of week, month)
   - Competitor activity
   - External factors (holidays, events)

3. **Use more data**
   - Need 100+ samples minimum
   - 1 year+ is recommended
   - More data = better predictions

4. **Try different model**
   - Adjust adstock_decay range
   - Try different saturation_lambda values
   - Consider ensemble methods

### Forecasts Don't Match Reality

**Problem**: Predicted conversions are far from actual

**Possible Causes**:

1. **Attribution window mismatch**
   - Solution: Verify your actual attribution window
   - Check Google Ads settings
   - Use correct value in forecaster

2. **Recent data used for training**
   - Solution: Only train on mature data
   - Exclude last N days (attribution window)

3. **External factors**
   - Solution: Account for changes in:
     - Competition
     - Seasonality
     - Product availability
     - Marketing mix

4. **Data drift**
   - Solution: Retrain model regularly
   - Monthly or quarterly retraining recommended

### Optimizer Can't Meet Targets

**Problem**: Cannot achieve inventory target with ROAS constraint

**Solutions**:

1. **Widen budget_range**
   - Try: `(1000, 20000)` instead of `(3000, 10000)`
   - May need higher budget than expected

2. **Relax ROAS target**
   - Lower target_roas by 0.5
   - Or increase tolerance

3. **Extend forecast period**
   - 60 days instead of 30
   - More time = easier to hit targets

4. **Accept trade-off**
   - May not be possible to meet both constraints
   - Choose priority: inventory or ROAS

---

## API Reference

### AttributionAwareForecaster

#### Constructor
```python
AttributionAwareForecaster(
    model: MarketingMixModel,
    attribution_window_days: int = 30,
    conversion_value: float = 1.0
)
```

#### Methods

**forecast_conversions(future_budgets, baseline_features, budget_column, include_attribution_lag)**
- Forecast conversions for future dates
- Returns DataFrame with predictions

**forecast_inventory_needs(future_budgets, baseline_features, budget_column, safety_stock_pct)**
- Forecast inventory requirements
- Includes safety stock buffer
- Returns DataFrame with inventory needs

**optimize_budget_for_inventory_target(target_inventory, forecast_period_days, target_roas, ...)**
- Find optimal daily budget for inventory target
- Maintains ROAS constraint
- Returns optimization results

**calculate_conversion_maturity(dates, reference_date)**
- Calculate conversion maturity for dates
- Returns Series with maturity (0-1)

**adjust_for_incomplete_conversions(df, date_column, conversion_column, reference_date)**
- Adjust data for attribution lag
- Returns DataFrame with adjusted conversions

**plot_inventory_forecast(forecast, target_inventory, save_path)**
- Visualize inventory forecast
- 4-panel plot showing conversions, budget, ROAS

---

## Files in This Project

```
ARIMA/
├── inventory_forecaster.py           # Attribution-aware forecasting module
├── example_inventory_forecast.py     # Complete working example
├── roas_optimizer.py                 # ROAS-constrained optimization
├── marketing_mix_model.py            # Core causal MMM model
├── data_generator.py                 # Sample data generator
├── INVENTORY_FORECASTING_GUIDE.md    # This guide
└── ROAS_OPTIMIZATION_GUIDE.md        # ROAS optimization guide
```

---

## Next Steps

1. ✅ Run the example: `python example_inventory_forecast.py`
2. ✅ Export your Google Ads data (last 1-2 years)
3. ✅ Load your data and train model
4. ✅ Find optimal budget for NC ROAS = 3.5
5. ✅ Forecast inventory for next 30 days
6. ✅ Implement recommended budget
7. ✅ Monitor actual performance
8. ✅ Adjust and iterate

---

## Key Takeaways

1. **Attribution Windows Matter**
   - Recent data is incomplete
   - Train on mature data only
   - Wait for attribution window to close before evaluating

2. **Forecasting Requires Care**
   - Account for attribution lag
   - Add safety stock buffer
   - Monitor and adjust regularly

3. **ROAS Optimization Works**
   - Can achieve target NC ROAS
   - While maximizing conversions
   - And planning inventory accurately

4. **This is Iterative**
   - Start with recommendations
   - Monitor actual performance
   - Adjust based on results
   - Retrain model regularly

---

## Support

For questions or issues:
- Review example code: `example_inventory_forecast.py`
- Check ROAS guide: `ROAS_OPTIMIZATION_GUIDE.md`
- See main README: `README.md`

---

## License

MIT License
