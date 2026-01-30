## Lifetime Attribution Google Ads Model

Complete guide for Google Ads campaigns with **LIFETIME** attribution windows, where conversions can be attributed years after the initial ad impression.

---

## The Lifetime Attribution Challenge

### What is Lifetime Attribution?

With **lifetime attribution**, conversions are attributed to the ad impression date **forever** - there's no time limit.

**Example:**
- **January 1, 2023**: User sees your Google Ad
- **December 15, 2024**: User makes a purchase (almost 2 years later!)
- **Google Ads reporting**: Conversion is attributed to **January 1, 2023**

### How This Differs from Standard Attribution

| Attribution Type | Window | Data Maturity | Forecasting Complexity |
|-----------------|--------|---------------|----------------------|
| **Standard** | 30-90 days | Recent data incomplete for 30-90 days | Moderate |
| **Lifetime** | Forever | Recent data incomplete for MONTHS/YEARS | Very High |

### Why This is Much More Challenging

1. **Recent data is EXTREMELY incomplete**
   - Yesterday's ad impressions will get conversions for years
   - Last month's data might be only 20-30% complete
   - Even 6-month-old data is still accumulating conversions

2. **Need cohort-based modeling**
   - Must understand how conversions accumulate over time
   - Different ad impressions have different conversion timing
   - Conversion curves are critical

3. **Long history required**
   - Need 1-2+ years of data for training
   - Can only use very old data (6+ months) reliably
   - Less data available for training

4. **Forecasting is complex**
   - Must predict not just total conversions, but timing
   - Attributed vs realized conversions are very different
   - Inventory planning requires modeling purchase timing

---

## Your Specific Situation

### Data You Have

**Budget Data:**
- ✅ Daily budget you set in Google Ads
- ✅ Conversions (attributed to ad impression date)
- ✅ NC ROAS
- ✅ Lifetime attribution window

**What You Don't Have (Yet):**
- ❌ Actual spend (what Google actually spent vs what you budgeted)

### Budget vs Actual Spend

**Important Distinction:**

- **Budget** = What you SET/control in Google Ads
  - Example: "My daily budget is $5,000"
  - This is YOUR input

- **Actual Spend** = What Google actually SPENT
  - Example: "Google spent $4,750 yesterday"
  - This is Google's output
  - Can differ from budget due to:
    - Competition levels
    - Bid adjustments
    - Daily spending limits
    - Budget pacing

**Why It Matters:**
- Budget-only model: Models effectiveness of your budget setting
- Budget+spend model: Models effectiveness of actual spending
- Spend model is more accurate (if you can get the data)

---

## The Solution: Lifetime Attribution Model

### Key Components

#### 1. **Conversion Curve Fitting**

Models how conversions accumulate over time using power law, exponential, or logistic curves.

```python
from lifetime_attribution_model import LifetimeAttributionModel

model = LifetimeAttributionModel(conversion_value=100.0, has_actual_spend=False)

# Fit conversion curve on old data
model.fit_conversion_curve(df, min_days_old=365)
```

**Example Output:**
```
Conversion Maturity by Age:
  7 days:   42.7% complete
  30 days:  55.4% complete
  90 days:  66.7% complete
  180 days: 75.0% complete
  365 days: 84.4% complete
  730 days: 94.8% complete
```

**Interpretation:**
- 7-day-old data has only collected 43% of eventual conversions
- Even 1-year-old data still missing ~15% of conversions
- Need 2+ years for nearly complete data

#### 2. **Mature Conversion Estimation**

Estimates final conversion counts for recent dates.

```python
df_adjusted = model.estimate_mature_conversions(df)

# Shows:
# - days_since_ad: age of ad impression
# - maturity_factor: % of conversions attributed so far
# - conversions_mature_estimate: estimated final count
```

#### 3. **Training on Old Data Only**

Critical: Only use data old enough to have reasonably complete conversions.

```python
trained_model = model.train_model(
    df,
    min_days_old=180,  # Use data at least 180 days old
    use_mature_estimates=True  # Adjust for incompleteness
)
```

**Recommended minimum ages:**
- **Conservative**: 365 days (1 year old)
- **Balanced**: 180 days (6 months old)
- **Aggressive**: 90 days (3 months old, less reliable)

#### 4. **Budget Optimization with ROAS Constraint**

Same as before, but accounts for lifetime attribution effects.

```python
from roas_optimizer import ROASConstrainedOptimizer

optimizer = ROASConstrainedOptimizer(trained_model, conversion_value=100.0)

result = optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_train,
    budget_range=(2000, 10000)
)

print(f"Optimal Daily Budget: ${result['optimal_budget']:,.2f}")
print(f"Expected Conversions (attributed over lifetime): {result['predicted_conversions']:.0f}")
print(f"Expected NC ROAS: {result['predicted_roas']:.2f}")
```

---

## Complete Workflow

### Step 1: Export Google Ads Data

**Required Columns:**
- `date` - Ad impression date
- `ad_budget` - Budget you set (or Campaign Budget)
- `conversions` - Conversions attributed to this date
- `ROAS` - Return on ad spend

**Optional but Recommended:**
- `actual_cost` - What Google actually spent (improves accuracy significantly)

**Time Period:**
- **Minimum**: 1 year of data
- **Recommended**: 2+ years of data
- **More is better** with lifetime attribution

### Step 2: Load and Inspect Data

```python
import pandas as pd
from lifetime_attribution_model import LifetimeAttributionModel

# Load data
df = pd.read_csv('google_ads_export.csv')
df['date'] = pd.to_datetime(df['date'])

# Check data characteristics
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total days: {len(df)}")

# Inspect lifetime attribution effect
reference_date = df['date'].max()
df['days_since_ad'] = (reference_date - df['date']).dt.days

# Compare recent vs old
recent = df[df['days_since_ad'] < 30]
old = df[df['days_since_ad'] > 365]

print(f"\nRecent data avg conversions: {recent['conversions'].mean():.1f}")
print(f"Old data avg conversions: {old['conversions'].mean():.1f}")
print(f"Difference: {(old['conversions'].mean() / recent['conversions'].mean() - 1)*100:.0f}%")
# Expect old data to have much higher conversions!
```

### Step 3: Initialize and Train Model

**Budget-Only Model (Your Current Situation):**

```python
# Initialize
model = LifetimeAttributionModel(
    conversion_value=100.0,  # Your average order value
    has_actual_spend=False   # Don't have actual spend data
)

# Train on data at least 180 days old
trained_model = model.train_model(
    df,
    min_days_old=180,
    use_mature_estimates=True,
    optimize_params=True
)

# This will:
# 1. Fit conversion curve on old data
# 2. Estimate mature conversions
# 3. Train marketing mix model
# 4. Optimize hyperparameters
```

**Budget+Spend Model (If You Get Actual Spend):**

```python
# Initialize with actual spend
model = LifetimeAttributionModel(
    conversion_value=100.0,
    has_actual_spend=True  # Have actual spend data!
)

# Train using actual spend column
trained_model = model.train_model(
    df,
    min_days_old=180,
    use_mature_estimates=True,
    optimize_params=True
)
# Model will automatically use 'actual_spend' column instead of 'ad_budget'
```

### Step 4: Find Optimal Budget for NC ROAS = 3.5

```python
from roas_optimizer import ROASConstrainedOptimizer

# Get training features
X_train, _ = model.prepare_training_data(df, min_days_old=180)

# Initialize optimizer
optimizer = ROASConstrainedOptimizer(trained_model, conversion_value=100.0)

# Find optimal budget
result = optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_train,
    budget_column='ad_budget',  # or 'actual_spend' if you have it
    budget_range=(2000, 10000),
    tolerance=0.1
)

print(f"\n🎯 OPTIMAL BUDGET:")
print(f"Daily Budget: ${result['optimal_budget']:,.2f}")
print(f"Expected Conversions: {result['predicted_conversions']:.0f} (attributed over lifetime)")
print(f"Expected NC ROAS: {result['predicted_roas']:.2f}")
print(f"Expected Revenue: ${result['revenue']:,.2f}")
print(f"Expected Profit: ${result['profit']:,.2f}")
```

### Step 5: Forecast Inventory (Complex with Lifetime Attribution)

With lifetime attribution, forecasting inventory is tricky because conversions attributed to "today" actually occur over months/years.

```python
# Create future budget plan
import pandas as pd
from datetime import datetime

future_budgets = pd.DataFrame({
    'date': pd.date_range(datetime.now(), periods=30, freq='D'),
    'ad_budget': result['optimal_budget']  # Use optimal budget
})

# Forecast with cohort adjustment
forecast = model.forecast_with_cohort_adjustment(
    future_budgets=future_budgets,
    baseline_features=X_train,
    forecast_horizon_days=30,
    conversion_realization_days=90  # Consider 90-day window for actual purchases
)

# Shows:
# - conversions_attributed: what Google Ads will show (attributed to date)
# - purchases_day_0, purchases_day_1, etc.: when purchases actually occur
```

**Key Insight:**
- `conversions_attributed` = What shows in Google Ads (attributed to ad date)
- `purchases_day_X` = When purchases actually happen (for inventory)
- These are VERY different with lifetime attribution!

---

## Understanding Conversion Maturity

### Maturity Factor

**Definition:** Percentage of eventual conversions that have been attributed so far

**Formula:**
```
maturity_factor = current_conversions / estimated_final_conversions
```

**Example with Power Law Curve:**

| Days Since Ad | Maturity | Example: Observed | Estimated Final |
|---------------|----------|-------------------|-----------------|
| 1 day | 35% | 50 | 143 |
| 7 days | 43% | 85 | 198 |
| 30 days | 55% | 125 | 227 |
| 90 days | 67% | 160 | 239 |
| 180 days | 75% | 180 | 240 |
| 365 days | 84% | 205 | 244 |
| 730 days | 95% | 233 | 245 |
| 1000+ days | 100% | 245 | 245 |

**Key Takeaway:** Even 6-month-old data is still missing ~25% of conversions!

---

## Budget vs Actual Spend Analysis

If you can get actual spend data, you can analyze budget efficiency.

```python
from lifetime_attribution_model import compare_budget_vs_spend_effectiveness

summary = compare_budget_vs_spend_effectiveness(
    df,
    budget_column='ad_budget',
    spend_column='actual_cost',  # From Google Ads export
    conversion_column='conversions',
    conversion_value=100.0
)

print(summary)
```

**Example Output:**
```
                      metric    value
                  Avg Budget  5692.34
                   Avg Spend  5472.56
              Avg Spend Rate     0.96
             Days Underspent  519
              Days Overspent  211
Avg Conversions per $1 Spend     0.05
           Avg ROAS on Spend     5.28
          Avg ROAS on Budget     5.05
```

**Interpretation:**

**If Avg Spend Rate < 0.90 (Underspending):**
- Google is not spending your full budget
- Possible reasons:
  - Budget too high for available demand
  - Bids too conservative
  - Poor ad quality/relevance
  - Targeting too narrow
- **Action**: Consider lowering budget or raising bids

**If Avg Spend Rate 0.90-1.10 (Good):**
- Budget utilization is healthy
- Google spending most/all of budget
- **Action**: Keep current strategy

**If Avg Spend Rate > 1.10 (Overspending):**
- Google spending more than budget
- Normal on individual days, but averages out monthly
- If consistent, may indicate aggressive bidding
- **Action**: Monitor monthly total, may need budget adjustment

---

## Best Practices for Lifetime Attribution

### 1. Data Collection

✅ **Collect Long History**
- Minimum: 1 year
- Recommended: 2+ years
- Ideal: 3+ years

✅ **Export Regularly**
- Weekly exports
- Keep all historical data
- Never discard old data (it's the most valuable!)

✅ **Include Actual Spend if Possible**
- Significantly improves accuracy
- Export from Google Ads reports
- Column usually named "Cost" or "Actual Cost"

### 2. Model Training

✅ **Use Old Data Only**
- Minimum age: 90 days (risky)
- Recommended: 180 days
- Conservative: 365 days

✅ **Understand Trade-offs**
- Older data minimum = More complete conversions, but less data
- Younger data minimum = More data, but incomplete conversions
- Sweet spot usually around 180 days

✅ **Enable Mature Estimates**
- Use `use_mature_estimates=True`
- Adjusts for incomplete conversions
- Based on fitted conversion curve

✅ **Optimize Hyperparameters**
- Always use `optimize_params=True`
- Higher adstock decay for lifetime attribution
- Model will find optimal values

### 3. Evaluation & Monitoring

✅ **Be Patient**
- Don't evaluate after 30 days (way too early!)
- Wait minimum 90 days to see results
- Best evaluation: 180+ days

✅ **Track Cohort Performance**
- Monitor how conversion curves evolve
- Compare cohorts (ads from different months)
- Look for changes in conversion timing

✅ **Focus on Trends, Not Daily Numbers**
- Daily numbers have high variance
- Look at weekly/monthly trends
- Compare month-over-month

✅ **Retrain Regularly**
- Every 6 months minimum
- When you collect more data
- If business/market changes

### 4. Forecasting & Inventory

✅ **Build Large Safety Buffers**
- Standard: 20% safety stock
- With lifetime attribution: 30-50% recommended
- Higher uncertainty requires larger buffers

✅ **Understand Attribution vs Realization**
- Attributed conversions ≠ when purchases occur
- Model both for inventory planning
- Use conversion timing distribution

✅ **Plan Conservatively**
- Forecast is less reliable with lifetime attribution
- Err on side of caution
- Better to have excess inventory than stockouts

### 5. Budget Setting

✅ **Make Gradual Changes**
- Don't dramatically change budget overnight
- Adjust by 10-20% at a time
- Wait 30-60 days to see effect

✅ **Set and Forget (Mostly)**
- Not a daily optimization problem
- Set budget based on analysis
- Review quarterly, adjust as needed

✅ **Monitor Actual Spend (if available)**
- If spending < 90% of budget: Budget may be too high
- If spending > 110%: Watch for overspend
- Adjust budget based on utilization

---

## Common Pitfalls and Solutions

### Pitfall 1: Using Recent Data for Training

**Problem:**
```python
# ❌ BAD: Using all data including recent
X = df[['ad_budget', 'ROAS']]
y = df['conversions']  # Recent data is incomplete!
model.fit(X, y)
```

**Solution:**
```python
# ✅ GOOD: Using only old data
model.train_model(df, min_days_old=180)  # Only use 180+ day old data
```

### Pitfall 2: Evaluating Too Soon

**Problem:**
"I changed my budget 2 weeks ago and conversions haven't improved!"

**Solution:**
- Wait minimum 90 days
- With lifetime attribution, effects take months to materialize
- Compare same time periods (e.g., Jan 2024 vs Jan 2023)

### Pitfall 3: Ignoring Conversion Curves

**Problem:**
Treating all dates equally, regardless of age

**Solution:**
```python
# Fit conversion curve to understand accumulation pattern
model.fit_conversion_curve(df, min_days_old=365)

# Use mature estimates
df_adjusted = model.estimate_mature_conversions(df)
```

### Pitfall 4: Confusing Budget and Spend

**Problem:**
"My budget is $5,000 but model uses $4,750"

**Solution:**
- Understand the difference: budget (what you set) vs spend (what Google used)
- If you have actual spend, use it (more accurate)
- If not, budget is fine (model still works)

### Pitfall 5: Expecting Immediate Inventory Impact

**Problem:**
"If I spend $5,000 today, how much inventory do I need tomorrow?"

**Solution:**
- With lifetime attribution, conversions attributed to today occur over months/years
- Model conversion realization timing
- Use safety buffers (30-50%)
- Consider historical timing patterns

---

## Advanced: Getting Actual Spend Data from Google Ads

### Why You Want This

- 10-20% accuracy improvement
- Better ROAS understanding
- Can analyze budget efficiency
- Identify optimization opportunities

### How to Export

1. **Google Ads Interface:**
   - Go to Campaigns
   - Click Reports → Predefined reports → Campaign performance
   - Date range: All time (or as long as possible)

2. **Columns to Include:**
   - Date
   - Campaign name
   - Budget (if available)
   - **Cost** (this is actual spend!)
   - Conversions
   - Cost / Conv (optional)
   - Conv. value / Cost (this is ROAS)

3. **Download:**
   - Click Download → CSV
   - Save with date in filename (e.g., `google_ads_2023_01_to_2024_12.csv`)

4. **In Your Code:**
```python
df = pd.read_csv('google_ads_export.csv')

# Rename 'Cost' column to 'actual_spend'
df = df.rename(columns={'Cost': 'actual_spend'})

# Now use has_actual_spend=True
model = LifetimeAttributionModel(conversion_value=100.0, has_actual_spend=True)
```

---

## Model Comparison: Standard vs Lifetime Attribution

| Aspect | Standard Attribution (30-90 days) | Lifetime Attribution |
|--------|-----------------------------------|---------------------|
| **Data Completeness** | Recent 30-90 days incomplete | Recent 6-12+ months incomplete |
| **Training Data** | Can use data 90+ days old | Must use data 180+ days old |
| **Historical Data Needed** | 6-12 months sufficient | 1-2+ years required |
| **Maturity at 30 days** | ~80-90% | ~50-60% |
| **Maturity at 180 days** | ~100% | ~70-80% |
| **Evaluation Time** | 30-60 days | 90-180 days |
| **Forecasting Complexity** | Moderate | High |
| **Inventory Safety Stock** | 20% | 30-50% |
| **Model Retraining Frequency** | Monthly/Quarterly | Quarterly/Semi-annually |
| **Budget Adjustment Frequency** | Weekly/Monthly | Monthly/Quarterly |

---

## Real-World Example

### Scenario

E-commerce company:
- Average order value: $100
- Target NC ROAS: 3.5
- Current daily budget: $7,000
- Have 18 months of historical Google Ads data
- Lifetime attribution window

### Analysis Steps

```python
# 1. Load 18 months of data
df = pd.read_csv('google_ads_18_months.csv')

# 2. Check data quality
print(f"Total days: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check lifetime attribution effect
df['days_since_ad'] = (df['date'].max() - pd.to_datetime(df['date'])).dt.days
recent = df[df['days_since_ad'] < 30]
old = df[df['days_since_ad'] > 180]
print(f"Recent avg conversions: {recent['conversions'].mean():.0f}")
print(f"Old avg conversions: {old['conversions'].mean():.0f}")
# Output: Recent: 145, Old: 285 (old has 2x more!)

# 3. Train model on 180+ day old data
model = LifetimeAttributionModel(conversion_value=100.0, has_actual_spend=False)
trained_model = model.train_model(df, min_days_old=180, optimize_params=True)

# 4. Find optimal budget for ROAS = 3.5
X_train, _ = model.prepare_training_data(df, min_days_old=180)
optimizer = ROASConstrainedOptimizer(trained_model, 100.0)

result = optimizer.optimize_for_target_roas(
    target_roas=3.5,
    baseline_X=X_train,
    budget_range=(3000, 12000)
)
```

### Results

```
Optimal Daily Budget: $8,250
Expected Conversions: 295 per day (attributed over lifetime)
Expected NC ROAS: 3.57
Expected Revenue: $29,500
Expected Profit: $21,250

Current Performance:
Daily Budget: $7,000
Observed Conversions: 145 per day (recent, incomplete!)
Current ROAS: 2.07 (based on incomplete recent data)

Recommendation:
- INCREASE budget from $7,000 to $8,250 (+18%)
- Expected mature conversions: 295/day (current is incomplete)
- Will achieve target ROAS of 3.5
- Wait 90-180 days to evaluate results
```

**Important Note:**
The current observed 145 conversions/day is from recent data and will grow to ~285+ as attribution window matures. The model accounts for this!

---

## Summary & Key Takeaways

### Main Differences from Standard Attribution

1. **Data Requirements**
   - Need much more historical data (1-2+ years vs 6-12 months)
   - Can only use very old data for training (180+ days vs 30-90 days)
   - Recent data is much more incomplete (50% vs 80% at 30 days)

2. **Modeling Approach**
   - Must model conversion curves explicitly
   - Use cohort-based analysis
   - Estimate mature conversions for all dates

3. **Evaluation & Monitoring**
   - Wait much longer to see results (90-180 days vs 30-60 days)
   - Focus on long-term trends, not daily numbers
   - Be more patient with changes

4. **Forecasting**
   - Much more complex (conversions attributed over months/years)
   - Need to model both attribution and realization timing
   - Use larger safety buffers (30-50% vs 20%)

### Your Action Plan

**Immediate (This Week):**
1. ✅ Export 1-2 years of Google Ads data
2. ✅ Include: date, budget, conversions, ROAS
3. ✅ Try to get actual spend data (Cost column)

**Short-term (This Month):**
4. ✅ Run lifetime attribution model on your data
5. ✅ Understand your conversion curve
6. ✅ Find optimal budget for NC ROAS = 3.5
7. ✅ Implement recommended budget

**Medium-term (Next 3-6 Months):**
8. ✅ Monitor performance (but be patient!)
9. ✅ Track conversion maturity over time
10. ✅ Adjust budget quarterly based on trends

**Long-term (Next Year):**
11. ✅ Collect more data continuously
12. ✅ Retrain model every 6 months
13. ✅ Refine forecasting based on realized patterns

### The Bottom Line

**Lifetime attribution is fundamentally different** from standard attribution windows. You need:
- More data (1-2+ years)
- Older data for training (180+ days)
- More patience (90-180 days to evaluate)
- Larger safety buffers (30-50%)
- Different mental model (conversions accumulate slowly)

But with the right approach, you can still:
- ✅ Find optimal budget for target NC ROAS
- ✅ Forecast conversions (with caveats)
- ✅ Plan inventory (with larger buffers)
- ✅ Make data-driven decisions

The key is understanding and accounting for the lifetime attribution effects in your analysis!

---

## Files & Resources

- `lifetime_attribution_model.py` - Core lifetime attribution model
- `example_lifetime_attribution.py` - Complete working example
- `LIFETIME_ATTRIBUTION_GUIDE.md` - This guide
- `roas_optimizer.py` - ROAS-constrained optimization
- `marketing_mix_model.py` - Base MMM model

---

## Support

For questions:
- Review example: `python example_lifetime_attribution.py`
- Check other guides: `ROAS_OPTIMIZATION_GUIDE.md`, `INVENTORY_FORECASTING_GUIDE.md`
- See main README: `README.md`

---

## License

MIT License
