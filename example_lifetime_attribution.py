"""
Complete Example: Lifetime Attribution Google Ads Model

This example handles the unique case where:
1. Attribution window is LIFETIME (not 30-90 days)
2. You may have budget data only, OR budget + actual spend data

Scenario:
---------
- User sees ad Jan 1, 2023
- User purchases Dec 15, 2024 (almost 2 years later!)
- Google Ads attributes conversion to Jan 1, 2023 (not Dec 15, 2024)

This means:
- Recent data is EXTREMELY incomplete (conversions coming in for years)
- Need cohort-based modeling to understand conversion timing
- Only very old data (6+ months) has reasonably complete conversions
- Forecasting requires understanding conversion curves

Budget vs Actual Spend:
------------------------
- **Budget**: What you SET in Google Ads (your control variable)
- **Actual Spend**: What Google actually SPENT (can differ)

If you only have budget data, the model works with that.
If you have actual spend too, the model can be more accurate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from lifetime_attribution_model import LifetimeAttributionModel, compare_budget_vs_spend_effectiveness
from data_generator import MarketingDataGenerator

sns.set_style('whitegrid')


def generate_lifetime_attribution_data(n_days: int = 730, has_actual_spend: bool = False):
    """
    Generate sample data with lifetime attribution characteristics.

    Parameters:
    -----------
    n_days : int
        Number of days of historical data
    has_actual_spend : bool
        Whether to include actual spend column

    Returns:
    --------
    pd.DataFrame
        Simulated Google Ads data
    """
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(
        n_periods=n_days,
        budget_mean=5000,
        budget_std=1500,
        adstock_decay=0.8,  # High decay for lifetime attribution
        saturation_lambda=4000
    )

    # Simulate lifetime attribution effect
    # Recent dates have fewer conversions (incomplete)
    # Old dates have more conversions (more complete)
    reference_date = pd.to_datetime(df['date'].max())
    df['days_since_ad'] = (reference_date - pd.to_datetime(df['date'])).dt.days

    # Apply maturity curve (power law: y = a * x^b)
    # Recent dates get lower multiplier, old dates get higher
    maturity = np.minimum((df['days_since_ad'] / 365) ** 0.3, 1.0)  # Slower accumulation
    df['conversions'] = df['conversions'] * (0.3 + 0.7 * maturity)  # Start at 30%, reach 100%

    # If including actual spend, simulate it
    if has_actual_spend:
        # Actual spend is typically 85-100% of budget (Google doesn't always spend full budget)
        # Sometimes exceeds budget (up to 2x on individual days, though averages out)
        spend_efficiency = np.random.uniform(0.85, 1.05, len(df))
        df['actual_spend'] = df['ad_budget'] * spend_efficiency

        # Occasional days of overspend (competitive auctions)
        overspend_days = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        df.loc[overspend_days, 'actual_spend'] *= np.random.uniform(1.1, 1.5, len(overspend_days))

        # Recalculate ROAS based on actual spend
        df['ROAS'] = (df['conversions'] * 100) / df['actual_spend']

    return df


def main_example():
    """
    Complete example showing lifetime attribution model with budget-only
    and budget+spend scenarios.
    """
    print("\n" + "="*80)
    print("  LIFETIME ATTRIBUTION GOOGLE ADS MODEL")
    print("  For campaigns with lifetime attribution windows")
    print("="*80)

    # =========================================================================
    # SCENARIO 1: Budget Only (Your Current Situation)
    # =========================================================================
    print("\n\n" + "="*80)
    print("  SCENARIO 1: BUDGET-ONLY MODEL (Your Current Data)")
    print("="*80)

    print("\n📊 STEP 1: Load Historical Data")
    print("-" * 80)

    # Generate 2 years of data (need long history for lifetime attribution)
    df_budget_only = generate_lifetime_attribution_data(n_days=730, has_actual_spend=False)

    print(f"✓ Loaded {len(df_budget_only)} days of historical data")
    print(f"  Date range: {df_budget_only['date'].min()} to {df_budget_only['date'].max()}")
    print(f"\n  Data characteristics:")
    print(f"  - Average daily budget: ${df_budget_only['ad_budget'].mean():,.2f}")
    print(f"  - Average daily conversions: {df_budget_only['conversions'].mean():.1f}")
    print(f"  - Average ROAS: {df_budget_only['ROAS'].mean():.2f}")

    # Show recent vs old data difference (lifetime attribution effect)
    recent = df_budget_only.tail(30)
    old = df_budget_only.head(30)
    print(f"\n  ⚠️  LIFETIME ATTRIBUTION EFFECT:")
    print(f"  - Recent 30 days avg conversions: {recent['conversions'].mean():.1f}")
    print(f"  - Oldest 30 days avg conversions: {old['conversions'].mean():.1f}")
    print(f"  - Difference: {((old['conversions'].mean() / recent['conversions'].mean() - 1) * 100):.0f}%")
    print(f"    (Old data has more complete conversions!)")

    print("\n\n🤖 STEP 2: Train Lifetime Attribution Model")
    print("-" * 80)

    conversion_value = 100.0  # Your AOV

    model = LifetimeAttributionModel(
        conversion_value=conversion_value,
        has_actual_spend=False
    )

    # Train on data at least 180 days old (more complete conversions)
    print(f"\n  Using data at least 180 days old for training...")
    print(f"  (Recent data too incomplete with lifetime attribution)")

    trained_model = model.train_model(
        df_budget_only,
        min_days_old=180,
        use_mature_estimates=True,
        optimize_params=True
    )

    print(f"\n✓ Model trained successfully")

    print("\n\n🔍 STEP 3: Understand Conversion Curve")
    print("-" * 80)

    # Show how conversions accumulate over time
    df_with_maturity = model.estimate_mature_conversions(df_budget_only)

    print(f"\n  Conversion Maturity by Age:")
    print(f"  (What % of final conversions have been attributed)")

    sample_ages = [7, 30, 90, 180, 365, 730]
    for age in sample_ages:
        subset = df_with_maturity[
            (df_with_maturity['days_since_ad'] >= age - 5) &
            (df_with_maturity['days_since_ad'] <= age + 5)
        ]
        if len(subset) > 0:
            avg_maturity = subset['maturity_factor'].mean()
            print(f"    {age:3d} days old: {avg_maturity*100:5.1f}% complete")

    print(f"\n  💡 Key Insight:")
    print(f"     Recent data (< 90 days) is very incomplete!")
    print(f"     Only use old data (180+ days) for training")

    print("\n\n💰 STEP 4: Find Optimal Budget for NC ROAS = 3.5")
    print("-" * 80)

    from roas_optimizer import ROASConstrainedOptimizer

    # Get training data features
    X_train, _ = model.prepare_training_data(df_budget_only, min_days_old=180)

    optimizer = ROASConstrainedOptimizer(trained_model, conversion_value)

    result = optimizer.optimize_for_target_roas(
        target_roas=3.5,
        baseline_X=X_train,
        budget_column='ad_budget',
        budget_range=(2000, 10000),
        tolerance=0.1
    )

    print(f"\n{'='*60}")
    print(f"  🎯 OPTIMAL BUDGET (Budget-Only Model)")
    print(f"{'='*60}")
    print(f"  Recommended Daily Budget: ${result['optimal_budget']:,.2f}")
    print(f"  Expected Conversions (attributed): {result['predicted_conversions']:.0f}")
    print(f"  Expected NC ROAS: {result['predicted_roas']:.2f}")
    print(f"  Expected Revenue: ${result['revenue']:,.2f}")
    print(f"  Expected Profit: ${result['profit']:,.2f}")
    print(f"{'='*60}\n")

    print(f"\n  ⏱️  Note about Lifetime Attribution:")
    print(f"     - These conversions will be attributed over time (not immediate)")
    print(f"     - Some purchases happen quickly, others take months/years")
    print(f"     - Total attributed conversions accumulate over lifetime")

    # =========================================================================
    # SCENARIO 2: Budget + Actual Spend (If You Get This Data)
    # =========================================================================
    print("\n\n" + "="*80)
    print("  SCENARIO 2: BUDGET + ACTUAL SPEND MODEL (Enhanced)")
    print("  (If you can get actual spend data from Google Ads)")
    print("="*80)

    print("\n📊 STEP 1: Load Data with Actual Spend")
    print("-" * 80)

    df_with_spend = generate_lifetime_attribution_data(n_days=730, has_actual_spend=True)

    print(f"✓ Loaded {len(df_with_spend)} days with budget AND actual spend")

    # Compare budget vs spend
    print(f"\n  Budget vs Actual Spend Comparison:")
    summary = compare_budget_vs_spend_effectiveness(
        df_with_spend,
        budget_column='ad_budget',
        spend_column='actual_spend',
        conversion_column='conversions',
        conversion_value=conversion_value
    )

    print("\n" + summary.to_string(index=False))

    print(f"\n  💡 Insights:")
    avg_spend_rate = (df_with_spend['actual_spend'] / df_with_spend['ad_budget']).mean()
    if avg_spend_rate < 0.95:
        print(f"     - Google is underspending your budget ({avg_spend_rate*100:.1f}% utilization)")
        print(f"     - Consider: Budget may be too high, or bids too conservative")
    elif avg_spend_rate > 1.05:
        print(f"     - Google is spending more than budget ({avg_spend_rate*100:.1f}% utilization)")
        print(f"     - This is normal (averages out monthly), but watch for overspend")
    else:
        print(f"     - Budget utilization is good ({avg_spend_rate*100:.1f}%)")

    print("\n\n🤖 STEP 2: Train Model with Actual Spend")
    print("-" * 80)

    model_with_spend = LifetimeAttributionModel(
        conversion_value=conversion_value,
        has_actual_spend=True
    )

    trained_model_spend = model_with_spend.train_model(
        df_with_spend,
        min_days_old=180,
        use_mature_estimates=True,
        optimize_params=True
    )

    print(f"\n✓ Model trained with actual spend data")
    print(f"  This model is more accurate because it uses what was actually spent,")
    print(f"  not just what was budgeted.")

    print("\n\n💰 STEP 3: Find Optimal Budget (Actual Spend Model)")
    print("-" * 80)

    X_train_spend, _ = model_with_spend.prepare_training_data(
        df_with_spend,
        min_days_old=180,
        spend_column='actual_spend'
    )

    optimizer_spend = ROASConstrainedOptimizer(trained_model_spend, conversion_value)

    result_spend = optimizer_spend.optimize_for_target_roas(
        target_roas=3.5,
        baseline_X=X_train_spend,
        budget_column='actual_spend',  # Now optimizing based on spend, not budget
        budget_range=(2000, 10000),
        tolerance=0.1
    )

    print(f"\n{'='*60}")
    print(f"  🎯 OPTIMAL SPEND (Budget+Spend Model)")
    print(f"{'='*60}")
    print(f"  Recommended Daily Spend: ${result_spend['optimal_budget']:,.2f}")
    print(f"  Expected Conversions: {result_spend['predicted_conversions']:.0f}")
    print(f"  Expected NC ROAS: {result_spend['predicted_roas']:.2f}")
    print(f"  Expected Revenue: ${result_spend['revenue']:,.2f}")
    print(f"  Expected Profit: ${result_spend['profit']:,.2f}")
    print(f"{'='*60}\n")

    # Translate back to budget (accounting for spend efficiency)
    avg_spend_efficiency = (df_with_spend['actual_spend'] / df_with_spend['ad_budget']).mean()
    recommended_budget = result_spend['optimal_budget'] / avg_spend_efficiency

    print(f"\n  💡 Budget Recommendation:")
    print(f"     Set daily budget to: ${recommended_budget:,.2f}")
    print(f"     (Accounts for ~{avg_spend_efficiency*100:.0f}% spend efficiency)")

    # =========================================================================
    # COMPARISON: Budget-Only vs Budget+Spend Models
    # =========================================================================
    print("\n\n" + "="*80)
    print("  📊 MODEL COMPARISON")
    print("="*80)

    comparison = pd.DataFrame({
        'Model': ['Budget-Only', 'Budget+Spend'],
        'Recommended (Budget)': [result['optimal_budget'], recommended_budget],
        'Recommended (Spend)': [result['optimal_budget'], result_spend['optimal_budget']],
        'Expected Conversions': [result['predicted_conversions'], result_spend['predicted_conversions']],
        'Expected ROAS': [result['predicted_roas'], result_spend['predicted_roas']],
        'Expected Profit': [result['profit'], result_spend['profit']]
    })

    print("\n" + comparison.to_string(index=False))

    print(f"\n  Key Differences:")
    print(f"  - Budget-only model: Simpler, uses what you control (budget)")
    print(f"  - Budget+spend model: More accurate, uses what Google actually spends")
    print(f"  - Difference in recommendations: {abs(recommended_budget - result['optimal_budget']):,.2f}")

    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n\n" + "="*80)
    print("  📋 FINAL RECOMMENDATIONS")
    print("="*80)

    print(f"""
  🎯 FOR LIFETIME ATTRIBUTION WITH BUDGET-ONLY DATA:

  1. DAILY BUDGET: ${result['optimal_budget']:,.2f}

  2. EXPECTED OUTCOMES (long-term attributed conversions):
     - Conversions: {result['predicted_conversions']:.0f} per day (attributed over lifetime)
     - NC ROAS: {result['predicted_roas']:.2f}
     - Revenue: ${result['revenue']:,.2f}
     - Profit: ${result['profit']:,.2f}

  3. KEY CONSIDERATIONS WITH LIFETIME ATTRIBUTION:

     ⏱️  TIMING:
     - Conversions attributed over months/years, not immediate
     - Recent data (< 6 months) is very incomplete
     - Only evaluate campaigns after sufficient time has passed
     - Budget changes take time to show full effect

     📊 DATA REQUIREMENTS:
     - Need 1-2+ years of historical data for accurate modeling
     - Must use data at least 180+ days old for training
     - Recent data is only useful for forecasting, not training

     📦 INVENTORY FORECASTING:
     - With lifetime attribution, forecasting is complex
     - Conversions attributed to today occur over months/years
     - Need to model when purchases actually happen (not when attributed)
     - Recommend building safety stock buffer of 30-50%

  4. IF YOU CAN GET ACTUAL SPEND DATA:

     ✅ ADVANTAGES:
     - More accurate predictions (uses what Google actually spent)
     - Can analyze budget efficiency
     - Better understanding of true ROAS
     - Can optimize for spend, not just budget

     📥 HOW TO GET IT:
     - Export from Google Ads: Campaigns → Download Report
     - Include columns: Date, Budget, Actual Cost, Conversions, ROAS
     - Use "Cost" column for actual spend

     💰 UPDATED RECOMMENDATION:
     - Set daily budget to: ${recommended_budget:,.2f}
     - Expected spend: ${result_spend['optimal_budget']:,.2f}
     - Expected conversions: {result_spend['predicted_conversions']:.0f}
     - Expected NC ROAS: {result_spend['predicted_roas']:.2f}

  5. MONITORING & ADJUSTMENT:

     📈 WHAT TO TRACK:
     - Daily conversions (attributed to each date)
     - Cumulative conversions (total attributed over time)
     - Budget utilization (if you have spend data)
     - ROAS (based on spend, not budget)

     🔄 WHEN TO ADJUST:
     - Wait at least 90 days before judging results
     - Check if conversions are trending toward targets
     - Adjust budget quarterly, not daily/weekly
     - Retrain model every 6 months with new data

  6. COMMON PITFALLS WITH LIFETIME ATTRIBUTION:

     ❌ DON'T:
     - Evaluate campaigns after just 30 days (way too early!)
     - Use recent data for training (it's incomplete!)
     - Expect immediate results (attribution takes time)
     - Compare day-to-day (high variance with long windows)

     ✅ DO:
     - Use old data (180+ days) for training
     - Wait 90+ days to evaluate performance
     - Focus on trends, not daily numbers
     - Build in safety buffers for uncertainty
     - Track cohort performance over time
    """)

    print("="*80)
    print("\n✅ Analysis complete!")
    print("\n📁 Key takeaways:")
    print("   1. With lifetime attribution, recent data is very incomplete")
    print("   2. Only train on data 180+ days old")
    print("   3. Getting actual spend data significantly improves accuracy")
    print("   4. Forecasting is complex - purchases happen over months/years")
    print("   5. Be patient - results take time to materialize")


if __name__ == '__main__':
    main_example()
