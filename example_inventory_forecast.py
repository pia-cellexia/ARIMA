"""
Complete Example: Google Ads Inventory Forecasting with Attribution Windows

This example demonstrates how to:
1. Handle Google Ads attribution windows (conversions attributed to ad impression date)
2. Forecast future conversions for inventory planning
3. Find optimal budget for target NC ROAS and inventory needs
4. Account for incomplete conversion data on recent dates

Use Case:
---------
You run Google Ads search campaigns where:
- All budget goes to search
- Conversions are attributed back to ad impression date (not purchase date)
- You need to forecast inventory requirements
- You want to achieve NC ROAS of 3.5

Example: User sees ad Jan 1, purchases Feb 13 → conversion attributed to Jan 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from marketing_mix_model import MarketingMixModel
from roas_optimizer import ROASConstrainedOptimizer
from inventory_forecaster import AttributionAwareForecaster, create_scenario_comparison
from data_generator import MarketingDataGenerator

sns.set_style('whitegrid')


def main_example():
    """
    Complete example: Google Ads inventory forecasting with target NC ROAS = 3.5
    """
    print("\n" + "="*80)
    print("  GOOGLE ADS INVENTORY FORECASTING")
    print("  With Attribution Windows & Target NC ROAS = 3.5")
    print("="*80)

    # =========================================================================
    # STEP 1: Load Historical Data
    # =========================================================================
    print("\n📊 STEP 1: Loading Historical Google Ads Data")
    print("-" * 80)

    # Generate sample data (in practice, export from Google Ads)
    # This simulates 1 year of Google Ads search campaign data
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(
        n_periods=365,
        budget_mean=5000,
        budget_std=1500,
        adstock_decay=0.6,  # Models attribution carryover
        saturation_lambda=3000
    )

    print(f"✓ Loaded {len(df)} days of Google Ads data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # =========================================================================
    # STEP 2: Handle Attribution Window
    # =========================================================================
    print("\n\n⏱️  STEP 2: Accounting for Attribution Window")
    print("-" * 80)

    attribution_window = 30  # Google Ads typical: 30 days
    conversion_value = 100.0  # Average order value

    print(f"  Attribution window: {attribution_window} days")
    print(f"  Conversion value: ${conversion_value:.2f}")
    print(f"\n  ℹ️  Note: Conversions from last {attribution_window} days are incomplete")
    print(f"     (purchases haven't all occurred yet)")

    # Identify mature vs incomplete data
    reference_date = pd.to_datetime(df['date']).max()
    cutoff_date = reference_date - timedelta(days=attribution_window)

    df_mature = df[pd.to_datetime(df['date']) <= cutoff_date].copy()
    df_incomplete = df[pd.to_datetime(df['date']) > cutoff_date].copy()

    print(f"\n  Mature data (complete conversions): {len(df_mature)} days")
    print(f"  Incomplete data (conversions still coming): {len(df_incomplete)} days")
    print(f"\n  💡 We'll train on mature data for accurate modeling")

    # =========================================================================
    # STEP 3: Train Marketing Mix Model
    # =========================================================================
    print("\n\n🤖 STEP 3: Training Marketing Mix Model")
    print("-" * 80)

    # Use only mature data for training (complete conversions)
    train_size = int(0.8 * len(df_mature))
    train_df = df_mature[:train_size]
    test_df = df_mature[train_size:]

    feature_cols = ['ad_budget', 'ROAS']
    X_train = train_df[feature_cols]
    y_train = train_df['conversions'].values
    X_test = test_df[feature_cols]
    y_test = test_df['conversions'].values

    print(f"  Training samples: {len(X_train)} (mature data only)")
    print(f"  Test samples: {len(X_test)}")

    # Train model with attribution-aware parameters
    model = MarketingMixModel(
        adstock_decay=0.6,  # Higher decay captures attribution lag
        saturation_lambda=1.0,
        use_saturation=True,
        use_adstock=True
    )

    print("\n  Training model with hyperparameter optimization...")
    model.fit(X_train, y_train, optimize_params=True)

    # Evaluate
    test_metrics = model.score(X_test, y_test)
    print(f"\n✓ Model trained successfully")
    print(f"  - R² Score: {test_metrics['R2']:.4f}")
    print(f"  - RMSE: {test_metrics['RMSE']:.2f}")
    print(f"  - MAPE: {test_metrics['MAPE']:.2f}%")

    # =========================================================================
    # STEP 4: Initialize Attribution-Aware Forecaster
    # =========================================================================
    print("\n\n🔮 STEP 4: Initialize Attribution-Aware Forecaster")
    print("-" * 80)

    forecaster = AttributionAwareForecaster(
        model=model,
        attribution_window_days=attribution_window,
        conversion_value=conversion_value
    )

    print(f"✓ Forecaster initialized")
    print(f"  - Attribution window: {attribution_window} days")
    print(f"  - Conversion value: ${conversion_value:.2f}")

    # =========================================================================
    # STEP 5: Forecast Future Conversions for Inventory Planning
    # =========================================================================
    print("\n\n📦 STEP 5: Forecast Inventory Needs")
    print("-" * 80)

    # Plan for next 30 days
    forecast_days = 30
    start_date = reference_date + timedelta(days=1)

    print(f"  Forecast period: {forecast_days} days")
    print(f"  Start date: {start_date.date()}")

    # Test different budget scenarios
    scenarios = [
        {'name': 'Conservative', 'daily_budget': 3000},
        {'name': 'Current Average', 'daily_budget': df_mature['ad_budget'].mean()},
        {'name': 'Moderate', 'daily_budget': 6000},
        {'name': 'Aggressive', 'daily_budget': 8000},
    ]

    print(f"\n  Comparing {len(scenarios)} budget scenarios...")

    comparison = create_scenario_comparison(
        forecaster=forecaster,
        scenarios=scenarios,
        baseline_features=X_test,
        forecast_period_days=forecast_days,
        start_date=start_date
    )

    print("\n  Scenario Comparison:")
    print(comparison.to_string(index=False))

    # =========================================================================
    # STEP 6: Find Optimal Budget for Target NC ROAS = 3.5
    # =========================================================================
    print("\n\n💰 STEP 6: Finding Optimal Budget for NC ROAS = 3.5")
    print("-" * 80)

    target_roas = 3.5

    # Use ROAS optimizer for single-period optimization
    roas_optimizer = ROASConstrainedOptimizer(
        model=model,
        conversion_value=conversion_value
    )

    result = roas_optimizer.optimize_for_target_roas(
        target_roas=target_roas,
        baseline_X=X_test,
        budget_column='ad_budget',
        budget_range=(2000, 10000),
        tolerance=0.1
    )

    print(f"\n{'='*60}")
    print(f"  🎯 OPTIMAL DAILY BUDGET FOR NC ROAS = 3.5")
    print(f"{'='*60}")
    print(f"  Recommended Daily Budget: ${result['optimal_budget']:,.2f}")
    print(f"  Expected Daily Conversions: {result['predicted_conversions']:.0f}")
    print(f"  Expected NC ROAS: {result['predicted_roas']:.2f}")
    print(f"  Expected Daily Revenue: ${result['revenue']:,.2f}")
    print(f"  Expected Daily Profit: ${result['profit']:,.2f}")
    print(f"{'='*60}\n")

    # =========================================================================
    # STEP 7: Forecast Inventory with Optimal Budget
    # =========================================================================
    print("\n📈 STEP 7: Forecast Inventory Needs with Optimal Budget")
    print("-" * 80)

    optimal_daily_budget = result['optimal_budget']

    # Create forecast with optimal budget
    future_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')
    future_budgets = pd.DataFrame({
        'date': future_dates,
        'ad_budget': optimal_daily_budget
    })

    inventory_forecast = forecaster.forecast_inventory_needs(
        future_budgets=future_budgets,
        baseline_features=X_test,
        safety_stock_pct=0.2  # 20% safety stock buffer
    )

    print(f"\n  Inventory Forecast Summary ({forecast_days} days):")
    print(f"  - Total conversions expected: {inventory_forecast['conversions_forecast'].sum():.0f}")
    print(f"  - Total inventory needed (with safety stock): {inventory_forecast['inventory_needed'].sum():.0f}")
    print(f"  - Average daily conversions: {inventory_forecast['conversions_forecast'].mean():.1f}")
    print(f"  - Total budget required: ${optimal_daily_budget * forecast_days:,.2f}")
    print(f"  - Average ROAS: {inventory_forecast['roas_predicted'].mean():.2f}")

    print(f"\n  Daily Breakdown (first 7 days):")
    print(inventory_forecast[['date', 'ad_budget', 'conversions_forecast',
                               'inventory_needed', 'roas_predicted']].head(7).to_string(index=False))

    # Visualize forecast
    forecaster.plot_inventory_forecast(
        inventory_forecast,
        target_inventory=None,
        save_path='inventory_forecast_optimal.png'
    )

    # =========================================================================
    # STEP 8: Optimize for Specific Inventory Target
    # =========================================================================
    print("\n\n🎯 STEP 8: Optimize Budget for Specific Inventory Target")
    print("-" * 80)

    # Example: Need 500 units over next 30 days with ROAS >= 3.5
    target_inventory = 500

    print(f"  Target inventory: {target_inventory} units")
    print(f"  Forecast period: {forecast_days} days")
    print(f"  Target NC ROAS: {target_roas}")
    print(f"\n  Finding optimal budget...")

    inventory_result = forecaster.optimize_budget_for_inventory_target(
        target_inventory=target_inventory,
        forecast_period_days=forecast_days,
        target_roas=target_roas,
        baseline_features=X_test,
        budget_range=(2000, 15000),
        start_date=start_date
    )

    print(f"\n{'='*60}")
    print(f"  📦 INVENTORY-OPTIMIZED BUDGET PLAN")
    print(f"{'='*60}")
    print(f"  Target: {target_inventory} units in {forecast_days} days")
    print(f"  Recommended Daily Budget: ${inventory_result['daily_budget']:,.2f}")
    print(f"  Total Budget Required: ${inventory_result['total_budget']:,.2f}")
    print(f"  Expected Conversions: {inventory_result['expected_conversions']:.0f}")
    print(f"  Expected NC ROAS: {inventory_result['expected_roas']:.2f}")
    print(f"  Meets Inventory Target: {'✓ Yes' if inventory_result['meets_inventory_target'] else '✗ No'}")
    print(f"  Meets ROAS Target (3.5): {'✓ Yes' if inventory_result['meets_roas_target'] else '✗ No'}")
    print(f"{'='*60}\n")

    # Visualize inventory-optimized forecast
    forecaster.plot_inventory_forecast(
        inventory_result['forecast'],
        target_inventory=target_inventory,
        save_path='inventory_forecast_target.png'
    )

    # =========================================================================
    # STEP 9: Handle Incomplete Recent Data
    # =========================================================================
    print("\n\n⚠️  STEP 9: Handling Incomplete Recent Conversion Data")
    print("-" * 80)

    print(f"\n  Recent {attribution_window} days have incomplete conversion data")
    print(f"  (purchases from these ad impressions haven't all occurred yet)")

    # Adjust for maturity
    df_adjusted = forecaster.adjust_for_incomplete_conversions(
        df,
        date_column='date',
        conversion_column='conversions',
        reference_date=reference_date
    )

    # Show adjustment for recent dates
    recent_adjusted = df_adjusted.tail(10)[['date', 'conversions', 'maturity',
                                             'conversions_expected']]

    print(f"\n  Example: Last 10 days (adjusted for maturity)")
    print(recent_adjusted.to_string(index=False))

    print(f"\n  💡 Interpretation:")
    print(f"     - maturity = 0.0: just happened, no conversions attributed yet")
    print(f"     - maturity = 0.5: halfway through attribution window, ~50% complete")
    print(f"     - maturity = 1.0: attribution window complete, 100% of conversions counted")
    print(f"     - conversions_expected: estimated final conversion count after window closes")

    # =========================================================================
    # STEP 10: Final Recommendations
    # =========================================================================
    print("\n\n" + "="*80)
    print("  📋 FINAL RECOMMENDATIONS")
    print("="*80)

    print(f"""
  🎯 FOR TARGET NC ROAS = 3.5:

  1. DAILY BUDGET: ${result['optimal_budget']:,.2f}

  2. EXPECTED OUTCOMES (per day):
     - Conversions: {result['predicted_conversions']:.0f} units
     - Revenue: ${result['revenue']:,.2f}
     - NC ROAS: {result['predicted_roas']:.2f}
     - Profit: ${result['profit']:,.2f}

  3. 30-DAY FORECAST:
     - Total conversions: {inventory_forecast['conversions_forecast'].sum():.0f} units
     - Total inventory needed (with 20% buffer): {inventory_forecast['inventory_needed'].sum():.0f} units
     - Total budget: ${optimal_daily_budget * forecast_days:,.2f}
     - Total revenue: ${inventory_forecast['revenue_predicted'].sum():,.2f}

  📦 FOR SPECIFIC INVENTORY TARGET ({target_inventory} units):

  1. RECOMMENDED DAILY BUDGET: ${inventory_result['daily_budget']:,.2f}
     (Total: ${inventory_result['total_budget']:,.2f} over {forecast_days} days)

  2. WILL DELIVER:
     - {inventory_result['expected_conversions']:.0f} units (target: {target_inventory})
     - NC ROAS: {inventory_result['expected_roas']:.2f} (target: {target_roas})

  ⏱️  ATTRIBUTION WINDOW CONSIDERATIONS:

  1. DATA MATURITY:
     - Use data older than {attribution_window} days for training (complete conversions)
     - Recent {attribution_window} days have incomplete conversion data
     - Factor in attribution lag when forecasting

  2. REPORTING:
     - Conversions will be attributed back to ad impression date
     - Real purchases happen over next {attribution_window} days
     - Total conversions for recent dates will increase as attribution window closes

  3. INVENTORY PLANNING:
     - Order inventory based on forecasted conversions (attributed to date)
     - Add 20% safety stock buffer
     - Monitor daily and adjust as needed

  🔄 NEXT STEPS:

  ✓ Set daily budget to ${result['optimal_budget']:,.2f}
  ✓ Monitor actual NC ROAS (target: 3.5)
  ✓ Order {inventory_forecast['inventory_needed'].sum():.0f} units inventory for next 30 days
  ✓ Review forecast weekly and adjust budget as needed
  ✓ Wait {attribution_window} days before evaluating final conversion counts
    """)

    print("="*80)
    print("\n✅ Analysis complete! Generated files:")
    print("   - inventory_forecast_optimal.png (optimal budget forecast)")
    print("   - inventory_forecast_target.png (inventory target forecast)")
    print("\n📊 Budget scenarios comparison saved above")


if __name__ == '__main__':
    main_example()
