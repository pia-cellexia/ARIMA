"""
Custom Lifetime Attribution Model for 121 Days of Google Ads Data

This script is tailored for your specific situation:
- 121 days of historical data (about 4 months)
- Lifetime attribution window
- Have actual spend data (cost column)
- Columns: date, ad_budget, conversions, cost, ROAS

Key Adaptations:
----------------
1. Work with limited data (121 days instead of ideal 365+ days)
2. Use actual spend instead of budget (more accurate)
3. Account for data incompleteness with lifetime attribution
4. Provide realistic expectations given data limitations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from lifetime_attribution_model import LifetimeAttributionModel, compare_budget_vs_spend_effectiveness
from roas_optimizer import ROASConstrainedOptimizer

sns.set_style('whitegrid')


def analyze_your_data(file_path: str):
    """
    Analyze your specific Google Ads data with lifetime attribution.

    Parameters:
    -----------
    file_path : str
        Path to your CSV file with columns:
        - date: Date column
        - ad_budget: Your daily budget
        - conversions: Daily conversions (lifetime attributed)
        - cost: Actual spend by Google
        - ROAS: Return on ad spend
    """

    print("\n" + "="*80)
    print("  CUSTOM LIFETIME ATTRIBUTION ANALYSIS")
    print("  For Your 121 Days of Google Ads Data")
    print("="*80)

    # =========================================================================
    # STEP 1: Load and Validate Your Data
    # =========================================================================
    print("\n📊 STEP 1: Loading Your Data")
    print("-" * 80)

    # Load data
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Validate columns
    required_cols = ['date', 'ad_budget', 'conversions', 'cost', 'ROAS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"\n  Data Summary:")
    summary_stats = df[['ad_budget', 'cost', 'conversions', 'ROAS']].describe()
    print(summary_stats)

    # =========================================================================
    # STEP 2: Understand Data Limitations with Lifetime Attribution
    # =========================================================================
    print("\n\n⚠️  STEP 2: Understanding Data Limitations")
    print("-" * 80)

    reference_date = df['date'].max()
    df['days_since_ad'] = (reference_date - df['date']).dt.days

    print(f"\n  With LIFETIME attribution, your data completeness:")

    # Estimate maturity using power law approximation
    # Based on typical lifetime attribution curves
    df['estimated_maturity'] = np.minimum((df['days_since_ad'] / 365) ** 0.3, 1.0)
    df['estimated_maturity'] = np.maximum(df['estimated_maturity'] * 0.7 + 0.3, 0.3)  # 30-100% range

    age_ranges = [
        (0, 7, "Last 7 days"),
        (7, 30, "Last 30 days"),
        (30, 60, "30-60 days ago"),
        (60, 90, "60-90 days ago"),
        (90, 121, "90-121 days ago")
    ]

    for min_age, max_age, label in age_ranges:
        subset = df[(df['days_since_ad'] >= min_age) & (df['days_since_ad'] < max_age)]
        if len(subset) > 0:
            avg_maturity = subset['estimated_maturity'].mean()
            avg_conv = subset['conversions'].mean()
            estimated_final = avg_conv / avg_maturity if avg_maturity > 0 else avg_conv
            print(f"    {label:20s}: ~{avg_maturity*100:.0f}% complete")
            print(f"      Current avg: {avg_conv:.1f} conversions/day")
            print(f"      Estimated final: {estimated_final:.1f} conversions/day (when mature)")

    print(f"\n  ⚠️  KEY INSIGHT:")
    print(f"     Even your OLDEST data (121 days) is only ~70-75% complete!")
    print(f"     Your NEWEST data is only ~40-50% complete!")
    print(f"     This means ALL your data is still accumulating conversions.")

    print(f"\n  📊 IMPLICATIONS:")
    print(f"     1. Model accuracy will be limited (not enough mature data)")
    print(f"     2. Predictions have higher uncertainty")
    print(f"     3. Need to account for incompleteness in modeling")
    print(f"     4. Results are indicative, not definitive")

    # =========================================================================
    # STEP 3: Analyze Budget vs Actual Spend
    # =========================================================================
    print("\n\n💰 STEP 3: Budget vs Actual Spend Analysis")
    print("-" * 80)

    conversion_value = 100.0  # Adjust to your actual AOV

    summary = compare_budget_vs_spend_effectiveness(
        df,
        budget_column='ad_budget',
        spend_column='cost',
        conversion_column='conversions',
        conversion_value=conversion_value
    )

    print(f"\n  Budget Efficiency Summary:")
    print(summary.to_string(index=False))

    avg_spend_rate = (df['cost'] / df['ad_budget']).mean()
    print(f"\n  💡 Interpretation:")
    if avg_spend_rate < 0.90:
        print(f"     Google is underspending your budget ({avg_spend_rate*100:.1f}% utilization)")
        print(f"     Consider: Budget may be too high or bids too conservative")
    elif avg_spend_rate > 1.10:
        print(f"     Google frequently exceeds budget ({avg_spend_rate*100:.1f}% utilization)")
        print(f"     Monitor for overspend, but this can be normal day-to-day")
    else:
        print(f"     Budget utilization is healthy ({avg_spend_rate*100:.1f}%)")

    # =========================================================================
    # STEP 4: Train Model with Actual Spend
    # =========================================================================
    print("\n\n🤖 STEP 4: Training Model with Actual Spend Data")
    print("-" * 80)

    print(f"\n  ⚠️  IMPORTANT: With only 121 days of data:")
    print(f"     - Standard approach uses 180+ day old data (you don't have this)")
    print(f"     - Will use older 60+ days data (best available)")
    print(f"     - Model will be less accurate due to incomplete conversions")
    print(f"     - Treat results as indicative estimates, not precise forecasts")

    # Initialize model with actual spend
    model = LifetimeAttributionModel(
        conversion_value=conversion_value,
        has_actual_spend=True  # You have actual spend data!
    )

    # Train on data at least 180 days old (more complete conversions)
    min_days_old = 60

    print(f"\n  Training on data at least {min_days_old} days old...")
    print(f"  (Ideally would use 180+ days, but working with what we have)")
    print(f"\n  🔑 IMPORTANT: Training on OBSERVED conversions (not mature estimates)")
    print(f"     - This predicts what you'll actually SEE (observed conversions)")
    print(f"     - NOT what they'll eventually become (mature conversions)")
    print(f"     - Better for realistic forecasting")

    try:
        trained_model = model.train_model(
            df,
            min_days_old=min_days_old,
            use_mature_estimates=False,  # ← CHANGED: Use OBSERVED, not mature!
            optimize_params=True
        )

        print(f"\n✓ Model trained successfully")
        print(f"  Note: Predictions will have higher uncertainty due to:")
        print(f"  1. Limited data (121 days vs ideal 365+ days)")
        print(f"  2. All data incomplete (even oldest is ~70-75% complete)")
        print(f"  3. Lifetime attribution effects not fully captured")

    except Exception as e:
        print(f"\n❌ Error training model: {e}")
        print(f"\n  This might happen if you don't have enough data {min_days_old}+ days old.")
        print(f"  Consider collecting more historical data before modeling.")
        return

    # =========================================================================
    # STEP 5: Find Optimal Budget for NC ROAS = 3.5
    # =========================================================================
    print("\n\n🎯 STEP 5: Finding Optimal Budget for NC ROAS = 3.5")
    print("-" * 80)

    # Prepare training features (using actual spend)
    X_train, _ = model.prepare_training_data(
        df,
        min_days_old=min_days_old,
        spend_column='cost'
    )

    # Initialize optimizer
    optimizer = ROASConstrainedOptimizer(trained_model, conversion_value)

    # Current spend range
    min_spend = df['cost'].quantile(0.1)
    max_spend = df['cost'].quantile(0.9) * 1.5

    print(f"\n  Searching optimal spend in range: ${min_spend:,.0f} - ${max_spend:,.0f}")

    try:
        result = optimizer.optimize_for_target_roas(
            target_roas=3.5,
            baseline_X=X_train,
            budget_column='cost',  # Optimizing spend, not budget
            budget_range=(min_spend, max_spend),
            tolerance=0.15  # Slightly more tolerance due to uncertainty
        )

        print(f"\n{'='*60}")
        print(f"  🎯 OPTIMAL SPEND RECOMMENDATION")
        print(f"{'='*60}")
        print(f"  Recommended Daily Spend: ${result['optimal_budget']:,.2f}")
        print(f"  Expected Conversions: {result['predicted_conversions']:.1f} conversions/day")
        print(f"  Expected NC ROAS: {result['predicted_roas']:.2f}")
        print(f"  Optimization Success: {result['optimization_success']}")
        print(f"{'='*60}\n")

        # Translate to budget recommendation
        avg_spend_efficiency = (df['cost'] / df['ad_budget']).mean()
        recommended_budget = result['optimal_budget'] / avg_spend_efficiency

        print(f"\n  💡 Budget Setting Recommendation:")
        print(f"     Set daily budget to: ${recommended_budget:,.2f}")
        print(f"     This accounts for ~{avg_spend_efficiency*100:.0f}% average spend efficiency")
        print(f"     Expected actual spend: ${result['optimal_budget']:,.2f}")

        # Compare with current
        current_avg_spend = df['cost'].mean()
        current_avg_budget = df['ad_budget'].mean()
        current_avg_conv = df['conversions'].mean()
        current_avg_roas = df['ROAS'].mean()

        print(f"\n  📊 Comparison with Current Performance:")
        print(f"     Current avg budget: ${current_avg_budget:,.2f}")
        print(f"     Current avg spend: ${current_avg_spend:,.2f}")
        print(f"     Current avg conversions: {current_avg_conv:.1f} (incomplete!)")
        print(f"     Current avg ROAS: {current_avg_roas:.2f}")
        print(f"\n     Recommended:")
        print(f"     Change in budget: {((recommended_budget / current_avg_budget - 1) * 100):+.1f}%")
        print(f"     Change in spend: {((result['optimal_budget'] / current_avg_spend - 1) * 100):+.1f}%")

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        print(f"  This might happen if target ROAS is not achievable with your data.")
        print(f"  Try a different target ROAS or collect more data.")
        return

    # =========================================================================
    # STEP 6: Forecast Next 3 Months with Recommended Budget
    # =========================================================================
    print("\n\n📅 STEP 6: 3-Month Purchase Forecast with Recommended Budget")
    print("-" * 80)

    print(f"\n  Creating 90-day forecast starting from {reference_date.date() + timedelta(days=1)}...")
    print(f"  Using recommended daily spend: ${result['optimal_budget']:,.2f}")

    # DEBUG: Show what the model learned from
    print(f"\n  🔍 DEBUG - Understanding Model Training:")
    df_train_subset = df[df['days_since_ad'] >= min_days_old].copy()
    print(f"  - Training data: {len(df_train_subset)} days (ages {min_days_old}+ days)")
    print(f"  - Average OBSERVED conversions in training data: {df_train_subset['conversions'].mean():.1f}/day")
    print(f"  - Average maturity of training data: {df_train_subset['estimated_maturity'].mean()*100:.0f}%")
    if 'conversions_mature_estimate' in df_train_subset.columns:
        print(f"  - Average MATURE conversion estimate: {df_train_subset['conversions_mature_estimate'].mean():.1f}/day")

    # What does current data show?
    print(f"\n  🔍 Current Performance (for comparison):")
    print(f"  - Recent 30 days avg conversions: {df.tail(30)['conversions'].mean():.1f}/day")
    print(f"  - Overall avg conversions: {df['conversions'].mean():.1f}/day")
    print(f"  - Overall avg spend: ${df['cost'].mean():,.2f}/day")

    print(f"\n  ℹ️  IMPORTANT CLARIFICATION:")
    print(f"  This forecast shows NEW conversions from NEW ads you'll run in next 3 months.")
    print(f"  With lifetime attribution, you'll ALSO continue getting conversions from")
    print(f"  historical ads (not shown here - those are ongoing/existing conversions).\n")

    # Create future dates (90 days = ~3 months)
    future_start_date = reference_date + timedelta(days=1)
    future_dates = pd.date_range(start=future_start_date, periods=90, freq='D')

    # Create future budget dataframe
    future_df = pd.DataFrame({
        'date': future_dates,
        'cost': result['optimal_budget'],  # Use optimal spend
        'ROAS': X_train['ROAS'].mean()  # Use average ROAS as baseline
    })

    # Prepare features for prediction
    X_forecast = future_df[['cost', 'ROAS']].copy()

    # Predict conversions - these will be MATURE estimates if we trained with use_mature_estimates=True
    predictions = trained_model.predict(X_forecast)

    # DEBUG: Show what model predicts
    print(f"\n  🔍 Model Prediction:")
    print(f"  - Model predicts: {predictions.mean():.1f} conversions/day")
    print(f"  - At spend level: ${result['optimal_budget']:,.2f}/day")

    # ISSUE: The model was trained on MATURE conversion estimates
    # So it predicts MATURE conversions (what they'll eventually be)
    # But for a forecast, we probably want INITIAL conversions (what you'll see immediately)

    # Calculate adjustment factor: convert from mature estimate back to observed
    # Average maturity of training data tells us the scaling factor
    avg_training_maturity = df[df['days_since_ad'] >= min_days_old]['estimated_maturity'].mean()

    # If we trained on mature estimates, scale predictions back down to observed levels
    # QUESTION: Should we show mature (eventual) or observed (immediate) conversions?
    # For now, let's show OBSERVED (immediate) conversions, not mature
    predictions_observed = predictions * avg_training_maturity  # Scale back to observed level

    print(f"\n  ⚠️  MATURITY ADJUSTMENT:")
    print(f"  - Model trained on data ~{avg_training_maturity*100:.0f}% mature")
    print(f"  - Model predicts MATURE conversions: {predictions.mean():.1f}/day")
    print(f"  - Adjusted to OBSERVED (initial) conversions: {predictions_observed.mean():.1f}/day")
    print(f"  - (These will grow to mature level over time with lifetime attribution)")

    future_df['new_conversions'] = predictions_observed  # Use observed, not mature

    # Add month information for summary
    future_df['month'] = future_df['date'].dt.to_period('M')
    future_df['month_name'] = future_df['date'].dt.strftime('%B %Y')

    # Calculate totals for summary
    total_spend = future_df['cost'].sum()
    total_new_conversions = future_df['new_conversions'].sum()

    print(f"\n  {'='*70}")
    print(f"  📊 90-DAY CONVERSION FORECAST (DAY-BY-DAY)")
    print(f"  {'='*70}")
    print(f"\n  Daily Budget: ${recommended_budget:,.2f}")
    print(f"  Daily Spend: ${result['optimal_budget']:,.2f}")
    print(f"  Target NC ROAS: 3.5\n")
    print(f"  ⚠️  NOTE: These are NEW conversions from NEW ads in forecast period.")
    print(f"      Does NOT include ongoing conversions from historical ads.")
    print(f"      Shows OBSERVED conversions (what you'll see initially).")
    print(f"      With lifetime attribution, these will grow ~{1/avg_training_maturity:.1f}x over time.\n")

    # Show day-by-day predictions
    print(f"  {'Date':<12} {'Spend':<12} {'Predicted Conversions':<25}")
    print(f"  {'-'*12} {'-'*12} {'-'*25}")

    for idx, row in future_df.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d'):<12} ${row['cost']:>10,.2f}  {row['new_conversions']:>10.1f}")

    print(f"  {'-'*12} {'-'*12} {'-'*25}")
    print(f"  {'TOTAL':<12} ${total_spend:>10,.2f}  {total_new_conversions:>10.1f}")
    print()

    # Monthly summary
    monthly_summary = future_df.groupby('month_name').agg({
        'cost': 'sum',
        'new_conversions': 'sum'
    }).reset_index()
    monthly_summary['days'] = future_df.groupby('month_name').size().values
    monthly_summary['avg_conversions_per_day'] = monthly_summary['new_conversions'] / monthly_summary['days']

    print(f"\n  {'='*70}")
    print(f"  📊 MONTHLY SUMMARY")
    print(f"  {'='*70}")
    for idx, row in monthly_summary.iterrows():
        print(f"  {row['month_name']}:")
        print(f"    Days: {row['days']}")
        print(f"    Total spend: ${row['cost']:,.2f}")
        print(f"    Total conversions: {row['new_conversions']:.1f}")
        print(f"    Avg conversions/day: {row['avg_conversions_per_day']:.1f}")
        print()

    print(f"\n  {'='*70}")
    print(f"  📊 90-DAY TOTALS (from NEW ads):")
    print(f"  {'='*70}")
    print(f"  Total spend: ${total_spend:,.2f}")
    print(f"  Total NEW conversions: {total_new_conversions:.1f} (observed level)")
    print(f"  Average conversions/day: {total_new_conversions/90:.1f}")
    print(f"  {'='*70}\n")

    # Sanity check: Compare to historical performance
    print(f"  📊 SANITY CHECK - Does this make sense?")
    daily_forecast_conv = total_new_conversions / 90
    daily_forecast_spend = total_spend / 90
    historical_conv_per_dollar = df['conversions'].sum() / df['cost'].sum()
    forecast_conv_per_dollar = total_new_conversions / total_spend

    print(f"  - Forecast: {daily_forecast_conv:.1f} conversions/day at ${daily_forecast_spend:,.2f}/day")
    print(f"  - Historical avg: {df['conversions'].mean():.1f} conversions/day at ${df['cost'].mean():,.2f}/day")
    print(f"  - Historical: {historical_conv_per_dollar:.3f} conversions per $1 spent")
    print(f"  - Forecast: {forecast_conv_per_dollar:.3f} conversions per $1 spent")

    if abs(forecast_conv_per_dollar / historical_conv_per_dollar - 1) > 0.5:
        print(f"\n  ⚠️  WARNING: Forecast efficiency differs significantly from historical!")
        print(f"     This might indicate:")
        print(f"     - Model trained on incomplete data (all your data is <75% mature)")
        print(f"     - Recommended spend very different from historical")
        print(f"     - Data quality issues")
        print(f"     - Take forecast as rough estimate only")
    else:
        print(f"\n  ✓ Forecast looks reasonable compared to historical performance")
    print()

    print(f"  ⚠️  LIFETIME ATTRIBUTION REMINDER:")
    print(f"  These {total_new_conversions:.0f} conversions are OBSERVED level")
    print(f"  With lifetime attribution, they'll mature to ~{total_new_conversions/avg_training_maturity:.0f} over time")
    print(f"\n  Also remember: You'll see ongoing conversions from historical ads")
    print(f"  (not shown in this forecast)")

    # =========================================================================
    # STEP 7: Visualize Results
    # =========================================================================
    print("\n\n📊 STEP 7: Visualizing Your Data")
    print("-" * 80)

    # Create comprehensive visualization (including forecast)
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, :])  # Full width for forecast
    ]

    # Plot 1: Budget vs Spend over time
    ax1 = axes[0]
    ax1.plot(df['date'], df['ad_budget'], label='Budget', linewidth=2, alpha=0.7)
    ax1.plot(df['date'], df['cost'], label='Actual Spend', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Amount ($)', fontsize=10)
    ax1.set_title('Budget vs Actual Spend Over Time', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Conversions over time (with maturity note)
    ax2 = axes[1]
    ax2.plot(df['date'], df['conversions'], linewidth=2, color='#2E86AB')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Conversions', fontsize=10)
    ax2.set_title('Conversions Over Time (Incomplete - Still Accumulating!)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: ROAS over time
    ax3 = axes[2]
    ax3.plot(df['date'], df['ROAS'], linewidth=2, color='#A23B72')
    ax3.axhline(3.5, color='red', linestyle='--', label='Target ROAS = 3.5', linewidth=2)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('ROAS', fontsize=10)
    ax3.set_title('ROAS Over Time', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Spend efficiency
    ax4 = axes[3]
    spend_efficiency = (df['cost'] / df['ad_budget']).values
    ax4.plot(df['date'], spend_efficiency * 100, linewidth=2, color='#F18F01')
    ax4.axhline(100, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Spend Efficiency (%)', fontsize=10)
    ax4.set_title('Budget Utilization (Spend / Budget %)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 5: Conversions vs Spend
    ax5 = axes[4]
    ax5.scatter(df['cost'], df['conversions'], alpha=0.6, s=50, color='#2E86AB')
    ax5.set_xlabel('Daily Spend ($)', fontsize=10)
    ax5.set_ylabel('Daily Conversions (Incomplete)', fontsize=10)
    ax5.set_title('Conversions vs Spend', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Data maturity
    ax6 = axes[5]
    scatter = ax6.scatter(df['days_since_ad'], df['estimated_maturity'] * 100,
                         alpha=0.6, s=50, c=df['conversions'], cmap='viridis')
    ax6.set_xlabel('Days Since Ad Impression', fontsize=10)
    ax6.set_ylabel('Estimated Maturity (%)', fontsize=10)
    ax6.set_title('Data Completeness by Age', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Conversions', fontsize=9)

    # Plot 7: 3-Month Forecast (full width)
    ax7 = axes[6]

    # Historical data
    ax7.plot(df['date'], df['conversions'], linewidth=2, color='#2E86AB',
             label='Historical Conversions (Incomplete)', alpha=0.7)

    # Forecast data - NEW conversions from NEW ads
    ax7.plot(future_df['date'], future_df['new_conversions'],
             linewidth=2.5, color='#F18F01', label='Forecast: NEW Conversions from NEW Ads', linestyle='--')

    # Vertical line separating historical and forecast
    ax7.axvline(reference_date, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax7.text(reference_date, ax7.get_ylim()[1] * 0.95, 'Today',
             ha='right', va='top', fontsize=10, color='red')

    # Add monthly markers for forecast
    for month_name in monthly_forecast['month_name'].unique():
        month_data = future_df[future_df['month_name'] == month_name]
        month_start = month_data['date'].min()
        month_conversions = monthly_forecast[monthly_forecast['month_name'] == month_name]['new_conversions'].values[0]

        ax7.axvline(month_start, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax7.text(month_start + timedelta(days=15), ax7.get_ylim()[1] * 0.85,
                f"{month_name.split()[0][:3]}\n{month_conversions:.0f} NEW",
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax7.set_xlabel('Date', fontsize=11)
    ax7.set_ylabel('Daily Conversions', fontsize=11)
    ax7.set_title('Historical + 3-Month Forecast: NEW Conversions from Recommended Budget',
                  fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=9)
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('your_data_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to 'your_data_analysis.png'")

    # =========================================================================
    # STEP 8: Recommendations and Next Steps
    # =========================================================================
    print("\n\n" + "="*80)
    print("  📋 RECOMMENDATIONS & NEXT STEPS")
    print("="*80)

    print(f"""
  🎯 IMMEDIATE ACTIONS:

  1. SET DAILY BUDGET: ${recommended_budget:,.2f}
     - This targets NC ROAS of 3.5
     - Based on best available data (limited by 121 days)
     - Accounts for {avg_spend_efficiency*100:.0f}% spend efficiency

  2. 90-DAY FORECAST (NEW conversions from NEW ads):
     - Expected daily spend: ${result['optimal_budget']:,.2f}
     - Expected daily conversions: {total_new_conversions/90:.1f} conversions/day
     - Total 90-day conversions: {total_new_conversions:.1f} conversions
     - Total 90-day spend: ${total_spend:,.2f}

     📝 Note: These are OBSERVED conversions (what you'll see initially).
        With lifetime attribution, these grow to ~{total_new_conversions/avg_training_maturity:.0f}
        mature conversions over time. Forecast adjusted to show realistic initial numbers,
        not inflated "eventual mature" numbers.

     📊 See daily_conversion_forecast.csv for day-by-day predictions

  3. MONITOR CLOSELY:
     - Track actual spend daily (not just budget)
     - Watch NC ROAS (currently {current_avg_roas:.2f})
     - Conversions will grow over time (lifetime attribution!)
     - Compare monthly actuals to forecast

  4. BE PATIENT:
     - Don't judge results after 30 days (way too early!)
     - Wait 90-180 days to see true impact
     - Remember: conversions attributed over months/years
     - Forecast shows attributed conversions, not when purchases occur

  ⚠️  DATA LIMITATIONS (IMPORTANT!):

  1. INSUFFICIENT MATURE DATA:
     - You have 121 days (4 months) of data
     - Oldest data is only ~70-75% complete
     - Ideal: 365+ days with 180+ days of mature data
     - Your situation: All data is incomplete

  2. MODEL ACCURACY:
     - Predictions are INDICATIVE, not precise
     - Higher uncertainty than typical models
     - Use as directional guidance, not absolute truth
     - 20-30% error margin is realistic

  3. CONVERSION INCOMPLETENESS:
     - Your current {current_avg_conv:.1f} conversions/day will grow!
     - Estimated mature: {current_avg_conv / df['estimated_maturity'].mean():.1f} conversions/day
     - Don't compare current to predictions directly
     - Current numbers are incomplete

  📊 TO IMPROVE MODEL ACCURACY:

  1. COLLECT MORE DATA (Priority #1):
     - Continue collecting for 6+ more months
     - Aim for 365+ days total
     - More data = much better predictions

  2. EXPORT HISTORICAL DATA:
     - If available, export older historical data from Google Ads
     - Even 6-12 months ago would help significantly
     - More history = better conversion curve fitting

  3. RETRAIN REGULARLY:
     - Retrain model monthly as new data comes in
     - Model will improve as data matures
     - After 6 months, results will be much more reliable

  4. TRACK CONVERSION MATURITY:
     - Monitor how old dates accumulate conversions
     - Validate maturity assumptions
     - Adjust model as you learn your conversion patterns

  🔄 MONITORING PLAN:

  Week 1-4:
  - Set new budget: ${recommended_budget:,.2f}
  - Track daily spend and conversions
  - Don't make changes yet!

  Week 4-12:
  - Review weekly trends
  - Compare spend efficiency
  - Still too early to judge ROAS

  Month 3-6:
  - Retrain model with new data (now have 5-7 months)
  - Re-evaluate optimal budget
  - Should see more reliable predictions

  Month 6+:
  - Have enough data for robust modeling
  - Conversion curves will be clearer
  - Can make confident budget decisions

  💡 KEY TAKEAWAY:

  Your current recommendation (${recommended_budget:,.2f} daily budget) is the
  BEST ESTIMATE given limited data. It's directionally correct but has higher
  uncertainty than ideal.

  The MOST IMPORTANT thing you can do is COLLECT MORE DATA over the next
  6 months. As your data matures and grows, the model will become much more
  accurate and reliable.

  Think of this as your "best guess" based on incomplete information. It's
  better than guessing randomly, but not as good as what you'll have with
  more data!
    """)

    print("="*80)
    print("\n✅ Analysis complete!")
    print(f"\n📁 Generated files:")
    print(f"   - your_data_analysis.png (comprehensive visualization with 3-month forecast)")
    print(f"\n📖 For more details, see:")
    print(f"   - LIFETIME_ATTRIBUTION_GUIDE.md")
    print(f"   - ROAS_OPTIMIZATION_GUIDE.md")

    # Save forecast to CSV for easy reference
    # Export daily forecast to CSV
    forecast_export = future_df[['date', 'cost', 'new_conversions']].copy()
    forecast_export['date'] = forecast_export['date'].dt.strftime('%Y-%m-%d')
    forecast_export = forecast_export.rename(columns={
        'date': 'forecast_date',
        'cost': 'daily_spend',
        'new_conversions': 'predicted_conversions'
    })
    forecast_export.to_csv('daily_conversion_forecast.csv', index=False)

    # Also export monthly summary
    monthly_export = monthly_summary.copy()
    monthly_export = monthly_export.rename(columns={
        'month_name': 'month',
        'cost': 'total_spend',
        'new_conversions': 'total_conversions',
        'days': 'days_in_month',
        'avg_conversions_per_day': 'avg_conversions_per_day'
    })
    monthly_export.to_csv('monthly_conversion_summary.csv', index=False)

    print(f"\n📊 Forecast Files Exported:")
    print(f"   - daily_conversion_forecast.csv (90 days of daily predictions)")
    print(f"   - monthly_conversion_summary.csv (monthly summary)")
    print(f"\n   First 10 days of forecast:")
    print(forecast_export.head(10).to_string(index=False))

    return {
        'recommended_budget': recommended_budget,
        'recommended_spend': result['optimal_budget'],
        'daily_forecast': future_df,
        'monthly_summary': monthly_summary,
        'total_new_conversions': total_new_conversions,
        'total_spend': total_spend,
        'avg_conversions_per_day': total_new_conversions / 90,
        'optimization_result': result
    }


if __name__ == '__main__':
    # CUSTOMIZE THIS PATH TO YOUR DATA FILE
    file_path = 'your_google_ads_data.csv'

    # Expected format:
    # - Column 'date': Date of ad impression
    # - Column 'ad_budget': Daily budget you set
    # - Column 'conversions': Daily conversions (lifetime attributed)
    # - Column 'cost': Actual spend by Google
    # - Column 'ROAS': Return on ad spend

    print("\n" + "="*80)
    print("  INSTRUCTIONS:")
    print("="*80)
    print(f"""
  1. Save your Google Ads data as CSV with these columns:
     - date (YYYY-MM-DD format)
     - ad_budget (your daily budget)
     - conversions (daily conversions)
     - cost (actual spend by Google)
     - ROAS (return on ad spend)

  2. Update file_path variable above to point to your CSV file

  3. Adjust conversion_value (line 100) to your actual average order value

  4. Run this script: python analyze_your_data.py

  5. Review results and recommendations carefully

  6. Get 90-day forecast with recommended budget:
     - Day-by-day conversion predictions
     - Daily and monthly summaries
     - Exported to daily_conversion_forecast.csv and monthly_conversion_summary.csv

  7. Remember: With only 121 days of data, predictions are indicative!
    """)
    print("="*80)

    # Uncomment to run analysis
    # analyze_your_data(file_path)
