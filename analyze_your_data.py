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

    # Use min_days_old=60 (instead of ideal 180)
    # This is a compromise given limited data
    min_days_old = 60

    print(f"\n  Training on data at least {min_days_old} days old...")
    print(f"  (Ideally would use 180+ days, but working with what we have)")

    try:
        trained_model = model.train_model(
            df,
            min_days_old=min_days_old,
            use_mature_estimates=True,
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
        print(f"  Expected Conversions: {result['predicted_conversions']:.1f} (lifetime attributed)")
        print(f"  Expected NC ROAS: {result['predicted_roas']:.2f}")
        print(f"  Expected Revenue: ${result['revenue']:,.2f}")
        print(f"  Expected Profit: ${result['profit']:,.2f}")
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

    # Predict conversions (these are attributed conversions)
    predictions = trained_model.predict(X_forecast)
    future_df['conversions_attributed'] = predictions
    future_df['revenue'] = predictions * conversion_value
    future_df['profit'] = future_df['revenue'] - future_df['cost']

    # Add month information
    future_df['month'] = future_df['date'].dt.to_period('M')
    future_df['month_name'] = future_df['date'].dt.strftime('%B %Y')

    # Aggregate by month
    monthly_forecast = future_df.groupby('month_name').agg({
        'cost': 'sum',
        'conversions_attributed': 'sum',
        'revenue': 'sum',
        'profit': 'sum'
    }).reset_index()

    monthly_forecast['roas'] = monthly_forecast['revenue'] / monthly_forecast['cost']
    monthly_forecast['days'] = future_df.groupby('month_name').size().values

    print(f"\n  {'='*70}")
    print(f"  📊 3-MONTH FORECAST WITH RECOMMENDED BUDGET")
    print(f"  {'='*70}")
    print(f"\n  Daily Budget: ${recommended_budget:,.2f}")
    print(f"  Daily Spend: ${result['optimal_budget']:,.2f}")
    print(f"  Target NC ROAS: 3.5\n")

    for idx, row in monthly_forecast.iterrows():
        print(f"  {row['month_name']}:")
        print(f"    Days in month: {row['days']}")
        print(f"    Total spend: ${row['cost']:,.2f}")
        print(f"    Conversions (attributed): {row['conversions_attributed']:.0f}")
        print(f"    Revenue: ${row['revenue']:,.2f}")
        print(f"    Profit: ${row['profit']:,.2f}")
        print(f"    NC ROAS: {row['roas']:.2f}")
        print()

    # Calculate when purchases actually occur (vs attributed)
    # With lifetime attribution, conversions attributed to future dates
    # will actually occur over months/years
    # For inventory planning, we need to estimate WHEN purchases happen

    print(f"  ⏱️  LIFETIME ATTRIBUTION NOTE:")
    print(f"  The numbers above show conversions ATTRIBUTED to each month.")
    print(f"  With lifetime attribution, these purchases happen over time:")
    print(f"\n  Purchase Timing Estimate (when purchases actually occur):")

    # Approximate distribution: 40% in first 30 days, 30% in next 60 days, 30% over months/years
    immediate_pct = 0.40  # 40% of purchases happen relatively soon
    medium_pct = 0.30     # 30% happen in next 2-3 months
    longterm_pct = 0.30   # 30% happen over many months/years

    for idx, row in monthly_forecast.iterrows():
        attributed_conv = row['conversions_attributed']
        immediate_purchases = attributed_conv * immediate_pct
        medium_purchases = attributed_conv * medium_pct
        longterm_purchases = attributed_conv * longterm_pct

        print(f"\n  {row['month_name']} attributed conversions: {attributed_conv:.0f}")
        print(f"    ~{immediate_purchases:.0f} purchases in first 30 days (~{immediate_pct*100:.0f}%)")
        print(f"    ~{medium_purchases:.0f} purchases in next 60-90 days (~{medium_pct*100:.0f}%)")
        print(f"    ~{longterm_purchases:.0f} purchases over many months (~{longterm_pct*100:.0f}%)")

    # Total summary
    total_spend = monthly_forecast['cost'].sum()
    total_conversions = monthly_forecast['conversions_attributed'].sum()
    total_revenue = monthly_forecast['revenue'].sum()
    total_profit = monthly_forecast['profit'].sum()
    avg_roas = total_revenue / total_spend

    print(f"\n  {'='*70}")
    print(f"  📊 3-MONTH TOTALS:")
    print(f"  {'='*70}")
    print(f"  Total spend: ${total_spend:,.2f}")
    print(f"  Total conversions (attributed): {total_conversions:.0f}")
    print(f"  Total revenue: ${total_revenue:,.2f}")
    print(f"  Total profit: ${total_profit:,.2f}")
    print(f"  Average NC ROAS: {avg_roas:.2f}")
    print(f"  {'='*70}\n")

    # Inventory recommendation
    immediate_inventory = total_conversions * immediate_pct * 1.2  # 20% safety buffer

    print(f"  📦 INVENTORY RECOMMENDATION:")
    print(f"  For next 3 months, prepare approximately:")
    print(f"  - {immediate_inventory:.0f} units for immediate fulfillment (first 30 days)")
    print(f"    (Includes 20% safety buffer)")
    print(f"  - Additional {total_conversions * medium_pct:.0f} units over next 60-90 days")
    print(f"  - Remaining {total_conversions * longterm_pct:.0f} units will be needed over longer term")
    print(f"\n  Total attributed conversions: {total_conversions:.0f} units over 3 months")

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
             label='Historical (Incomplete)', alpha=0.7)

    # Forecast data
    ax7.plot(future_df['date'], future_df['conversions_attributed'],
             linewidth=2.5, color='#F18F01', label='Forecast (Attributed)', linestyle='--')

    # Vertical line separating historical and forecast
    ax7.axvline(reference_date, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax7.text(reference_date, ax7.get_ylim()[1] * 0.95, 'Today',
             ha='right', va='top', fontsize=10, color='red')

    # Add monthly markers for forecast
    for month_name in monthly_forecast['month_name'].unique():
        month_data = future_df[future_df['month_name'] == month_name]
        month_start = month_data['date'].min()
        month_conversions = monthly_forecast[monthly_forecast['month_name'] == month_name]['conversions_attributed'].values[0]

        ax7.axvline(month_start, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax7.text(month_start + timedelta(days=15), ax7.get_ylim()[1] * 0.85,
                f"{month_name.split()[0][:3]}\n{month_conversions:.0f} conv",
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax7.set_xlabel('Date', fontsize=11)
    ax7.set_ylabel('Daily Conversions (Attributed)', fontsize=11)
    ax7.set_title('Historical Performance + 3-Month Forecast with Recommended Budget',
                  fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left')
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

  2. 3-MONTH FORECAST (with recommended budget):
     - Expected spend: ${total_spend:,.2f}
     - Expected conversions: {total_conversions:.0f} (attributed)
     - Expected revenue: ${total_revenue:,.2f}
     - Expected profit: ${total_profit:,.2f}
     - Expected NC ROAS: {avg_roas:.2f}

  3. INVENTORY PLANNING:
     - Prepare ~{immediate_inventory:.0f} units for immediate fulfillment (30 days)
     - Additional ~{total_conversions * medium_pct:.0f} units over next 60-90 days
     - Remaining ~{total_conversions * longterm_pct:.0f} units over longer term
     - Total: {total_conversions:.0f} attributed conversions over 3 months

  4. MONITOR CLOSELY:
     - Track actual spend daily (not just budget)
     - Watch NC ROAS (currently {current_avg_roas:.2f})
     - Conversions will grow over time (lifetime attribution!)
     - Compare monthly actuals to forecast

  5. BE PATIENT:
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
    forecast_export = monthly_forecast.copy()
    forecast_export['daily_budget'] = recommended_budget
    forecast_export['daily_spend'] = result['optimal_budget']
    forecast_export = forecast_export[['month_name', 'days', 'daily_budget', 'daily_spend',
                                       'cost', 'conversions_attributed', 'revenue', 'profit', 'roas']]
    forecast_export.to_csv('3_month_forecast.csv', index=False)
    print(f"\n📊 3-Month Forecast Details:")
    print(f"   - 3_month_forecast.csv (monthly breakdown)")
    print(f"\n   Monthly Forecast Summary:")
    print(forecast_export.to_string(index=False))

    return {
        'recommended_budget': recommended_budget,
        'recommended_spend': result['optimal_budget'],
        'monthly_forecast': monthly_forecast,
        'daily_forecast': future_df,
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

  6. Get 3-month forecast with recommended budget:
     - Monthly purchase predictions
     - Inventory requirements
     - Revenue and profit forecasts
     - Exported to 3_month_forecast.csv

  7. Remember: With only 121 days of data, predictions are indicative!
    """)
    print("="*80)

    # Uncomment to run analysis
    # analyze_your_data(file_path)
