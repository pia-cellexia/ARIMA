"""
Example: ROAS-Constrained Budget Optimization

This example demonstrates how to find the optimal budget that:
1. Maximizes conversions (purchases)
2. Achieves a target NC ROAS of 3.5

Use Case:
---------
You have:
- Historical data on ad budget, conversions, and ROAS
- Complete control over ad budget
- A target NC ROAS of 3.5 that you need to achieve

You want to:
- Predict future conversions
- Find the ideal budget to maximize purchases while hitting your ROAS target
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from marketing_mix_model import MarketingMixModel
from roas_optimizer import ROASConstrainedOptimizer, optimize_multi_channel_with_roas
from data_generator import MarketingDataGenerator

sns.set_style('whitegrid')


def main_example():
    """
    Complete example: Find optimal budget for target ROAS of 3.5
    """
    print("\n" + "="*80)
    print("  ROAS-CONSTRAINED MARKETING BUDGET OPTIMIZATION")
    print("  Target: Maximize Conversions with NC ROAS = 3.5")
    print("="*80)

    # =========================================================================
    # STEP 1: Load or Generate Data
    # =========================================================================
    print("\n📊 STEP 1: Generating Sample Marketing Data")
    print("-" * 80)

    # Generate synthetic data (in practice, load your own CSV)
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(
        n_periods=365,  # 1 year of daily data
        budget_mean=5000,
        budget_std=1500,
        adstock_decay=0.6,
        saturation_lambda=3000
    )

    print(f"✓ Generated {len(df)} days of marketing data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n  Data summary:")
    print(df[['ad_budget', 'conversions', 'ROAS']].describe())

    # =========================================================================
    # STEP 2: Train Marketing Mix Model
    # =========================================================================
    print("\n\n🤖 STEP 2: Training Marketing Mix Model")
    print("-" * 80)

    # Split into train/test
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Prepare features
    feature_cols = ['ad_budget', 'ROAS']
    X_train = train_df[feature_cols]
    y_train = train_df['conversions'].values
    X_test = test_df[feature_cols]
    y_test = test_df['conversions'].values

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Initialize and train model
    model = MarketingMixModel(
        adstock_decay=0.5,
        saturation_lambda=1.0,
        use_saturation=True,
        use_adstock=True
    )

    print("\n  Training model with hyperparameter optimization...")
    model.fit(X_train, y_train, optimize_params=True)

    print(f"✓ Model trained successfully")
    print(f"  - Optimized adstock decay: {model.adstock_decay:.3f}")
    print(f"  - Optimized saturation lambda: {model.saturation_lambda:.3f}")

    # Evaluate model
    test_metrics = model.score(X_test, y_test)
    print(f"\n  Model Performance:")
    print(f"  - R² Score: {test_metrics['R2']:.4f}")
    print(f"  - RMSE: {test_metrics['RMSE']:.2f}")
    print(f"  - MAPE: {test_metrics['MAPE']:.2f}%")

    # =========================================================================
    # STEP 3: Initialize ROAS-Constrained Optimizer
    # =========================================================================
    print("\n\n🎯 STEP 3: Initialize ROAS-Constrained Optimizer")
    print("-" * 80)

    # Assume each conversion is worth $100 in revenue
    # (adjust this based on your actual conversion value)
    conversion_value = 100.0

    optimizer = ROASConstrainedOptimizer(
        model=model,
        conversion_value=conversion_value
    )

    print(f"✓ Optimizer initialized")
    print(f"  - Conversion value: ${conversion_value:.2f}")

    # =========================================================================
    # STEP 4: Find Optimal Budget for Target ROAS = 3.5
    # =========================================================================
    print("\n\n💰 STEP 4: Finding Optimal Budget for Target ROAS = 3.5")
    print("-" * 80)

    target_roas = 3.5
    budget_range = (1000, 20000)  # Search between $1K and $20K

    result = optimizer.optimize_for_target_roas(
        target_roas=target_roas,
        baseline_X=X_test,
        budget_column='ad_budget',
        budget_range=budget_range,
        tolerance=0.1
    )

    print(f"\n{'='*60}")
    print(f"  🎉 OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"  Target ROAS: {target_roas}")
    print(f"  Optimal Budget: ${result['optimal_budget']:,.2f}")
    print(f"  Predicted Conversions: {result['predicted_conversions']:.0f}")
    print(f"  Predicted ROAS: {result['predicted_roas']:.2f}")
    print(f"  Expected Revenue: ${result['revenue']:,.2f}")
    print(f"  Expected Profit: ${result['profit']:,.2f}")
    print(f"  Optimization Success: {result['optimization_success']}")
    print(f"{'='*60}\n")

    # =========================================================================
    # STEP 5: Analyze Budget Recommendations for Multiple ROAS Targets
    # =========================================================================
    print("\n📈 STEP 5: Budget Recommendations for Different ROAS Targets")
    print("-" * 80)

    roas_targets = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    recommendations = optimizer.get_budget_recommendations(
        baseline_X=X_test,
        budget_column='ad_budget',
        budget_range=budget_range,
        roas_targets=roas_targets
    )

    print("\n  Budget Recommendations Table:")
    print(recommendations.to_string(index=False))

    # =========================================================================
    # STEP 6: Visualize Optimization Landscape
    # =========================================================================
    print("\n\n📊 STEP 6: Visualizing Optimization Landscape")
    print("-" * 80)

    optimizer.plot_optimization_landscape(
        baseline_X=X_test,
        budget_column='ad_budget',
        budget_range=budget_range,
        target_roas=target_roas,
        optimal_budget=result['optimal_budget'],
        save_path='roas_optimization_landscape.png'
    )

    # =========================================================================
    # STEP 7: Analyze Trade-offs
    # =========================================================================
    print("\n\n⚖️  STEP 7: Analyzing Budget-ROAS-Conversions Trade-offs")
    print("-" * 80)

    tradeoff_df = optimizer.analyze_budget_roas_tradeoff(
        baseline_X=X_test,
        budget_column='ad_budget',
        budget_range=budget_range,
        n_points=50
    )

    print("\n  Sample of trade-off analysis:")
    sample_indices = [0, 12, 24, 36, 49]
    print(tradeoff_df.iloc[sample_indices].to_string(index=False))

    # Plot trade-off curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Budget vs ROAS with target line
    axes[0].plot(tradeoff_df['budget'], tradeoff_df['roas'],
                linewidth=2, color='#2E86AB')
    axes[0].axhline(target_roas, color='red', linestyle='--',
                   label=f'Target ROAS = {target_roas}', linewidth=2)
    axes[0].axvline(result['optimal_budget'], color='green', linestyle='--',
                   label=f"Optimal Budget = ${result['optimal_budget']:,.0f}",
                   linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Budget ($)', fontsize=12)
    axes[0].set_ylabel('ROAS', fontsize=12)
    axes[0].set_title('Budget vs ROAS Trade-off', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Conversions vs ROAS
    axes[1].scatter(tradeoff_df['conversions'], tradeoff_df['roas'],
                   c=tradeoff_df['budget'], cmap='viridis', s=50, alpha=0.7)
    axes[1].axhline(target_roas, color='red', linestyle='--',
                   label=f'Target ROAS = {target_roas}', linewidth=2)
    axes[1].scatter([result['predicted_conversions']], [result['predicted_roas']],
                   color='red', s=300, marker='*', edgecolors='black',
                   linewidth=2, label='Optimal Point', zorder=5)
    axes[1].set_xlabel('Conversions', fontsize=12)
    axes[1].set_ylabel('ROAS', fontsize=12)
    axes[1].set_title('Conversions vs ROAS (Efficiency Frontier)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Budget ($)')
    plt.tight_layout()
    plt.savefig('roas_tradeoff_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved trade-off analysis to 'roas_tradeoff_analysis.png'")

    # =========================================================================
    # STEP 8: What-If Scenarios
    # =========================================================================
    print("\n\n🔮 STEP 8: What-If Scenarios")
    print("-" * 80)

    scenarios = [
        ("Current average budget", X_test['ad_budget'].mean()),
        ("Optimal budget (ROAS=3.5)", result['optimal_budget']),
        ("Conservative budget", 3000),
        ("Aggressive budget", 10000),
    ]

    print("\n  Scenario Analysis:")
    for scenario_name, budget in scenarios:
        X_scenario = X_test.mean().to_frame().T
        X_scenario['ad_budget'] = budget

        conversions = model.predict(X_scenario)[0]
        revenue = conversions * conversion_value
        roas = revenue / budget
        profit = revenue - budget

        print(f"\n  {scenario_name}:")
        print(f"    Budget: ${budget:,.2f}")
        print(f"    Conversions: {conversions:.0f}")
        print(f"    ROAS: {roas:.2f}")
        print(f"    Revenue: ${revenue:,.2f}")
        print(f"    Profit: ${profit:,.2f}")

    # =========================================================================
    # STEP 9: Recommendations Summary
    # =========================================================================
    print("\n\n" + "="*80)
    print("  📋 FINAL RECOMMENDATIONS")
    print("="*80)

    print(f"""
  To achieve your target NC ROAS of 3.5 and maximize conversions:

  1. RECOMMENDED BUDGET: ${result['optimal_budget']:,.2f} per period

  2. EXPECTED OUTCOMES:
     - Conversions: {result['predicted_conversions']:.0f} purchases
     - Revenue: ${result['revenue']:,.2f}
     - ROAS: {result['predicted_roas']:.2f} (meets target of 3.5)
     - Profit: ${result['profit']:,.2f}

  3. KEY INSIGHTS:
     - Current average budget: ${X_test['ad_budget'].mean():,.2f}
     - Budget change needed: ${result['optimal_budget'] - X_test['ad_budget'].mean():+,.2f} ({((result['optimal_budget'] / X_test['ad_budget'].mean() - 1) * 100):+.1f}%)
     - This budget maximizes conversions while meeting your ROAS constraint

  4. NEXT STEPS:
     ✓ Review the optimization_landscape.png for visual insights
     ✓ Adjust conversion_value if your actual value differs from ${conversion_value}
     ✓ Monitor actual ROAS and adjust budget dynamically
     ✓ Consider testing different ROAS targets using the recommendations table
    """)

    print("="*80)
    print("\n✅ Analysis complete! Generated files:")
    print("   - roas_optimization_landscape.png")
    print("   - roas_tradeoff_analysis.png")


def multi_channel_example():
    """
    Example: Multi-channel budget optimization with ROAS constraint
    """
    print("\n\n" + "="*80)
    print("  MULTI-CHANNEL ROAS-CONSTRAINED OPTIMIZATION")
    print("="*80)

    # Generate multi-channel data
    generator = MarketingDataGenerator(seed=123)
    df = generator.generate_multi_channel_data(n_periods=365)

    print(f"\n✓ Generated {len(df)} days of multi-channel data")

    # Prepare data
    train_size = int(0.8 * len(df))
    budget_cols = ['search_budget', 'social_budget', 'display_budget']
    feature_cols = budget_cols + ['ROAS']

    X_train = df[:train_size][feature_cols]
    y_train = df[:train_size]['conversions'].values
    X_test = df[train_size:][feature_cols]

    # Train model
    print("\n🤖 Training multi-channel model...")
    model = MarketingMixModel(use_saturation=True, use_adstock=True)
    model.fit(X_train, y_train, optimize_params=True)

    print(f"✓ Model trained (R² = {model.score(X_test, df[train_size:]['conversions'].values)['R2']:.4f})")

    # Optimize with ROAS constraint
    print("\n💰 Optimizing budget allocation across channels...")
    print("   Total budget: $10,000")
    print("   Target ROAS: 3.5")

    optimal_allocation = optimize_multi_channel_with_roas(
        model=model,
        total_budget=10000,
        target_roas=3.5,
        baseline_X=X_test,
        budget_columns=budget_cols,
        conversion_value=100.0,
        tolerance=0.1
    )

    print("\n📊 Optimal Budget Allocation:")
    for channel, budget in optimal_allocation.items():
        pct = (budget / 10000) * 100
        print(f"   {channel.replace('_budget', '').capitalize()}: ${budget:,.2f} ({pct:.1f}%)")

    # Compare with current allocation
    current = X_test[budget_cols].mean()
    print("\n📊 Current Average Allocation:")
    for col in budget_cols:
        pct = (current[col] / current.sum()) * 100
        print(f"   {col.replace('_budget', '').capitalize()}: ${current[col]:,.2f} ({pct:.1f}%)")


if __name__ == '__main__':
    # Run main example
    main_example()

    # Optionally run multi-channel example
    # Uncomment to run:
    # multi_channel_example()
