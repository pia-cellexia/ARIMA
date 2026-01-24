"""
Example Usage of Causal Marketing Mix Model

This script demonstrates how to:
1. Generate synthetic marketing data
2. Train a marketing mix model
3. Make predictions
4. Estimate causal effects
5. Optimize budget allocation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from marketing_mix_model import MarketingMixModel, calculate_roi, calculate_roas
from data_generator import MarketingDataGenerator

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def example_1_single_channel():
    """
    Example 1: Single-channel marketing mix model.

    This example shows how to predict conversions based on ad budget
    when you have data on budget, conversions, and ROAS.
    """
    print("="*70)
    print("EXAMPLE 1: Single-Channel Marketing Mix Model")
    print("="*70)

    # Step 1: Generate or load data
    print("\n1. Generating sample data...")
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(
        n_periods=365,
        budget_mean=5000,
        budget_std=1500,
        adstock_decay=0.6,
        saturation_lambda=3000
    )

    print(f"   Generated {len(df)} days of data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n   Sample data:")
    print(df.head())

    # Step 2: Prepare data for modeling
    print("\n2. Preparing data...")

    # Split into train and test (80/20)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Prepare features and target
    # Features: ad_budget, ROAS, and optionally time features
    feature_cols = ['ad_budget', 'ROAS']
    X_train = train_df[feature_cols]
    y_train = train_df['conversions'].values
    X_test = test_df[feature_cols]
    y_test = test_df['conversions'].values

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Step 3: Train the model
    print("\n3. Training Marketing Mix Model...")

    # Initialize model with adstock and saturation effects
    model = MarketingMixModel(
        adstock_decay=0.5,  # Start with initial guess
        saturation_lambda=1.0,
        use_saturation=True,
        use_adstock=True
    )

    # Fit model (with hyperparameter optimization)
    model.fit(X_train, y_train, optimize_params=True)

    print(f"   Optimized adstock decay: {model.adstock_decay:.3f}")
    print(f"   Optimized saturation lambda: {model.saturation_lambda:.3f}")

    # Step 4: Evaluate model performance
    print("\n4. Evaluating model performance...")

    train_metrics = model.score(X_train, y_train)
    test_metrics = model.score(X_test, y_test)

    print("\n   Training metrics:")
    for metric, value in train_metrics.items():
        print(f"   - {metric}: {value:.4f}")

    print("\n   Test metrics:")
    for metric, value in test_metrics.items():
        print(f"   - {metric}: {value:.4f}")

    # Step 5: Make predictions
    print("\n5. Making predictions...")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Plot predictions vs actuals
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Conversions')
    plt.ylabel('Predicted Conversions')
    plt.title('Training Set: Predicted vs Actual')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Conversions')
    plt.ylabel('Predicted Conversions')
    plt.title('Test Set: Predicted vs Actual')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 'predictions_vs_actual.png'")

    # Step 6: Feature importance
    print("\n6. Analyzing feature importance...")

    importance = model.get_feature_importance()
    print("\n   Feature importance:")
    print(importance)

    # Step 7: Estimate causal effects
    print("\n7. Estimating causal effects...")

    # What happens if we increase budget by $1000?
    effect_increase = model.estimate_causal_effect(
        X_test,
        treatment_col='ad_budget',
        treatment_change=1000
    )

    print("\n   Effect of increasing ad budget by $1000:")
    print(f"   - Mean effect on conversions: {effect_increase['mean_effect']:.2f}")
    print(f"   - Total effect: {effect_increase['total_effect']:.2f}")
    print(f"   - Range: [{effect_increase['min_effect']:.2f}, {effect_increase['max_effect']:.2f}]")

    # What happens if we decrease budget by $500?
    effect_decrease = model.estimate_causal_effect(
        X_test,
        treatment_col='ad_budget',
        treatment_change=-500
    )

    print("\n   Effect of decreasing ad budget by $500:")
    print(f"   - Mean effect on conversions: {effect_decrease['mean_effect']:.2f}")
    print(f"   - Total effect: {effect_decrease['total_effect']:.2f}")

    # Step 8: Visualize the response curve
    print("\n8. Visualizing budget response curve...")

    budget_range = np.linspace(1000, 10000, 50)
    X_test_sample = X_test.iloc[[0]].copy()  # Take one sample

    conversions_at_budgets = []
    for budget in budget_range:
        X_test_sample['ad_budget'] = budget
        conv = model.predict(X_test_sample)[0]
        conversions_at_budgets.append(conv)

    plt.figure(figsize=(10, 6))
    plt.plot(budget_range, conversions_at_budgets, linewidth=2)
    plt.xlabel('Ad Budget ($)')
    plt.ylabel('Predicted Conversions')
    plt.title('Budget Response Curve (with Adstock & Saturation)')
    plt.grid(True, alpha=0.3)
    plt.savefig('budget_response_curve.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 'budget_response_curve.png'")

    return model, df


def example_2_multi_channel():
    """
    Example 2: Multi-channel marketing mix model.

    Shows how to handle multiple marketing channels and optimize
    budget allocation across them.
    """
    print("\n\n")
    print("="*70)
    print("EXAMPLE 2: Multi-Channel Marketing Mix Model")
    print("="*70)

    # Step 1: Generate multi-channel data
    print("\n1. Generating multi-channel data...")

    generator = MarketingDataGenerator(seed=123)
    df = generator.generate_multi_channel_data(
        n_periods=365,
        channels={
            'search': {
                'budget_mean': 3000,
                'budget_std': 800,
                'adstock_decay': 0.3,
                'saturation_lambda': 2000,
                'effectiveness': 0.06
            },
            'social': {
                'budget_mean': 2500,
                'budget_std': 700,
                'adstock_decay': 0.5,
                'saturation_lambda': 3000,
                'effectiveness': 0.04
            },
            'display': {
                'budget_mean': 2000,
                'budget_std': 600,
                'adstock_decay': 0.7,
                'saturation_lambda': 2500,
                'effectiveness': 0.03
            }
        }
    )

    print(f"   Generated {len(df)} days of data")
    print(f"\n   Sample data:")
    print(df.head())

    # Step 2: Prepare data
    print("\n2. Preparing multi-channel data...")

    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Features: all budget columns and ROAS
    budget_cols = ['search_budget', 'social_budget', 'display_budget']
    feature_cols = budget_cols + ['ROAS']

    X_train = train_df[feature_cols]
    y_train = train_df['conversions'].values
    X_test = test_df[feature_cols]
    y_test = test_df['conversions'].values

    # Step 3: Train model
    print("\n3. Training multi-channel model...")

    model = MarketingMixModel(
        adstock_decay=0.5,
        saturation_lambda=1.0,
        use_saturation=True,
        use_adstock=True
    )

    model.fit(X_train, y_train, optimize_params=True)

    print(f"   Optimized adstock decay: {model.adstock_decay:.3f}")
    print(f"   Optimized saturation lambda: {model.saturation_lambda:.3f}")

    # Step 4: Evaluate
    print("\n4. Evaluating model...")

    test_metrics = model.score(X_test, y_test)
    print("\n   Test metrics:")
    for metric, value in test_metrics.items():
        print(f"   - {metric}: {value:.4f}")

    # Step 5: Feature importance
    print("\n5. Channel effectiveness...")

    importance = model.get_feature_importance()
    print("\n   Channel importance:")
    print(importance)

    # Visualize
    plt.figure(figsize=(10, 6))
    channel_importance = importance[importance['feature'].str.contains('budget')]
    plt.barh(channel_importance['feature'], channel_importance['coefficient'])
    plt.xlabel('Coefficient (Contribution to Conversions)')
    plt.title('Channel Effectiveness Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('channel_effectiveness.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 'channel_effectiveness.png'")

    # Step 6: Optimize budget allocation
    print("\n6. Optimizing budget allocation...")

    total_budget = 7500  # $7,500 total budget
    baseline_X = X_test.mean().to_frame().T

    optimal_allocation = model.optimize_budget_allocation(
        total_budget=total_budget,
        baseline_X=baseline_X,
        budget_columns=budget_cols
    )

    print(f"\n   Optimal allocation for ${total_budget:,.0f} budget:")
    for channel, budget in optimal_allocation.items():
        channel_name = channel.replace('_budget', '')
        pct = (budget / total_budget) * 100
        print(f"   - {channel_name.capitalize()}: ${budget:,.2f} ({pct:.1f}%)")

    # Compare with current allocation
    current_allocation = X_test[budget_cols].mean()
    current_total = current_allocation.sum()

    print(f"\n   Current average allocation (${current_total:,.0f}):")
    for col in budget_cols:
        channel_name = col.replace('_budget', '')
        pct = (current_allocation[col] / current_total) * 100
        print(f"   - {channel_name.capitalize()}: ${current_allocation[col]:,.2f} ({pct:.1f}%)")

    # Visualize allocation comparison
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    channels_names = [col.replace('_budget', '').capitalize() for col in budget_cols]
    current_vals = [current_allocation[col] for col in budget_cols]
    optimal_vals = [optimal_allocation[col] for col in budget_cols]

    x = np.arange(len(channels_names))
    width = 0.35

    ax[0].bar(x - width/2, current_vals, width, label='Current', alpha=0.8)
    ax[0].bar(x + width/2, optimal_vals, width, label='Optimal', alpha=0.8)
    ax[0].set_xlabel('Channel')
    ax[0].set_ylabel('Budget ($)')
    ax[0].set_title('Budget Allocation Comparison')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(channels_names)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Percentage comparison
    current_pcts = [v/sum(current_vals)*100 for v in current_vals]
    optimal_pcts = [v/sum(optimal_vals)*100 for v in optimal_vals]

    ax[1].bar(x - width/2, current_pcts, width, label='Current', alpha=0.8)
    ax[1].bar(x + width/2, optimal_pcts, width, label='Optimal', alpha=0.8)
    ax[1].set_xlabel('Channel')
    ax[1].set_ylabel('Budget Share (%)')
    ax[1].set_title('Budget Allocation % Comparison')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(channels_names)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('budget_optimization.png', dpi=150, bbox_inches='tight')
    print("\n   Saved plot to 'budget_optimization.png'")

    return model, df


def example_3_causal_inference():
    """
    Example 3: Causal inference and what-if scenarios.

    Demonstrates how to use the model for causal reasoning
    and answering what-if questions.
    """
    print("\n\n")
    print("="*70)
    print("EXAMPLE 3: Causal Inference & What-If Scenarios")
    print("="*70)

    # Generate data
    generator = MarketingDataGenerator(seed=999)
    df = generator.generate_data(n_periods=180)

    # Prepare data
    train_size = int(0.8 * len(df))
    X_train = df[:train_size][['ad_budget', 'ROAS']]
    y_train = df[:train_size]['conversions'].values
    X_test = df[train_size:][['ad_budget', 'ROAS']]
    y_test = df[train_size:]['conversions'].values

    # Train model
    model = MarketingMixModel(use_saturation=True, use_adstock=True)
    model.fit(X_train, y_train, optimize_params=True)

    print("\n1. What-If Scenarios:")
    print("   " + "-"*50)

    scenarios = [
        ("Double the budget", 2.0),
        ("50% increase", 0.5),
        ("25% increase", 0.25),
        ("No change", 0.0),
        ("25% decrease", -0.25),
        ("50% decrease", -0.5),
    ]

    results = []

    for scenario_name, multiplier in scenarios:
        current_budget = X_test['ad_budget'].mean()
        change = current_budget * multiplier

        effect = model.estimate_causal_effect(
            X_test,
            treatment_col='ad_budget',
            treatment_change=change
        )

        results.append({
            'Scenario': scenario_name,
            'Budget Change': f"${change:,.0f}",
            'Mean Effect': f"{effect['mean_effect']:.2f}",
            'Total Effect': f"{effect['total_effect']:.2f}"
        })

        print(f"\n   {scenario_name}:")
        print(f"   - Budget change: ${change:,.0f}")
        print(f"   - Expected conversion change: {effect['mean_effect']:.2f} per period")
        print(f"   - Total conversion change: {effect['total_effect']:.2f}")

    # Create results table
    results_df = pd.DataFrame(results)
    print("\n\n2. Summary Table:")
    print(results_df.to_string(index=False))

    return model


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "CAUSAL MARKETING MIX MODEL - EXAMPLES" + " "*21 + "║")
    print("╚" + "="*68 + "╝")

    # Run Example 1
    model1, df1 = example_1_single_channel()

    # Run Example 2
    model2, df2 = example_2_multi_channel()

    # Run Example 3
    model3 = example_3_causal_inference()

    print("\n\n")
    print("="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - predictions_vs_actual.png")
    print("  - budget_response_curve.png")
    print("  - channel_effectiveness.png")
    print("  - budget_optimization.png")
    print("\nYou can now use the MarketingMixModel class in your own projects!")


if __name__ == '__main__':
    main()
