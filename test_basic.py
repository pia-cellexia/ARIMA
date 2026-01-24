"""
Basic test script to verify the model works correctly.
"""

import numpy as np
import pandas as pd
from marketing_mix_model import MarketingMixModel
from data_generator import MarketingDataGenerator

def test_data_generator():
    """Test that data generator works."""
    print("Testing data generator...")
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(n_periods=100)

    assert len(df) == 100
    assert 'ad_budget' in df.columns
    assert 'conversions' in df.columns
    assert 'ROAS' in df.columns
    assert df['conversions'].min() >= 0
    assert df['ad_budget'].min() >= 0

    print("✓ Data generator works!")
    return df

def test_model_fit():
    """Test that model can fit and predict."""
    print("\nTesting model fit and predict...")

    # Generate data
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(n_periods=200)

    # Prepare data
    X = df[['ad_budget', 'ROAS']]
    y = df['conversions'].values

    # Train/test split
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Fit model
    model = MarketingMixModel(
        adstock_decay=0.5,
        saturation_lambda=1.0,
        use_saturation=True,
        use_adstock=True
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert not np.any(np.isnan(y_pred))

    # Score
    metrics = model.score(X_test, y_test)
    assert 'R2' in metrics
    assert 'RMSE' in metrics

    print(f"✓ Model trained successfully!")
    print(f"  R² Score: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    return model, X_test

def test_causal_effect():
    """Test causal effect estimation."""
    print("\nTesting causal effect estimation...")

    # Generate data and fit model
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_data(n_periods=200)
    X = df[['ad_budget', 'ROAS']]
    y = df['conversions'].values

    model = MarketingMixModel()
    model.fit(X, y)

    # Estimate effect
    effect = model.estimate_causal_effect(
        X.tail(50),
        treatment_col='ad_budget',
        treatment_change=1000
    )

    assert 'mean_effect' in effect
    assert 'total_effect' in effect
    assert not np.isnan(effect['mean_effect'])

    print(f"✓ Causal effect estimation works!")
    print(f"  Effect of +$1000 budget: {effect['mean_effect']:.2f} conversions")

    return effect

def test_multi_channel():
    """Test multi-channel data generation."""
    print("\nTesting multi-channel data generation...")

    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_multi_channel_data(n_periods=100)

    assert 'search_budget' in df.columns
    assert 'social_budget' in df.columns
    assert 'display_budget' in df.columns
    assert 'conversions' in df.columns

    print("✓ Multi-channel data generation works!")
    print(f"  Channels: {[col for col in df.columns if 'budget' in col]}")

    return df

def main():
    """Run all tests."""
    print("="*60)
    print("Running Basic Tests")
    print("="*60)

    try:
        # Test 1: Data generator
        df = test_data_generator()

        # Test 2: Model fit and predict
        model, X_test = test_model_fit()

        # Test 3: Causal effect
        effect = test_causal_effect()

        # Test 4: Multi-channel
        df_multi = test_multi_channel()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe model is working correctly. You can now:")
        print("1. Run 'python example_usage.py' for full examples")
        print("2. Use the MarketingMixModel in your own code")
        print("3. Generate data with 'python data_generator.py'")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise

if __name__ == '__main__':
    main()
