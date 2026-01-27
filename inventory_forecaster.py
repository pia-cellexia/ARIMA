"""
Attribution-Aware Inventory Forecasting for Google Ads Campaigns

This module handles the unique attribution behavior of Google Ads where conversions
are back-attributed to the ad impression date, not the purchase date.

Example:
--------
If a user sees an ad on January 1 and purchases on February 13, the conversion
is attributed to January 1 in Google Ads reporting.

This is critical for:
1. Inventory forecasting (knowing future demand)
2. Budget planning (accounting for delayed conversions)
3. ROAS optimization (understanding true campaign performance)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from marketing_mix_model import MarketingMixModel
from roas_optimizer import ROASConstrainedOptimizer

sns.set_style('whitegrid')


class AttributionAwareForecaster:
    """
    Forecaster that handles Google Ads attribution windows for inventory planning.

    Google Ads attributes conversions to the ad impression date, not the purchase date.
    This means:
    - Recent dates have incomplete conversion data (some purchases haven't occurred yet)
    - Historical dates have mature/complete conversion data
    - Forecasting must account for attribution lag
    """

    def __init__(
        self,
        model: MarketingMixModel,
        attribution_window_days: int = 30,
        conversion_value: float = 1.0
    ):
        """
        Initialize the attribution-aware forecaster.

        Parameters:
        -----------
        model : MarketingMixModel
            Fitted marketing mix model
        attribution_window_days : int, default=30
            Attribution window in days (how far back conversions are attributed)
            Common values: 30, 60, 90 days
        conversion_value : float, default=1.0
            Revenue value per conversion
        """
        if not model.is_fitted_:
            raise ValueError("Model must be fitted before forecasting")

        self.model = model
        self.attribution_window_days = attribution_window_days
        self.conversion_value = conversion_value

    def calculate_conversion_maturity(
        self,
        dates: pd.Series,
        reference_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Calculate what percentage of conversions are likely complete for each date.

        Recent dates have lower maturity (conversions still coming in).
        Dates older than attribution window have 100% maturity.

        Parameters:
        -----------
        dates : pd.Series
            Date column
        reference_date : datetime, optional
            Reference date (today). If None, uses max date in data

        Returns:
        --------
        pd.Series
            Maturity percentage (0-1) for each date
        """
        if reference_date is None:
            reference_date = pd.to_datetime(dates).max()
        else:
            reference_date = pd.to_datetime(reference_date)

        dates = pd.to_datetime(dates)
        days_since = (reference_date - dates).dt.days

        # Linear maturity model: 0% at day 0, 100% at attribution_window_days
        maturity = np.minimum(days_since / self.attribution_window_days, 1.0)

        return pd.Series(maturity, index=dates.index)

    def adjust_for_incomplete_conversions(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        conversion_column: str = 'conversions',
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Adjust recent conversion data for incomplete attribution.

        Parameters:
        -----------
        df : pd.DataFrame
            Historical data with dates and conversions
        date_column : str
            Name of date column
        conversion_column : str
            Name of conversion column
        reference_date : datetime, optional
            Reference date (today)

        Returns:
        --------
        pd.DataFrame
            DataFrame with additional columns:
            - maturity: conversion maturity (0-1)
            - conversions_adjusted: conversions adjusted for maturity
            - conversions_expected: expected final conversions
        """
        df = df.copy()

        # Calculate maturity
        df['maturity'] = self.calculate_conversion_maturity(
            df[date_column],
            reference_date
        )

        # Adjusted conversions (what we've observed)
        df['conversions_adjusted'] = df[conversion_column]

        # Expected final conversions (accounting for incomplete data)
        df['conversions_expected'] = np.where(
            df['maturity'] > 0,
            df[conversion_column] / df['maturity'],
            df[conversion_column]
        )

        # For recent dates (low maturity), expected conversions is an estimate
        # For old dates (high maturity), expected = observed

        return df

    def forecast_conversions(
        self,
        future_budgets: pd.DataFrame,
        baseline_features: Optional[pd.DataFrame] = None,
        budget_column: str = 'ad_budget',
        include_attribution_lag: bool = True
    ) -> pd.DataFrame:
        """
        Forecast conversions for future dates given planned budgets.

        Parameters:
        -----------
        future_budgets : pd.DataFrame
            Future dates and planned budgets
            Must have columns: date, ad_budget (or custom budget_column)
        baseline_features : pd.DataFrame, optional
            Baseline values for other features (e.g., ROAS)
        budget_column : str
            Name of budget column
        include_attribution_lag : bool
            If True, distributes conversions across attribution window

        Returns:
        --------
        pd.DataFrame
            Forecast with columns:
            - date
            - ad_budget
            - conversions_predicted (attributed to this date)
            - conversions_realized (when purchases actually occur)
            - revenue_predicted
            - roas_predicted
        """
        future_budgets = future_budgets.copy()

        # Ensure date column exists
        if 'date' not in future_budgets.columns:
            raise ValueError("future_budgets must have 'date' column")

        # Prepare features for prediction
        if baseline_features is None:
            # Use simple baseline (you might want to customize this)
            X_forecast = future_budgets[[budget_column]]
        else:
            X_forecast = future_budgets.copy()
            # Add baseline features if missing
            for col in baseline_features.columns:
                if col not in X_forecast.columns and col != budget_column:
                    X_forecast[col] = baseline_features[col].mean()

        # Ensure correct column order matching training data
        if hasattr(self.model, 'feature_names_'):
            feature_cols = [col for col in self.model.feature_names_ if col in X_forecast.columns]
            X_forecast = X_forecast[feature_cols]

        # Predict conversions
        conversions_predicted = self.model.predict(X_forecast)

        # Build forecast dataframe
        forecast = future_budgets[['date', budget_column]].copy()
        forecast['conversions_predicted'] = conversions_predicted
        forecast['revenue_predicted'] = conversions_predicted * self.conversion_value
        forecast['roas_predicted'] = forecast['revenue_predicted'] / forecast[budget_column]

        if include_attribution_lag:
            # Distribute conversions across attribution window
            # (conversions attributed to date X actually occur over next N days)
            forecast['conversions_realized'] = self._distribute_conversions_over_window(
                forecast['conversions_predicted']
            )
        else:
            forecast['conversions_realized'] = forecast['conversions_predicted']

        return forecast

    def _distribute_conversions_over_window(
        self,
        conversions: pd.Series,
        decay_rate: float = 0.5
    ) -> pd.Series:
        """
        Distribute conversions over attribution window with decay.

        Most conversions happen soon after ad view, fewer later.
        Uses exponential decay to model this.
        """
        # Simple approximation: assume conversions follow exponential decay
        # In practice, you'd calibrate this to your actual conversion delay distribution
        return conversions * (1 - decay_rate)

    def forecast_inventory_needs(
        self,
        future_budgets: pd.DataFrame,
        baseline_features: Optional[pd.DataFrame] = None,
        budget_column: str = 'ad_budget',
        safety_stock_pct: float = 0.2
    ) -> pd.DataFrame:
        """
        Forecast inventory requirements based on predicted conversions.

        Parameters:
        -----------
        future_budgets : pd.DataFrame
            Future dates and planned budgets
        baseline_features : pd.DataFrame, optional
            Baseline feature values
        budget_column : str
            Budget column name
        safety_stock_pct : float, default=0.2
            Safety stock percentage (e.g., 0.2 = 20% buffer)

        Returns:
        --------
        pd.DataFrame
            Inventory forecast with:
            - date
            - conversions_forecast
            - inventory_needed (with safety stock)
            - cumulative_inventory_needed
        """
        forecast = self.forecast_conversions(
            future_budgets,
            baseline_features,
            budget_column,
            include_attribution_lag=False  # Use attributed conversions
        )

        # Add safety stock
        forecast['conversions_forecast'] = forecast['conversions_predicted']
        forecast['safety_stock'] = forecast['conversions_forecast'] * safety_stock_pct
        forecast['inventory_needed'] = forecast['conversions_forecast'] + forecast['safety_stock']

        # Cumulative inventory needed
        forecast['cumulative_inventory_needed'] = forecast['inventory_needed'].cumsum()

        # Round up for inventory
        forecast['inventory_needed'] = np.ceil(forecast['inventory_needed'])
        forecast['cumulative_inventory_needed'] = np.ceil(forecast['cumulative_inventory_needed'])

        return forecast

    def optimize_budget_for_inventory_target(
        self,
        target_inventory: int,
        forecast_period_days: int,
        target_roas: float,
        baseline_features: pd.DataFrame,
        budget_column: str = 'ad_budget',
        budget_range: Tuple[float, float] = (1000, 50000),
        start_date: Optional[datetime] = None
    ) -> Dict:
        """
        Find optimal daily budget to achieve inventory target with target ROAS.

        Parameters:
        -----------
        target_inventory : int
            Target number of conversions (inventory units) needed
        forecast_period_days : int
            Number of days in forecast period
        target_roas : float
            Target ROAS to maintain
        baseline_features : pd.DataFrame
            Baseline feature values
        budget_column : str
            Budget column name
        budget_range : Tuple[float, float]
            (min, max) daily budget to consider
        start_date : datetime, optional
            Start date for forecast

        Returns:
        --------
        Dict
            Optimal budget plan with:
            - daily_budget: recommended daily budget
            - total_budget: total budget for period
            - expected_conversions: expected total conversions
            - expected_roas: expected ROAS
            - meets_inventory_target: bool
            - meets_roas_target: bool
            - forecast: detailed daily forecast
        """
        if start_date is None:
            start_date = datetime.now()

        # Create date range
        dates = pd.date_range(start=start_date, periods=forecast_period_days, freq='D')

        # Binary search for optimal daily budget
        min_budget, max_budget = budget_range
        tolerance = target_inventory * 0.05  # 5% tolerance

        best_result = None

        for _ in range(20):  # Max iterations
            test_budget = (min_budget + max_budget) / 2

            # Create budget plan
            future_budgets = pd.DataFrame({
                'date': dates,
                budget_column: test_budget
            })

            # Forecast
            forecast = self.forecast_inventory_needs(
                future_budgets,
                baseline_features,
                budget_column
            )

            total_conversions = forecast['conversions_forecast'].sum()
            avg_roas = forecast['roas_predicted'].mean()

            # Check if targets are met
            meets_inventory = total_conversions >= (target_inventory - tolerance)
            meets_roas = avg_roas >= (target_roas - 0.1)

            result = {
                'daily_budget': test_budget,
                'total_budget': test_budget * forecast_period_days,
                'expected_conversions': total_conversions,
                'expected_roas': avg_roas,
                'meets_inventory_target': meets_inventory,
                'meets_roas_target': meets_roas,
                'forecast': forecast
            }

            if meets_inventory and meets_roas:
                best_result = result
                # Try lower budget
                max_budget = test_budget
            else:
                # Need higher budget
                min_budget = test_budget

            # Check if we've converged
            if abs(total_conversions - target_inventory) < tolerance:
                break

        if best_result is None:
            # Use the last result even if targets not met
            best_result = result

        return best_result

    def plot_inventory_forecast(
        self,
        forecast: pd.DataFrame,
        target_inventory: Optional[int] = None,
        save_path: str = 'inventory_forecast.png'
    ) -> None:
        """
        Visualize inventory forecast.

        Parameters:
        -----------
        forecast : pd.DataFrame
            Forecast dataframe from forecast_inventory_needs()
        target_inventory : int, optional
            Target inventory line to draw
        save_path : str
            Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Daily conversions
        ax1 = axes[0, 0]
        ax1.bar(range(len(forecast)), forecast['conversions_forecast'],
               color='#2E86AB', alpha=0.7, label='Forecast Conversions')
        if 'safety_stock' in forecast.columns:
            ax1.bar(range(len(forecast)), forecast['safety_stock'],
                   bottom=forecast['conversions_forecast'],
                   color='#F18F01', alpha=0.5, label='Safety Stock')
        ax1.set_xlabel('Days', fontsize=11)
        ax1.set_ylabel('Conversions', fontsize=11)
        ax1.set_title('Daily Conversion Forecast', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative inventory
        ax2 = axes[0, 1]
        ax2.plot(range(len(forecast)), forecast['cumulative_inventory_needed'],
                linewidth=2, color='#2E86AB', marker='o', markersize=4,
                label='Cumulative Inventory Needed')
        if target_inventory is not None:
            ax2.axhline(target_inventory, color='red', linestyle='--',
                       linewidth=2, label=f'Target: {target_inventory}')
        ax2.set_xlabel('Days', fontsize=11)
        ax2.set_ylabel('Units', fontsize=11)
        ax2.set_title('Cumulative Inventory Needs', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Daily budget
        ax3 = axes[1, 0]
        ax3.bar(range(len(forecast)), forecast['ad_budget'],
               color='#A23B72', alpha=0.7)
        ax3.set_xlabel('Days', fontsize=11)
        ax3.set_ylabel('Budget ($)', fontsize=11)
        ax3.set_title('Daily Budget Plan', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Daily ROAS
        ax4 = axes[1, 1]
        ax4.plot(range(len(forecast)), forecast['roas_predicted'],
                linewidth=2, color='#00A896', marker='o', markersize=4)
        ax4.axhline(forecast['roas_predicted'].mean(), color='red',
                   linestyle='--', linewidth=1.5,
                   label=f"Avg ROAS: {forecast['roas_predicted'].mean():.2f}")
        ax4.set_xlabel('Days', fontsize=11)
        ax4.set_ylabel('ROAS', fontsize=11)
        ax4.set_title('Daily ROAS Forecast', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved inventory forecast to '{save_path}'")


def create_scenario_comparison(
    forecaster: AttributionAwareForecaster,
    scenarios: List[Dict],
    baseline_features: pd.DataFrame,
    forecast_period_days: int = 30,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Compare multiple budget scenarios for inventory planning.

    Parameters:
    -----------
    forecaster : AttributionAwareForecaster
        Fitted forecaster
    scenarios : List[Dict]
        List of scenarios, each with:
        - name: str
        - daily_budget: float
    baseline_features : pd.DataFrame
        Baseline feature values
    forecast_period_days : int
        Forecast period in days
    start_date : datetime, optional
        Start date

    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics for each scenario
    """
    if start_date is None:
        start_date = datetime.now()

    dates = pd.date_range(start=start_date, periods=forecast_period_days, freq='D')

    results = []

    for scenario in scenarios:
        name = scenario['name']
        daily_budget = scenario['daily_budget']

        future_budgets = pd.DataFrame({
            'date': dates,
            'ad_budget': daily_budget
        })

        forecast = forecaster.forecast_inventory_needs(
            future_budgets,
            baseline_features
        )

        results.append({
            'scenario': name,
            'daily_budget': daily_budget,
            'total_budget': daily_budget * forecast_period_days,
            'total_conversions': forecast['conversions_forecast'].sum(),
            'total_inventory_needed': forecast['inventory_needed'].sum(),
            'avg_roas': forecast['roas_predicted'].mean(),
            'total_revenue': forecast['revenue_predicted'].sum(),
            'total_profit': forecast['revenue_predicted'].sum() - (daily_budget * forecast_period_days)
        })

    return pd.DataFrame(results)
