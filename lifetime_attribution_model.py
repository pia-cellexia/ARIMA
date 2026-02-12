"""
Lifetime Attribution Marketing Mix Model for Google Ads

This module handles Google Ads campaigns with LIFETIME attribution windows, where
conversions can be attributed years after the initial ad impression.

Key Differences from Standard Attribution:
------------------------------------------
- Standard: 30-90 day attribution window
- Lifetime: Conversions attributed forever (e.g., ad shown Jan 2023, purchase Dec 2024 → attributed to Jan 2023)

This creates unique challenges:
1. Recent data is EXTREMELY incomplete (conversions still coming in for years)
2. Need cohort-based modeling to understand conversion curves over time
3. Only very old data has "mature" conversion counts
4. Forecasting requires predicting conversion timing, not just total conversions

Budget vs Actual Spend:
-----------------------
- **Budget**: What you SET in Google Ads (your control variable)
- **Actual Spend**: What Google actually SPENT (can differ due to competition, bid adjustments)

This model can work with:
- Budget only (if actual spend not available)
- Budget + Actual Spend (for more accurate modeling)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from marketing_mix_model import MarketingMixModel
from roas_optimizer import ROASConstrainedOptimizer

sns.set_style('whitegrid')


class LifetimeAttributionModel:
    """
    Marketing mix model for lifetime attribution windows.

    Handles the unique challenges of lifetime attribution by:
    1. Modeling conversion curves over time (cohort-based approach)
    2. Estimating conversion maturity for recent dates
    3. Forecasting using historical conversion patterns
    4. Supporting both budget-only and budget+spend modeling
    """

    def __init__(
        self,
        conversion_value: float = 1.0,
        has_actual_spend: bool = False
    ):
        """
        Initialize lifetime attribution model.

        Parameters:
        -----------
        conversion_value : float
            Revenue per conversion
        has_actual_spend : bool
            Whether actual spend data is available (vs budget only)
        """
        self.conversion_value = conversion_value
        self.has_actual_spend = has_actual_spend
        self.model = None
        self.conversion_curve_params = None

    def fit_conversion_curve(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        conversion_column: str = 'conversions',
        min_days_old: int = 365,
        curve_type: str = 'power_law'
    ) -> Dict[str, float]:
        """
        Fit a conversion curve to understand how conversions accumulate over time.

        With lifetime attribution, conversions continue to be attributed long after
        the ad impression. This function models that curve.

        Parameters:
        -----------
        df : pd.DataFrame
            Historical data with dates and conversions
        date_column : str
            Name of date column
        conversion_column : str
            Name of conversion column
        min_days_old : int
            Minimum age (days) of data to use for curve fitting
            Older data is more "mature" (more complete conversions)
        curve_type : str
            Type of curve: 'power_law', 'exponential', or 'logistic'

        Returns:
        --------
        Dict[str, float]
            Fitted curve parameters
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        reference_date = df['date'].max()

        # Calculate days since ad for each record
        df['days_since_ad'] = (reference_date - df['date']).dt.days

        # Group by days_since_ad and sum conversions
        # This shows average conversion count by age of ad impression
        curve_data = df.groupby('days_since_ad')[conversion_column].mean().reset_index()
        curve_data = curve_data[curve_data['days_since_ad'] >= 30]  # Need some minimum age
        curve_data = curve_data.sort_values('days_since_ad')

        days = curve_data['days_since_ad'].values
        conversions = curve_data[conversion_column].values

        # Fit curve based on type
        if curve_type == 'power_law':
            # y = a * x^b (common for long-tail processes)
            def func(x, a, b):
                return a * np.power(x, b)

            try:
                params, _ = curve_fit(func, days, conversions, p0=[1.0, 0.5], maxfev=5000)
                self.conversion_curve_params = {'type': 'power_law', 'a': params[0], 'b': params[1]}
            except:
                # Fallback to simple linear
                self.conversion_curve_params = {'type': 'linear', 'slope': conversions.mean() / days.mean()}

        elif curve_type == 'exponential':
            # y = a * (1 - exp(-b * x)) (saturation curve)
            def func(x, a, b):
                return a * (1 - np.exp(-b * x))

            try:
                params, _ = curve_fit(func, days, conversions, p0=[conversions.max(), 0.01], maxfev=5000)
                self.conversion_curve_params = {'type': 'exponential', 'a': params[0], 'b': params[1]}
            except:
                self.conversion_curve_params = {'type': 'linear', 'slope': conversions.mean() / days.mean()}

        else:  # logistic
            # y = L / (1 + exp(-k*(x-x0))) (S-curve)
            def func(x, L, k, x0):
                return L / (1 + np.exp(-k * (x - x0)))

            try:
                params, _ = curve_fit(func, days, conversions,
                                     p0=[conversions.max(), 0.01, days.mean()],
                                     maxfev=5000)
                self.conversion_curve_params = {'type': 'logistic', 'L': params[0], 'k': params[1], 'x0': params[2]}
            except:
                self.conversion_curve_params = {'type': 'linear', 'slope': conversions.mean() / days.mean()}

        return self.conversion_curve_params

    def estimate_mature_conversions(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        conversion_column: str = 'conversions',
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Estimate mature (final) conversion counts for recent dates.

        With lifetime attribution, recent dates have very incomplete conversion data.
        This function estimates what the final conversion count will be based on
        the fitted conversion curve.

        Parameters:
        -----------
        df : pd.DataFrame
            Data with dates and observed conversions
        date_column : str
            Date column name
        conversion_column : str
            Conversion column name
        reference_date : datetime, optional
            Reference date (today)

        Returns:
        --------
        pd.DataFrame
            Data with additional columns:
            - days_since_ad: age of ad impression
            - maturity_factor: estimated % of final conversions observed
            - conversions_mature_estimate: estimated final conversion count
        """
        if self.conversion_curve_params is None:
            raise ValueError("Must fit conversion curve first using fit_conversion_curve()")

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        if reference_date is None:
            reference_date = df['date'].max()
        else:
            reference_date = pd.to_datetime(reference_date)

        # Days since ad impression
        df['days_since_ad'] = (reference_date - df['date']).dt.days

        # Estimate maturity factor based on conversion curve
        # Maturity = observed / expected_at_maturity
        # For lifetime attribution, we assume maturity approaches 1.0 after ~2-3 years

        params = self.conversion_curve_params
        curve_type = params['type']

        if curve_type == 'power_law':
            # Reference: conversions at 1000 days (very mature)
            mature_days = 1000
            current_expected = params['a'] * np.power(df['days_since_ad'], params['b'])
            mature_expected = params['a'] * np.power(mature_days, params['b'])
            df['maturity_factor'] = current_expected / mature_expected

        elif curve_type == 'exponential':
            mature_days = 1000
            current_expected = params['a'] * (1 - np.exp(-params['b'] * df['days_since_ad']))
            mature_expected = params['a'] * (1 - np.exp(-params['b'] * mature_days))
            df['maturity_factor'] = current_expected / mature_expected

        elif curve_type == 'logistic':
            current_expected = params['L'] / (1 + np.exp(-params['k'] * (df['days_since_ad'] - params['x0'])))
            df['maturity_factor'] = current_expected / params['L']

        else:  # linear fallback
            # Simple heuristic: linear growth up to 1000 days
            df['maturity_factor'] = np.minimum(df['days_since_ad'] / 1000, 1.0)

        # Clip maturity factor to [0, 1]
        df['maturity_factor'] = df['maturity_factor'].clip(0, 1)

        # Estimate mature conversions
        df['conversions_mature_estimate'] = np.where(
            df['maturity_factor'] > 0.01,
            df[conversion_column] / df['maturity_factor'],
            df[conversion_column]
        )

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        min_days_old: int = 180,
        use_mature_estimates: bool = True,
        budget_column: str = 'ad_budget',
        spend_column: Optional[str] = 'actual_spend',
        conversion_column: str = 'conversions',
        roas_column: str = 'ROAS'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for lifetime attribution model.

        Key decisions:
        1. How old should data be to be considered "complete enough"?
        2. Should we use mature estimates for slightly recent data?
        3. Should we use budget or actual spend as predictor?

        Parameters:
        -----------
        df : pd.DataFrame
            Historical data
        min_days_old : int
            Minimum age (days) for data to be included
            Higher = more complete conversions, less data
            Lower = more data, less complete conversions
        use_mature_estimates : bool
            If True, uses estimated mature conversions for recent-ish data
            If False, only uses very old data with actual mature conversions
        budget_column : str
            Budget column name
        spend_column : str, optional
            Actual spend column (if available)
        conversion_column : str
            Conversion column name
        roas_column : str
            ROAS column name

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (Features, Target)
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        reference_date = df['date'].max()

        # Calculate age
        df['days_since_ad'] = (reference_date - df['date']).dt.days

        # Filter by minimum age
        df_training = df[df['days_since_ad'] >= min_days_old].copy()

        if len(df_training) < 30:
            raise ValueError(f"Not enough mature data (only {len(df_training)} records). Lower min_days_old or collect more historical data.")

        # Prepare features
        if self.has_actual_spend and spend_column and spend_column in df.columns:
            # Use actual spend if available
            feature_cols = [spend_column, roas_column]
            X = df_training[feature_cols]
        else:
            # Use budget
            feature_cols = [budget_column, roas_column]
            X = df_training[feature_cols]

        # Prepare target
        if use_mature_estimates and self.conversion_curve_params is not None:
            # Use estimated mature conversions
            df_training = self.estimate_mature_conversions(
                df_training,
                conversion_column=conversion_column,
                reference_date=reference_date
            )
            y = df_training['conversions_mature_estimate'].values
        else:
            # Use observed conversions (assumes data is old enough to be mostly complete)
            y = df_training[conversion_column].values

        return X, y

    def train_model(
        self,
        df: pd.DataFrame,
        min_days_old: int = 180,
        use_mature_estimates: bool = True,
        **kwargs
    ) -> MarketingMixModel:
        """
        Train marketing mix model on lifetime attribution data.

        Parameters:
        -----------
        df : pd.DataFrame
            Historical data
        min_days_old : int
            Minimum age for training data
        use_mature_estimates : bool
            Use estimated mature conversions
        **kwargs : dict
            Additional arguments passed to MarketingMixModel

        Returns:
        --------
        MarketingMixModel
            Trained model
        """
        # Fit conversion curve first
        print(f"  Fitting conversion curve on data {min_days_old}+ days old...")
        self.fit_conversion_curve(df, min_days_old=min_days_old)
        print(f"  ✓ Conversion curve fitted: {self.conversion_curve_params}")

        # Prepare training data
        print(f"\n  Preparing training data (min age: {min_days_old} days)...")
        X, y = self.prepare_training_data(df, min_days_old=min_days_old, use_mature_estimates=use_mature_estimates)
        print(f"  ✓ Training samples: {len(X)}")

        # Train model
        print(f"\n  Training marketing mix model...")
        self.model = MarketingMixModel(
            adstock_decay=kwargs.get('adstock_decay', 0.7),  # Higher for lifetime attribution
            saturation_lambda=kwargs.get('saturation_lambda', 1.0),
            use_saturation=kwargs.get('use_saturation', True),
            use_adstock=kwargs.get('use_adstock', True)
        )

        self.model.fit(X, y, optimize_params=kwargs.get('optimize_params', True))

        return self.model

    def forecast_with_cohort_adjustment(
        self,
        future_budgets: pd.DataFrame,
        baseline_features: pd.DataFrame,
        forecast_horizon_days: int = 30,
        conversion_realization_days: int = 90,
        budget_column: str = 'ad_budget'
    ) -> pd.DataFrame:
        """
        Forecast conversions accounting for lifetime attribution.

        Key insight: With lifetime attribution, when you spend budget on Day 1:
        - Some conversions attributed to Day 1 happen immediately
        - More conversions attributed to Day 1 happen over next weeks/months/years
        - For inventory planning, need to know WHEN purchases actually occur

        Parameters:
        -----------
        future_budgets : pd.DataFrame
            Future dates and planned budgets
        baseline_features : pd.DataFrame
            Baseline feature values
        forecast_horizon_days : int
            How far to forecast (days)
        conversion_realization_days : int
            Time window to consider for actual purchase timing
            E.g., 90 days = consider purchases in next 90 days
        budget_column : str
            Budget column name

        Returns:
        --------
        pd.DataFrame
            Forecast with:
            - conversions_attributed: conversions attributed to this date (Google Ads view)
            - conversions_realized_day_X: when purchases actually occur
            - inventory_needed_by_day: when inventory is actually needed
        """
        if self.model is None:
            raise ValueError("Must train model first")

        future_budgets = future_budgets.copy()

        # Prepare features for prediction
        X_forecast = future_budgets[[budget_column]].copy()
        for col in baseline_features.columns:
            if col not in X_forecast.columns and col != budget_column:
                X_forecast[col] = baseline_features[col].mean()

        # Ensure correct column order
        if hasattr(self.model, 'feature_names_'):
            feature_cols = [col for col in self.model.feature_names_ if col in X_forecast.columns]
            X_forecast = X_forecast[feature_cols]

        # Predict conversions (these are attributed conversions, not realized purchases)
        conversions_attributed = self.model.predict(X_forecast)

        forecast = future_budgets[['date', budget_column]].copy()
        forecast['conversions_attributed'] = conversions_attributed
        forecast['revenue_total'] = conversions_attributed * self.conversion_value
        forecast['roas'] = forecast['revenue_total'] / forecast[budget_column]

        # Distribute conversions over realization period
        # Use conversion curve to estimate timing
        if self.conversion_curve_params is not None:
            # Create distribution of when purchases actually occur
            # For simplicity, use exponential decay (most purchases soon after ad)
            decay_rate = 0.05  # 5% daily decay (calibrate to your data)
            days_array = np.arange(conversion_realization_days)
            purchase_timing_dist = np.exp(-decay_rate * days_array)
            purchase_timing_dist = purchase_timing_dist / purchase_timing_dist.sum()

            # Store daily distribution
            for day_offset in range(min(30, conversion_realization_days)):
                col_name = f'purchases_day_{day_offset}'
                forecast[col_name] = conversions_attributed * purchase_timing_dist[day_offset]

        return forecast


def compare_budget_vs_spend_effectiveness(
    df: pd.DataFrame,
    budget_column: str = 'ad_budget',
    spend_column: str = 'actual_spend',
    conversion_column: str = 'conversions',
    conversion_value: float = 100.0
) -> pd.DataFrame:
    """
    Compare effectiveness of budget (planned) vs actual spend.

    Useful when you have both budget and actual spend data to understand:
    - How efficiently is budget being used?
    - Is there a budget ceiling effect?
    - Should you increase/decrease budget?

    Parameters:
    -----------
    df : pd.DataFrame
        Data with budget, actual spend, and conversions
    budget_column : str
        Budget column
    spend_column : str
        Actual spend column
    conversion_column : str
        Conversions column
    conversion_value : float
        Revenue per conversion

    Returns:
    --------
    pd.DataFrame
        Analysis showing budget utilization and effectiveness
    """
    df = df.copy()

    # Budget utilization
    df['spend_rate'] = df[spend_column] / df[budget_column]
    df['underspend'] = np.maximum(0, df[budget_column] - df[spend_column])
    df['overspend'] = np.maximum(0, df[spend_column] - df[budget_column])

    # Effectiveness metrics
    df['conversions_per_spend'] = df[conversion_column] / df[spend_column]
    df['conversions_per_budget'] = df[conversion_column] / df[budget_column]
    df['revenue'] = df[conversion_column] * conversion_value
    df['roas_on_spend'] = df['revenue'] / df[spend_column]
    df['roas_on_budget'] = df['revenue'] / df[budget_column]

    # Summary
    summary = pd.DataFrame({
        'metric': [
            'Avg Budget',
            'Avg Spend',
            'Avg Spend Rate',
            'Days Underspent',
            'Days Overspent',
            'Avg Conversions per $1 Spend',
            'Avg ROAS on Spend',
            'Avg ROAS on Budget'
        ],
        'value': [
            df[budget_column].mean(),
            df[spend_column].mean(),
            df['spend_rate'].mean(),
            (df['underspend'] > 0).sum(),
            (df['overspend'] > 0).sum(),
            df['conversions_per_spend'].mean(),
            df['roas_on_spend'].mean(),
            df['roas_on_budget'].mean()
        ]
    })

    return summary
