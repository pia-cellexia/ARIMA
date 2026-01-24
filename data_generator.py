"""
Data Generator for Marketing Mix Model

Generates synthetic marketing data with realistic causal relationships,
including adstock effects, saturation, and seasonality.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict


class MarketingDataGenerator:
    """
    Generates synthetic marketing data for testing and demonstration.

    The data includes:
    - Ad budget (controlled variable)
    - Conversions (target, with causal relationship to budget)
    - ROAS (derived from conversions and budget)
    - Seasonality effects
    - Noise and randomness
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the data generator.

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_data(
        self,
        n_periods: int = 365,
        start_date: str = '2023-01-01',
        base_conversions: float = 100.0,
        budget_mean: float = 5000.0,
        budget_std: float = 1500.0,
        adstock_decay: float = 0.6,
        saturation_lambda: float = 3000.0,
        seasonality_amplitude: float = 0.3,
        noise_std: float = 15.0,
        conversion_value: float = 50.0
    ) -> pd.DataFrame:
        """
        Generate synthetic marketing data.

        Parameters:
        -----------
        n_periods : int, default=365
            Number of time periods (days)
        start_date : str, default='2023-01-01'
            Start date for the time series
        base_conversions : float, default=100.0
            Baseline number of conversions without marketing
        budget_mean : float, default=5000.0
            Mean ad budget
        budget_std : float, default=1500.0
            Standard deviation of ad budget
        adstock_decay : float, default=0.6
            Adstock decay rate (carryover effect)
        saturation_lambda : float, default=3000.0
            Saturation parameter (diminishing returns)
        seasonality_amplitude : float, default=0.3
            Amplitude of seasonal fluctuations (0-1)
        noise_std : float, default=15.0
            Standard deviation of noise in conversions
        conversion_value : float, default=50.0
            Value per conversion (for calculating ROAS)

        Returns:
        --------
        pd.DataFrame
            Generated marketing data with columns:
            - date: Date
            - ad_budget: Ad budget spent
            - conversions: Number of conversions (target variable)
            - ROAS: Return on ad spend
            - day_of_week: Day of week (0=Monday)
            - week_of_year: Week of year
            - month: Month
        """
        # Generate dates
        start = pd.to_datetime(start_date)
        dates = [start + timedelta(days=i) for i in range(n_periods)]

        # Generate ad budget (with some trends and variation)
        trend = np.linspace(0, 0.5, n_periods)  # Slight upward trend
        budget_noise = np.random.normal(0, budget_std, n_periods)
        ad_budget = budget_mean * (1 + trend) + budget_noise
        ad_budget = np.maximum(ad_budget, 0)  # No negative budget

        # Add weekly patterns (higher budget on weekdays)
        weekday = np.array([d.weekday() for d in dates])
        weekend_effect = np.where(weekday >= 5, 0.7, 1.0)  # 30% less on weekends
        ad_budget = ad_budget * weekend_effect

        # Apply adstock transformation to budget
        adstocked_budget = self._adstock_transform(ad_budget, adstock_decay)

        # Apply saturation transformation (diminishing returns)
        saturated_budget = self._saturation_transform(adstocked_budget, saturation_lambda)

        # Generate seasonality (annual cycle)
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * day_of_year / 365.25)

        # Generate conversions with causal relationship to budget
        # conversions = base + effect_of_budget + seasonality + noise
        budget_effect = saturated_budget * 0.05  # Each unit of saturated budget adds conversions

        conversions = (
            base_conversions * (1 + seasonality) +
            budget_effect +
            np.random.normal(0, noise_std, n_periods)
        )
        conversions = np.maximum(conversions, 0)  # No negative conversions

        # Calculate ROAS
        revenue = conversions * conversion_value
        roas = revenue / (ad_budget + 1e-10)

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'ad_budget': ad_budget,
            'conversions': conversions,
            'ROAS': roas,
            'day_of_week': weekday,
            'week_of_year': [d.isocalendar()[1] for d in dates],
            'month': [d.month for d in dates]
        })

        return df

    def generate_multi_channel_data(
        self,
        n_periods: int = 365,
        start_date: str = '2023-01-01',
        channels: Optional[Dict[str, Dict]] = None,
        base_conversions: float = 100.0,
        seasonality_amplitude: float = 0.3,
        noise_std: float = 15.0,
        conversion_value: float = 50.0
    ) -> pd.DataFrame:
        """
        Generate multi-channel marketing data.

        Parameters:
        -----------
        n_periods : int, default=365
            Number of time periods
        start_date : str, default='2023-01-01'
            Start date
        channels : Dict[str, Dict], optional
            Dictionary of channel configurations. If None, uses default channels.
            Format: {
                'channel_name': {
                    'budget_mean': float,
                    'budget_std': float,
                    'adstock_decay': float,
                    'saturation_lambda': float,
                    'effectiveness': float  # Conversion rate per unit
                }
            }
        base_conversions : float, default=100.0
            Baseline conversions
        seasonality_amplitude : float, default=0.3
            Seasonal effect amplitude
        noise_std : float, default=15.0
            Noise standard deviation
        conversion_value : float, default=50.0
            Value per conversion

        Returns:
        --------
        pd.DataFrame
            Multi-channel marketing data
        """
        if channels is None:
            channels = {
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

        # Generate dates
        start = pd.to_datetime(start_date)
        dates = [start + timedelta(days=i) for i in range(n_periods)]

        # Initialize DataFrame
        df = pd.DataFrame({'date': dates})

        # Generate seasonality
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * day_of_year / 365.25)

        # Generate budget and effects for each channel
        total_effect = 0

        for channel_name, config in channels.items():
            # Generate budget
            budget = np.random.normal(
                config['budget_mean'],
                config['budget_std'],
                n_periods
            )
            budget = np.maximum(budget, 0)

            # Apply adstock
            adstocked = self._adstock_transform(budget, config['adstock_decay'])

            # Apply saturation
            saturated = self._saturation_transform(adstocked, config['saturation_lambda'])

            # Calculate effect
            effect = saturated * config['effectiveness']
            total_effect += effect

            # Add to DataFrame
            df[f'{channel_name}_budget'] = budget

        # Generate conversions
        conversions = (
            base_conversions * (1 + seasonality) +
            total_effect +
            np.random.normal(0, noise_std, n_periods)
        )
        conversions = np.maximum(conversions, 0)

        df['conversions'] = conversions

        # Calculate total budget and ROAS
        budget_cols = [col for col in df.columns if 'budget' in col]
        df['total_budget'] = df[budget_cols].sum(axis=1)

        revenue = conversions * conversion_value
        df['ROAS'] = revenue / (df['total_budget'] + 1e-10)

        # Add time features
        df['day_of_week'] = [d.weekday() for d in dates]
        df['week_of_year'] = [d.isocalendar()[1] for d in dates]
        df['month'] = [d.month for d in dates]

        return df

    @staticmethod
    def _adstock_transform(x: np.ndarray, decay: float) -> np.ndarray:
        """Apply adstock transformation."""
        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]

        for t in range(1, len(x)):
            adstocked[t] = x[t] + decay * adstocked[t-1]

        return adstocked

    @staticmethod
    def _saturation_transform(x: np.ndarray, lambda_: float) -> np.ndarray:
        """Apply saturation transformation."""
        return x / (lambda_ + x) * lambda_


def main():
    """Example usage of the data generator."""
    generator = MarketingDataGenerator(seed=42)

    # Generate single-channel data
    print("Generating single-channel data...")
    df_single = generator.generate_data(n_periods=365)
    print(f"\nGenerated {len(df_single)} days of data")
    print("\nFirst few rows:")
    print(df_single.head(10))
    print("\nSummary statistics:")
    print(df_single.describe())

    # Save to CSV
    df_single.to_csv('marketing_data_single_channel.csv', index=False)
    print("\nSaved to 'marketing_data_single_channel.csv'")

    # Generate multi-channel data
    print("\n" + "="*60)
    print("Generating multi-channel data...")
    df_multi = generator.generate_multi_channel_data(n_periods=365)
    print(f"\nGenerated {len(df_multi)} days of multi-channel data")
    print("\nFirst few rows:")
    print(df_multi.head(10))
    print("\nSummary statistics:")
    print(df_multi.describe())

    # Save to CSV
    df_multi.to_csv('marketing_data_multi_channel.csv', index=False)
    print("\nSaved to 'marketing_data_multi_channel.csv'")


if __name__ == '__main__':
    main()
