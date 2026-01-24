"""
ROAS-Constrained Marketing Mix Model

This module extends the base MarketingMixModel with optimization capabilities
for finding the ideal budget that maximizes conversions while achieving a target ROAS.

Use Case:
---------
You have complete control over ad budget and want to:
1. Maximize purchases (conversions)
2. Achieve a specific NC ROAS target (e.g., 3.5)

The optimizer finds the optimal budget allocation that satisfies both objectives.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from marketing_mix_model import MarketingMixModel, calculate_roas

sns.set_style('whitegrid')


class ROASConstrainedOptimizer:
    """
    Optimizer for finding optimal budget allocation given ROAS constraints.

    This class works with a trained MarketingMixModel to find the budget
    that maximizes conversions while meeting a target ROAS threshold.
    """

    def __init__(
        self,
        model: MarketingMixModel,
        conversion_value: float = 1.0
    ):
        """
        Initialize the ROAS-constrained optimizer.

        Parameters:
        -----------
        model : MarketingMixModel
            A fitted marketing mix model
        conversion_value : float, default=1.0
            Revenue value per conversion (used to calculate ROAS)
        """
        if not model.is_fitted_:
            raise ValueError("Model must be fitted before optimization")

        self.model = model
        self.conversion_value = conversion_value

    def optimize_for_target_roas(
        self,
        target_roas: float,
        baseline_X: pd.DataFrame,
        budget_column: str = 'ad_budget',
        budget_range: Tuple[float, float] = (100, 50000),
        other_features: Optional[Dict[str, float]] = None,
        tolerance: float = 0.1
    ) -> Dict[str, float]:
        """
        Find the optimal budget that maximizes conversions while achieving target ROAS.

        Parameters:
        -----------
        target_roas : float
            Target ROAS to achieve (e.g., 3.5 means $3.50 revenue per $1 spend)
        baseline_X : pd.DataFrame
            Baseline feature values (for non-budget features)
        budget_column : str, default='ad_budget'
            Name of the budget column to optimize
        budget_range : Tuple[float, float], default=(100, 50000)
            (min, max) budget to consider
        other_features : Dict[str, float], optional
            Fixed values for other features. If None, uses baseline_X means
        tolerance : float, default=0.1
            ROAS tolerance (target_roas ± tolerance is acceptable)

        Returns:
        --------
        Dict[str, float]
            Dictionary with:
            - optimal_budget: The recommended budget
            - predicted_conversions: Expected conversions at optimal budget
            - predicted_roas: Expected ROAS at optimal budget
            - revenue: Expected revenue
            - profit: Expected profit (revenue - budget)
        """
        # Prepare baseline features
        if other_features is None:
            X_base = baseline_X.mean().to_frame().T
        else:
            X_base = pd.DataFrame([other_features])

        # Ensure all required columns exist
        for col in baseline_X.columns:
            if col not in X_base.columns and col != budget_column:
                X_base[col] = baseline_X[col].mean()

        def objective(budget):
            """Maximize conversions (minimize negative conversions)"""
            X_test = X_base.copy()
            X_test[budget_column] = budget

            # Ensure correct column order
            X_test = X_test[baseline_X.columns]

            conversions = self.model.predict(X_test)[0]
            return -conversions  # Negative because we minimize

        def roas_constraint(budget):
            """ROAS must be >= target_roas - tolerance"""
            X_test = X_base.copy()
            X_test[budget_column] = budget
            X_test = X_test[baseline_X.columns]

            conversions = self.model.predict(X_test)[0]
            revenue = conversions * self.conversion_value
            roas = revenue / (budget + 1e-10)

            # Return positive if constraint is satisfied
            return roas - (target_roas - tolerance)

        # Optimize with constraint
        result = minimize(
            objective,
            x0=[(budget_range[0] + budget_range[1]) / 2],  # Start in middle
            method='SLSQP',
            bounds=[budget_range],
            constraints={'type': 'ineq', 'fun': roas_constraint}
        )

        if not result.success:
            # Try global optimization with a wrapper that includes constraint
            def penalized_objective(budget_array):
                budget = budget_array[0]
                obj_val = objective(budget)
                constraint_val = roas_constraint(budget)
                # Add penalty if constraint is violated
                if constraint_val < 0:
                    return obj_val + 1000 * abs(constraint_val)
                return obj_val

            result = differential_evolution(
                penalized_objective,
                bounds=[budget_range],
                seed=42
            )

        optimal_budget = result.x[0]

        # Calculate final metrics
        X_optimal = X_base.copy()
        X_optimal[budget_column] = optimal_budget
        X_optimal = X_optimal[baseline_X.columns]

        predicted_conversions = self.model.predict(X_optimal)[0]
        revenue = predicted_conversions * self.conversion_value
        predicted_roas = revenue / optimal_budget
        profit = revenue - optimal_budget

        return {
            'optimal_budget': optimal_budget,
            'predicted_conversions': predicted_conversions,
            'predicted_roas': predicted_roas,
            'revenue': revenue,
            'profit': profit,
            'optimization_success': result.success
        }

    def find_budget_for_conversion_target(
        self,
        target_conversions: float,
        baseline_X: pd.DataFrame,
        budget_column: str = 'ad_budget',
        budget_range: Tuple[float, float] = (100, 50000)
    ) -> Dict[str, float]:
        """
        Find the minimum budget needed to achieve a target number of conversions.

        Parameters:
        -----------
        target_conversions : float
            Target number of conversions to achieve
        baseline_X : pd.DataFrame
            Baseline feature values
        budget_column : str, default='ad_budget'
            Name of the budget column
        budget_range : Tuple[float, float]
            (min, max) budget to consider

        Returns:
        --------
        Dict[str, float]
            Dictionary with optimal budget and predicted metrics
        """
        X_base = baseline_X.mean().to_frame().T

        def objective(budget):
            """Minimize budget"""
            return budget

        def conversion_constraint(budget):
            """Conversions must be >= target"""
            X_test = X_base.copy()
            X_test[budget_column] = budget
            X_test = X_test[baseline_X.columns]

            conversions = self.model.predict(X_test)[0]
            return conversions - target_conversions

        result = minimize(
            objective,
            x0=[(budget_range[0] + budget_range[1]) / 2],
            method='SLSQP',
            bounds=[budget_range],
            constraints={'type': 'ineq', 'fun': conversion_constraint}
        )

        optimal_budget = result.x[0]

        X_optimal = X_base.copy()
        X_optimal[budget_column] = optimal_budget
        X_optimal = X_optimal[baseline_X.columns]

        predicted_conversions = self.model.predict(X_optimal)[0]
        revenue = predicted_conversions * self.conversion_value
        predicted_roas = revenue / optimal_budget

        return {
            'optimal_budget': optimal_budget,
            'predicted_conversions': predicted_conversions,
            'predicted_roas': predicted_roas,
            'revenue': revenue,
            'optimization_success': result.success
        }

    def analyze_budget_roas_tradeoff(
        self,
        baseline_X: pd.DataFrame,
        budget_column: str = 'ad_budget',
        budget_range: Tuple[float, float] = (100, 50000),
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Analyze the trade-off between budget, conversions, and ROAS.

        Parameters:
        -----------
        baseline_X : pd.DataFrame
            Baseline feature values
        budget_column : str
            Name of budget column
        budget_range : Tuple[float, float]
            Range of budgets to analyze
        n_points : int
            Number of points to evaluate

        Returns:
        --------
        pd.DataFrame
            DataFrame with budget, conversions, ROAS, and efficiency metrics
        """
        budgets = np.linspace(budget_range[0], budget_range[1], n_points)
        X_base = baseline_X.mean().to_frame().T

        results = []

        for budget in budgets:
            X_test = X_base.copy()
            X_test[budget_column] = budget
            X_test = X_test[baseline_X.columns]

            conversions = self.model.predict(X_test)[0]
            revenue = conversions * self.conversion_value
            roas = revenue / budget
            profit = revenue - budget
            roi = profit / budget
            cpa = budget / (conversions + 1e-10)  # Cost per acquisition

            results.append({
                'budget': budget,
                'conversions': conversions,
                'revenue': revenue,
                'roas': roas,
                'profit': profit,
                'roi': roi,
                'cpa': cpa
            })

        return pd.DataFrame(results)

    def plot_optimization_landscape(
        self,
        baseline_X: pd.DataFrame,
        budget_column: str = 'ad_budget',
        budget_range: Tuple[float, float] = (100, 50000),
        target_roas: Optional[float] = None,
        optimal_budget: Optional[float] = None,
        save_path: str = 'optimization_landscape.png'
    ) -> None:
        """
        Visualize the optimization landscape showing budget vs conversions vs ROAS.

        Parameters:
        -----------
        baseline_X : pd.DataFrame
            Baseline features
        budget_column : str
            Budget column name
        budget_range : Tuple[float, float]
            Budget range to plot
        target_roas : float, optional
            Target ROAS line to draw
        optimal_budget : float, optional
            Optimal budget point to highlight
        save_path : str
            Path to save the plot
        """
        # Analyze trade-off
        df = self.analyze_budget_roas_tradeoff(
            baseline_X, budget_column, budget_range, n_points=100
        )

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Budget vs Conversions
        ax1 = axes[0, 0]
        ax1.plot(df['budget'], df['conversions'], linewidth=2, color='#2E86AB')
        ax1.set_xlabel('Budget ($)', fontsize=11)
        ax1.set_ylabel('Conversions', fontsize=11)
        ax1.set_title('Budget vs Conversions (with Saturation)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        if optimal_budget is not None:
            opt_row = df.iloc[(df['budget'] - optimal_budget).abs().argmin()]
            ax1.axvline(optimal_budget, color='red', linestyle='--', alpha=0.7, label='Optimal Budget')
            ax1.scatter([optimal_budget], [opt_row['conversions']],
                       color='red', s=100, zorder=5, marker='o')
            ax1.legend()

        # Plot 2: Budget vs ROAS
        ax2 = axes[0, 1]
        ax2.plot(df['budget'], df['roas'], linewidth=2, color='#A23B72')
        ax2.set_xlabel('Budget ($)', fontsize=11)
        ax2.set_ylabel('ROAS', fontsize=11)
        ax2.set_title('Budget vs ROAS (Diminishing Returns)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        if target_roas is not None:
            ax2.axhline(target_roas, color='green', linestyle='--',
                       alpha=0.7, label=f'Target ROAS = {target_roas}')
            ax2.legend()

        if optimal_budget is not None:
            opt_row = df.iloc[(df['budget'] - optimal_budget).abs().argmin()]
            ax2.axvline(optimal_budget, color='red', linestyle='--', alpha=0.7)
            ax2.scatter([optimal_budget], [opt_row['roas']],
                       color='red', s=100, zorder=5, marker='o')

        # Plot 3: Budget vs Profit
        ax3 = axes[1, 0]
        ax3.plot(df['budget'], df['profit'], linewidth=2, color='#F18F01')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax3.set_xlabel('Budget ($)', fontsize=11)
        ax3.set_ylabel('Profit ($)', fontsize=11)
        ax3.set_title('Budget vs Profit', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        if optimal_budget is not None:
            opt_row = df.iloc[(df['budget'] - optimal_budget).abs().argmin()]
            ax3.axvline(optimal_budget, color='red', linestyle='--', alpha=0.7)
            ax3.scatter([optimal_budget], [opt_row['profit']],
                       color='red', s=100, zorder=5, marker='o')

        # Plot 4: ROAS vs Conversions (efficiency frontier)
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['conversions'], df['roas'],
                             c=df['budget'], cmap='viridis',
                             s=50, alpha=0.6)
        ax4.set_xlabel('Conversions', fontsize=11)
        ax4.set_ylabel('ROAS', fontsize=11)
        ax4.set_title('Efficiency Frontier (colored by budget)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Budget ($)', fontsize=10)

        if target_roas is not None:
            ax4.axhline(target_roas, color='green', linestyle='--',
                       alpha=0.7, linewidth=2)

        if optimal_budget is not None:
            opt_row = df.iloc[(df['budget'] - optimal_budget).abs().argmin()]
            ax4.scatter([opt_row['conversions']], [opt_row['roas']],
                       color='red', s=200, zorder=5, marker='*',
                       edgecolors='black', linewidth=1.5,
                       label='Optimal Point')
            ax4.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved optimization landscape to '{save_path}'")

    def get_budget_recommendations(
        self,
        baseline_X: pd.DataFrame,
        budget_column: str = 'ad_budget',
        budget_range: Tuple[float, float] = (100, 50000),
        roas_targets: List[float] = [2.0, 2.5, 3.0, 3.5, 4.0]
    ) -> pd.DataFrame:
        """
        Get budget recommendations for multiple ROAS targets.

        Parameters:
        -----------
        baseline_X : pd.DataFrame
            Baseline features
        budget_column : str
            Budget column name
        budget_range : Tuple[float, float]
            Budget search range
        roas_targets : List[float]
            List of target ROAS values

        Returns:
        --------
        pd.DataFrame
            Recommendations table with budget, conversions, and ROAS for each target
        """
        recommendations = []

        for target in roas_targets:
            try:
                result = self.optimize_for_target_roas(
                    target_roas=target,
                    baseline_X=baseline_X,
                    budget_column=budget_column,
                    budget_range=budget_range,
                    tolerance=0.05
                )

                recommendations.append({
                    'target_roas': target,
                    'recommended_budget': result['optimal_budget'],
                    'predicted_conversions': result['predicted_conversions'],
                    'predicted_roas': result['predicted_roas'],
                    'revenue': result['revenue'],
                    'profit': result['profit'],
                    'feasible': result['optimization_success']
                })
            except Exception as e:
                recommendations.append({
                    'target_roas': target,
                    'recommended_budget': np.nan,
                    'predicted_conversions': np.nan,
                    'predicted_roas': np.nan,
                    'revenue': np.nan,
                    'profit': np.nan,
                    'feasible': False
                })

        return pd.DataFrame(recommendations)


def optimize_multi_channel_with_roas(
    model: MarketingMixModel,
    total_budget: float,
    target_roas: float,
    baseline_X: pd.DataFrame,
    budget_columns: List[str],
    conversion_value: float = 1.0,
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Optimize budget allocation across multiple channels with ROAS constraint.

    Parameters:
    -----------
    model : MarketingMixModel
        Fitted model
    total_budget : float
        Total budget to allocate
    target_roas : float
        Target ROAS to achieve
    baseline_X : pd.DataFrame
        Baseline features
    budget_columns : List[str]
        List of budget column names
    conversion_value : float
        Revenue per conversion
    tolerance : float
        ROAS tolerance

    Returns:
    --------
    Dict[str, float]
        Optimal allocation across channels
    """
    n_channels = len(budget_columns)
    X_base = baseline_X.mean().to_frame().T

    def objective(budgets):
        """Maximize conversions"""
        X_test = X_base.copy()
        for i, col in enumerate(budget_columns):
            X_test[col] = budgets[i]
        X_test = X_test[baseline_X.columns]

        conversions = model.predict(X_test)[0]
        return -conversions

    def budget_constraint(budgets):
        """Total budget constraint"""
        return total_budget - np.sum(budgets)

    def roas_constraint(budgets):
        """ROAS constraint"""
        X_test = X_base.copy()
        for i, col in enumerate(budget_columns):
            X_test[col] = budgets[i]
        X_test = X_test[baseline_X.columns]

        conversions = model.predict(X_test)[0]
        revenue = conversions * conversion_value
        total_spend = np.sum(budgets)
        roas = revenue / (total_spend + 1e-10)

        return roas - (target_roas - tolerance)

    # Initial guess
    x0 = np.array([total_budget / n_channels] * n_channels)

    # Bounds
    bounds = [(0, total_budget) for _ in range(n_channels)]

    # Optimize
    result = minimize(
        objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=[
            {'type': 'eq', 'fun': budget_constraint},
            {'type': 'ineq', 'fun': roas_constraint}
        ]
    )

    optimal_allocation = {
        budget_columns[i]: result.x[i]
        for i in range(n_channels)
    }

    return optimal_allocation
