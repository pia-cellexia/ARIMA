"""
Causal Marketing Mix Model (MMM)

This module implements a causal marketing mix model that can predict conversions
based on ad budget, ROAS, and other marketing variables. It includes:
- Adstock transformation (carryover effects)
- Saturation effects (diminishing returns)
- Causal inference capabilities
- Bayesian and frequentist approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import curve_fit, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class MarketingMixModel:
    """
    A causal marketing mix model for predicting conversions based on marketing budget.

    This model incorporates:
    1. Adstock effect: Marketing has a carryover effect over time
    2. Saturation effect: Diminishing returns as budget increases
    3. Causal estimation: Estimates the true causal effect of budget on conversions
    """

    def __init__(
        self,
        adstock_decay: float = 0.5,
        saturation_lambda: float = 1.0,
        use_saturation: bool = True,
        use_adstock: bool = True
    ):
        """
        Initialize the Marketing Mix Model.

        Parameters:
        -----------
        adstock_decay : float, default=0.5
            Decay rate for adstock transformation (0-1). Higher values mean
            longer carryover effects.
        saturation_lambda : float, default=1.0
            Lambda parameter for saturation curve. Controls the rate of diminishing returns.
        use_saturation : bool, default=True
            Whether to apply saturation transformation to budget data.
        use_adstock : bool, default=True
            Whether to apply adstock transformation to budget data.
        """
        self.adstock_decay = adstock_decay
        self.saturation_lambda = saturation_lambda
        self.use_saturation = use_saturation
        self.use_adstock = use_adstock

        self.scaler = StandardScaler()
        self.coefficients_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def adstock_transform(
        self,
        x: np.ndarray,
        decay: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply adstock (carryover) transformation to a time series.

        The adstock effect captures the idea that marketing has a lasting impact
        beyond the immediate period. Formula: x_adstock[t] = x[t] + decay * x_adstock[t-1]

        Parameters:
        -----------
        x : np.ndarray
            Input time series
        decay : float, optional
            Decay rate (0-1). If None, uses self.adstock_decay

        Returns:
        --------
        np.ndarray
            Transformed time series with carryover effects
        """
        if decay is None:
            decay = self.adstock_decay

        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]

        for t in range(1, len(x)):
            adstocked[t] = x[t] + decay * adstocked[t-1]

        return adstocked

    def saturation_transform(
        self,
        x: np.ndarray,
        lambda_: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply saturation (diminishing returns) transformation.

        Uses the Hill saturation curve: f(x) = x^alpha / (lambda^alpha + x^alpha)
        where alpha is fixed at 1 for simplicity.

        Parameters:
        -----------
        x : np.ndarray
            Input values
        lambda_ : float, optional
            Saturation parameter. If None, uses self.saturation_lambda

        Returns:
        --------
        np.ndarray
            Transformed values with diminishing returns
        """
        if lambda_ is None:
            lambda_ = self.saturation_lambda

        # Avoid division by zero
        return x / (lambda_ + x)

    def transform_features(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply adstock and saturation transformations to features.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features (should include ad_budget and optionally other channels)

        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Transformed features and their names
        """
        X_transformed = X.copy()
        feature_names = []

        for col in X.columns:
            if 'budget' in col.lower() or 'spend' in col.lower():
                transformed_col = X[col].values

                # Apply adstock transformation
                if self.use_adstock:
                    transformed_col = self.adstock_transform(transformed_col)

                # Apply saturation transformation
                if self.use_saturation:
                    transformed_col = self.saturation_transform(transformed_col)

                X_transformed[col] = transformed_col
                feature_names.append(col)
            else:
                feature_names.append(col)

        return X_transformed.values, feature_names

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        optimize_params: bool = False
    ) -> 'MarketingMixModel':
        """
        Fit the marketing mix model to data.

        Parameters:
        -----------
        X : pd.DataFrame
            Features (ad_budget, ROAS, and other covariates)
        y : np.ndarray
            Target variable (conversions)
        optimize_params : bool, default=False
            Whether to optimize adstock_decay and saturation_lambda parameters

        Returns:
        --------
        self
            Fitted model
        """
        if optimize_params:
            self._optimize_hyperparameters(X, y)

        # Transform features
        X_transformed, feature_names = self.transform_features(X)
        self.feature_names_ = feature_names

        # Standardize features
        X_scaled = self.scaler.fit_transform(X_transformed)

        # Fit linear regression (ordinary least squares)
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])

        # Solve using normal equation: beta = (X'X)^(-1) X'y
        try:
            coeffs = np.linalg.solve(
                X_with_intercept.T @ X_with_intercept,
                X_with_intercept.T @ y
            )
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            coeffs = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)

        self.intercept_ = coeffs[0]
        self.coefficients_ = coeffs[1:]
        self.is_fitted_ = True

        return self

    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> None:
        """
        Optimize adstock_decay and saturation_lambda using cross-validation.

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : np.ndarray
            Target variable
        """
        def objective(params):
            self.adstock_decay = params[0]
            self.saturation_lambda = params[1]

            # Transform and fit
            X_transformed, _ = self.transform_features(X)
            X_scaled = StandardScaler().fit_transform(X_transformed)
            X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])

            try:
                coeffs = np.linalg.solve(
                    X_with_intercept.T @ X_with_intercept,
                    X_with_intercept.T @ y
                )
            except np.linalg.LinAlgError:
                coeffs = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)

            y_pred = X_with_intercept @ coeffs
            mse = mean_squared_error(y, y_pred)
            return mse

        # Optimize
        result = minimize(
            objective,
            x0=[self.adstock_decay, self.saturation_lambda],
            bounds=[(0.0, 0.99), (0.01, 10.0)],
            method='L-BFGS-B'
        )

        self.adstock_decay = result.x[0]
        self.saturation_lambda = result.x[1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict conversions for given features.

        Parameters:
        -----------
        X : pd.DataFrame
            Features (same structure as training data)

        Returns:
        --------
        np.ndarray
            Predicted conversions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Transform features
        X_transformed, _ = self.transform_features(X)

        # Standardize
        X_scaled = self.scaler.transform(X_transformed)

        # Predict
        y_pred = self.intercept_ + X_scaled @ self.coefficients_

        return y_pred

    def score(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics.

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : np.ndarray
            True conversions

        Returns:
        --------
        Dict[str, float]
            Dictionary with R2, RMSE, MAE, and MAPE
        """
        y_pred = self.predict(X)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100

        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (contribution to conversions).

        Returns:
        --------
        pd.DataFrame
            Feature names and their importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")

        importance = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': self.coefficients_,
            'abs_coefficient': np.abs(self.coefficients_)
        }).sort_values('abs_coefficient', ascending=False)

        return importance

    def estimate_causal_effect(
        self,
        X: pd.DataFrame,
        treatment_col: str,
        treatment_change: float
    ) -> Dict[str, float]:
        """
        Estimate the causal effect of changing a treatment variable (e.g., budget).

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        treatment_col : str
            Name of the treatment column (e.g., 'ad_budget')
        treatment_change : float
            Amount to change the treatment (can be positive or negative)

        Returns:
        --------
        Dict[str, float]
            Dictionary with estimated effect on conversions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before estimating causal effects")

        # Predict with original data
        y_pred_original = self.predict(X)

        # Create counterfactual data
        X_counterfactual = X.copy()
        X_counterfactual[treatment_col] = X_counterfactual[treatment_col] + treatment_change

        # Predict with counterfactual
        y_pred_counterfactual = self.predict(X_counterfactual)

        # Calculate effect
        effect = y_pred_counterfactual - y_pred_original

        return {
            'mean_effect': np.mean(effect),
            'total_effect': np.sum(effect),
            'min_effect': np.min(effect),
            'max_effect': np.max(effect),
            'std_effect': np.std(effect)
        }

    def optimize_budget_allocation(
        self,
        total_budget: float,
        baseline_X: pd.DataFrame,
        budget_columns: List[str],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Optimize budget allocation across channels to maximize conversions.

        Parameters:
        -----------
        total_budget : float
            Total budget to allocate
        baseline_X : pd.DataFrame
            Baseline features (will use mean values for non-budget features)
        budget_columns : List[str]
            Names of budget columns to optimize
        constraints : Dict[str, Tuple[float, float]], optional
            Min/max constraints for each budget column

        Returns:
        --------
        Dict[str, float]
            Optimal budget allocation
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before optimizing budget")

        n_channels = len(budget_columns)

        # Set up constraints
        if constraints is None:
            constraints = {col: (0, total_budget) for col in budget_columns}

        def objective(budgets):
            """Negative conversions (since we minimize)"""
            X_test = baseline_X.copy()
            for i, col in enumerate(budget_columns):
                X_test[col] = budgets[i]
            y_pred = self.predict(X_test)
            return -np.sum(y_pred)  # Negative because we minimize

        def budget_constraint(budgets):
            """Total budget constraint"""
            return total_budget - np.sum(budgets)

        # Initial guess (equal allocation)
        x0 = np.array([total_budget / n_channels] * n_channels)

        # Bounds
        bounds = [constraints.get(col, (0, total_budget)) for col in budget_columns]

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': budget_constraint}
        )

        optimal_allocation = {
            budget_columns[i]: result.x[i]
            for i in range(n_channels)
        }

        return optimal_allocation


class BayesianMarketingMixModel:
    """
    Bayesian Marketing Mix Model using PyMC.

    This provides uncertainty quantification and credible intervals for all estimates.
    """

    def __init__(
        self,
        adstock_decay_prior: Tuple[float, float] = (0.3, 0.7),
        saturation_lambda_prior: Tuple[float, float] = (0.5, 2.0)
    ):
        """
        Initialize Bayesian MMM.

        Parameters:
        -----------
        adstock_decay_prior : Tuple[float, float]
            (mean, std) for adstock decay prior
        saturation_lambda_prior : Tuple[float, float]
            (mean, std) for saturation lambda prior
        """
        self.adstock_decay_prior = adstock_decay_prior
        self.saturation_lambda_prior = saturation_lambda_prior
        self.trace = None
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs):
        """
        Fit Bayesian model using MCMC.

        Note: This is a placeholder for a full PyMC implementation.
        Full implementation would require defining priors and running MCMC sampling.
        """
        raise NotImplementedError(
            "Bayesian model requires PyMC installation and full implementation. "
            "Use MarketingMixModel for frequentist approach."
        )


def calculate_roi(
    conversions: np.ndarray,
    budget: np.ndarray,
    conversion_value: float = 1.0
) -> np.ndarray:
    """
    Calculate Return on Investment (ROI) for marketing budget.

    Parameters:
    -----------
    conversions : np.ndarray
        Number of conversions
    budget : np.ndarray
        Marketing budget
    conversion_value : float, default=1.0
        Value per conversion

    Returns:
    --------
    np.ndarray
        ROI for each observation
    """
    revenue = conversions * conversion_value
    roi = (revenue - budget) / (budget + 1e-10)
    return roi


def calculate_roas(
    conversions: np.ndarray,
    budget: np.ndarray,
    conversion_value: float = 1.0
) -> np.ndarray:
    """
    Calculate Return on Ad Spend (ROAS).

    Parameters:
    -----------
    conversions : np.ndarray
        Number of conversions
    budget : np.ndarray
        Marketing budget
    conversion_value : float, default=1.0
        Value per conversion

    Returns:
    --------
    np.ndarray
        ROAS for each observation
    """
    revenue = conversions * conversion_value
    roas = revenue / (budget + 1e-10)
    return roas
