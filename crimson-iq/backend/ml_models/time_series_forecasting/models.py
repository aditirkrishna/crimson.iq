# Time series forecasting models

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

class BaseTimeSeriesModel:
    """Base class for time series forecasting models"""
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mape': mean_absolute_percentage_error(y, predictions) * 100,
            'r2': self._calculate_r2(y, predictions)
        }
        
        return metrics
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model_name': self.model_name,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'model': self.model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.feature_columns = model_data['feature_columns']
        self.model = model_data['model']
        
        logger.info(f"Model loaded from {filepath}")

class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA model for time series forecasting"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        super().__init__("ARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.config = {
            'order': order,
            'seasonal_order': seasonal_order
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit ARIMA model to univariate time series"""
        # For ARIMA, we need univariate data
        if len(X.shape) > 1:
            # Use the first feature or target as the time series
            time_series = y.flatten() if len(y.shape) > 1 else y
        else:
            time_series = X
        
        # Check for stationarity
        if not self._is_stationary(time_series):
            logger.warning("Time series is not stationary. Consider differencing.")
        
        # Fit ARIMA model
        if self.seasonal_order:
            self.model = sm.tsa.statespace.SARIMAX(
                time_series, 
                order=self.order, 
                seasonal_order=self.seasonal_order
            )
        else:
            self.model = ARIMA(time_series, order=self.order)
        
        self.model = self.model.fit()
        self.is_fitted = True
        
        logger.info(f"ARIMA model fitted with order {self.order}")
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get forecast
        forecast = self.model.forecast(steps=steps)
        
        # Reshape to match expected output format
        if len(X.shape) > 1:
            # Return predictions for each sample
            return np.tile(forecast, (X.shape[0], 1))
        else:
            return forecast
    
    def _is_stationary(self, time_series: np.ndarray) -> bool:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(time_series)
        return result[1] < 0.05  # p-value < 0.05 indicates stationarity

class ExponentialSmoothingModel(BaseTimeSeriesModel):
    """Exponential Smoothing model for time series forecasting"""
    
    def __init__(self, trend: Optional[str] = 'add', 
                 seasonal: Optional[str] = 'add',
                 seasonal_periods: Optional[int] = None):
        super().__init__("ExponentialSmoothing")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.config = {
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Exponential Smoothing model"""
        # For Exponential Smoothing, we need univariate data
        if len(X.shape) > 1:
            time_series = y.flatten() if len(y.shape) > 1 else y
        else:
            time_series = X
        
        # Create model
        self.model = ExponentialSmoothing(
            time_series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods
        )
        
        # Fit model
        self.model = self.model.fit()
        self.is_fitted = True
        
        logger.info(f"Exponential Smoothing model fitted")
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get forecast
        forecast = self.model.forecast(steps=steps)
        
        # Reshape to match expected output format
        if len(X.shape) > 1:
            return np.tile(forecast, (X.shape[0], 1))
        else:
            return forecast

class RandomForestTimeSeriesModel(BaseTimeSeriesModel):
    """Random Forest model for time series forecasting with engineered features"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Random Forest model"""
        # Reshape data for Random Forest
        if len(X.shape) == 3:
            # Convert 3D (samples, timesteps, features) to 2D (samples, timesteps*features)
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        # Flatten target if needed
        if len(y.shape) > 1:
            y_reshaped = y.flatten()
        else:
            y_reshaped = y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Create and fit model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y_reshaped)
        self.is_fitted = True
        
        logger.info(f"Random Forest model fitted with {self.n_estimators} estimators")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Reshape data
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Reshape back if needed
        if len(X.shape) == 3 and X.shape[0] > 1:
            # For multiple samples, reshape to match original target shape
            return predictions.reshape(-1, 1)
        else:
            return predictions

class LinearRegressionTimeSeriesModel(BaseTimeSeriesModel):
    """Linear Regression model for time series forecasting"""
    
    def __init__(self):
        super().__init__("LinearRegression")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Linear Regression model"""
        # Reshape data
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        if len(y.shape) > 1:
            y_reshaped = y.flatten()
        else:
            y_reshaped = y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Create and fit model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_reshaped)
        self.is_fitted = True
        
        logger.info("Linear Regression model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Reshape data
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Reshape back if needed
        if len(X.shape) == 3 and X.shape[0] > 1:
            return predictions.reshape(-1, 1)
        else:
            return predictions

class TimeSeriesModelEnsemble:
    """Ensemble of multiple time series models"""
    
    def __init__(self, models: List[BaseTimeSeriesModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit all models in the ensemble"""
        for model in self.models:
            model.fit(X, y, **kwargs)
        
        logger.info(f"Ensemble fitted with {len(self.models)} models")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        predictions = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mape': mean_absolute_percentage_error(y, predictions) * 100
        }
        
        return metrics

def create_model_factory():
    """Factory function to create different types of time series models"""
    
    def create_model(model_type: str, **kwargs) -> BaseTimeSeriesModel:
        if model_type.lower() == 'arima':
            return ARIMAModel(**kwargs)
        elif model_type.lower() == 'exponential_smoothing':
            return ExponentialSmoothingModel(**kwargs)
        elif model_type.lower() == 'random_forest':
            return RandomForestTimeSeriesModel(**kwargs)
        elif model_type.lower() == 'linear_regression':
            return LinearRegressionTimeSeriesModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return create_model

def main():
    """Example usage of time series models"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_timesteps = 24
    
    # Generate synthetic time series data
    X = np.random.randn(n_samples, n_timesteps, n_features)
    y = np.random.poisson(15, (n_samples, 7))  # 7-day forecast
    
    # Split data
    train_size = int(0.7 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create models
    models = {
        'random_forest': RandomForestTimeSeriesModel(n_estimators=50),
        'linear_regression': LinearRegressionTimeSeriesModel()
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics
        
        print(f"{name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Create ensemble
    ensemble = TimeSeriesModelEnsemble(list(models.values()))
    ensemble.fit(X_train, y_train)
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    
    print(f"\nEnsemble Results:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()
