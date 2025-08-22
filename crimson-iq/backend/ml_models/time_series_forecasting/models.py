"""
Phase 1.2: Baseline Time-Series Forecasting Models
Implements ARIMA, Exponential Smoothing, and Random Forest models for blood demand prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Base class for time series forecasting models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'demand_quantity'
        
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesForecaster':
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, data: pd.DataFrame, steps: int = 24) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        return metrics
        
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
            
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

class ARIMAForecaster(TimeSeriesForecaster):
    """ARIMA model for time series forecasting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ARIMA", config)
        self.order = config.get('order', (1, 1, 1))  # (p, d, q)
        self.seasonal_order = config.get('seasonal_order', (1, 1, 1, 24))  # (P, D, Q, s)
        
    def _check_stationarity(self, series: pd.Series) -> bool:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
        
    def _make_stationary(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """Make series stationary by differencing"""
        original_series = series.copy()
        d = 0
        
        while not self._check_stationarity(series) and d < 2:
            series = series.diff().dropna()
            d += 1
            
        return series, d
        
    def fit(self, data: pd.DataFrame) -> 'ARIMAForecaster':
        """Fit ARIMA model"""
        logger.info(f"Fitting ARIMA model with order {self.order}")
        
        # Prepare time series data
        if isinstance(data, pd.DataFrame):
            # Aggregate by timestamp if multiple columns
            if len(data.columns) > 1:
                series = data[self.target_column].groupby(data.index).sum()
            else:
                series = data.iloc[:, 0]
        else:
            series = data
            
        # Make series stationary
        stationary_series, d = self._make_stationary(series)
        
        # Update order if differencing was needed
        if d > 0:
            p, _, q = self.order
            self.order = (p, d, q)
            logger.info(f"Updated ARIMA order to {self.order} after differencing")
        
        # Fit ARIMA model
        try:
            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            # Try with simpler order
            self.order = (1, 1, 0)
            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            logger.info("ARIMA model fitted with simplified order")
            
        return self
        
    def predict(self, data: pd.DataFrame = None, steps: int = 24) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast.values
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            return np.zeros(steps)

class ExponentialSmoothingForecaster(TimeSeriesForecaster):
    """Exponential Smoothing model for time series forecasting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ExponentialSmoothing", config)
        self.trend = config.get('trend', 'add')
        self.seasonal = config.get('seasonal', 'add')
        self.seasonal_periods = config.get('seasonal_periods', 24)  # 24 hours
        self.damped_trend = config.get('damped_trend', False)
        
    def fit(self, data: pd.DataFrame) -> 'ExponentialSmoothingForecaster':
        """Fit Exponential Smoothing model"""
        logger.info(f"Fitting Exponential Smoothing model")
        
        # Prepare time series data
        if isinstance(data, pd.DataFrame):
            if len(data.columns) > 1:
                series = data[self.target_column].groupby(data.index).sum()
            else:
                series = data.iloc[:, 0]
        else:
            series = data
            
        # Fit model
        try:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend
            )
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            logger.info("Exponential Smoothing model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Exponential Smoothing model: {e}")
            # Try with simpler configuration
            self.model = ExponentialSmoothing(series, trend='add')
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            logger.info("Exponential Smoothing model fitted with simplified config")
            
        return self
        
    def predict(self, data: pd.DataFrame = None, steps: int = 24) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast.values
        except Exception as e:
            logger.error(f"Error making Exponential Smoothing predictions: {e}")
            return np.zeros(steps)

class RandomForestForecaster(TimeSeriesForecaster):
    """Random Forest model for time series forecasting with feature engineering"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("RandomForest", config)
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 10)
        self.random_state = config.get('random_state', 42)
        self.lag_features = config.get('lag_features', [1, 2, 3, 6, 12, 24])
        
    def _create_lag_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create lag features for time series"""
        df = data.copy()
        
        for lag in self.lag_features:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        # Create rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            
        return df
        
    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = data.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
            
        return df
        
    def fit(self, data: pd.DataFrame) -> 'RandomForestForecaster':
        """Fit Random Forest model"""
        logger.info(f"Fitting Random Forest model with {self.n_estimators} estimators")
        
        # Prepare features
        df = data.copy()
        if self.target_column in df.columns:
            df = self._create_lag_features(df, self.target_column)
            df = self._create_time_features(df)
            
            # Drop rows with NaN values (from lag features)
            df = df.dropna()
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != self.target_column and not col.startswith('timestamp')]
            self.feature_columns = feature_cols
            
            X = df[feature_cols]
            y = df[self.target_column]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit model
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            logger.info("Random Forest model fitted successfully")
        else:
            logger.error(f"Target column '{self.target_column}' not found in data")
            
        return self
        
    def predict(self, data: pd.DataFrame, steps: int = 24) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = []
        df = data.copy()
        
        for i in range(steps):
            # Prepare features for current step
            df = self._create_lag_features(df, self.target_column)
            df = self._create_time_features(df)
            
            # Get latest row
            latest_row = df.iloc[-1:][self.feature_columns]
            
            # Scale features
            latest_scaled = self.scaler.transform(latest_row)
            
            # Make prediction
            pred = self.model.predict(latest_scaled)[0]
            predictions.append(pred)
            
            # Add prediction to dataframe for next iteration
            new_row = df.iloc[-1:].copy()
            new_row[self.target_column] = pred
            df = pd.concat([df, new_row], ignore_index=True)
            
        return np.array(predictions)

class ModelEvaluator:
    """Evaluates and compares different forecasting models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model: TimeSeriesForecaster):
        """Add a model to the evaluator"""
        self.models[name] = model
        
    def evaluate_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on train/test data"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name} model...")
            
            try:
                # Fit model on training data
                model.fit(train_data)
                
                # Make predictions on test data
                y_true = test_data[model.target_column].values
                y_pred = model.predict(test_data, steps=len(test_data))
                
                # Ensure predictions match test data length
                if len(y_pred) > len(y_true):
                    y_pred = y_pred[:len(y_true)]
                elif len(y_pred) < len(y_true):
                    y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)), mode='edge')
                
                # Calculate metrics
                metrics = model.evaluate(y_true, y_pred)
                results[name] = metrics
                
                logger.info(f"{name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%")
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
                results[name] = {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}
                
        self.results = results
        return results
        
    def plot_predictions(self, test_data: pd.DataFrame, save_path: str = None):
        """Plot predictions for all models"""
        fig, axes = plt.subplots(len(self.models), 1, figsize=(15, 5*len(self.models)))
        if len(self.models) == 1:
            axes = [axes]
            
        for i, (name, model) in enumerate(self.models.items()):
            if name in self.results:
                y_true = test_data[model.target_column].values
                y_pred = model.predict(test_data, steps=len(test_data))
                
                # Ensure predictions match test data length
                if len(y_pred) > len(y_true):
                    y_pred = y_pred[:len(y_true)]
                elif len(y_pred) < len(y_true):
                    y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)), mode='edge')
                
                axes[i].plot(y_true, label='Actual', alpha=0.7)
                axes[i].plot(y_pred, label='Predicted', alpha=0.7)
                axes[i].set_title(f'{name} Model Predictions')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Demand')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
        
    def get_best_model(self) -> Tuple[str, TimeSeriesForecaster]:
        """Get the best performing model based on RMSE"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models first.")
            
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        return best_model_name, self.models[best_model_name]

def main():
    """Main function for model training and evaluation"""
    
    # Load processed data
    try:
        demand_data = pd.read_parquet("processed_data/demand_processed.parquet")
        logger.info("Loaded processed demand data")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run the data preprocessing pipeline first.")
        return
    
    # Prepare data for modeling
    # Aggregate demand by timestamp and blood group
    demand_agg = demand_data.groupby(['timestamp', 'blood_group'])['demand_quantity'].sum().reset_index()
    
    # Focus on one blood group for demonstration (O+ is most common)
    o_positive_data = demand_agg[demand_agg['blood_group'] == 'O+'].set_index('timestamp')['demand_quantity']
    
    # Split data into train/test
    split_idx = int(len(o_positive_data) * 0.8)
    train_data = o_positive_data[:split_idx]
    test_data = o_positive_data[split_idx:]
    
    logger.info(f"Training data: {len(train_data)} samples")
    logger.info(f"Test data: {len(test_data)} samples")
    
    # Initialize models
    models = {
        'ARIMA': ARIMAForecaster({'order': (1, 1, 1)}),
        'ExponentialSmoothing': ExponentialSmoothingForecaster({
            'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 24
        }),
        'RandomForest': RandomForestForecaster({
            'n_estimators': 100, 'max_depth': 10, 'lag_features': [1, 2, 3, 6, 12, 24]
        })
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    for name, model in models.items():
        evaluator.add_model(name, model)
    
    # Evaluate models
    results = evaluator.evaluate_models(train_data, test_data)
    
    # Plot predictions
    evaluator.plot_predictions(test_data, save_path="forecast_predictions.png")
    
    # Get best model
    best_model_name, best_model = evaluator.get_best_model()
    logger.info(f"Best model: {best_model_name}")
    
    # Save best model
    best_model.save_model(f"models/{best_model_name.lower()}_model.pkl")
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    for name, metrics in results.items():
        print(f"{name:20} | MAE: {metrics['mae']:6.2f} | RMSE: {metrics['rmse']:6.2f} | MAPE: {metrics['mape']:6.2f}%")
    print("="*50)

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    main()
