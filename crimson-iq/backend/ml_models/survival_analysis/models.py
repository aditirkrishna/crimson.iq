# Survival analysis models

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

# Survival analysis libraries
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

class BaseSurvivalModel:
    """Base class for survival analysis models"""
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
        
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: Surv) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        # Calculate concordance index
        c_index = concordance_index_censored(
            y['event'], y['time'], predictions
        )[0]
        
        metrics = {
            'concordance_index': c_index,
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions)
        }
        
        return metrics
    
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

class CoxProportionalHazardsModel(BaseSurvivalModel):
    """Cox Proportional Hazards model for survival analysis"""
    
    def __init__(self, alpha: float = 1.0, ties: str = 'breslow'):
        super().__init__("CoxProportionalHazards")
        self.alpha = alpha
        self.ties = ties
        self.config = {
            'alpha': alpha,
            'ties': ties
        }
    
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit Cox Proportional Hazards model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha, ties=self.ties)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Cox Proportional Hazards model fitted with alpha={self.alpha}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher values indicate higher risk)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_survival_function(X_scaled, times)

class RandomSurvivalForestModel(BaseSurvivalModel):
    """Random Survival Forest model for survival analysis"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42):
        super().__init__("RandomSurvivalForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
    
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit Random Survival Forest model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Random Survival Forest model fitted with {self.n_estimators} estimators")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_survival_function(X_scaled, times)

class BloodViabilityPredictor:
    """Specialized predictor for blood unit viability"""
    
    def __init__(self, model: BaseSurvivalModel, blood_expiry_days: int = 42):
        self.model = model
        self.blood_expiry_days = blood_expiry_days  # Standard blood expiry period
        
    def predict_viability_risk(self, X: np.ndarray, 
                             collection_dates: np.ndarray,
                             current_date: Optional[datetime] = None) -> Dict[str, np.ndarray]:
        """Predict blood unit viability risk"""
        if current_date is None:
            current_date = datetime.now()
        
        # Predict survival function
        times = np.arange(1, self.blood_expiry_days + 1)  # Days from collection
        survival_function = self.model.predict_survival_function(X, times)
        
        # Calculate days since collection
        days_since_collection = np.array([
            (current_date - pd.to_datetime(date)).days 
            for date in collection_dates
        ])
        
        # Calculate current survival probability
        current_survival = np.array([
            survival_function[i, min(int(days), self.blood_expiry_days - 1)]
            for i, days in enumerate(days_since_collection)
        ])
        
        # Calculate risk score (1 - survival probability)
        risk_score = 1 - current_survival
        
        # Calculate predicted expiry date
        # Find the time point where survival probability drops below 0.5
        predicted_expiry_days = np.array([
            np.argmax(survival_function[i, :] < 0.5) + 1
            for i in range(len(X))
        ])
        
        predicted_expiry_dates = np.array([
            collection_dates[i] + timedelta(days=int(predicted_expiry_days[i]))
            for i in range(len(X))
        ])
        
        return {
            'survival_probability': current_survival,
            'risk_score': risk_score,
            'predicted_expiry_days': predicted_expiry_days,
            'predicted_expiry_date': predicted_expiry_dates,
            'days_since_collection': days_since_collection
        }
    
    def get_viability_alerts(self, risk_scores: np.ndarray, 
                           threshold: float = 0.7) -> np.ndarray:
        """Get alerts for blood units with high viability risk"""
        return risk_scores > threshold

def main():
    """Example usage of survival analysis models"""
    print("Survival analysis models implemented successfully!")
    print("Available models:")
    print("- CoxProportionalHazardsModel")
    print("- RandomSurvivalForestModel")
    print("- BloodViabilityPredictor")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

# Survival analysis libraries
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

class BaseSurvivalModel:
    """Base class for survival analysis models"""
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
        
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: Surv) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        # Calculate concordance index
        c_index = concordance_index_censored(
            y['event'], y['time'], predictions
        )[0]
        
        metrics = {
            'concordance_index': c_index,
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions)
        }
        
        return metrics
    
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

class CoxProportionalHazardsModel(BaseSurvivalModel):
    """Cox Proportional Hazards model for survival analysis"""
    
    def __init__(self, alpha: float = 1.0, ties: str = 'breslow'):
        super().__init__("CoxProportionalHazards")
        self.alpha = alpha
        self.ties = ties
        self.config = {
            'alpha': alpha,
            'ties': ties
        }
    
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit Cox Proportional Hazards model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha, ties=self.ties)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Cox Proportional Hazards model fitted with alpha={self.alpha}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher values indicate higher risk)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_survival_function(X_scaled, times)

class RandomSurvivalForestModel(BaseSurvivalModel):
    """Random Survival Forest model for survival analysis"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42):
        super().__init__("RandomSurvivalForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.config = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
    
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit Random Survival Forest model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit model
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Random Survival Forest model fitted with {self.n_estimators} estimators")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_survival_function(X_scaled, times)

class KaplanMeierModel(BaseSurvivalModel):
    """Kaplan-Meier estimator for non-parametric survival analysis"""
    
    def __init__(self):
        super().__init__("KaplanMeier")
    
    def fit(self, X: np.ndarray, y: Surv, **kwargs):
        """Fit Kaplan-Meier estimator"""
        # For KM, we don't actually need features, but we store the survival data
        self.survival_data = y
        self.is_fitted = True
        
        logger.info("Kaplan-Meier estimator fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict survival times (returns median survival time for all samples)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Calculate median survival time
        times, survival_probs = kaplan_meier_estimator(
            self.survival_data['event'], 
            self.survival_data['time']
        )
        
        # Find median survival time
        median_idx = np.argmax(survival_probs <= 0.5)
        median_survival = times[median_idx] if median_idx < len(times) else times[-1]
        
        # Return median survival time for all samples
        return np.full(X.shape[0], median_survival)
    
    def predict_survival_function(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Predict survival function at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Calculate KM survival function
        km_times, km_survival = kaplan_meier_estimator(
            self.survival_data['event'], 
            self.survival_data['time']
        )
        
        # Interpolate to requested times
        survival_at_times = np.interp(times, km_times, km_survival, right=0)
        
        # Return same survival function for all samples
        return np.tile(survival_at_times, (X.shape[0], 1))

class BloodViabilityPredictor:
    """Specialized predictor for blood unit viability"""
    
    def __init__(self, model: BaseSurvivalModel, blood_expiry_days: int = 42):
        self.model = model
        self.blood_expiry_days = blood_expiry_days  # Standard blood expiry period
        
    def predict_viability_risk(self, X: np.ndarray, 
                             collection_dates: np.ndarray,
                             current_date: Optional[datetime] = None) -> Dict[str, np.ndarray]:
        """Predict blood unit viability risk"""
        if current_date is None:
            current_date = datetime.now()
        
        # Predict survival function
        times = np.arange(1, self.blood_expiry_days + 1)  # Days from collection
        survival_function = self.model.predict_survival_function(X, times)
        
        # Calculate days since collection
        days_since_collection = np.array([
            (current_date - pd.to_datetime(date)).days 
            for date in collection_dates
        ])
        
        # Calculate current survival probability
        current_survival = np.array([
            survival_function[i, min(int(days), self.blood_expiry_days - 1)]
            for i, days in enumerate(days_since_collection)
        ])
        
        # Calculate risk score (1 - survival probability)
        risk_score = 1 - current_survival
        
        # Calculate predicted expiry date
        # Find the time point where survival probability drops below 0.5
        predicted_expiry_days = np.array([
            np.argmax(survival_function[i, :] < 0.5) + 1
            for i in range(len(X))
        ])
        
        predicted_expiry_dates = np.array([
            collection_dates[i] + timedelta(days=int(predicted_expiry_days[i]))
            for i in range(len(X))
        ])
        
        return {
            'survival_probability': current_survival,
            'risk_score': risk_score,
            'predicted_expiry_days': predicted_expiry_days,
            'predicted_expiry_date': predicted_expiry_dates,
            'days_since_collection': days_since_collection
        }
    
    def get_viability_alerts(self, risk_scores: np.ndarray, 
                           threshold: float = 0.7) -> np.ndarray:
        """Get alerts for blood units with high viability risk"""
        return risk_scores > threshold

class SurvivalDataProcessor:
    """Data processor for survival analysis"""
    
    def __init__(self):
        self.encoder = OneHotEncoder()
        
    def prepare_survival_data(self, df: pd.DataFrame, 
                            time_column: str = 'time_to_event',
                            event_column: str = 'event',
                            feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, Surv]:
        """Prepare data for survival analysis"""
        
        # Select features
        if feature_columns is None:
            # Exclude time and event columns from features
            exclude_cols = [time_column, event_column, 'timestamp', 'collection_date', 'expiry_date']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].values
        
        # Create survival data structure
        y = Surv.from_dataframe(event_column, time_column, df)
        
        # Encode categorical features
        categorical_mask = df[feature_columns].dtypes == 'object'
        if categorical_mask.any():
            X = self.encoder.fit_transform(X)
        
        return X, y
    
    def create_synthetic_survival_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, Surv, pd.DataFrame]:
        """Create synthetic survival data for blood units"""
        
        np.random.seed(42)
        
        # Generate features
        blood_groups = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_samples)
        temperatures = np.random.normal(4.0, 1.0, n_samples)  # Refrigeration temperature
        humidity = np.random.normal(60.0, 10.0, n_samples)
        volumes = np.random.choice([450, 500, 550], n_samples)
        
        # Generate survival times (days until expiry)
        # Base survival time is 42 days (standard blood expiry)
        base_survival = 42
        
        # Add effects of features
        temp_effect = -2 * np.abs(temperatures - 4.0)  # Temperature deviation reduces survival
        humidity_effect = -0.5 * np.abs(humidity - 60.0)  # Humidity deviation reduces survival
        volume_effect = 0.1 * (volumes - 500)  # Volume effect
        
        # Blood group effects (some blood groups have different shelf lives)
        blood_group_effects = {
            'A+': 0, 'A-': 2, 'B+': 0, 'B-': 2, 
            'AB+': -1, 'AB-': 1, 'O+': 0, 'O-': 3
        }
        group_effects = np.array([blood_group_effects[bg] for bg in blood_groups])
        
        # Calculate survival times
        survival_times = base_survival + temp_effect + humidity_effect + volume_effect + group_effects
        survival_times = np.maximum(1, survival_times)  # Minimum 1 day
        
        # Add some random noise
        survival_times += np.random.normal(0, 2, n_samples)
        survival_times = np.maximum(1, survival_times)
        
        # Generate events (1 = expired, 0 = censored)
        # Most blood units expire, but some are used before expiry
        events = np.random.binomial(1, 0.8, n_samples)
        
        # Create feature matrix
        features = np.column_stack([
            temperatures, humidity, volumes, group_effects
        ])
        
        # Create survival data structure
        y = Surv.from_arrays(event=events.astype(bool), time=survival_times)
        
        # Create DataFrame for reference
        df = pd.DataFrame({
            'blood_group': blood_groups,
            'temperature': temperatures,
            'humidity': humidity,
            'volume': volumes,
            'time_to_event': survival_times,
            'event': events
        })
        
        return features, y, df

def main():
    """Example usage of survival analysis models"""
    
    # Create data processor
    processor = SurvivalDataProcessor()
    
    # Generate synthetic data
    X, y, df = processor.create_synthetic_survival_data(n_samples=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create models
    models = {
        'cox_ph': CoxProportionalHazardsModel(alpha=1.0),
        'random_survival_forest': RandomSurvivalForestModel(n_estimators=50),
        'kaplan_meier': KaplanMeierModel()
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
            print(f"  {metric}: {value:.4f}")
    
    # Test blood viability prediction
    print("\nTesting blood viability prediction...")
    
    # Create sample blood units
    sample_features = np.array([
        [4.0, 60.0, 500, 0],  # Normal conditions
        [6.0, 80.0, 450, 0],  # High temp and humidity
        [2.0, 40.0, 550, 2],  # Low temp and humidity, A- blood
    ])
    
    collection_dates = np.array([
        datetime.now() - timedelta(days=10),
        datetime.now() - timedelta(days=20),
        datetime.now() - timedelta(days=5)
    ])
    
    # Use Cox model for viability prediction
    cox_model = models['cox_ph']
    viability_predictor = BloodViabilityPredictor(cox_model)
    
    viability_results = viability_predictor.predict_viability_risk(
        sample_features, collection_dates
    )
    
    print("Viability Predictions:")
    for i in range(len(sample_features)):
        print(f"Blood Unit {i+1}:")
        print(f"  Survival Probability: {viability_results['survival_probability'][i]:.3f}")
        print(f"  Risk Score: {viability_results['risk_score'][i]:.3f}")
        print(f"  Predicted Expiry: {viability_results['predicted_expiry_date'][i].strftime('%Y-%m-%d')}")
        print(f"  Days Since Collection: {viability_results['days_since_collection'][i]}")

if __name__ == "__main__":
    main()
