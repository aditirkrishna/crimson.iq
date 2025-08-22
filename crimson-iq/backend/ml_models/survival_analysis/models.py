"""
Phase 1.3: Survival Analysis Models for Blood Viability
Implements Cox Proportional Hazards and Kaplan-Meier estimators for blood unit survival analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from enum import Enum

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class BloodGroup(Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class SurvivalModel:
    """Base class for survival analysis models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        
    def fit(self, data: pd.DataFrame) -> 'SurvivalModel':
        """Fit the survival model"""
        raise NotImplementedError
        
    def predict_survival(self, data: pd.DataFrame, times: List[float]) -> np.ndarray:
        """Predict survival probabilities at given times"""
        raise NotImplementedError
        
    def predict_hazard(self, data: pd.DataFrame, times: List[float]) -> np.ndarray:
        """Predict hazard rates at given times"""
        raise NotImplementedError
        
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        raise NotImplementedError
        
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

class CoxProportionalHazardsModel(SurvivalModel):
    """Cox Proportional Hazards model for survival analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("CoxPH", config)
        self.penalizer = config.get('penalizer', 0.1)
        self.l1_ratio = config.get('l1_ratio', 0.0)
        
    def fit(self, data: pd.DataFrame) -> 'CoxProportionalHazardsModel':
        """Fit Cox Proportional Hazards model"""
        logger.info("Fitting Cox Proportional Hazards model")
        
        # Prepare data for lifelines
        # lifelines expects: duration (time to event), event (1 if event occurred, 0 if censored)
        if 'duration' not in data.columns or 'event' not in data.columns:
            raise ValueError("Data must contain 'duration' and 'event' columns")
            
        # Identify feature columns (exclude duration, event, and timestamp columns)
        exclude_cols = ['duration', 'event', 'timestamp', 'entry_time', 'exit_time']
        self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Create model
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        
        # Fit model
        try:
            self.model.fit(data, duration_col='duration', event_col='event')
            self.is_fitted = True
            logger.info("Cox Proportional Hazards model fitted successfully")
            
            # Print model summary
            logger.info("Model Summary:")
            logger.info(f"Concordance Index: {self.model.concordance_index_:.4f}")
            logger.info(f"Log-likelihood: {self.model.log_likelihood_:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting Cox model: {e}")
            raise
            
        return self
        
    def predict_survival(self, data: pd.DataFrame, times: List[float]) -> np.ndarray:
        """Predict survival probabilities at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            survival_probs = self.model.predict_survival_function(data, times=times)
            return survival_probs.values.T  # Transpose to get shape (n_samples, n_times)
        except Exception as e:
            logger.error(f"Error predicting survival: {e}")
            return np.ones((len(data), len(times)))
            
    def predict_hazard(self, data: pd.DataFrame, times: List[float]) -> np.ndarray:
        """Predict hazard rates at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            hazard_rates = self.model.predict_partial_hazard(data)
            return hazard_rates.values.reshape(-1, 1)  # Shape (n_samples, 1)
        except Exception as e:
            logger.error(f"Error predicting hazard: {e}")
            return np.ones((len(data), 1))
            
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        try:
            # Calculate concordance index
            c_index = self.model.concordance_index_
            
            # Calculate log-likelihood
            log_likelihood = self.model.log_likelihood_
            
            # Calculate AIC
            aic = self.model.AIC_
            
            # Calculate BIC
            bic = self.model.BIC_
            
            metrics = {
                'concordance_index': c_index,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {
                'concordance_index': 0.0,
                'log_likelihood': float('-inf'),
                'aic': float('inf'),
                'bic': float('inf')
            }
            
    def plot_partial_effects(self, feature: str, save_path: str = None):
        """Plot partial effects of a feature on survival"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        try:
            plt.figure(figsize=(10, 6))
            self.model.plot_partial_effects(feature)
            plt.title(f'Partial Effects of {feature} on Survival')
            plt.xlabel('Time (days)')
            plt.ylabel('Survival Probability')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Partial effects plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting partial effects: {e}")

class KaplanMeierModel(SurvivalModel):
    """Kaplan-Meier estimator for non-parametric survival analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("KaplanMeier", config)
        self.confidence_interval = config.get('confidence_interval', 0.95)
        
    def fit(self, data: pd.DataFrame) -> 'KaplanMeierModel':
        """Fit Kaplan-Meier estimator"""
        logger.info("Fitting Kaplan-Meier estimator")
        
        # Prepare data
        if 'duration' not in data.columns or 'event' not in data.columns:
            raise ValueError("Data must contain 'duration' and 'event' columns")
            
        # Create model
        self.model = KaplanMeierFitter()
        
        # Fit model
        try:
            self.model.fit(data['duration'], data['event'])
            self.is_fitted = True
            logger.info("Kaplan-Meier estimator fitted successfully")
            
            # Print summary
            logger.info("Model Summary:")
            logger.info(f"Median survival time: {self.model.median_survival_time_:.2f} days")
            logger.info(f"Mean survival time: {self.model.mean_survival_time_:.2f} days")
            
        except Exception as e:
            logger.error(f"Error fitting Kaplan-Meier model: {e}")
            raise
            
        return self
        
    def predict_survival(self, data: pd.DataFrame = None, times: List[float] = None) -> np.ndarray:
        """Predict survival probabilities at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            if times is None:
                # Use the times from the fitted model
                survival_probs = self.model.survival_function_.values
                return survival_probs.reshape(1, -1)  # Shape (1, n_times)
            else:
                survival_probs = self.model.survival_function_at_times(times)
                return survival_probs.reshape(1, -1)  # Shape (1, n_times)
                
        except Exception as e:
            logger.error(f"Error predicting survival: {e}")
            return np.ones((1, len(times) if times else 1))
            
    def predict_hazard(self, data: pd.DataFrame = None, times: List[float] = None) -> np.ndarray:
        """Predict hazard rates at given times"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            if times is None:
                # Use the times from the fitted model
                hazard_rates = self.model.hazard_at_times(self.model.timeline)
                return hazard_rates.reshape(1, -1)  # Shape (1, n_times)
            else:
                hazard_rates = self.model.hazard_at_times(times)
                return hazard_rates.reshape(1, -1)  # Shape (1, n_times)
                
        except Exception as e:
            logger.error(f"Error predicting hazard: {e}")
            return np.zeros((1, len(times) if times else 1))
            
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        try:
            # Calculate median survival time
            median_survival = self.model.median_survival_time_
            
            # Calculate mean survival time
            mean_survival = self.model.mean_survival_time_
            
            # Calculate survival probability at specific times
            survival_at_7_days = self.model.survival_function_at_times(7)
            survival_at_14_days = self.model.survival_function_at_times(14)
            survival_at_30_days = self.model.survival_function_at_times(30)
            
            metrics = {
                'median_survival_time': median_survival,
                'mean_survival_time': mean_survival,
                'survival_at_7_days': survival_at_7_days,
                'survival_at_14_days': survival_at_14_days,
                'survival_at_30_days': survival_at_30_days
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {
                'median_survival_time': 0.0,
                'mean_survival_time': 0.0,
                'survival_at_7_days': 0.0,
                'survival_at_14_days': 0.0,
                'survival_at_30_days': 0.0
            }
            
    def plot_survival_curve(self, save_path: str = None):
        """Plot survival curve"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot survival function
            plt.subplot(2, 2, 1)
            self.model.plot_survival_function()
            plt.title('Survival Function')
            plt.xlabel('Time (days)')
            plt.ylabel('Survival Probability')
            plt.grid(True, alpha=0.3)
            
            # Plot cumulative hazard
            plt.subplot(2, 2, 2)
            self.model.plot_cumulative_hazard()
            plt.title('Cumulative Hazard')
            plt.xlabel('Time (days)')
            plt.ylabel('Cumulative Hazard')
            plt.grid(True, alpha=0.3)
            
            # Plot hazard function
            plt.subplot(2, 2, 3)
            self.model.plot_hazard()
            plt.title('Hazard Function')
            plt.xlabel('Time (days)')
            plt.ylabel('Hazard Rate')
            plt.grid(True, alpha=0.3)
            
            # Plot cumulative density
            plt.subplot(2, 2, 4)
            self.model.plot_cumulative_density()
            plt.title('Cumulative Density')
            plt.xlabel('Time (days)')
            plt.ylabel('Cumulative Density')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Survival curves plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting survival curves: {e}")

class SurvivalDataProcessor:
    """Processes data for survival analysis"""
    
    def __init__(self):
        self.blood_shelf_life = 42  # days
        
    def prepare_survival_data(self, inventory_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare inventory data for survival analysis"""
        logger.info("Preparing survival analysis data")
        
        # Create survival analysis dataset
        survival_data = []
        
        for _, row in inventory_data.iterrows():
            # Calculate duration (time from donation to expiry or use)
            donation_date = pd.to_datetime(row['donation_date'])
            expiry_date = pd.to_datetime(row['expiry_date'])
            current_time = pd.to_datetime(row['timestamp'])
            
            # Calculate time to event
            if row['status'] == 'used':
                # Blood was used - event occurred
                event_time = current_time
                event = 1
            elif current_time >= expiry_date:
                # Blood expired - event occurred
                event_time = expiry_date
                event = 1
            else:
                # Blood still available - censored
                event_time = current_time
                event = 0
                
            # Calculate duration in days
            duration = (event_time - donation_date).days
            
            # Only include blood units that have been in inventory for at least 1 day
            if duration > 0:
                survival_data.append({
                    'blood_unit_id': row['blood_unit_id'],
                    'pod_id': row['pod_id'],
                    'blood_group': row['blood_group'],
                    'donation_date': donation_date,
                    'entry_time': donation_date,
                    'exit_time': event_time,
                    'duration': duration,
                    'event': event,
                    'status': row['status'],
                    'temperature': row['temperature'],
                    'quantity': row['quantity']
                })
        
        survival_df = pd.DataFrame(survival_data)
        
        # Add time-based features
        survival_df = self._add_time_features(survival_df)
        
        # Add temperature-based features
        survival_df = self._add_temperature_features(survival_df)
        
        logger.info(f"Prepared survival data with {len(survival_df)} blood units")
        return survival_df
        
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = data.copy()
        
        # Add day of week features
        df['donation_day_of_week'] = df['donation_date'].dt.dayofweek
        df['donation_month'] = df['donation_date'].dt.month
        df['donation_quarter'] = df['donation_date'].dt.quarter
        
        # Add seasonal features
        df['is_weekend_donation'] = df['donation_day_of_week'].isin([5, 6]).astype(int)
        df['is_summer'] = df['donation_month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['donation_month'].isin([12, 1, 2]).astype(int)
        
        return df
        
    def _add_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temperature-based features"""
        df = data.copy()
        
        # Temperature categories
        df['temp_category'] = pd.cut(
            df['temperature'], 
            bins=[-np.inf, 2.5, 4.0, 5.5, np.inf],
            labels=['very_cold', 'cold', 'optimal', 'warm']
        )
        
        # Temperature deviation from optimal
        df['temp_deviation'] = abs(df['temperature'] - 4.0)
        
        # Temperature stability (simplified - could be enhanced with rolling statistics)
        df['temp_stable'] = (df['temp_deviation'] < 0.5).astype(int)
        
        return df
        
    def create_stratified_data(self, data: pd.DataFrame, stratify_by: str = 'blood_group') -> Dict[str, pd.DataFrame]:
        """Create stratified datasets for different groups"""
        stratified_data = {}
        
        for group in data[stratify_by].unique():
            group_data = data[data[stratify_by] == group].copy()
            stratified_data[group] = group_data
            
        return stratified_data

class SurvivalModelEvaluator:
    """Evaluates and compares survival analysis models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model: SurvivalModel):
        """Add a model to the evaluator"""
        self.models[name] = model
        
    def evaluate_models(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on the data"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name} model...")
            
            try:
                # Fit model
                model.fit(data)
                
                # Evaluate model
                metrics = model.evaluate(data)
                results[name] = metrics
                
                logger.info(f"{name} evaluation completed")
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
                results[name] = {}
                
        self.results = results
        return results
        
    def compare_survival_curves(self, data: pd.DataFrame, stratify_by: str = 'blood_group', save_path: str = None):
        """Compare survival curves across different groups"""
        
        # Create stratified data
        processor = SurvivalDataProcessor()
        stratified_data = processor.create_stratified_data(data, stratify_by)
        
        # Fit Kaplan-Meier models for each group
        km_models = {}
        for group, group_data in stratified_data.items():
            if len(group_data) > 10:  # Only fit if enough data
                km_model = KaplanMeierModel()
                try:
                    km_model.fit(group_data)
                    km_models[group] = km_model
                except Exception as e:
                    logger.warning(f"Could not fit KM model for {group}: {e}")
        
        # Plot comparison
        if km_models:
            plt.figure(figsize=(12, 8))
            
            for group, model in km_models.items():
                model.model.plot_survival_function(label=group)
            
            plt.title(f'Survival Curves by {stratify_by}')
            plt.xlabel('Time (days)')
            plt.ylabel('Survival Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Survival curves comparison saved to {save_path}")
            
            plt.show()
            
            # Perform log-rank test for pairwise comparisons
            if len(km_models) > 1:
                self._perform_logrank_tests(km_models, data, stratify_by)
        
    def _perform_logrank_tests(self, km_models: Dict[str, KaplanMeierModel], data: pd.DataFrame, stratify_by: str):
        """Perform log-rank tests for pairwise comparisons"""
        groups = list(km_models.keys())
        
        print(f"\nLog-Rank Test Results for {stratify_by}:")
        print("=" * 50)
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                
                # Get data for each group
                group1_data = data[data[stratify_by] == group1]
                group2_data = data[data[stratify_by] == group2]
                
                # Perform log-rank test
                try:
                    test_result = logrank_test(
                        group1_data['duration'], group2_data['duration'],
                        group1_data['event'], group2_data['event']
                    )
                    
                    print(f"{group1} vs {group2}:")
                    print(f"  Test statistic: {test_result.test_statistic:.4f}")
                    print(f"  p-value: {test_result.p_value:.4f}")
                    print(f"  Significant: {'Yes' if test_result.p_value < 0.05 else 'No'}")
                    print()
                    
                except Exception as e:
                    logger.warning(f"Could not perform log-rank test for {group1} vs {group2}: {e}")

def main():
    """Main function for survival analysis pipeline"""
    
    # Load processed data
    try:
        inventory_data = pd.read_parquet("processed_data/inventory_processed.parquet")
        logger.info("Loaded processed inventory data")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run the data preprocessing pipeline first.")
        return
    
    # Prepare survival analysis data
    processor = SurvivalDataProcessor()
    survival_data = processor.prepare_survival_data(inventory_data)
    
    # Split data for evaluation
    train_size = int(0.8 * len(survival_data))
    train_data = survival_data[:train_size]
    test_data = survival_data[train_size:]
    
    logger.info(f"Training data: {len(train_data)} blood units")
    logger.info(f"Test data: {len(test_data)} blood units")
    
    # Initialize models
    models = {
        'CoxPH': CoxProportionalHazardsModel({'penalizer': 0.1}),
        'KaplanMeier': KaplanMeierModel()
    }
    
    # Initialize evaluator
    evaluator = SurvivalModelEvaluator()
    for name, model in models.items():
        evaluator.add_model(name, model)
    
    # Evaluate models
    results = evaluator.evaluate_models(train_data)
    
    # Compare survival curves by blood group
    evaluator.compare_survival_curves(
        survival_data, 
        stratify_by='blood_group', 
        save_path="survival_curves_comparison.png"
    )
    
    # Plot individual model results
    for name, model in models.items():
        if name == 'KaplanMeier':
            model.plot_survival_curve(save_path=f"{name.lower()}_survival_curves.png")
        elif name == 'CoxPH' and 'blood_group' in model.feature_columns:
            model.plot_partial_effects('blood_group', save_path=f"{name.lower()}_partial_effects.png")
    
    # Save best model (Cox PH for multivariate analysis)
    if 'CoxPH' in models:
        models['CoxPH'].save_model("models/coxph_survival_model.pkl")
    
    # Print summary
    print("\n" + "="*50)
    print("SURVIVAL ANALYSIS SUMMARY")
    print("="*50)
    for name, metrics in results.items():
        print(f"\n{name} Model:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("="*50)

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    main()
