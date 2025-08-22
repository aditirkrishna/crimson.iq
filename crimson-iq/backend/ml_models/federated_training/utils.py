"""
Utility functions for federated learning workflows
Includes model adaptation, data loading, and evaluation metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
import os
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FederatedModelAdapter:
    """Adapter for converting between federated and local model formats"""
    
    @staticmethod
    def model_to_weights(model: Any) -> List[np.ndarray]:
        """Convert model to weight arrays"""
        if hasattr(model, 'params_'):
            # For lifelines models (Cox Proportional Hazards)
            params = model.params_
            weights = [params.values.reshape(-1, 1)]
            return weights
        elif hasattr(model, 'get_weights'):
            # For custom models with get_weights method
            return model.get_weights()
        else:
            # Fallback for unknown model types
            logger.warning("Unknown model type, returning placeholder weights")
            return [np.random.randn(10, 1)]
            
    @staticmethod
    def weights_to_model(weights: List[np.ndarray], model_type: str, 
                        model_config: Dict[str, Any]) -> Any:
        """Convert weight arrays back to model"""
        if model_type == "survival_analysis":
            # For survival analysis models, we need to reconstruct the model
            # This is a simplified approach - in practice, you'd need more sophisticated reconstruction
            from ml_models.survival_analysis.models import CoxProportionalHazardsModel
            
            model = CoxProportionalHazardsModel(model_config)
            # Note: This is a placeholder - actual weight reconstruction would be more complex
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    @staticmethod
    def validate_model_compatibility(model1: Any, model2: Any) -> bool:
        """Validate if two models are compatible for federated learning"""
        # Check if models have the same structure
        if hasattr(model1, 'params_') and hasattr(model2, 'params_'):
            return model1.params_.shape == model2.params_.shape
        elif hasattr(model1, 'get_weights') and hasattr(model2, 'get_weights'):
            weights1 = model1.get_weights()
            weights2 = model2.get_weights()
            if len(weights1) != len(weights2):
                return False
            return all(w1.shape == w2.shape for w1, w2 in zip(weights1, weights2))
        else:
            return False

class FederatedDataLoader:
    """Data loader for federated learning scenarios"""
    
    @staticmethod
    def load_local_data(data_path: str, model_type: str) -> pd.DataFrame:
        """Load local data for federated learning"""
        try:
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                logger.info(f"Loaded {len(data)} samples from {data_path}")
                return data
            else:
                logger.warning(f"Data file {data_path} not found, generating synthetic data")
                return FederatedDataLoader.generate_synthetic_data(model_type)
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return FederatedDataLoader.generate_synthetic_data(model_type)
            
    @staticmethod
    def generate_synthetic_data(model_type: str, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic data for federated learning"""
        np.random.seed(42)  # For reproducibility
        
        if model_type == "survival_analysis":
            return FederatedDataLoader._generate_survival_data(n_samples)
        elif model_type == "time_series":
            return FederatedDataLoader._generate_time_series_data(n_samples)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    @staticmethod
    def _generate_survival_data(n_samples: int) -> pd.DataFrame:
        """Generate synthetic survival analysis data"""
        # Generate synthetic blood unit data
        blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        temperatures = np.random.normal(4.0, 1.0, n_samples)
        quantities = np.random.uniform(200, 500, n_samples)
        
        # Generate donation dates
        base_date = datetime.now() - pd.Timedelta(days=365)
        donation_dates = [base_date + pd.Timedelta(days=np.random.randint(0, 365)) 
                         for _ in range(n_samples)]
        
        # Generate survival times
        survival_times = np.random.exponential(30, n_samples)
        events = np.random.binomial(1, 0.7, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'blood_unit_id': [f"UNIT_{i:06d}" for i in range(n_samples)],
            'blood_group': np.random.choice(blood_groups, n_samples),
            'temperature': temperatures,
            'quantity': quantities,
            'donation_date': donation_dates,
            'duration': survival_times,
            'event': events,
            'pod_id': np.random.randint(1, 11, n_samples),
            'status': np.random.choice(['available', 'reserved', 'expired'], n_samples),
            'donation_day_of_week': [d.weekday() for d in donation_dates],
            'donation_month': [d.month for d in donation_dates],
            'is_weekend_donation': [(d.weekday() >= 5) for d in donation_dates],
            'is_summer': [(d.month in [6, 7, 8]) for d in donation_dates],
            'is_winter': [(d.month in [12, 1, 2]) for d in donation_dates],
            'temp_category': pd.cut(temperatures, 
                                  bins=[-np.inf, 2.5, 4.0, 5.5, np.inf],
                                  labels=['very_cold', 'cold', 'optimal', 'warm']),
            'temp_deviation': np.abs(temperatures - 4.0),
            'temp_stable': (np.abs(temperatures - 4.0) < 0.5).astype(int)
        })
        
        return data
        
    @staticmethod
    def _generate_time_series_data(n_samples: int) -> pd.DataFrame:
        """Generate synthetic time series data"""
        # Generate daily demand and supply data
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Generate demand with seasonal patterns
        base_demand = 100
        seasonal_demand = base_demand + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        demand = np.maximum(0, seasonal_demand + np.random.normal(0, 10, n_samples))
        
        # Generate supply (correlated with demand but with some lag)
        supply = np.roll(demand, 7) + np.random.normal(0, 5, n_samples)
        supply = np.maximum(0, supply)
        
        # Generate temperature data
        temperature = 4.0 + np.random.normal(0, 1, n_samples)
        
        data = pd.DataFrame({
            'date': dates,
            'demand': demand,
            'supply': supply,
            'temperature': temperature,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.binomial(1, 0.05, n_samples)  # 5% chance of holiday
        })
        
        return data

class FederatedMetrics:
    """Metrics and evaluation utilities for federated learning"""
    
    @staticmethod
    def calculate_aggregation_metrics(client_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated metrics from client results"""
        if not client_metrics:
            return {}
            
        # Calculate averages
        avg_loss = np.mean([m.get("loss", 0) for m in client_metrics])
        avg_accuracy = np.mean([m.get("accuracy", 0) for m in client_metrics])
        avg_response_time = np.mean([m.get("response_time", 0) for m in client_metrics])
        
        # Calculate standard deviations
        std_loss = np.std([m.get("loss", 0) for m in client_metrics])
        std_accuracy = np.std([m.get("accuracy", 0) for m in client_metrics])
        
        # Calculate client participation
        total_clients = len(client_metrics)
        successful_clients = sum(1 for m in client_metrics if m.get("success", True))
        
        return {
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "avg_response_time": avg_response_time,
            "std_loss": std_loss,
            "std_accuracy": std_accuracy,
            "total_clients": total_clients,
            "successful_clients": successful_clients,
            "participation_rate": successful_clients / total_clients if total_clients > 0 else 0
        }
        
    @staticmethod
    def calculate_privacy_metrics(epsilon: float, delta: float, num_rounds: int) -> Dict[str, float]:
        """Calculate privacy metrics for federated learning"""
        # Cumulative privacy budget
        cumulative_epsilon = epsilon * np.sqrt(2 * num_rounds * np.log(1 / delta))
        cumulative_delta = delta * num_rounds
        
        # Estimate membership inference risk
        if cumulative_epsilon <= 0.1:
            membership_risk = 0.01
        elif cumulative_epsilon <= 1.0:
            membership_risk = 0.05
        elif cumulative_epsilon <= 5.0:
            membership_risk = 0.15
        else:
            membership_risk = 0.30
            
        return {
            "epsilon": epsilon,
            "delta": delta,
            "cumulative_epsilon": cumulative_epsilon,
            "cumulative_delta": cumulative_delta,
            "membership_inference_risk": membership_risk,
            "privacy_level": "high" if cumulative_epsilon <= 1.0 else "medium" if cumulative_epsilon <= 5.0 else "low"
        }
        
    @staticmethod
    def calculate_communication_efficiency(num_clients: int, num_rounds: int, 
                                         avg_message_size: float) -> Dict[str, float]:
        """Calculate communication efficiency metrics"""
        total_messages = num_clients * num_rounds * 2  # 2 messages per round per client
        total_data_transferred = total_messages * avg_message_size
        
        return {
            "total_messages": total_messages,
            "total_data_transferred_mb": total_data_transferred / (1024 * 1024),
            "avg_messages_per_round": num_clients * 2,
            "communication_efficiency": num_clients / total_messages if total_messages > 0 else 0
        }

class FederatedModelPersistence:
    """Utilities for saving and loading federated models"""
    
    @staticmethod
    def save_federated_model(model: Any, model_type: str, filepath: str, 
                           metadata: Optional[Dict[str, Any]] = None):
        """Save federated model with metadata"""
        try:
            # Save model
            joblib.dump(model, filepath)
            
            # Save metadata
            metadata_file = filepath.replace('.pkl', '_metadata.json')
            metadata = metadata or {}
            metadata.update({
                "model_type": model_type,
                "saved_at": datetime.now().isoformat(),
                "filepath": filepath
            })
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved federated model to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save federated model: {e}")
            raise
            
    @staticmethod
    def load_federated_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
        """Load federated model and metadata"""
        try:
            # Load model
            model = joblib.load(filepath)
            
            # Load metadata
            metadata_file = filepath.replace('.pkl', '_metadata.json')
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
            logger.info(f"Loaded federated model from {filepath}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load federated model: {e}")
            raise

class FederatedTrainingMonitor:
    """Monitor federated training progress"""
    
    def __init__(self):
        self.training_history = []
        self.metrics_history = []
        
    def log_round(self, round_num: int, client_metrics: List[Dict[str, Any]], 
                  aggregation_metrics: Dict[str, Any]):
        """Log training round information"""
        round_info = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "client_metrics": client_metrics,
            "aggregation_metrics": aggregation_metrics
        }
        
        self.training_history.append(round_info)
        
        # Extract key metrics for history
        metrics_summary = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "avg_loss": aggregation_metrics.get("avg_loss", 0),
            "avg_accuracy": aggregation_metrics.get("avg_accuracy", 0),
            "participation_rate": aggregation_metrics.get("participation_rate", 0),
            "num_clients": aggregation_metrics.get("total_clients", 0)
        }
        
        self.metrics_history.append(metrics_summary)
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        if not self.metrics_history:
            return {}
            
        losses = [m["avg_loss"] for m in self.metrics_history]
        accuracies = [m["avg_accuracy"] for m in self.metrics_history]
        
        return {
            "total_rounds": len(self.metrics_history),
            "final_loss": losses[-1] if losses else 0,
            "final_accuracy": accuracies[-1] if accuracies else 0,
            "best_loss": min(losses) if losses else 0,
            "best_accuracy": max(accuracies) if accuracies else 0,
            "avg_participation_rate": np.mean([m["participation_rate"] for m in self.metrics_history]),
            "training_start": self.metrics_history[0]["timestamp"] if self.metrics_history else None,
            "training_end": self.metrics_history[-1]["timestamp"] if self.metrics_history else None
        }
        
    def save_training_log(self, filepath: str):
        """Save training log to file"""
        try:
            log_data = {
                "training_history": self.training_history,
                "metrics_history": self.metrics_history,
                "summary": self.get_training_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            logger.info(f"Saved training log to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save training log: {e}")
            raise
