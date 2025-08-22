"""
Federated Learning Client for Crimson IQ Inventory System
Implements Flower client for federated training with local ML models
"""

import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.common import (
    FitRes, EvaluateRes, Parameters, Scalar, NDArrays, 
    parameters_to_ndarrays, ndarrays_to_parameters
)

from ml_models.survival_analysis.models import CoxProportionalHazardsModel, SurvivalModel
from ml_models.inference import MLPredictor

logger = logging.getLogger(__name__)

class CrimsonFLClient(fl.client.NumPyClient):
    """Federated Learning Client for Crimson IQ"""
    
    def __init__(self, 
                 node_name: str,
                 node_type: str,
                 model_type: str = "survival_analysis",
                 data_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.node_name = node_name
        self.node_type = node_type
        self.model_type = model_type
        self.data_path = data_path
        self.config = config or {}
        
        # Initialize local model
        self.model = self._initialize_model()
        self.local_data = None
        self.is_training = False
        
        # Load local data
        self._load_local_data()
        
        # Metrics tracking
        self.training_rounds = 0
        self.total_training_time = 0.0
        
    def _initialize_model(self) -> SurvivalModel:
        """Initialize the local ML model"""
        if self.model_type == "survival_analysis":
            model_config = {
                "penalizer": self.config.get("penalizer", 0.1),
                "l1_ratio": self.config.get("l1_ratio", 0.0)
            }
            return CoxProportionalHazardsModel(model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _load_local_data(self):
        """Load local training data"""
        try:
            if self.data_path and os.path.exists(self.data_path):
                # Load from file
                self.local_data = pd.read_csv(self.data_path)
                logger.info(f"Loaded {len(self.local_data)} samples from {self.data_path}")
            else:
                # Generate synthetic data for demonstration
                self.local_data = self._generate_synthetic_data()
                logger.info(f"Generated {len(self.local_data)} synthetic samples")
                
        except Exception as e:
            logger.error(f"Failed to load local data: {e}")
            self.local_data = self._generate_synthetic_data()
            
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data for federated learning"""
        np.random.seed(42)  # For reproducibility
        
        n_samples = 1000
        
        # Generate synthetic blood unit data
        blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        temperatures = np.random.normal(4.0, 1.0, n_samples)  # Optimal storage temperature
        quantities = np.random.uniform(200, 500, n_samples)  # ml
        
        # Generate donation dates
        base_date = datetime.now() - pd.Timedelta(days=365)
        donation_dates = [base_date + pd.Timedelta(days=np.random.randint(0, 365)) 
                         for _ in range(n_samples)]
        
        # Generate survival times (days until expiration or usage)
        survival_times = np.random.exponential(30, n_samples)  # Mean 30 days
        events = np.random.binomial(1, 0.7, n_samples)  # 70% used, 30% expired
        
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
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters"""
        if self.model.model is None:
            # Return random parameters if model not trained
            return [np.random.randn(10, 1)]  # Placeholder
            
        # Extract parameters from the model
        if hasattr(self.model.model, 'params_'):
            # For lifelines models
            params = self.model.model.params_
            return [params.values.reshape(-1, 1)]
        else:
            # Fallback to random parameters
            return [np.random.randn(10, 1)]
            
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters"""
        if not parameters:
            return
            
        # Update model parameters
        if hasattr(self.model.model, 'params_'):
            # For lifelines models, we need to reconstruct the model
            # This is a simplified approach
            pass
            
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data"""
        start_time = time.time()
        self.is_training = True
        
        try:
            # Set parameters
            self.set_parameters(parameters)
            
            # Prepare data for training
            if self.local_data is None or len(self.local_data) == 0:
                raise ValueError("No local data available for training")
                
            # Train the model
            self.model.fit(self.local_data)
            
            # Evaluate local performance
            metrics = self.model.evaluate(self.local_data)
            
            # Get updated parameters
            updated_parameters = self.get_parameters(config)
            
            # Calculate training time
            training_time = time.time() - start_time
            self.total_training_time += training_time
            self.training_rounds += 1
            
            # Prepare response metrics
            response_metrics = {
                "client_size": len(self.local_data),
                "loss": metrics.get("concordance_index", 0.0),
                "accuracy": metrics.get("concordance_index", 0.0),
                "response_time": training_time,
                "node_name": self.node_name,
                "node_type": self.node_type,
                "training_rounds": self.training_rounds,
                "total_training_time": self.total_training_time
            }
            
            logger.info(f"Local training completed in {training_time:.2f}s, "
                       f"Loss: {response_metrics['loss']:.4f}")
            
            return updated_parameters, len(self.local_data), response_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Return original parameters on failure
            return parameters, 0, {"error": str(e)}
        finally:
            self.is_training = False
            
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local data"""
        try:
            # Set parameters
            self.set_parameters(parameters)
            
            if self.local_data is None or len(self.local_data) == 0:
                return 0.0, 0, {"error": "No local data available"}
                
            # Evaluate the model
            metrics = self.model.evaluate(self.local_data)
            
            # Calculate loss (1 - concordance index for survival models)
            loss = 1.0 - metrics.get("concordance_index", 0.0)
            
            evaluation_metrics = {
                "concordance_index": metrics.get("concordance_index", 0.0),
                "node_name": self.node_name,
                "node_type": self.node_type
            }
            
            return loss, len(self.local_data), evaluation_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 1.0, 0, {"error": str(e)}
            
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information"""
        return {
            "node_name": self.node_name,
            "node_type": self.node_type,
            "model_type": self.model_type,
            "data_size": len(self.local_data) if self.local_data is not None else 0,
            "training_rounds": self.training_rounds,
            "total_training_time": self.total_training_time,
            "is_training": self.is_training
        }

class FederatedClientManager:
    """Manager for federated learning clients"""
    
    def __init__(self, server_address: str = "localhost:8080"):
        self.server_address = server_address
        self.client = None
        self.is_connected = False
        
    def start_client(self, node_name: str, node_type: str, 
                    model_type: str = "survival_analysis",
                    data_path: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None):
        """Start federated learning client"""
        try:
            # Create client
            self.client = CrimsonFLClient(
                node_name=node_name,
                node_type=node_type,
                model_type=model_type,
                data_path=data_path,
                config=config
            )
            
            # Start Flower client
            fl.client.start_numpy_client(
                server_address=self.server_address,
                client=self.client
            )
            
            self.is_connected = True
            logger.info(f"Federated learning client started for {node_name}")
            
        except Exception as e:
            logger.error(f"Failed to start client: {e}")
            raise
            
    def stop_client(self):
        """Stop federated learning client"""
        self.is_connected = False
        logger.info("Federated learning client stopped")
        
    def get_client_status(self) -> Dict[str, Any]:
        """Get client status"""
        if self.client is None:
            return {"status": "not_initialized"}
            
        return {
            "status": "connected" if self.is_connected else "disconnected",
            "client_info": self.client.get_client_info(),
            "server_address": self.server_address
        }

def main():
    """Main entry point for federated learning client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crimson IQ Federated Learning Client")
    parser.add_argument("--server", type=str, default="localhost:8080", 
                       help="Server address")
    parser.add_argument("--node-name", type=str, required=True, 
                       help="Node name")
    parser.add_argument("--node-type", type=str, default="hospital", 
                       choices=["hospital", "blood_bank", "clinic"],
                       help="Node type")
    parser.add_argument("--model-type", type=str, default="survival_analysis",
                       choices=["survival_analysis", "time_series", "reinforcement"],
                       help="Model type")
    parser.add_argument("--data-path", type=str, help="Path to local data file")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create and start client manager
    client_manager = FederatedClientManager(args.server)
    
    try:
        client_manager.start_client(
            node_name=args.node_name,
            node_type=args.node_type,
            model_type=args.model_type,
            data_path=args.data_path,
            config=config
        )
        
        # Keep client running
        while client_manager.is_connected:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        client_manager.stop_client()

if __name__ == "__main__":
    main()
