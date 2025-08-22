"""
Phase 1.4: Inference Pipelines and APIs
Provides batch and streaming inference for time-series forecasting and survival analysis models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import joblib
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Import our models
from time_series_forecasting.models import (
    ARIMAForecaster, ExponentialSmoothingForecaster, RandomForestForecaster
)
from survival_analysis.models import (
    CoxProportionalHazardsModel, KaplanMeierModel, SurvivalDataProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        
    def register_model(self, model_name: str, model_path: str, model_type: str):
        """Register a model in the registry"""
        registry_file = self.models_dir / "model_registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
            
        registry[model_name] = {
            'path': model_path,
            'type': model_type,
            'registered_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            
        logger.info(f"Registered model: {model_name}")
        
    def load_model(self, model_name: str) -> Any:
        """Load a model from the registry"""
        registry_file = self.models_dir / "model_registry.json"
        
        if not registry_file.exists():
            raise ValueError("Model registry not found")
            
        with open(registry_file, 'r') as f:
            registry = json.load(f)
            
        if model_name not in registry:
            raise ValueError(f"Model {model_name} not found in registry")
            
        model_info = registry[model_name]
        model_path = model_info['path']
        model_type = model_info['type']
        
        # Load model based on type
        if model_type == 'time_series':
            model = self._load_time_series_model(model_path)
        elif model_type == 'survival':
            model = self._load_survival_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.loaded_models[model_name] = model
        logger.info(f"Loaded model: {model_name}")
        
        return model
        
    def _load_time_series_model(self, model_path: str):
        """Load a time series model"""
        model = joblib.load(model_path)
        return model
        
    def _load_survival_model(self, model_path: str):
        """Load a survival analysis model"""
        model = joblib.load(model_path)
        return model
        
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available models"""
        registry_file = self.models_dir / "model_registry.json"
        
        if not registry_file.exists():
            return {}
            
        with open(registry_file, 'r') as f:
            registry = json.load(f)
            
        return registry

class TimeSeriesInferenceEngine:
    """Inference engine for time series forecasting models"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.models = {}
        
    def load_models(self, model_names: List[str]):
        """Load multiple time series models"""
        for model_name in model_names:
            try:
                model = self.registry.load_model(model_name)
                self.models[model_name] = model
                logger.info(f"Loaded time series model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                
    def predict_demand(self, data: pd.DataFrame, model_name: str, 
                      steps: int = 24, blood_group: str = None) -> Dict[str, Any]:
        """Predict blood demand using specified model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model = self.models[model_name]
        
        try:
            # Prepare data for prediction
            if blood_group:
                # Filter by blood group if specified
                if 'blood_group' in data.columns:
                    data = data[data['blood_group'] == blood_group]
                    
            # Make prediction
            predictions = model.predict(data, steps=steps)
            
            # Prepare response
            response = {
                'model_name': model_name,
                'blood_group': blood_group,
                'prediction_steps': steps,
                'predictions': predictions.tolist(),
                'timestamp': datetime.now().isoformat(),
                'confidence_interval': self._calculate_confidence_interval(predictions)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error making demand prediction: {e}")
            raise
            
    def batch_predict_demand(self, data_batch: List[pd.DataFrame], 
                           model_name: str, steps: int = 24) -> List[Dict[str, Any]]:
        """Make batch predictions for multiple datasets"""
        
        results = []
        
        for i, data in enumerate(data_batch):
            try:
                result = self.predict_demand(data, model_name, steps)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
        return results
        
    def _calculate_confidence_interval(self, predictions: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, List[float]]:
        """Calculate confidence intervals for predictions"""
        # Simple confidence interval calculation
        # In practice, this would be more sophisticated based on model type
        
        std_dev = np.std(predictions)
        z_score = 1.96  # 95% confidence level
        
        margin_of_error = z_score * std_dev / np.sqrt(len(predictions))
        
        return {
            'lower_bound': (predictions - margin_of_error).tolist(),
            'upper_bound': (predictions + margin_of_error).tolist(),
            'confidence_level': confidence_level
        }

class SurvivalInferenceEngine:
    """Inference engine for survival analysis models"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.models = {}
        self.data_processor = SurvivalDataProcessor()
        
    def load_models(self, model_names: List[str]):
        """Load multiple survival analysis models"""
        for model_name in model_names:
            try:
                model = self.registry.load_model(model_name)
                self.models[model_name] = model
                logger.info(f"Loaded survival model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                
    def predict_viability_risk(self, blood_unit_data: pd.DataFrame, 
                             model_name: str, prediction_times: List[float] = None) -> Dict[str, Any]:
        """Predict blood unit viability risk using survival analysis"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        model = self.models[model_name]
        
        try:
            # Prepare data for survival analysis
            survival_data = self.data_processor.prepare_survival_data(blood_unit_data)
            
            if prediction_times is None:
                prediction_times = [7, 14, 21, 30, 35, 42]  # Days from donation
                
            # Make predictions
            survival_probs = model.predict_survival(survival_data, prediction_times)
            hazard_rates = model.predict_hazard(survival_data, prediction_times)
            
            # Calculate risk scores
            risk_scores = self._calculate_risk_scores(survival_probs, prediction_times)
            
            # Prepare response
            response = {
                'model_name': model_name,
                'prediction_times': prediction_times,
                'survival_probabilities': survival_probs.tolist(),
                'hazard_rates': hazard_rates.tolist(),
                'risk_scores': risk_scores,
                'timestamp': datetime.now().isoformat(),
                'blood_units_analyzed': len(survival_data)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error predicting viability risk: {e}")
            raise
            
    def batch_predict_viability(self, blood_unit_batches: List[pd.DataFrame], 
                              model_name: str) -> List[Dict[str, Any]]:
        """Make batch viability predictions"""
        
        results = []
        
        for i, batch_data in enumerate(blood_unit_batches):
            try:
                result = self.predict_viability_risk(batch_data, model_name)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch viability prediction {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
        return results
        
    def _calculate_risk_scores(self, survival_probs: np.ndarray, 
                             prediction_times: List[float]) -> Dict[str, float]:
        """Calculate risk scores based on survival probabilities"""
        
        # Risk score is inverse of survival probability
        risk_scores = 1 - survival_probs
        
        # Calculate various risk metrics
        risk_metrics = {
            'immediate_risk_7d': float(risk_scores[0, 0]) if risk_scores.shape[0] > 0 else 0.0,
            'short_term_risk_14d': float(risk_scores[0, 1]) if risk_scores.shape[1] > 1 else 0.0,
            'medium_term_risk_30d': float(risk_scores[0, 3]) if risk_scores.shape[1] > 3 else 0.0,
            'expiry_risk_42d': float(risk_scores[0, -1]) if risk_scores.shape[1] > 0 else 0.0,
            'average_risk': float(np.mean(risk_scores)),
            'max_risk': float(np.max(risk_scores))
        }
        
        return risk_metrics

class InferenceAPI:
    """REST API for inference endpoints"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.ts_engine = TimeSeriesInferenceEngine(model_registry)
        self.survival_engine = SurvivalInferenceEngine(model_registry)
        
    def initialize_models(self):
        """Initialize all available models"""
        available_models = self.registry.get_available_models()
        
        ts_models = [name for name, info in available_models.items() 
                    if info['type'] == 'time_series']
        survival_models = [name for name, info in available_models.items() 
                          if info['type'] == 'survival']
        
        if ts_models:
            self.ts_engine.load_models(ts_models)
            
        if survival_models:
            self.survival_engine.load_models(survival_models)
            
        logger.info(f"Initialized {len(ts_models)} time series models and {len(survival_models)} survival models")
        
    def demand_forecast_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for demand forecasting"""
        
        try:
            # Extract parameters
            data = pd.DataFrame(request_data.get('data', []))
            model_name = request_data.get('model_name', 'RandomForest')
            steps = request_data.get('steps', 24)
            blood_group = request_data.get('blood_group')
            
            # Validate input
            if data.empty:
                return {'error': 'No data provided'}
                
            # Make prediction
            result = self.ts_engine.predict_demand(data, model_name, steps, blood_group)
            
            return {
                'status': 'success',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error in demand forecast endpoint: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def viability_risk_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for viability risk assessment"""
        
        try:
            # Extract parameters
            data = pd.DataFrame(request_data.get('data', []))
            model_name = request_data.get('model_name', 'CoxPH')
            prediction_times = request_data.get('prediction_times', [7, 14, 21, 30, 35, 42])
            
            # Validate input
            if data.empty:
                return {'error': 'No data provided'}
                
            # Make prediction
            result = self.survival_engine.predict_viability_risk(data, model_name, prediction_times)
            
            return {
                'status': 'success',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error in viability risk endpoint: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def batch_inference_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for batch inference"""
        
        try:
            inference_type = request_data.get('type')  # 'demand' or 'viability'
            data_batches = [pd.DataFrame(batch) for batch in request_data.get('data_batches', [])]
            model_name = request_data.get('model_name')
            
            if inference_type == 'demand':
                steps = request_data.get('steps', 24)
                results = self.ts_engine.batch_predict_demand(data_batches, model_name, steps)
            elif inference_type == 'viability':
                results = self.survival_engine.batch_predict_viability(data_batches, model_name)
            else:
                return {'error': 'Invalid inference type'}
                
            return {
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in batch inference endpoint: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

class StreamingInferenceEngine:
    """Real-time streaming inference engine"""
    
    def __init__(self, inference_api: InferenceAPI):
        self.api = inference_api
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = asyncio.Queue()
        
    async def start_streaming(self):
        """Start the streaming inference engine"""
        logger.info("Starting streaming inference engine")
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._process_queue()),
            asyncio.create_task(self._health_check())
        ]
        
        await asyncio.gather(*tasks)
        
    async def _process_queue(self):
        """Process items from the inference queue"""
        while True:
            try:
                # Get item from queue
                item = await self.processing_queue.get()
                
                # Process inference request
                if item['type'] == 'demand':
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, 
                        self.api.demand_forecast_endpoint, 
                        item['data']
                    )
                elif item['type'] == 'viability':
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.api.viability_risk_endpoint,
                        item['data']
                    )
                else:
                    result = {'error': 'Unknown inference type'}
                    
                # Send result back
                if 'callback' in item:
                    await item['callback'](result)
                    
            except Exception as e:
                logger.error(f"Error processing queue item: {e}")
                
            finally:
                self.processing_queue.task_done()
                
    async def _health_check(self):
        """Periodic health check"""
        while True:
            try:
                queue_size = self.processing_queue.qsize()
                logger.info(f"Streaming engine health check - Queue size: {queue_size}")
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
    async def submit_inference_request(self, inference_type: str, data: Dict[str, Any], 
                                     callback=None) -> str:
        """Submit an inference request to the streaming engine"""
        
        request_id = f"{inference_type}_{int(time.time() * 1000)}"
        
        await self.processing_queue.put({
            'id': request_id,
            'type': inference_type,
            'data': data,
            'callback': callback,
            'timestamp': datetime.now().isoformat()
        })
        
        return request_id

def main():
    """Main function for inference pipeline"""
    
    # Initialize model registry
    registry = ModelRegistry()
    
    # Register models (assuming they exist)
    try:
        registry.register_model("randomforest_forecast", "models/randomforest_model.pkl", "time_series")
        registry.register_model("coxph_survival", "models/coxph_survival_model.pkl", "survival")
        logger.info("Registered models in registry")
    except Exception as e:
        logger.warning(f"Could not register models: {e}")
    
    # Initialize inference API
    api = InferenceAPI(registry)
    api.initialize_models()
    
    # Example usage
    logger.info("Inference pipeline initialized successfully")
    
    # Test demand forecasting
    try:
        # Load sample data
        demand_data = pd.read_parquet("processed_data/demand_processed.parquet")
        
        # Test demand forecast
        forecast_request = {
            'data': demand_data.head(100).to_dict('records'),
            'model_name': 'RandomForest',
            'steps': 24,
            'blood_group': 'O+'
        }
        
        forecast_result = api.demand_forecast_endpoint(forecast_request)
        logger.info("Demand forecast test completed")
        
    except Exception as e:
        logger.warning(f"Demand forecast test failed: {e}")
    
    # Test viability risk assessment
    try:
        # Load sample inventory data
        inventory_data = pd.read_parquet("processed_data/inventory_processed.parquet")
        
        # Test viability risk
        viability_request = {
            'data': inventory_data.head(50).to_dict('records'),
            'model_name': 'CoxPH',
            'prediction_times': [7, 14, 21, 30, 35, 42]
        }
        
        viability_result = api.viability_risk_endpoint(viability_request)
        logger.info("Viability risk assessment test completed")
        
    except Exception as e:
        logger.warning(f"Viability risk assessment test failed: {e}")
    
    logger.info("Inference pipeline testing completed")

if __name__ == "__main__":
    main()
