# ML Predictor API and inference pipeline

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Import our models
import sys
sys.path.append('../../ml_models')

from time_series_forecasting.models import (
    RandomForestTimeSeriesModel, 
    LinearRegressionTimeSeriesModel,
    TimeSeriesModelEnsemble
)
from time_series_forecasting.dataset import BloodInventoryDataProcessor
from survival_analysis.models import (
    CoxProportionalHazardsModel,
    BloodViabilityPredictor
)
from inventory.models import (
    BloodGroup, BloodUnitStatus, TemperatureStatus,
    MLPredictionRequest, MLPredictionResponse,
    DemandForecastCreate, ViabilityPredictionCreate
)

logger = logging.getLogger(__name__)

class MLPredictorService:
    """Main ML prediction service for blood inventory management"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.demand_models = {}
        self.viability_models = {}
        self.data_processor = BloodInventoryDataProcessor()
        
        # Load or initialize models
        self._initialize_models()
        
        logger.info("ML Predictor Service initialized")
    
    def _initialize_models(self):
        """Initialize or load trained models"""
        try:
            # Try to load existing models
            self._load_demand_models()
            self._load_viability_models()
            logger.info("Loaded existing trained models")
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")
            logger.info("Initializing new models...")
            self._initialize_new_models()
    
    def _load_demand_models(self):
        """Load trained demand forecasting models"""
        demand_model_files = {
            'random_forest': 'demand_rf_model.pkl',
            'linear_regression': 'demand_lr_model.pkl',
            'ensemble': 'demand_ensemble_model.pkl'
        }
        
        for model_name, filename in demand_model_files.items():
            model_file = self.model_path / filename
            if model_file.exists():
                if model_name == 'ensemble':
                    # Load ensemble models
                    models = []
                    for i in range(2):  # RF and LR
                        model = RandomForestTimeSeriesModel() if i == 0 else LinearRegressionTimeSeriesModel()
                        model.load_model(str(model_file.with_suffix(f'.{i}.pkl')))
                        models.append(model)
                    self.demand_models[model_name] = TimeSeriesModelEnsemble(models)
                else:
                    # Load individual model
                    model = RandomForestTimeSeriesModel() if model_name == 'random_forest' else LinearRegressionTimeSeriesModel()
                    model.load_model(str(model_file))
                    self.demand_models[model_name] = model
    
    def _load_viability_models(self):
        """Load trained viability prediction models"""
        viability_model_file = self.model_path / 'viability_cox_model.pkl'
        if viability_model_file.exists():
            model = CoxProportionalHazardsModel()
            model.load_model(str(viability_model_file))
            self.viability_models['cox_ph'] = model
    
    def _initialize_new_models(self):
        """Initialize new models with default configurations"""
        # Demand forecasting models
        self.demand_models['random_forest'] = RandomForestTimeSeriesModel(
            n_estimators=100, max_depth=10
        )
        self.demand_models['linear_regression'] = LinearRegressionTimeSeriesModel()
        
        # Create ensemble
        ensemble_models = [
            self.demand_models['random_forest'],
            self.demand_models['linear_regression']
        ]
        self.demand_models['ensemble'] = TimeSeriesModelEnsemble(ensemble_models)
        
        # Viability prediction model
        self.viability_models['cox_ph'] = CoxProportionalHazardsModel(alpha=1.0)
        
        logger.info("New models initialized")
    
    def train_models(self, training_data_path: str = None):
        """Train all models with provided or synthetic data"""
        logger.info("Starting model training...")
        
        try:
            # Load or generate training data
            if training_data_path and Path(training_data_path).exists():
                df = self.data_processor.load_sample_data(training_data_path)
            else:
                df = self.data_processor.load_sample_data("")  # Generate synthetic data
            
            # Clean and engineer features
            df_clean = self.data_processor.clean_data(df)
            df_features = self.data_processor.engineer_features(df_clean)
            
            # Prepare time series data
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
                self.data_processor.prepare_time_series_data(
                    df_features, sequence_length=24, forecast_horizon=7
                )
            
            # Train demand forecasting models
            self._train_demand_models(X_train, y_train, X_val, y_val)
            
            # Train viability models (simplified for now)
            self._train_viability_models(df_features)
            
            # Save models
            self._save_models()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def _train_demand_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray):
        """Train demand forecasting models"""
        logger.info("Training demand forecasting models...")
        
        # Train individual models
        for name, model in self.demand_models.items():
            if name != 'ensemble':
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = model.evaluate(X_val, y_val)
                logger.info(f"{name} validation metrics: {metrics}")
        
        # Train ensemble
        logger.info("Training ensemble...")
        self.demand_models['ensemble'].fit(X_train, y_train)
        ensemble_metrics = self.demand_models['ensemble'].evaluate(X_val, y_val)
        logger.info(f"Ensemble validation metrics: {ensemble_metrics}")
    
    def _train_viability_models(self, df: pd.DataFrame):
        """Train viability prediction models"""
        logger.info("Training viability prediction models...")
        
        # For now, we'll use a simplified approach
        # In a real implementation, you'd need proper survival data
        # This is a placeholder for demonstration
        
        # Create synthetic survival data
        n_samples = len(df)
        collection_dates = pd.date_range(
            start=datetime.now() - timedelta(days=60),
            periods=n_samples,
            freq='H'
        )
        
        # Simulate survival times (days until expiry)
        base_survival = 42  # Standard blood expiry
        temperature_effect = -2 * np.abs(df['temperature_celsius'] - 4.0)
        survival_times = base_survival + temperature_effect + np.random.normal(0, 2, n_samples)
        survival_times = np.maximum(1, survival_times)
        
        # Create events (1 = expired, 0 = censored)
        events = np.random.binomial(1, 0.8, n_samples)
        
        # Create feature matrix for survival analysis
        feature_cols = ['temperature_celsius', 'humidity_percent', 'volume_ml']
        X_survival = df[feature_cols].values
        
        # Create survival data structure
        from sksurv.util import Surv
        y_survival = Surv.from_arrays(event=events.astype(bool), time=survival_times)
        
        # Train Cox model
        self.viability_models['cox_ph'].fit(X_survival, y_survival)
        
        logger.info("Viability models trained")
    
    def _save_models(self):
        """Save trained models"""
        logger.info("Saving trained models...")
        
        # Save demand models
        for name, model in self.demand_models.items():
            if name == 'ensemble':
                # Save individual models in ensemble
                for i, ensemble_model in enumerate(model.models):
                    model_file = self.model_path / f'demand_ensemble_model.{i}.pkl'
                    ensemble_model.save_model(str(model_file))
            else:
                model_file = self.model_path / f'demand_{name}_model.pkl'
                model.save_model(str(model_file))
        
        # Save viability models
        for name, model in self.viability_models.items():
            model_file = self.model_path / f'viability_{name}_model.pkl'
            model.save_model(str(model_file))
        
        logger.info("Models saved successfully")
    
    def predict_demand(self, request: MLPredictionRequest) -> Dict[str, Any]:
        """Predict blood demand for specified parameters"""
        try:
            # Generate sample data for prediction
            # In a real implementation, you'd use actual historical data
            df = self.data_processor.load_sample_data("")
            df_clean = self.data_processor.clean_data(df)
            df_features = self.data_processor.engineer_features(df_clean)
            
            # Filter by blood group and pod if specified
            if request.blood_group:
                df_features = df_features[df_features['blood_group'] == request.blood_group.value]
            if request.pod_id:
                df_features = df_features[df_features['pod_id'] == request.pod_id]
            
            # Prepare data for prediction
            X_pred, _, _, _, _, _, feature_cols = \
                self.data_processor.prepare_time_series_data(
                    df_features, sequence_length=24, forecast_horizon=request.forecast_horizon_days
                )
            
            # Make predictions with ensemble model
            predictions = self.demand_models['ensemble'].predict(X_pred[-1:])  # Use last sample
            
            # Format predictions
            forecast_dates = pd.date_range(
                start=datetime.now(),
                periods=request.forecast_horizon_days,
                freq='D'
            )
            
            prediction_results = []
            for i, (date, pred) in enumerate(zip(forecast_dates, predictions[0])):
                result = {
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_demand': float(pred),
                    'confidence_interval_lower': float(pred * 0.8),  # Simplified
                    'confidence_interval_upper': float(pred * 1.2)   # Simplified
                }
                prediction_results.append(result)
            
            return {
                'predictions': prediction_results,
                'model_metadata': {
                    'model_type': 'ensemble',
                    'forecast_horizon': request.forecast_horizon_days,
                    'blood_group': request.blood_group.value if request.blood_group else 'all',
                    'pod_id': request.pod_id or 'all',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in demand prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def predict_viability(self, blood_unit_ids: List[str], 
                         features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict blood unit viability"""
        try:
            if not self.viability_models:
                raise ValueError("No viability models available")
            
            # Convert features to numpy array
            feature_cols = ['temperature_celsius', 'humidity_percent', 'volume_ml']
            X = np.array([[f.get(col, 0) for col in feature_cols] for f in features])
            
            # Create collection dates (simplified)
            collection_dates = np.array([datetime.now() - timedelta(days=10)] * len(X))
            
            # Create viability predictor
            viability_predictor = BloodViabilityPredictor(self.viability_models['cox_ph'])
            
            # Predict viability
            viability_results = viability_predictor.predict_viability_risk(
                X, collection_dates
            )
            
            # Format results
            results = []
            for i, blood_unit_id in enumerate(blood_unit_ids):
                result = {
                    'blood_unit_id': blood_unit_id,
                    'survival_probability': float(viability_results['survival_probability'][i]),
                    'risk_score': float(viability_results['risk_score'][i]),
                    'predicted_expiry_date': viability_results['predicted_expiry_date'][i].strftime('%Y-%m-%d'),
                    'days_since_collection': int(viability_results['days_since_collection'][i]),
                    'alert': bool(viability_results['risk_score'][i] > 0.7)
                }
                results.append(result)
            
            return {
                'viability_predictions': results,
                'model_metadata': {
                    'model_type': 'cox_proportional_hazards',
                    'timestamp': datetime.now().isoformat(),
                    'total_units': len(results),
                    'high_risk_units': sum(1 for r in results if r['alert'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in viability prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Viability prediction error: {str(e)}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            'demand_models': {},
            'viability_models': {},
            'last_training': None,
            'model_path': str(self.model_path)
        }
        
        # Check demand models
        for name, model in self.demand_models.items():
            status['demand_models'][name] = {
                'is_fitted': model.is_fitted,
                'model_type': model.model_name
            }
        
        # Check viability models
        for name, model in self.viability_models.items():
            status['viability_models'][name] = {
                'is_fitted': model.is_fitted,
                'model_type': model.model_name
            }
        
        return status

# FastAPI application
app = FastAPI(title="Blood Inventory ML Predictor", version="1.0.0")

# Global predictor service
predictor_service = MLPredictorService()

@app.on_event("startup")
async def startup_event():
    """Initialize the ML predictor service on startup"""
    global predictor_service
    predictor_service = MLPredictorService()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Blood Inventory ML Predictor API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models/status")
async def get_model_status():
    """Get status of all ML models"""
    return predictor_service.get_model_status()

@app.post("/models/train")
async def train_models(background_tasks: BackgroundTasks, training_data_path: str = None):
    """Train all ML models"""
    background_tasks.add_task(predictor_service.train_models, training_data_path)
    return {"message": "Model training started in background", "status": "training"}

@app.post("/predict/demand")
async def predict_demand(request: MLPredictionRequest):
    """Predict blood demand"""
    return predictor_service.predict_demand(request)

@app.post("/predict/viability")
async def predict_viability(blood_unit_ids: List[str], features: List[Dict[str, Any]]):
    """Predict blood unit viability"""
    return predictor_service.predict_viability(blood_unit_ids, features)

@app.get("/predict/demand/sample")
async def sample_demand_prediction():
    """Sample demand prediction with default parameters"""
    request = MLPredictionRequest(
        blood_group=BloodGroup.A_POSITIVE,
        pod_id="GZ_POD_1",
        forecast_horizon_days=7,
        include_confidence_intervals=True
    )
    return predictor_service.predict_demand(request)

@app.get("/predict/viability/sample")
async def sample_viability_prediction():
    """Sample viability prediction"""
    blood_unit_ids = ["BU001", "BU002", "BU003"]
    features = [
        {"temperature_celsius": 4.0, "humidity_percent": 60.0, "volume_ml": 500},
        {"temperature_celsius": 6.0, "humidity_percent": 80.0, "volume_ml": 450},
        {"temperature_celsius": 2.0, "humidity_percent": 40.0, "volume_ml": 550}
    ]
    return predictor_service.predict_viability(blood_unit_ids, features)

def main():
    """Run the ML predictor API server"""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
