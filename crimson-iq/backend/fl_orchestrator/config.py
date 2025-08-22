"""
Configuration for federated learning orchestrator
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class FLConfig:
    """Federated Learning Configuration"""
    
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    num_rounds: int = 10
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    
    # Privacy settings
    enable_differential_privacy: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_noise_multiplier: float = 1.1
    
    # Model settings
    model_type: str = "survival_analysis"  # survival_analysis, time_series, reinforcement
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, fednova
    
    # Communication settings
    max_message_size: int = 100 * 1024 * 1024  # 100MB
    connection_timeout: int = 60
    keep_alive_timeout: int = 30
    
    # Security settings
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    certificate_path: str = ""
    private_key_path: str = ""
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "fl_orchestrator.log"
    
    # Performance settings
    max_concurrent_clients: int = 10
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    client_timeout: int = 300  # seconds
    
    @classmethod
    def from_env(cls) -> 'FLConfig':
        """Create config from environment variables"""
        return cls(
            server_host=os.getenv("FL_SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("FL_SERVER_PORT", "8080")),
            num_rounds=int(os.getenv("FL_NUM_ROUNDS", "10")),
            min_fit_clients=int(os.getenv("FL_MIN_FIT_CLIENTS", "2")),
            min_evaluate_clients=int(os.getenv("FL_MIN_EVALUATE_CLIENTS", "2")),
            min_available_clients=int(os.getenv("FL_MIN_AVAILABLE_CLIENTS", "2")),
            enable_differential_privacy=os.getenv("FL_ENABLE_DP", "true").lower() == "true",
            dp_epsilon=float(os.getenv("FL_DP_EPSILON", "1.0")),
            dp_delta=float(os.getenv("FL_DP_DELTA", "1e-5")),
            dp_noise_multiplier=float(os.getenv("FL_DP_NOISE_MULTIPLIER", "1.1")),
            model_type=os.getenv("FL_MODEL_TYPE", "survival_analysis"),
            aggregation_strategy=os.getenv("FL_AGGREGATION_STRATEGY", "fedavg"),
            enable_encryption=os.getenv("FL_ENABLE_ENCRYPTION", "true").lower() == "true",
            log_level=os.getenv("FL_LOG_LEVEL", "INFO"),
            max_concurrent_clients=int(os.getenv("FL_MAX_CONCURRENT_CLIENTS", "10")),
            batch_size=int(os.getenv("FL_BATCH_SIZE", "32")),
            learning_rate=float(os.getenv("FL_LEARNING_RATE", "0.01")),
        )

# Default configuration
DEFAULT_CONFIG = FLConfig.from_env()

# Model-specific configurations
MODEL_CONFIGS = {
    "survival_analysis": {
        "model_class": "CoxProportionalHazardsModel",
        "features": [
            "blood_group", "temperature", "quantity", "donation_day_of_week",
            "donation_month", "is_weekend_donation", "is_summer", "is_winter",
            "temp_category", "temp_deviation", "temp_stable"
        ],
        "target": "duration",
        "event_col": "event",
        "penalizer": 0.1,
        "l1_ratio": 0.0
    },
    "time_series": {
        "model_class": "LSTMForecaster",
        "sequence_length": 30,
        "prediction_horizon": 7,
        "features": ["demand", "supply", "temperature", "day_of_week", "month"],
        "target": "demand"
    },
    "reinforcement_learning": {
        "model_class": "DQNAgent",
        "state_size": 10,
        "action_size": 4,
        "learning_rate": 0.001,
        "epsilon": 0.1,
        "epsilon_decay": 0.995
    }
}
