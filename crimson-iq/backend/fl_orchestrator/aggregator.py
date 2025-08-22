"""
Aggregation algorithms for federated learning orchestrator
Implements various aggregation strategies with privacy support
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import time
import statistics

from .security import SecurityManager

logger = logging.getLogger(__name__)

class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies"""
    
    @abstractmethod
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                  client_sizes: List[int], 
                  **kwargs) -> List[np.ndarray]:
        """Aggregate client weights"""
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass

class FedAvgStrategy(AggregationStrategy):
    """Federated Averaging (FedAvg) strategy"""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager
        
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                  client_sizes: List[int], 
                  **kwargs) -> List[np.ndarray]:
        """Aggregate weights using FedAvg"""
        if not client_weights:
            raise ValueError("No client weights provided")
            
        # Calculate total samples
        total_samples = sum(client_sizes)
        
        # Weighted average aggregation
        aggregated_weights = []
        num_layers = len(client_weights[0])
        
        for layer_idx in range(num_layers):
            # Initialize aggregated layer weights
            layer_shape = client_weights[0][layer_idx].shape
            aggregated_layer = np.zeros(layer_shape)
            
            # Weighted sum of layer weights
            for client_idx, weights in enumerate(client_weights):
                weight = client_sizes[client_idx] / total_samples
                aggregated_layer += weight * weights[layer_idx]
                
            aggregated_weights.append(aggregated_layer)
            
        # Apply security measures if available
        if self.security_manager:
            batch_size = kwargs.get('batch_size', 32)
            learning_rate = kwargs.get('learning_rate', 0.01)
            aggregated_weights = self.security_manager.add_privacy_noise(
                aggregated_weights, batch_size, learning_rate
            )
            
        logger.info(f"FedAvg aggregated weights from {len(client_weights)} clients")
        return aggregated_weights
        
    def get_name(self) -> str:
        return "fedavg"

class FedProxStrategy(AggregationStrategy):
    """Federated Proximal (FedProx) strategy"""
    
    def __init__(self, mu: float = 0.01, security_manager: Optional[SecurityManager] = None):
        self.mu = mu
        self.security_manager = security_manager
        self.global_weights = None
        
    def set_global_weights(self, global_weights: List[np.ndarray]):
        """Set global weights for proximal term"""
        self.global_weights = global_weights
        
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                  client_sizes: List[int], 
                  **kwargs) -> List[np.ndarray]:
        """Aggregate weights using FedProx"""
        if not client_weights:
            raise ValueError("No client weights provided")
            
        if self.global_weights is None:
            logger.warning("Global weights not set, falling back to FedAvg")
            return FedAvgStrategy(self.security_manager).aggregate(client_weights, client_sizes, **kwargs)
            
        # First perform standard FedAvg
        fedavg_weights = FedAvgStrategy(self.security_manager).aggregate(client_weights, client_sizes, **kwargs)
        
        # Apply proximal term
        proximal_weights = []
        for fedavg_w, global_w in zip(fedavg_weights, self.global_weights):
            proximal_w = fedavg_w + self.mu * (global_w - fedavg_w)
            proximal_weights.append(proximal_w)
            
        logger.info(f"FedProx aggregated weights from {len(client_weights)} clients with mu={self.mu}")
        return proximal_weights
        
    def get_name(self) -> str:
        return "fedprox"

class FedNovaStrategy(AggregationStrategy):
    """Federated Normalized Averaging (FedNova) strategy"""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager
        
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                  client_sizes: List[int], 
                  client_epochs: List[int] = None,
                  **kwargs) -> List[np.ndarray]:
        """Aggregate weights using FedNova"""
        if not client_weights:
            raise ValueError("No client weights provided")
            
        if client_epochs is None:
            # Default to 1 epoch per client
            client_epochs = [1] * len(client_weights)
            
        # Calculate normalized weights
        total_epochs = sum(client_epochs)
        normalized_weights = [epochs / total_epochs for epochs in client_epochs]
        
        # Weighted average aggregation with normalization
        aggregated_weights = []
        num_layers = len(client_weights[0])
        
        for layer_idx in range(num_layers):
            layer_shape = client_weights[0][layer_idx].shape
            aggregated_layer = np.zeros(layer_shape)
            
            for client_idx, weights in enumerate(client_weights):
                weight = normalized_weights[client_idx]
                aggregated_layer += weight * weights[layer_idx]
                
            aggregated_weights.append(aggregated_layer)
            
        # Apply security measures if available
        if self.security_manager:
            batch_size = kwargs.get('batch_size', 32)
            learning_rate = kwargs.get('learning_rate', 0.01)
            aggregated_weights = self.security_manager.add_privacy_noise(
                aggregated_weights, batch_size, learning_rate
            )
            
        logger.info(f"FedNova aggregated weights from {len(client_weights)} clients")
        return aggregated_weights
        
    def get_name(self) -> str:
        return "fednova"

class RobustAggregationStrategy(AggregationStrategy):
    """Robust aggregation strategy with outlier detection"""
    
    def __init__(self, outlier_threshold: float = 2.0, security_manager: Optional[SecurityManager] = None):
        self.outlier_threshold = outlier_threshold
        self.security_manager = security_manager
        
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                  client_sizes: List[int], 
                  **kwargs) -> List[np.ndarray]:
        """Aggregate weights with outlier detection"""
        if not client_weights:
            raise ValueError("No client weights provided")
            
        # Detect and remove outliers
        filtered_weights, filtered_sizes = self._remove_outliers(client_weights, client_sizes)
        
        if len(filtered_weights) == 0:
            raise ValueError("All clients were identified as outliers")
            
        # Use FedAvg for aggregation
        return FedAvgStrategy(self.security_manager).aggregate(filtered_weights, filtered_sizes, **kwargs)
        
    def _remove_outliers(self, client_weights: List[List[np.ndarray]], 
                        client_sizes: List[int]) -> Tuple[List[List[np.ndarray]], List[int]]:
        """Remove outlier clients based on weight statistics"""
        if len(client_weights) <= 2:
            return client_weights, client_sizes
            
        # Calculate weight statistics for each layer
        num_layers = len(client_weights[0])
        outlier_scores = []
        
        for client_idx, weights in enumerate(client_weights):
            client_score = 0.0
            for layer_idx in range(num_layers):
                # Calculate distance from mean for this layer
                layer_weights = [w[layer_idx] for w in client_weights]
                mean_weight = np.mean(layer_weights, axis=0)
                distance = np.linalg.norm(weights[layer_idx] - mean_weight)
                client_score += distance
                
            outlier_scores.append(client_score)
            
        # Calculate outlier threshold
        mean_score = statistics.mean(outlier_scores)
        std_score = statistics.stdev(outlier_scores) if len(outlier_scores) > 1 else 0
        
        # Filter out outliers
        filtered_weights = []
        filtered_sizes = []
        
        for idx, (weights, size) in enumerate(zip(client_weights, client_sizes)):
            if abs(outlier_scores[idx] - mean_score) <= self.outlier_threshold * std_score:
                filtered_weights.append(weights)
                filtered_sizes.append(size)
            else:
                logger.warning(f"Client {idx} identified as outlier (score: {outlier_scores[idx]:.4f})")
                
        logger.info(f"Removed {len(client_weights) - len(filtered_weights)} outliers")
        return filtered_weights, filtered_sizes
        
    def get_name(self) -> str:
        return "robust"

class Aggregator:
    """Main aggregator for federated learning"""
    
    def __init__(self, config: Dict[str, Any], security_manager: Optional[SecurityManager] = None):
        self.config = config
        self.security_manager = security_manager
        self.strategies = self._initialize_strategies()
        self.current_strategy = config.get("aggregation_strategy", "fedavg")
        
        # Metrics
        self.aggregation_count = 0
        self.aggregation_times = []
        
    def _initialize_strategies(self) -> Dict[str, AggregationStrategy]:
        """Initialize available aggregation strategies"""
        strategies = {
            "fedavg": FedAvgStrategy(self.security_manager),
            "fedprox": FedProxStrategy(
                mu=self.config.get("fedprox_mu", 0.01),
                security_manager=self.security_manager
            ),
            "fednova": FedNovaStrategy(self.security_manager),
            "robust": RobustAggregationStrategy(
                outlier_threshold=self.config.get("outlier_threshold", 2.0),
                security_manager=self.security_manager
            )
        }
        return strategies
        
    def set_strategy(self, strategy_name: str):
        """Set the aggregation strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown aggregation strategy: {strategy_name}")
        self.current_strategy = strategy_name
        logger.info(f"Set aggregation strategy to {strategy_name}")
        
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                  client_sizes: List[int], 
                  **kwargs) -> List[np.ndarray]:
        """Aggregate client weights using the current strategy"""
        start_time = time.time()
        
        try:
            strategy = self.strategies[self.current_strategy]
            aggregated_weights = strategy.aggregate(client_weights, client_sizes, **kwargs)
            
            # Update metrics
            aggregation_time = time.time() - start_time
            self.aggregation_count += 1
            self.aggregation_times.append(aggregation_time)
            
            logger.info(f"Aggregation completed in {aggregation_time:.4f}s using {self.current_strategy}")
            return aggregated_weights
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregator metrics"""
        return {
            "aggregation_count": self.aggregation_count,
            "current_strategy": self.current_strategy,
            "avg_aggregation_time": statistics.mean(self.aggregation_times) if self.aggregation_times else 0,
            "available_strategies": list(self.strategies.keys())
        }
        
    def validate_weights(self, client_weights: List[List[np.ndarray]]) -> bool:
        """Validate client weights before aggregation"""
        if not client_weights:
            return False
            
        # Check if all clients have the same number of layers
        num_layers = len(client_weights[0])
        for weights in client_weights:
            if len(weights) != num_layers:
                logger.error("Client weights have different number of layers")
                return False
                
        # Check if all layers have compatible shapes
        for layer_idx in range(num_layers):
            layer_shape = client_weights[0][layer_idx].shape
            for weights in client_weights:
                if weights[layer_idx].shape != layer_shape:
                    logger.error(f"Layer {layer_idx} has incompatible shapes")
                    return False
                    
        return True
        
    def calculate_aggregation_weights(self, client_sizes: List[int], 
                                    client_trust_scores: List[float] = None) -> List[float]:
        """Calculate aggregation weights based on data size and trust scores"""
        if client_trust_scores is None:
            client_trust_scores = [1.0] * len(client_sizes)
            
        # Combine data size and trust score
        total_weight = 0
        weights = []
        
        for size, trust in zip(client_sizes, client_trust_scores):
            weight = size * trust
            weights.append(weight)
            total_weight += weight
            
        # Normalize weights
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
            
        return weights
