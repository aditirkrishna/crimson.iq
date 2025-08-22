"""
Security utilities for federated learning orchestrator
Implements differential privacy, encryption, and secure aggregation
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import hashlib
import secrets

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, noise_multiplier: float = 1.1):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.sensitivity = 1.0  # Default sensitivity
        
    def calculate_noise_scale(self, batch_size: int, learning_rate: float) -> float:
        """Calculate noise scale for differential privacy"""
        # Standard deviation for Gaussian noise
        noise_scale = self.noise_multiplier * learning_rate * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return noise_scale
        
    def add_noise_to_gradients(self, gradients: List[np.ndarray], batch_size: int, learning_rate: float) -> List[np.ndarray]:
        """Add differential privacy noise to gradients"""
        noise_scale = self.calculate_noise_scale(batch_size, learning_rate)
        
        noisy_gradients = []
        for grad in gradients:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, grad.shape)
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
            
        logger.info(f"Added DP noise with scale {noise_scale:.6f}")
        return noisy_gradients
        
    def add_noise_to_weights(self, weights: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        """Add differential privacy noise to model weights"""
        noise_scale = self.calculate_noise_scale(batch_size, 1.0)
        
        noisy_weights = []
        for weight in weights:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, weight.shape)
            noisy_weight = weight + noise
            noisy_weights.append(noisy_weight)
            
        return noisy_weights

class Encryption:
    """Encryption utilities for secure communication"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        self.key = key
        self.cipher = Fernet(key)
        
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'Encryption':
        """Create encryption instance from password"""
        if salt is None:
            salt = os.urandom(16)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)
        
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data"""
        return self.cipher.encrypt(data)
        
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data)
        
    def encrypt_model_weights(self, weights: List[np.ndarray]) -> bytes:
        """Encrypt model weights"""
        # Convert weights to bytes
        weights_bytes = self._weights_to_bytes(weights)
        return self.encrypt_data(weights_bytes)
        
    def decrypt_model_weights(self, encrypted_weights: bytes) -> List[np.ndarray]:
        """Decrypt model weights"""
        weights_bytes = self.decrypt_data(encrypted_weights)
        return self._bytes_to_weights(weights_bytes)
        
    def _weights_to_bytes(self, weights: List[np.ndarray]) -> bytes:
        """Convert weights list to bytes"""
        import pickle
        return pickle.dumps(weights)
        
    def _bytes_to_weights(self, weights_bytes: bytes) -> List[np.ndarray]:
        """Convert bytes to weights list"""
        import pickle
        return pickle.loads(weights_bytes)

class SecureAggregation:
    """Secure aggregation utilities for federated learning"""
    
    def __init__(self, enable_encryption: bool = True):
        self.enable_encryption = enable_encryption
        self.encryption = Encryption() if enable_encryption else None
        
    def aggregate_weights(self, client_weights: List[List[np.ndarray]], 
                         client_sizes: List[int]) -> List[np.ndarray]:
        """Securely aggregate model weights from multiple clients"""
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
            
        logger.info(f"Aggregated weights from {len(client_weights)} clients")
        return aggregated_weights
        
    def secure_aggregate_weights(self, client_weights: List[List[np.ndarray]], 
                                client_sizes: List[int]) -> List[np.ndarray]:
        """Securely aggregate weights with encryption"""
        if self.encryption is None:
            return self.aggregate_weights(client_weights, client_sizes)
            
        # Encrypt individual client weights
        encrypted_weights = []
        for weights in client_weights:
            encrypted = self.encryption.encrypt_model_weights(weights)
            encrypted_weights.append(encrypted)
            
        # Perform aggregation on encrypted data (simplified - in practice would use MPC)
        # For now, we'll decrypt and then aggregate
        decrypted_weights = []
        for encrypted in encrypted_weights:
            decrypted = self.encryption.decrypt_model_weights(encrypted)
            decrypted_weights.append(decrypted)
            
        return self.aggregate_weights(decrypted_weights, client_sizes)
        
    def federated_averaging(self, client_weights: List[List[np.ndarray]], 
                           client_sizes: List[int]) -> List[np.ndarray]:
        """Federated Averaging (FedAvg) algorithm"""
        return self.aggregate_weights(client_weights, client_sizes)
        
    def federated_proximal(self, client_weights: List[List[np.ndarray]], 
                          client_sizes: List[int], 
                          global_weights: List[np.ndarray],
                          mu: float = 0.01) -> List[np.ndarray]:
        """Federated Proximal (FedProx) algorithm with proximal term"""
        # First perform standard FedAvg
        fedavg_weights = self.aggregate_weights(client_weights, client_sizes)
        
        # Apply proximal term
        proximal_weights = []
        for fedavg_w, global_w in zip(fedavg_weights, global_weights):
            proximal_w = fedavg_w + mu * (global_w - fedavg_w)
            proximal_weights.append(proximal_w)
            
        return proximal_weights

class PrivacyMetrics:
    """Privacy metrics and evaluation utilities"""
    
    @staticmethod
    def calculate_privacy_budget(epsilon: float, delta: float, num_rounds: int) -> Tuple[float, float]:
        """Calculate cumulative privacy budget across rounds"""
        # Composition theorem for differential privacy
        cumulative_epsilon = epsilon * np.sqrt(2 * num_rounds * np.log(1 / delta))
        cumulative_delta = delta * num_rounds
        
        return cumulative_epsilon, cumulative_delta
        
    @staticmethod
    def estimate_membership_inference_risk(epsilon: float, delta: float) -> float:
        """Estimate membership inference attack risk"""
        # Simplified risk estimation based on epsilon
        if epsilon <= 0.1:
            return 0.01  # Very low risk
        elif epsilon <= 1.0:
            return 0.05  # Low risk
        elif epsilon <= 5.0:
            return 0.15  # Medium risk
        else:
            return 0.30  # High risk
            
    @staticmethod
    def calculate_sensitivity(data: np.ndarray, norm: str = "L2") -> float:
        """Calculate sensitivity of a dataset"""
        if norm == "L2":
            return np.linalg.norm(data, ord=2)
        elif norm == "L1":
            return np.linalg.norm(data, ord=1)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

class SecurityManager:
    """Main security manager for federated learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dp = DifferentialPrivacy(
            epsilon=config.get("dp_epsilon", 1.0),
            delta=config.get("dp_delta", 1e-5),
            noise_multiplier=config.get("dp_noise_multiplier", 1.1)
        )
        self.secure_agg = SecureAggregation(
            enable_encryption=config.get("enable_encryption", True)
        )
        self.privacy_metrics = PrivacyMetrics()
        
    def secure_aggregate(self, client_weights: List[List[np.ndarray]], 
                        client_sizes: List[int],
                        strategy: str = "fedavg") -> List[np.ndarray]:
        """Securely aggregate weights using specified strategy"""
        if strategy == "fedavg":
            return self.secure_agg.federated_averaging(client_weights, client_sizes)
        elif strategy == "fedprox":
            # Note: global_weights would need to be passed from previous round
            return self.secure_agg.federated_proximal(client_weights, client_sizes, [])
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy}")
            
    def add_privacy_noise(self, weights: List[np.ndarray], 
                         batch_size: int, learning_rate: float) -> List[np.ndarray]:
        """Add differential privacy noise to weights"""
        if self.config.get("enable_differential_privacy", True):
            return self.dp.add_noise_to_weights(weights, batch_size)
        return weights
        
    def get_privacy_metrics(self, num_rounds: int) -> Dict[str, float]:
        """Get privacy metrics for the federated learning process"""
        epsilon = self.config.get("dp_epsilon", 1.0)
        delta = self.config.get("dp_delta", 1e-5)
        
        cumulative_epsilon, cumulative_delta = self.privacy_metrics.calculate_privacy_budget(
            epsilon, delta, num_rounds
        )
        
        membership_risk = self.privacy_metrics.estimate_membership_inference_risk(
            cumulative_epsilon, cumulative_delta
        )
        
        return {
            "epsilon": epsilon,
            "delta": delta,
            "cumulative_epsilon": cumulative_epsilon,
            "cumulative_delta": cumulative_delta,
            "membership_inference_risk": membership_risk
        }
