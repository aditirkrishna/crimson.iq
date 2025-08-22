#!/usr/bin/env python3
"""
Test script for Crimson IQ Federated Learning Setup
Demonstrates the federated learning system with a simple example
"""

import os
import sys
import time
import logging
import subprocess
import threading
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fl_orchestrator.config import FLConfig
from fl_orchestrator.main import FederatedLearningOrchestrator
from ml_models.federated_training.utils import FederatedTrainingMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FederatedLearningDemo:
    """Demo class for federated learning setup"""
    
    def __init__(self):
        self.orchestrator = None
        self.client_processes = []
        self.monitor = FederatedTrainingMonitor()
        
    def setup_orchestrator(self) -> bool:
        """Setup the federated learning orchestrator"""
        try:
            # Create configuration
            config = FLConfig(
                server_host="localhost",
                server_port=8080,
                num_rounds=5,  # Short demo
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                model_type="survival_analysis",
                aggregation_strategy="fedavg",
                enable_differential_privacy=True,
                dp_epsilon=1.0,
                dp_delta=1e-5
            )
            
            # Create orchestrator
            self.orchestrator = FederatedLearningOrchestrator(config)
            
            logger.info("Federated learning orchestrator setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup orchestrator: {e}")
            return False
            
    def start_clients(self, num_clients: int = 3) -> bool:
        """Start federated learning clients"""
        try:
            client_configs = [
                {
                    "node_name": f"Hospital {i+1}",
                    "node_type": "hospital",
                    "model_type": "survival_analysis"
                }
                for i in range(num_clients)
            ]
            
            for i, config in enumerate(client_configs):
                # Create client command
                cmd = [
                    sys.executable,
                    os.path.join("inventory", "federated_client.py"),
                    "--server", "localhost:8080",
                    "--node-name", config["node_name"],
                    "--node-type", config["node_type"],
                    "--model-type", config["model_type"]
                ]
                
                # Start client process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.client_processes.append(process)
                logger.info(f"Started client: {config['node_name']}")
                
                # Small delay between client starts
                time.sleep(1)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start clients: {e}")
            return False
            
    def run_demo(self, duration: int = 60):
        """Run the federated learning demo"""
        try:
            logger.info("Starting Federated Learning Demo")
            logger.info("=" * 50)
            
            # Setup orchestrator
            if not self.setup_orchestrator():
                return False
                
            # Start orchestrator
            self.orchestrator.start_server()
            logger.info("Orchestrator server started")
            
            # Wait a moment for server to be ready
            time.sleep(2)
            
            # Start clients
            if not self.start_clients(3):
                return False
                
            logger.info("All clients started")
            logger.info("=" * 50)
            
            # Monitor training
            start_time = time.time()
            while time.time() - start_time < duration:
                if not self.orchestrator.is_running:
                    logger.warning("Orchestrator stopped unexpectedly")
                    break
                    
                # Get status
                status = self.orchestrator.get_server_status()
                
                logger.info(f"Active clients: {status['client_stats']['active_clients']}")
                logger.info(f"Current round: {status.get('current_round', 0)}")
                
                # Check if training is complete
                if status.get('current_round', 0) >= self.orchestrator.config.num_rounds:
                    logger.info("Training completed!")
                    break
                    
                time.sleep(10)  # Check every 10 seconds
                
            logger.info("=" * 50)
            logger.info("Demo completed")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return False
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        # Stop client processes
        for process in self.client_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                pass
                
        # Stop orchestrator
        if self.orchestrator and self.orchestrator.is_running:
            self.orchestrator.stop_server()
            
        logger.info("Cleanup complete")

def test_security_features():
    """Test security features"""
    logger.info("Testing Security Features")
    logger.info("-" * 30)
    
    from fl_orchestrator.security import SecurityManager, DifferentialPrivacy, Encryption
    
    # Test differential privacy
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    test_weights = [np.random.randn(10, 1) for _ in range(3)]
    noisy_weights = dp.add_noise_to_weights(test_weights, batch_size=32)
    
    logger.info(f"Differential privacy test: {len(noisy_weights)} weights processed")
    
    # Test encryption
    encryption = Encryption()
    test_data = b"test model weights"
    encrypted = encryption.encrypt_data(test_data)
    decrypted = encryption.decrypt_data(encrypted)
    
    assert test_data == decrypted
    logger.info("Encryption test: PASSED")
    
    # Test security manager
    config = {
        "enable_differential_privacy": True,
        "dp_epsilon": 1.0,
        "dp_delta": 1e-5,
        "enable_encryption": True
    }
    
    security_manager = SecurityManager(config)
    privacy_metrics = security_manager.get_privacy_metrics(num_rounds=10)
    
    logger.info(f"Privacy metrics: epsilon={privacy_metrics['epsilon']}, "
               f"risk={privacy_metrics['membership_inference_risk']:.3f}")

def test_aggregation_strategies():
    """Test aggregation strategies"""
    logger.info("Testing Aggregation Strategies")
    logger.info("-" * 30)
    
    from fl_orchestrator.aggregator import Aggregator, FedAvgStrategy, FedProxStrategy
    from fl_orchestrator.security import SecurityManager
    
    # Create test data
    client_weights = [
        [np.random.randn(5, 3) for _ in range(2)]  # 2 layers, 5x3 weights
        for _ in range(3)  # 3 clients
    ]
    client_sizes = [100, 150, 200]
    
    # Test FedAvg
    fedavg = FedAvgStrategy()
    aggregated = fedavg.aggregate(client_weights, client_sizes)
    
    logger.info(f"FedAvg test: {len(aggregated)} aggregated layers")
    
    # Test FedProx
    fedprox = FedProxStrategy(mu=0.01)
    global_weights = [np.random.randn(5, 3) for _ in range(2)]
    fedprox.set_global_weights(global_weights)
    proximal_aggregated = fedprox.aggregate(client_weights, client_sizes)
    
    logger.info(f"FedProx test: {len(proximal_aggregated)} aggregated layers")
    
    # Test full aggregator
    config = {"enable_encryption": False}
    security_manager = SecurityManager(config)
    aggregator = Aggregator(config, security_manager)
    
    aggregator.set_strategy("fedavg")
    result = aggregator.aggregate(client_weights, client_sizes)
    
    logger.info(f"Full aggregator test: {len(result)} aggregated layers")

def test_client_manager():
    """Test client manager"""
    logger.info("Testing Client Manager")
    logger.info("-" * 30)
    
    from fl_orchestrator.client_manager import ClientManager
    
    config = {
        "health_check_interval": 30,
        "client_timeout": 300,
        "min_available_clients": 2
    }
    
    client_manager = ClientManager(config)
    
    # Register clients
    client_id1 = client_manager.register_client(
        "Hospital Alpha", "hospital", "192.168.1.100", 8081,
        ["survival_analysis"], ["survival_analysis"], 1000
    )
    
    client_id2 = client_manager.register_client(
        "Blood Bank Beta", "blood_bank", "192.168.1.101", 8082,
        ["survival_analysis", "time_series"], ["survival_analysis"], 2000
    )
    
    logger.info(f"Registered clients: {client_id1}, {client_id2}")
    
    # Get stats
    stats = client_manager.get_client_stats()
    logger.info(f"Client stats: {stats['total_clients']} total, {stats['active_clients']} active")
    
    # Get active clients
    active_clients = client_manager.get_active_clients("survival_analysis")
    logger.info(f"Active clients for survival analysis: {len(active_clients)}")

def main():
    """Main test function"""
    logger.info("Crimson IQ Federated Learning Test Suite")
    logger.info("=" * 50)
    
    # Test individual components
    try:
        test_security_features()
        test_aggregation_strategies()
        test_client_manager()
        
        logger.info("\nAll component tests passed!")
        logger.info("=" * 50)
        
        # Run demo
        demo = FederatedLearningDemo()
        success = demo.run_demo(duration=120)  # 2 minutes
        
        if success:
            logger.info("Federated Learning Demo completed successfully!")
        else:
            logger.error("Federated Learning Demo failed!")
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    import numpy as np
    
    success = main()
    if success:
        print("\n✅ All tests passed! Federated learning setup is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the logs for details.")
        
    sys.exit(0 if success else 1)
