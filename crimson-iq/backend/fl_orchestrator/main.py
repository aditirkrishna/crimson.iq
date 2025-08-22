"""
Main entry point for federated learning orchestrator
Implements Flower server with custom strategies and privacy enhancements
"""

import logging
import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.server import ServerConfig, History
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes, Parameters, Scalar, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
)

from .config import FLConfig, DEFAULT_CONFIG, MODEL_CONFIGS
from .security import SecurityManager
from .client_manager import ClientManager
from .aggregator import Aggregator, FedAvgStrategy, FedProxStrategy, RobustAggregationStrategy

logger = logging.getLogger(__name__)

class CrimsonFLStrategy(FedAvg):
    """Custom federated learning strategy for Crimson IQ"""
    
    def __init__(self, 
                 config: FLConfig,
                 security_manager: SecurityManager,
                 client_manager: ClientManager,
                 aggregator: Aggregator,
                 *args, **kwargs):
        self.config = config
        self.security_manager = security_manager
        self.client_manager = client_manager
        self.aggregator = aggregator
        self.current_round = 0
        self.global_weights = None
        
        # Initialize base FedAvg strategy
        super().__init__(
            min_fit_clients=config.min_fit_clients,
            min_evaluate_clients=config.min_evaluate_clients,
            min_available_clients=config.min_available_clients,
            *args, **kwargs
        )
        
    def aggregate_fit(self, server_round: int, results: List[Tuple], failures: List) -> Optional[Parameters]:
        """Aggregate fit results using custom aggregator"""
        if not results:
            logger.warning("No fit results to aggregate")
            return None
            
        # Extract weights and metrics
        client_weights = []
        client_sizes = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            if fit_res.status.is_success():
                # Convert parameters to numpy arrays
                weights = parameters_to_ndarrays(fit_res.parameters)
                client_weights.append(weights)
                
                # Get client size from metrics
                client_size = fit_res.metrics.get("client_size", 1)
                client_sizes.append(client_size)
                
                # Store metrics
                client_metrics.append(fit_res.metrics)
                
                # Update client manager metrics
                client_id = str(client_proxy.cid)
                response_time = fit_res.metrics.get("response_time", 0.0)
                self.client_manager.update_client_metrics(client_id, True, response_time)
            else:
                # Handle failed clients
                client_id = str(client_proxy.cid)
                self.client_manager.update_client_metrics(client_id, False, 0.0)
                logger.warning(f"Client {client_id} failed during fit")
                
        if not client_weights:
            logger.error("No successful fit results")
            return None
            
        # Validate weights
        if not self.aggregator.validate_weights(client_weights):
            logger.error("Invalid client weights")
            return None
            
        # Aggregate weights using custom aggregator
        try:
            aggregated_weights = self.aggregator.aggregate(
                client_weights=client_weights,
                client_sizes=client_sizes,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate
            )
            
            # Store global weights for next round
            self.global_weights = aggregated_weights
            
            # Update FedProx strategy if using it
            if isinstance(self.aggregator.strategies[self.aggregator.current_strategy], FedProxStrategy):
                self.aggregator.strategies[self.aggregator.current_strategy].set_global_weights(aggregated_weights)
                
            # Convert back to parameters
            parameters = ndarrays_to_parameters(aggregated_weights)
            
            # Log aggregation metrics
            self._log_aggregation_metrics(server_round, client_metrics, len(client_weights))
            
            return parameters
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return None
            
    def aggregate_evaluate(self, server_round: int, results: List[Tuple], failures: List) -> Optional[float]:
        """Aggregate evaluation results"""
        if not results:
            return None
            
        # Calculate weighted average of evaluation metrics
        total_loss = 0.0
        total_samples = 0
        
        for client_proxy, eval_res in results:
            if eval_res.status.is_success():
                loss = eval_res.loss
                num_examples = eval_res.num_examples
                total_loss += loss * num_examples
                total_samples += num_examples
                
        if total_samples == 0:
            return None
            
        return total_loss / total_samples
        
    def _log_aggregation_metrics(self, server_round: int, client_metrics: List[Dict], num_clients: int):
        """Log aggregation metrics"""
        if not client_metrics:
            return
            
        # Calculate average metrics
        avg_loss = sum(m.get("loss", 0) for m in client_metrics) / len(client_metrics)
        avg_accuracy = sum(m.get("accuracy", 0) for m in client_metrics) / len(client_metrics)
        avg_response_time = sum(m.get("response_time", 0) for m in client_metrics) / len(client_metrics)
        
        logger.info(f"Round {server_round} - Clients: {num_clients}, "
                   f"Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, "
                   f"Avg Response Time: {avg_response_time:.2f}s")

class FederatedLearningOrchestrator:
    """Main federated learning orchestrator"""
    
    def __init__(self, config: Optional[FLConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.setup_logging()
        
        # Initialize components
        self.security_manager = SecurityManager(self.config.__dict__)
        self.client_manager = ClientManager(self.config.__dict__)
        self.aggregator = Aggregator(self.config.__dict__, self.security_manager)
        
        # Initialize strategy
        self.strategy = CrimsonFLStrategy(
            config=self.config,
            security_manager=self.security_manager,
            client_manager=self.client_manager,
            aggregator=self.aggregator
        )
        
        # Server state
        self.server = None
        self.history = None
        self.is_running = False
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        
    def start_server(self):
        """Start the federated learning server"""
        if self.is_running:
            logger.warning("Server is already running")
            return
            
        try:
            # Start client manager
            self.client_manager.start()
            
            # Configure server
            server_config = ServerConfig(
                num_rounds=self.config.num_rounds,
                round_timeout=self.config.connection_timeout
            )
            
            # Start Flower server
            self.server = fl.server.start_server(
                server_address=f"{self.config.server_host}:{self.config.server_port}",
                config=server_config,
                strategy=self.strategy
            )
            
            self.is_running = True
            logger.info(f"Federated learning server started on {self.config.server_host}:{self.config.server_port}")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
            
    def stop_server(self):
        """Stop the federated learning server"""
        if not self.is_running:
            return
            
        try:
            # Stop client manager
            self.client_manager.stop()
            
            # Stop server
            if self.server:
                self.server.stop()
                
            self.is_running = False
            logger.info("Federated learning server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and metrics"""
        return {
            "is_running": self.is_running,
            "config": self.config.__dict__,
            "client_stats": self.client_manager.get_client_stats(),
            "aggregator_metrics": self.aggregator.get_metrics(),
            "privacy_metrics": self.security_manager.get_privacy_metrics(self.config.num_rounds),
            "current_round": getattr(self.strategy, 'current_round', 0),
            "server_address": f"{self.config.server_host}:{self.config.server_port}"
        }
        
    def register_client(self, node_name: str, node_type: str, ip_address: str, 
                       port: int, capabilities: List[str], model_types: List[str], 
                       data_size: int = 0) -> str:
        """Register a new federated learning client"""
        return self.client_manager.register_client(
            node_name, node_type, ip_address, port, capabilities, model_types, data_size
        )
        
    def set_aggregation_strategy(self, strategy_name: str):
        """Set the aggregation strategy"""
        self.aggregator.set_strategy(strategy_name)
        
    def export_client_data(self) -> str:
        """Export client data"""
        return self.client_manager.export_client_data()
        
    def import_client_data(self, json_data: str):
        """Import client data"""
        self.client_manager.import_client_data(json_data)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crimson IQ Federated Learning Orchestrator")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients required")
    parser.add_argument("--strategy", type=str, default="fedavg", 
                       choices=["fedavg", "fedprox", "fednova", "robust"],
                       help="Aggregation strategy")
    
    args = parser.parse_args()
    
    # Create configuration
    config = FLConfig(
        server_host=args.host,
        server_port=args.port,
        num_rounds=args.rounds,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        aggregation_strategy=args.strategy
    )
    
    # Create and start orchestrator
    orchestrator = FederatedLearningOrchestrator(config)
    
    try:
        orchestrator.start_server()
        
        # Keep server running
        while orchestrator.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        orchestrator.stop_server()

if __name__ == "__main__":
    main()
