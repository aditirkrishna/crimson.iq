"""
Federated Training Scripts for Crimson IQ ML Models
Orchestrates federated learning training jobs for different model types
"""

import logging
import os
import sys
import time
import json
import subprocess
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fl_orchestrator.config import FLConfig, MODEL_CONFIGS
from fl_orchestrator.main import FederatedLearningOrchestrator

logger = logging.getLogger(__name__)

class FederatedTrainingOrchestrator:
    """Orchestrates federated training jobs"""
    
    def __init__(self, config: FLConfig):
        self.config = config
        self.orchestrator = FederatedLearningOrchestrator(config)
        self.training_jobs = {}
        self.client_processes = []
        
    def start_training_job(self, job_id: str, model_type: str, 
                          client_configs: List[Dict[str, Any]]) -> bool:
        """Start a federated training job"""
        try:
            logger.info(f"Starting federated training job {job_id} for {model_type}")
            
            # Update orchestrator config for this model type
            self.orchestrator.config.model_type = model_type
            
            # Start the orchestrator server
            self.orchestrator.start_server()
            
            # Start client processes
            self._start_clients(client_configs)
            
            # Store job information
            self.training_jobs[job_id] = {
                "model_type": model_type,
                "start_time": datetime.now(),
                "status": "running",
                "client_configs": client_configs
            }
            
            logger.info(f"Federated training job {job_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training job {job_id}: {e}")
            return False
            
    def _start_clients(self, client_configs: List[Dict[str, Any]]):
        """Start federated learning clients"""
        for i, config in enumerate(client_configs):
            try:
                # Create client command
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "..", "..", "inventory", "federated_client.py"),
                    "--server", f"{self.config.server_host}:{self.config.server_port}",
                    "--node-name", config.get("node_name", f"client_{i}"),
                    "--node-type", config.get("node_type", "hospital"),
                    "--model-type", config.get("model_type", self.config.model_type),
                    "--data-path", config.get("data_path", ""),
                ]
                
                # Add config file if provided
                if config.get("config_path"):
                    cmd.extend(["--config", config["config_path"]])
                    
                # Start client process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.client_processes.append(process)
                logger.info(f"Started client {config.get('node_name', f'client_{i}')}")
                
            except Exception as e:
                logger.error(f"Failed to start client {i}: {e}")
                
    def stop_training_job(self, job_id: str):
        """Stop a federated training job"""
        if job_id not in self.training_jobs:
            logger.warning(f"Training job {job_id} not found")
            return
            
        try:
            # Stop client processes
            for process in self.client_processes:
                process.terminate()
                process.wait(timeout=5)
                
            # Stop orchestrator
            self.orchestrator.stop_server()
            
            # Update job status
            self.training_jobs[job_id]["status"] = "stopped"
            self.training_jobs[job_id]["end_time"] = datetime.now()
            
            logger.info(f"Federated training job {job_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping training job {job_id}: {e}")
            
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        if job_id not in self.training_jobs:
            return None
            
        job = self.training_jobs[job_id].copy()
        
        # Add server status
        if self.orchestrator.is_running:
            job["server_status"] = self.orchestrator.get_server_status()
            
        return job
        
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs"""
        return [
            {"job_id": job_id, **job_info}
            for job_id, job_info in self.training_jobs.items()
        ]
        
    def cleanup(self):
        """Cleanup resources"""
        # Stop all jobs
        for job_id in list(self.training_jobs.keys()):
            self.stop_training_job(job_id)
            
        # Stop orchestrator
        if self.orchestrator.is_running:
            self.orchestrator.stop_server()

def create_sample_client_configs(num_clients: int = 3) -> List[Dict[str, Any]]:
    """Create sample client configurations"""
    node_types = ["hospital", "blood_bank", "clinic"]
    node_names = [
        "General Hospital Alpha",
        "Central Blood Bank",
        "Community Clinic Beta",
        "Regional Medical Center",
        "Emergency Care Unit"
    ]
    
    configs = []
    for i in range(num_clients):
        config = {
            "node_name": node_names[i % len(node_names)],
            "node_type": node_types[i % len(node_types)],
            "model_type": "survival_analysis",
            "data_path": "",  # Will use synthetic data
            "config_path": ""
        }
        configs.append(config)
        
    return configs

def run_survival_analysis_training():
    """Run federated training for survival analysis models"""
    # Configuration for survival analysis
    config = FLConfig(
        model_type="survival_analysis",
        num_rounds=15,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        aggregation_strategy="fedavg",
        enable_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5
    )
    
    # Create orchestrator
    orchestrator = FederatedTrainingOrchestrator(config)
    
    try:
        # Create client configurations
        client_configs = create_sample_client_configs(3)
        
        # Start training job
        job_id = f"survival_analysis_{int(time.time())}"
        success = orchestrator.start_training_job(job_id, "survival_analysis", client_configs)
        
        if success:
            logger.info("Survival analysis federated training started")
            
            # Monitor training
            while orchestrator.orchestrator.is_running:
                status = orchestrator.get_job_status(job_id)
                if status:
                    logger.info(f"Job status: {status['status']}")
                time.sleep(30)
                
        else:
            logger.error("Failed to start survival analysis training")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        orchestrator.cleanup()

def run_time_series_training():
    """Run federated training for time series forecasting models"""
    # Configuration for time series forecasting
    config = FLConfig(
        model_type="time_series",
        num_rounds=20,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        aggregation_strategy="fedprox",
        enable_differential_privacy=True,
        dp_epsilon=0.8,
        dp_delta=1e-5
    )
    
    # Create orchestrator
    orchestrator = FederatedTrainingOrchestrator(config)
    
    try:
        # Create client configurations
        client_configs = create_sample_client_configs(4)
        
        # Start training job
        job_id = f"time_series_{int(time.time())}"
        success = orchestrator.start_training_job(job_id, "time_series", client_configs)
        
        if success:
            logger.info("Time series federated training started")
            
            # Monitor training
            while orchestrator.orchestrator.is_running:
                status = orchestrator.get_job_status(job_id)
                if status:
                    logger.info(f"Job status: {status['status']}")
                time.sleep(30)
                
        else:
            logger.error("Failed to start time series training")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        orchestrator.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crimson IQ Federated Training Orchestrator")
    parser.add_argument("--model-type", type=str, default="survival_analysis",
                       choices=["survival_analysis", "time_series", "reinforcement"],
                       help="Model type to train")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--strategy", type=str, default="fedavg",
                       choices=["fedavg", "fedprox", "fednova", "robust"],
                       help="Aggregation strategy")
    parser.add_argument("--enable-dp", action="store_true", help="Enable differential privacy")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = FLConfig(
        model_type=args.model_type,
        num_rounds=args.rounds,
        aggregation_strategy=args.strategy,
        enable_differential_privacy=args.enable_dp
    )
    
    # Create orchestrator
    orchestrator = FederatedTrainingOrchestrator(config)
    
    try:
        # Create client configurations
        client_configs = create_sample_client_configs(args.clients)
        
        # Start training job
        job_id = f"{args.model_type}_{int(time.time())}"
        success = orchestrator.start_training_job(job_id, args.model_type, client_configs)
        
        if success:
            logger.info(f"{args.model_type} federated training started")
            
            # Monitor training
            while orchestrator.orchestrator.is_running:
                status = orchestrator.get_job_status(job_id)
                if status:
                    logger.info(f"Job status: {status['status']}")
                time.sleep(30)
                
        else:
            logger.error(f"Failed to start {args.model_type} training")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        orchestrator.cleanup()

if __name__ == "__main__":
    main()
