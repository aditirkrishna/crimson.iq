"""
Client manager for federated learning orchestrator
Handles client registration, health tracking, and coordination
"""

import logging
import time
import threading
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json

logger = logging.getLogger(__name__)

@dataclass
class ClientInfo:
    """Information about a federated learning client"""
    client_id: str
    node_name: str
    node_type: str  # hospital, blood_bank, etc.
    ip_address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    model_types: List[str] = field(default_factory=list)
    data_size: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    trust_score: float = 1.0
    participation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    registered_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "client_id": self.client_id,
            "node_name": self.node_name,
            "node_type": self.node_type,
            "ip_address": self.ip_address,
            "port": self.port,
            "capabilities": self.capabilities,
            "model_types": self.model_types,
            "data_size": self.data_size,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "is_active": self.is_active,
            "trust_score": self.trust_score,
            "participation_count": self.participation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_response_time": self.avg_response_time,
            "registered_at": self.registered_at.isoformat()
        }

class ClientManager:
    """Manages federated learning clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients: Dict[str, ClientInfo] = {}
        self.active_clients: Set[str] = set()
        self.health_check_interval = config.get("health_check_interval", 30)
        self.client_timeout = config.get("client_timeout", 300)
        self.min_available_clients = config.get("min_available_clients", 2)
        
        # Threading
        self.lock = threading.RLock()
        self.health_check_thread = None
        self.running = False
        
        # Metrics
        self.total_registrations = 0
        self.total_deregistrations = 0
        self.health_check_count = 0
        
    def start(self):
        """Start the client manager"""
        if self.running:
            logger.warning("Client manager is already running")
            return
            
        self.running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        logger.info("Client manager started")
        
    def stop(self):
        """Stop the client manager"""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Client manager stopped")
        
    def register_client(self, node_name: str, node_type: str, ip_address: str, 
                       port: int, capabilities: List[str], model_types: List[str], 
                       data_size: int = 0) -> str:
        """Register a new federated learning client"""
        with self.lock:
            client_id = str(uuid.uuid4())
            
            client_info = ClientInfo(
                client_id=client_id,
                node_name=node_name,
                node_type=node_type,
                ip_address=ip_address,
                port=port,
                capabilities=capabilities,
                model_types=model_types,
                data_size=data_size
            )
            
            self.clients[client_id] = client_info
            self.active_clients.add(client_id)
            self.total_registrations += 1
            
            logger.info(f"Registered client {client_id} ({node_name}) at {ip_address}:{port}")
            return client_id
            
    def deregister_client(self, client_id: str) -> bool:
        """Deregister a federated learning client"""
        with self.lock:
            if client_id in self.clients:
                client_info = self.clients[client_id]
                del self.clients[client_id]
                self.active_clients.discard(client_id)
                self.total_deregistrations += 1
                
                logger.info(f"Deregistered client {client_id} ({client_info.node_name})")
                return True
            return False
            
    def update_heartbeat(self, client_id: str) -> bool:
        """Update client heartbeat"""
        with self.lock:
            if client_id in self.clients:
                self.clients[client_id].last_heartbeat = datetime.now()
                return True
            return False
            
    def get_active_clients(self, model_type: Optional[str] = None) -> List[ClientInfo]:
        """Get list of active clients, optionally filtered by model type"""
        with self.lock:
            active_clients = []
            current_time = datetime.now()
            
            for client_id in self.active_clients:
                client = self.clients.get(client_id)
                if client and client.is_active:
                    # Check if client supports the requested model type
                    if model_type is None or model_type in client.model_types:
                        active_clients.append(client)
                        
            return active_clients
            
    def get_available_clients(self, model_type: str, min_clients: int = None) -> List[ClientInfo]:
        """Get available clients for federated learning"""
        if min_clients is None:
            min_clients = self.min_available_clients
            
        active_clients = self.get_active_clients(model_type)
        
        if len(active_clients) < min_clients:
            logger.warning(f"Insufficient clients available: {len(active_clients)} < {min_clients}")
            return []
            
        # Sort by trust score and data size
        sorted_clients = sorted(
            active_clients,
            key=lambda c: (c.trust_score, c.data_size),
            reverse=True
        )
        
        return sorted_clients[:min(len(sorted_clients), min_clients * 2)]  # Return up to 2x min_clients
        
    def update_client_metrics(self, client_id: str, success: bool, response_time: float):
        """Update client performance metrics"""
        with self.lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.participation_count += 1
                
                if success:
                    client.success_count += 1
                    # Update trust score positively
                    client.trust_score = min(1.0, client.trust_score + 0.01)
                else:
                    client.failure_count += 1
                    # Update trust score negatively
                    client.trust_score = max(0.0, client.trust_score - 0.05)
                    
                # Update average response time
                if client.avg_response_time == 0.0:
                    client.avg_response_time = response_time
                else:
                    client.avg_response_time = 0.9 * client.avg_response_time + 0.1 * response_time
                    
    def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """Get client information"""
        with self.lock:
            return self.clients.get(client_id)
            
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client manager statistics"""
        with self.lock:
            total_clients = len(self.clients)
            active_clients = len(self.active_clients)
            
            # Calculate average trust score
            avg_trust_score = 0.0
            if total_clients > 0:
                trust_scores = [c.trust_score for c in self.clients.values()]
                avg_trust_score = sum(trust_scores) / len(trust_scores)
                
            # Count by node type
            node_type_counts = {}
            for client in self.clients.values():
                node_type_counts[client.node_type] = node_type_counts.get(client.node_type, 0) + 1
                
            return {
                "total_clients": total_clients,
                "active_clients": active_clients,
                "avg_trust_score": avg_trust_score,
                "node_type_distribution": node_type_counts,
                "total_registrations": self.total_registrations,
                "total_deregistrations": self.total_deregistrations,
                "health_check_count": self.health_check_count
            }
            
    def _health_check_loop(self):
        """Health check loop for monitoring client status"""
        while self.running:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                
    def _perform_health_check(self):
        """Perform health check on all clients"""
        with self.lock:
            current_time = datetime.now()
            timeout_threshold = current_time - timedelta(seconds=self.client_timeout)
            
            clients_to_deactivate = []
            
            for client_id, client in self.clients.items():
                if client.last_heartbeat < timeout_threshold:
                    clients_to_deactivate.append(client_id)
                    
            # Deactivate timed out clients
            for client_id in clients_to_deactivate:
                if client_id in self.clients:
                    self.clients[client_id].is_active = False
                    self.active_clients.discard(client_id)
                    logger.warning(f"Client {client_id} timed out and deactivated")
                    
            self.health_check_count += 1
            
    def cleanup_inactive_clients(self, max_inactive_days: int = 7):
        """Clean up inactive clients older than specified days"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(days=max_inactive_days)
            clients_to_remove = []
            
            for client_id, client in self.clients.items():
                if not client.is_active and client.last_heartbeat < cutoff_time:
                    clients_to_remove.append(client_id)
                    
            for client_id in clients_to_remove:
                del self.clients[client_id]
                logger.info(f"Removed inactive client {client_id}")
                
    def export_client_data(self) -> str:
        """Export client data to JSON"""
        with self.lock:
            client_data = {
                "clients": [client.to_dict() for client in self.clients.values()],
                "stats": self.get_client_stats(),
                "exported_at": datetime.now().isoformat()
            }
            return json.dumps(client_data, indent=2)
            
    def import_client_data(self, json_data: str):
        """Import client data from JSON"""
        try:
            data = json.loads(json_data)
            with self.lock:
                for client_dict in data.get("clients", []):
                    client_info = ClientInfo(
                        client_id=client_dict["client_id"],
                        node_name=client_dict["node_name"],
                        node_type=client_dict["node_type"],
                        ip_address=client_dict["ip_address"],
                        port=client_dict["port"],
                        capabilities=client_dict["capabilities"],
                        model_types=client_dict["model_types"],
                        data_size=client_dict["data_size"],
                        last_heartbeat=datetime.fromisoformat(client_dict["last_heartbeat"]),
                        is_active=client_dict["is_active"],
                        trust_score=client_dict["trust_score"],
                        participation_count=client_dict["participation_count"],
                        success_count=client_dict["success_count"],
                        failure_count=client_dict["failure_count"],
                        avg_response_time=client_dict["avg_response_time"],
                        registered_at=datetime.fromisoformat(client_dict["registered_at"])
                    )
                    self.clients[client_info.client_id] = client_info
                    if client_info.is_active:
                        self.active_clients.add(client_info.client_id)
                        
            logger.info(f"Imported {len(data.get('clients', []))} clients")
        except Exception as e:
            logger.error(f"Error importing client data: {e}")
            raise
