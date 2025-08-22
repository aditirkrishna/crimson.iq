"""
Reinforcement Learning Inference Module for Blood Inventory Management
Provides decision support for inventory ordering and reallocation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime, timedelta
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, A2C
from env import BloodInventoryEnv, NodeType, BloodGroup, InventoryAction

logger = logging.getLogger(__name__)

class RLInferenceEngine:
    """Reinforcement Learning inference engine for inventory decisions"""
    
    def __init__(self, model_path: str, algorithm: str, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.algorithm = algorithm
        self.config = config or {}
        
        # Load trained model
        self.model = self._load_model(model_path, algorithm)
        
        # Create environment for inference
        self.env = BloodInventoryEnv(
            num_hospitals=self.config.get("num_hospitals", 3),
            num_blood_banks=self.config.get("num_blood_banks", 2),
            num_pods=self.config.get("num_pods", 5),
            time_horizon=self.config.get("time_horizon", 168),
            max_inventory=self.config.get("max_inventory", 200),
            min_temp=self.config.get("min_temp", 2.0),
            max_temp=self.config.get("max_temp", 6.0),
            optimal_temp=self.config.get("optimal_temp", 4.0)
        )
        
        # Decision tracking
        self.decision_history = []
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "avg_reward": 0.0,
            "last_decision_time": None
        }
        
        # Fallback mechanisms
        self.fallback_threshold = self.config.get("fallback_threshold", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        logger.info(f"RL Inference Engine initialized with {algorithm} model")
    
    def _load_model(self, model_path: str, algorithm: str):
        """Load trained RL model"""
        try:
            if algorithm == "dqn":
                model = DQN.load(model_path)
            elif algorithm == "ppo":
                model = PPO.load(model_path)
            elif algorithm == "a2c":
                model = A2C.load(model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            logger.info(f"Successfully loaded {algorithm} model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_inventory_decision(self, 
                             current_state: Dict[str, Any],
                             confidence_required: bool = True) -> Dict[str, Any]:
        """
        Get inventory management decision based on current state
        
        Args:
            current_state: Current inventory and system state
            confidence_required: Whether to require confidence threshold
            
        Returns:
            Decision with action, confidence, and reasoning
        """
        try:
            # Convert current state to environment observation
            obs = self._state_to_observation(current_state)
            
            # Get model prediction
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Get action confidence (for DQN, we can estimate confidence from Q-values)
            confidence = self._get_action_confidence(obs, action)
            
            # Check if confidence meets threshold
            if confidence_required and confidence < self.confidence_threshold:
                logger.warning(f"Low confidence ({confidence:.3f}), using fallback policy")
                action, confidence = self._get_fallback_decision(current_state)
            
            # Convert action to human-readable decision
            decision = self._action_to_decision(action, current_state)
            decision["confidence"] = confidence
            decision["algorithm"] = self.algorithm
            decision["timestamp"] = datetime.now().isoformat()
            
            # Track decision
            self._track_decision(decision, current_state)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting inventory decision: {e}")
            return self._get_fallback_decision(current_state)
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert current state to environment observation"""
        # This is a simplified conversion - in practice, you'd need more sophisticated mapping
        obs = np.zeros(self.env.observation_space.shape[0], dtype=np.float32)
        
        # Map inventory levels
        if "inventory_levels" in state:
            inventory = state["inventory_levels"]
            if isinstance(inventory, dict):
                # Convert dict to array format expected by environment
                inventory_array = np.zeros((self.env.total_nodes, self.env.num_blood_groups))
                for node_id, node_inventory in inventory.items():
                    node_idx = int(node_id) if node_id.isdigit() else 0
                    for bg, level in node_inventory.items():
                        bg_idx = list(BloodGroup).index(BloodGroup(bg))
                        inventory_array[node_idx, bg_idx] = level / self.env.max_inventory
                
                obs[:self.env.total_nodes * self.env.num_blood_groups] = inventory_array.flatten()
        
        # Map temperatures
        if "temperatures" in state:
            temps = state["temperatures"]
            if isinstance(temps, list) and len(temps) >= self.env.total_nodes:
                normalized_temps = [(t - self.env.min_temp) / (self.env.max_temp - self.env.min_temp) 
                                  for t in temps[:self.env.total_nodes]]
                start_idx = self.env.total_nodes * self.env.num_blood_groups
                obs[start_idx:start_idx + self.env.total_nodes] = normalized_temps
        
        # Add time features
        current_time = datetime.now()
        hour = current_time.hour / 24.0
        day = current_time.weekday() / 7.0
        week = current_time.isocalendar()[1] / 53.0
        month = current_time.month / 12.0
        season = (current_time.month % 12) // 3 / 4.0
        
        # Find time features position in observation
        time_start_idx = -8  # Approximate position for time features
        obs[time_start_idx:time_start_idx + 5] = [hour, day, week, month, season]
        
        return obs
    
    def _get_action_confidence(self, obs: np.ndarray, action: int) -> float:
        """Get confidence score for the selected action"""
        if self.algorithm == "dqn":
            # For DQN, we can estimate confidence from Q-value distribution
            try:
                # Get Q-values for all actions
                q_values = self.model.q_net(torch.FloatTensor(obs).unsqueeze(0)).detach().numpy()[0]
                
                # Calculate confidence as normalized Q-value for selected action
                max_q = np.max(q_values)
                min_q = np.min(q_values)
                if max_q != min_q:
                    confidence = (q_values[action] - min_q) / (max_q - min_q)
                else:
                    confidence = 0.5
                
                return float(confidence)
                
            except Exception as e:
                logger.warning(f"Could not calculate Q-value confidence: {e}")
                return 0.5
        
        elif self.algorithm in ["ppo", "a2c"]:
            # For policy gradient methods, we can use action probabilities
            try:
                # Get action probabilities
                action_probs = self.model.policy.get_distribution(obs).distribution.probs.detach().numpy()
                confidence = float(action_probs[action])
                return confidence
                
            except Exception as e:
                logger.warning(f"Could not calculate policy confidence: {e}")
                return 0.5
        
        else:
            return 0.5  # Default confidence
    
    def _action_to_decision(self, action: int, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action to human-readable decision"""
        action_enum = InventoryAction(action)
        
        decision = {
            "action": action_enum.name,
            "action_id": action,
            "reasoning": self._get_action_reasoning(action_enum, current_state),
            "expected_impact": self._get_expected_impact(action_enum, current_state)
        }
        
        return decision
    
    def _get_action_reasoning(self, action: InventoryAction, state: Dict[str, Any]) -> str:
        """Get reasoning for the selected action"""
        reasoning_map = {
            InventoryAction.MAINTAIN_TEMP: "Maintain current temperature settings for optimal blood storage",
            InventoryAction.ADJUST_TEMP_UP: "Increase temperature to prevent freezing and maintain blood viability",
            InventoryAction.ADJUST_TEMP_DOWN: "Decrease temperature to prevent spoilage and extend shelf life",
            InventoryAction.ORDER_SMALL: "Place small order to replenish low inventory levels",
            InventoryAction.ORDER_MEDIUM: "Place medium order to address moderate inventory shortages",
            InventoryAction.ORDER_LARGE: "Place large order to address significant inventory shortages",
            InventoryAction.REDISTRIBUTE_LOCAL: "Redistribute inventory within same facility to balance stock levels",
            InventoryAction.REDISTRIBUTE_REGIONAL: "Redistribute inventory between facilities to optimize regional supply",
            InventoryAction.EMERGENCY_ORDER: "Place emergency order due to critical inventory shortage",
            InventoryAction.ACTIVATE_BACKUP: "Activate backup storage systems for temperature control",
            InventoryAction.NO_ACTION: "No action required - current state is optimal"
        }
        
        return reasoning_map.get(action, "Action selected based on current inventory and temperature conditions")
    
    def _get_expected_impact(self, action: InventoryAction, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get expected impact of the action"""
        impact = {
            "inventory_change": 0,
            "temperature_change": 0,
            "cost_impact": 0,
            "risk_level": "low"
        }
        
        if action in [InventoryAction.ORDER_SMALL, InventoryAction.ORDER_MEDIUM, InventoryAction.ORDER_LARGE]:
            order_sizes = {
                InventoryAction.ORDER_SMALL: 15,
                InventoryAction.ORDER_MEDIUM: 30,
                InventoryAction.ORDER_LARGE: 50
            }
            impact["inventory_change"] = order_sizes[action]
            impact["cost_impact"] = order_sizes[action] * 0.1  # Cost per unit
            impact["risk_level"] = "low"
            
        elif action in [InventoryAction.ADJUST_TEMP_UP, InventoryAction.ADJUST_TEMP_DOWN]:
            impact["temperature_change"] = 0.3
            impact["cost_impact"] = 0.2
            impact["risk_level"] = "medium"
            
        elif action == InventoryAction.EMERGENCY_ORDER:
            impact["inventory_change"] = 40
            impact["cost_impact"] = 40 * 0.3  # Higher cost for emergency orders
            impact["risk_level"] = "high"
            
        elif action == InventoryAction.ACTIVATE_BACKUP:
            impact["cost_impact"] = 0.1
            impact["risk_level"] = "low"
        
        return impact
    
    def _get_fallback_decision(self, current_state: Dict[str, Any]) -> Tuple[int, float]:
        """Get fallback decision when RL model fails or has low confidence"""
        # Simple heuristic-based fallback
        inventory_levels = current_state.get("inventory_levels", {})
        temperatures = current_state.get("temperatures", [])
        
        # Check for critical issues
        if temperatures:
            avg_temp = np.mean(temperatures)
            if avg_temp > 5.5:  # Too warm
                return InventoryAction.ADJUST_TEMP_DOWN.value, 0.8
            elif avg_temp < 2.5:  # Too cold
                return InventoryAction.ADJUST_TEMP_UP.value, 0.8
        
        # Check inventory levels
        if inventory_levels:
            avg_inventory = np.mean([level for node_inv in inventory_levels.values() 
                                   for level in node_inv.values()])
            if avg_inventory < 0.1:  # Critical shortage
                return InventoryAction.EMERGENCY_ORDER.value, 0.9
            elif avg_inventory < 0.2:  # Low inventory
                return InventoryAction.ORDER_MEDIUM.value, 0.7
            elif avg_inventory < 0.3:  # Moderate inventory
                return InventoryAction.ORDER_SMALL.value, 0.6
        
        # Default: maintain temperature
        return InventoryAction.MAINTAIN_TEMP.value, 0.5
    
    def _track_decision(self, decision: Dict[str, Any], state: Dict[str, Any]):
        """Track decision for monitoring and learning"""
        decision_record = {
            "timestamp": decision["timestamp"],
            "action": decision["action"],
            "confidence": decision["confidence"],
            "state_summary": {
                "avg_inventory": np.mean([level for node_inv in state.get("inventory_levels", {}).values() 
                                        for level in node_inv.values()]) if state.get("inventory_levels") else 0,
                "avg_temperature": np.mean(state.get("temperatures", [4.0])),
                "num_shortages": state.get("shortages", 0),
                "num_violations": state.get("temperature_violations", 0)
            },
            "expected_impact": decision["expected_impact"]
        }
        
        self.decision_history.append(decision_record)
        
        # Update performance metrics
        self.performance_metrics["total_decisions"] += 1
        self.performance_metrics["last_decision_time"] = decision["timestamp"]
        
        if decision["confidence"] > self.confidence_threshold:
            self.performance_metrics["successful_decisions"] += 1
        else:
            self.performance_metrics["failed_decisions"] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        metrics = self.performance_metrics.copy()
        
        if metrics["total_decisions"] > 0:
            metrics["success_rate"] = metrics["successful_decisions"] / metrics["total_decisions"]
            metrics["failure_rate"] = metrics["failed_decisions"] / metrics["total_decisions"]
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
        
        # Calculate average confidence
        if self.decision_history:
            avg_confidence = np.mean([d["confidence"] for d in self.decision_history])
            metrics["avg_confidence"] = avg_confidence
        else:
            metrics["avg_confidence"] = 0.0
        
        return metrics
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        return self.decision_history[-limit:] if self.decision_history else []
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "avg_reward": 0.0,
            "last_decision_time": None
        }
        self.decision_history = []
        logger.info("Performance metrics reset")

class RLDecisionSupport:
    """High-level decision support system using RL models"""
    
    def __init__(self, models_config: Dict[str, str]):
        """
        Initialize decision support system with multiple RL models
        
        Args:
            models_config: Dictionary mapping algorithm names to model paths
        """
        self.models = {}
        self.engines = {}
        
        for algorithm, model_path in models_config.items():
            try:
                engine = RLInferenceEngine(model_path, algorithm)
                self.engines[algorithm] = engine
                logger.info(f"Loaded {algorithm} inference engine")
            except Exception as e:
                logger.error(f"Failed to load {algorithm} model: {e}")
        
        if not self.engines:
            raise ValueError("No RL models could be loaded")
        
        # Set default engine
        self.default_engine = list(self.engines.values())[0]
        
        logger.info(f"RL Decision Support initialized with {len(self.engines)} models")
    
    def get_optimal_decision(self, 
                           current_state: Dict[str, Any],
                           preferred_algorithm: Optional[str] = None,
                           ensemble: bool = False) -> Dict[str, Any]:
        """
        Get optimal decision using RL models
        
        Args:
            current_state: Current system state
            preferred_algorithm: Preferred RL algorithm to use
            ensemble: Whether to use ensemble of all available models
            
        Returns:
            Optimal decision with confidence and reasoning
        """
        if ensemble and len(self.engines) > 1:
            return self._get_ensemble_decision(current_state)
        else:
            # Use preferred algorithm or default
            engine = self.engines.get(preferred_algorithm, self.default_engine)
            return engine.get_inventory_decision(current_state)
    
    def _get_ensemble_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get decision using ensemble of all available models"""
        decisions = []
        confidences = []
        
        for algorithm, engine in self.engines.items():
            try:
                decision = engine.get_inventory_decision(current_state, confidence_required=False)
                decisions.append(decision)
                confidences.append(decision["confidence"])
            except Exception as e:
                logger.warning(f"Error getting decision from {algorithm}: {e}")
        
        if not decisions:
            # Fallback to default engine
            return self.default_engine.get_inventory_decision(current_state)
        
        # Weight decisions by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weights = [c / total_confidence for c in confidences]
            
            # Weighted voting for action selection
            action_votes = {}
            for decision, weight in zip(decisions, weights):
                action = decision["action"]
                action_votes[action] = action_votes.get(action, 0) + weight
            
            # Select action with highest weighted vote
            best_action = max(action_votes.items(), key=lambda x: x[1])[0]
            
            # Get average confidence
            avg_confidence = np.mean(confidences)
            
            # Create ensemble decision
            ensemble_decision = {
                "action": best_action,
                "confidence": avg_confidence,
                "algorithm": "ensemble",
                "timestamp": datetime.now().isoformat(),
                "reasoning": f"Ensemble decision from {len(decisions)} models",
                "expected_impact": decisions[0]["expected_impact"],  # Use first decision's impact
                "ensemble_details": {
                    "num_models": len(decisions),
                    "individual_confidences": confidences,
                    "action_votes": action_votes
                }
            }
            
            return ensemble_decision
        else:
            # Fallback
            return self.default_engine.get_inventory_decision(current_state)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and performance"""
        health = {
            "total_engines": len(self.engines),
            "active_engines": 0,
            "overall_performance": {},
            "engine_details": {}
        }
        
        total_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "avg_confidence": 0.0
        }
        
        for algorithm, engine in self.engines.items():
            try:
                metrics = engine.get_performance_metrics()
                health["engine_details"][algorithm] = metrics
                health["active_engines"] += 1
                
                # Aggregate metrics
                total_metrics["total_decisions"] += metrics["total_decisions"]
                total_metrics["successful_decisions"] += metrics["successful_decisions"]
                total_metrics["failed_decisions"] += metrics["failed_decisions"]
                total_metrics["avg_confidence"] += metrics["avg_confidence"]
                
            except Exception as e:
                logger.error(f"Error getting metrics for {algorithm}: {e}")
                health["engine_details"][algorithm] = {"error": str(e)}
        
        # Calculate overall performance
        if total_metrics["total_decisions"] > 0:
            health["overall_performance"] = {
                "total_decisions": total_metrics["total_decisions"],
                "success_rate": total_metrics["successful_decisions"] / total_metrics["total_decisions"],
                "failure_rate": total_metrics["failed_decisions"] / total_metrics["total_decisions"],
                "avg_confidence": total_metrics["avg_confidence"] / len(self.engines)
            }
        else:
            health["overall_performance"] = {
                "total_decisions": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "avg_confidence": 0.0
            }
        
        return health
    
    def update_model(self, algorithm: str, new_model_path: str):
        """Update RL model for a specific algorithm"""
        try:
            engine = RLInferenceEngine(new_model_path, algorithm)
            self.engines[algorithm] = engine
            logger.info(f"Updated {algorithm} model successfully")
        except Exception as e:
            logger.error(f"Failed to update {algorithm} model: {e}")
            raise

def create_rl_decision_support(models_dir: str = "./models") -> RLDecisionSupport:
    """Create RL decision support system from saved models"""
    models_config = {}
    
    # Look for saved models
    for algorithm in ["dqn", "ppo", "a2c"]:
        model_path = os.path.join(models_dir, f"{algorithm}_best", "best_model")
        if os.path.exists(model_path + ".zip"):
            models_config[algorithm] = model_path
    
    if not models_config:
        raise ValueError(f"No RL models found in {models_dir}")
    
    return RLDecisionSupport(models_config)

if __name__ == "__main__":
    # Example usage
    try:
        # Create decision support system
        decision_support = create_rl_decision_support()
        
        # Example current state
        current_state = {
            "inventory_levels": {
                "0": {"A+": 0.3, "A-": 0.1, "B+": 0.2, "B-": 0.05, 
                      "AB+": 0.05, "AB-": 0.02, "O+": 0.25, "O-": 0.03},
                "1": {"A+": 0.2, "A-": 0.08, "B+": 0.15, "B-": 0.03, 
                      "AB+": 0.04, "AB-": 0.01, "O+": 0.3, "O-": 0.05}
            },
            "temperatures": [4.2, 3.8, 4.5, 4.1, 4.0],
            "shortages": 2,
            "temperature_violations": 0
        }
        
        # Get decision
        decision = decision_support.get_optimal_decision(current_state, ensemble=True)
        
        print("RL Decision:")
        print(f"Action: {decision['action']}")
        print(f"Confidence: {decision['confidence']:.3f}")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Expected Impact: {decision['expected_impact']}")
        
        # Get system health
        health = decision_support.get_system_health()
        print(f"\nSystem Health: {health['overall_performance']}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
        print("Please ensure RL models are trained and saved before running inference")
