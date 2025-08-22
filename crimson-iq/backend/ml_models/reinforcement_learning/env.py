# Environment simulating inventory & cold chain state
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import random
from enum import Enum

class BloodGroup(Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class InventoryAction(Enum):
    MAINTAIN_TEMP = 0
    ADJUST_TEMP_UP = 1
    ADJUST_TEMP_DOWN = 2
    REORDER_BLOOD = 3
    REDISTRIBUTE_INVENTORY = 4
    NO_ACTION = 5

class BloodInventoryEnv(gym.Env):
    """
    Reinforcement Learning Environment for Blood Inventory Management
    
    This environment simulates:
    - Blood inventory levels across different blood groups
    - Temperature monitoring and control
    - Demand forecasting and fulfillment
    - Cold chain optimization
    """
    
    def __init__(self, 
                 num_pods: int = 5,
                 time_horizon: int = 168,  # 1 week in hours
                 max_inventory: int = 100,
                 min_temp: float = 2.0,
                 max_temp: float = 6.0,
                 optimal_temp: float = 4.0):
        
        super().__init__()
        
        # Environment parameters
        self.num_pods = num_pods
        self.time_horizon = time_horizon
        self.max_inventory = max_inventory
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.optimal_temp = optimal_temp
        
        # Blood groups
        self.blood_groups = list(BloodGroup)
        self.num_blood_groups = len(self.blood_groups)
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(len(InventoryAction))
        
        # Observation space: [inventory_levels, temperatures, demand_forecast, time_features]
        obs_dim = (self.num_pods * self.num_blood_groups +  # inventory levels
                  self.num_pods +                           # temperatures
                  self.num_pods * self.num_blood_groups +   # demand forecast
                  4)                                        # time features (hour, day, week, month)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Environment state
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize inventory levels (random but realistic)
        self.inventory = np.random.randint(
            low=10, high=self.max_inventory//2, 
            size=(self.num_pods, self.num_blood_groups)
        )
        
        # Initialize temperatures (around optimal with some variation)
        self.temperatures = np.random.normal(
            loc=self.optimal_temp, scale=0.5,
            size=self.num_pods
        )
        self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
        
        # Initialize demand forecast (based on historical patterns)
        self.demand_forecast = self._generate_demand_forecast()
        
        # Time tracking
        self.current_time = 0
        self.current_datetime = datetime.now()
        
        # Performance metrics
        self.total_reward = 0
        self.inventory_shortages = 0
        self.temperature_violations = 0
        self.successful_deliveries = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Store previous state for reward calculation
        prev_inventory = self.inventory.copy()
        prev_temperatures = self.temperatures.copy()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update environment state
        self._update_demand()
        self._update_temperatures()
        self._update_inventory()
        
        # Calculate reward
        step_reward = self._calculate_reward(prev_inventory, prev_temperatures)
        reward += step_reward
        
        # Update metrics
        self.total_reward += reward
        self.current_time += 1
        
        # Check if episode is done
        done = self.current_time >= self.time_horizon
        
        return self._get_observation(), reward, done, False, self._get_info()
    
    def _execute_action(self, action: int) -> float:
        """Execute the selected action and return immediate reward"""
        action_enum = InventoryAction(action)
        reward = 0
        
        if action_enum == InventoryAction.MAINTAIN_TEMP:
            # Small positive reward for maintaining temperature
            reward += 0.1
            
        elif action_enum == InventoryAction.ADJUST_TEMP_UP:
            # Adjust temperature up (with some randomness)
            adjustment = np.random.normal(0.5, 0.1, self.num_pods)
            self.temperatures += adjustment
            self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
            reward -= 0.2  # Cost of adjustment
            
        elif action_enum == InventoryAction.ADJUST_TEMP_DOWN:
            # Adjust temperature down (with some randomness)
            adjustment = np.random.normal(0.5, 0.1, self.num_pods)
            self.temperatures -= adjustment
            self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
            reward -= 0.2  # Cost of adjustment
            
        elif action_enum == InventoryAction.REORDER_BLOOD:
            # Reorder blood units (simplified)
            for pod in range(self.num_pods):
                for bg in range(self.num_blood_groups):
                    if self.inventory[pod, bg] < self.max_inventory // 4:
                        self.inventory[pod, bg] += np.random.randint(5, 15)
                        self.inventory[pod, bg] = min(self.inventory[pod, bg], self.max_inventory)
            reward -= 0.5  # Cost of reordering
            
        elif action_enum == InventoryAction.REDISTRIBUTE_INVENTORY:
            # Redistribute inventory between pods
            for bg in range(self.num_blood_groups):
                total_inventory = np.sum(self.inventory[:, bg])
                if total_inventory > 0:
                    # Simple redistribution: move from high to low inventory pods
                    high_pods = np.where(self.inventory[:, bg] > self.max_inventory // 2)[0]
                    low_pods = np.where(self.inventory[:, bg] < self.max_inventory // 4)[0]
                    
                    for high_pod in high_pods:
                        for low_pod in low_pods:
                            if self.inventory[high_pod, bg] > 10 and self.inventory[low_pod, bg] < self.max_inventory // 2:
                                transfer = min(5, self.inventory[high_pod, bg] - 5)
                                self.inventory[high_pod, bg] -= transfer
                                self.inventory[low_pod, bg] += transfer
            reward -= 0.3  # Cost of redistribution
            
        return reward
    
    def _update_demand(self):
        """Update demand forecast based on time and patterns"""
        # Simulate demand patterns (higher during certain hours/days)
        hour = self.current_datetime.hour
        day_of_week = self.current_datetime.weekday()
        
        # Higher demand during business hours and weekdays
        time_multiplier = 1.0
        if 8 <= hour <= 18:  # Business hours
            time_multiplier = 1.5
        if day_of_week < 5:  # Weekdays
            time_multiplier *= 1.2
            
        # Add some randomness
        time_multiplier *= np.random.normal(1.0, 0.2)
        
        self.demand_forecast = self._generate_demand_forecast() * time_multiplier
    
    def _update_temperatures(self):
        """Update temperatures with natural drift and external factors"""
        # Natural temperature drift
        drift = np.random.normal(0, 0.1, self.num_pods)
        self.temperatures += drift
        
        # External factors (time of day, equipment wear, etc.)
        hour = self.current_datetime.hour
        if 2 <= hour <= 6:  # Early morning - equipment might be less efficient
            self.temperatures += np.random.normal(0.1, 0.05, self.num_pods)
        
        # Clip to valid range
        self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
        
        # Update datetime
        self.current_datetime += timedelta(hours=1)
    
    def _update_inventory(self):
        """Update inventory based on demand and expiry"""
        # Simulate demand fulfillment
        for pod in range(self.num_pods):
            for bg in range(self.num_blood_groups):
                demand = int(self.demand_forecast[pod, bg])
                available = self.inventory[pod, bg]
                
                if demand > 0 and available > 0:
                    fulfilled = min(demand, available)
                    self.inventory[pod, bg] -= fulfilled
                    self.successful_deliveries += fulfilled
                    
                    if demand > available:
                        self.inventory_shortages += (demand - available)
        
        # Simulate blood expiry (simplified - 1% chance per unit per hour)
        for pod in range(self.num_pods):
            for bg in range(self.num_blood_groups):
                if self.inventory[pod, bg] > 0:
                    expiry_count = np.random.binomial(self.inventory[pod, bg], 0.01)
                    self.inventory[pod, bg] -= expiry_count
    
    def _calculate_reward(self, prev_inventory: np.ndarray, prev_temperatures: np.ndarray) -> float:
        """Calculate reward based on state changes and performance metrics"""
        reward = 0
        
        # Inventory management rewards
        for pod in range(self.num_pods):
            for bg in range(self.num_blood_groups):
                # Reward for maintaining adequate inventory
                if self.inventory[pod, bg] >= self.max_inventory // 4:
                    reward += 0.1
                else:
                    reward -= 0.2  # Penalty for low inventory
                
                # Penalty for overstocking
                if self.inventory[pod, bg] > self.max_inventory * 0.9:
                    reward -= 0.1
        
        # Temperature control rewards
        for pod in range(self.num_pods):
            temp = self.temperatures[pod]
            if self.min_temp <= temp <= self.max_temp:
                # Reward for optimal temperature
                if abs(temp - self.optimal_temp) < 0.5:
                    reward += 0.2
                else:
                    reward += 0.1
            else:
                reward -= 0.5  # Penalty for temperature violations
                self.temperature_violations += 1
        
        # Demand fulfillment rewards
        if self.successful_deliveries > 0:
            reward += 0.3
        
        # Penalty for shortages
        if self.inventory_shortages > 0:
            reward -= 0.4
        
        return reward
    
    def _generate_demand_forecast(self) -> np.ndarray:
        """Generate realistic demand forecast"""
        # Base demand varies by blood group (O+ and A+ are most common)
        base_demand = np.array([0.3, 0.1, 0.2, 0.05, 0.05, 0.02, 0.25, 0.03])  # A+, A-, B+, B-, AB+, AB-, O+, O-
        
        # Generate demand for each pod and blood group
        forecast = np.zeros((self.num_pods, self.num_blood_groups))
        for pod in range(self.num_pods):
            for bg in range(self.num_blood_groups):
                # Base demand with some randomness
                demand = base_demand[bg] * np.random.normal(1.0, 0.3)
                demand = max(0, demand)  # Ensure non-negative
                forecast[pod, bg] = demand
        
        return forecast
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as numpy array"""
        obs = []
        
        # Inventory levels (normalized)
        obs.extend(self.inventory.flatten() / self.max_inventory)
        
        # Temperatures (normalized)
        obs.extend((self.temperatures - self.min_temp) / (self.max_temp - self.min_temp))
        
        # Demand forecast (normalized)
        max_demand = np.max(self.demand_forecast) if np.max(self.demand_forecast) > 0 else 1
        obs.extend(self.demand_forecast.flatten() / max_demand)
        
        # Time features
        hour = self.current_datetime.hour / 24.0
        day = self.current_datetime.weekday() / 7.0
        week = self.current_datetime.isocalendar()[1] / 53.0
        month = self.current_datetime.month / 12.0
        
        obs.extend([hour, day, week, month])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state"""
        return {
            'current_time': self.current_time,
            'current_datetime': self.current_datetime,
            'total_reward': self.total_reward,
            'inventory_shortages': self.inventory_shortages,
            'temperature_violations': self.temperature_violations,
            'successful_deliveries': self.successful_deliveries,
            'avg_inventory_level': np.mean(self.inventory),
            'avg_temperature': np.mean(self.temperatures),
            'min_inventory': np.min(self.inventory),
            'max_inventory': np.max(self.inventory),
        }
    
    def render(self):
        """Render the current state (for debugging)"""
        print(f"Time: {self.current_time}/{self.time_horizon}")
        print(f"DateTime: {self.current_datetime}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Average Temperature: {np.mean(self.temperatures):.2f}°C")
        print(f"Average Inventory: {np.mean(self.inventory):.1f}")
        print(f"Shortages: {self.inventory_shortages}")
        print(f"Temperature Violations: {self.temperature_violations}")
        print(f"Successful Deliveries: {self.successful_deliveries}")
        print("-" * 50)
