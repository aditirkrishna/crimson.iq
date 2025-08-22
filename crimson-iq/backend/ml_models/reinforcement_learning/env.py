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

class NodeType(Enum):
    HOSPITAL = "hospital"
    BLOOD_BANK = "blood_bank"
    POD = "pod"

class InventoryAction(Enum):
    # Temperature control actions
    MAINTAIN_TEMP = 0
    ADJUST_TEMP_UP = 1
    ADJUST_TEMP_DOWN = 2
    
    # Inventory management actions
    ORDER_SMALL = 3      # Order 10-20 units
    ORDER_MEDIUM = 4     # Order 20-40 units
    ORDER_LARGE = 5      # Order 40-60 units
    
    # Redistribution actions
    REDISTRIBUTE_LOCAL = 6   # Within same facility
    REDISTRIBUTE_REGIONAL = 7  # Between facilities
    
    # Emergency actions
    EMERGENCY_ORDER = 8
    ACTIVATE_BACKUP = 9
    
    # No action
    NO_ACTION = 10

class BloodUnit:
    """Represents a blood unit with age and expiry tracking"""
    
    def __init__(self, blood_group: BloodGroup, donation_date: datetime, 
                 initial_quantity: int = 1, storage_location: str = "main"):
        self.blood_group = blood_group
        self.donation_date = donation_date
        self.quantity = initial_quantity
        self.storage_location = storage_location
        self.temperature_history = []
        self.is_expired = False
        
    def update_age(self, current_date: datetime) -> int:
        """Update age and check expiry"""
        age = (current_date - self.donation_date).days
        # Blood expires after 42 days
        if age > 42:
            self.is_expired = True
        return age
    
    def add_temperature_reading(self, temp: float, timestamp: datetime):
        """Add temperature reading to history"""
        self.temperature_history.append((temp, timestamp))
        
    def get_temperature_violations(self) -> int:
        """Count temperature violations (>6°C or <2°C)"""
        violations = 0
        for temp, _ in self.temperature_history:
            if temp > 6.0 or temp < 2.0:
                violations += 1
        return violations

class BloodInventoryEnv(gym.Env):
    """
    Enhanced Reinforcement Learning Environment for Blood Inventory Management
    
    This environment simulates:
    - Blood inventory levels across different blood groups and locations
    - Age/expiry tracking for each blood unit
    - Temperature monitoring and cold chain compliance
    - Supply chain dynamics with ordering and delivery delays
    - Multi-node simulation (hospitals, blood banks, pods)
    - Stochastic demand patterns and seasonal variations
    """
    
    def __init__(self, 
                 num_hospitals: int = 3,
                 num_blood_banks: int = 2,
                 num_pods: int = 5,
                 time_horizon: int = 168,  # 1 week in hours
                 max_inventory: int = 200,
                 min_temp: float = 2.0,
                 max_temp: float = 6.0,
                 optimal_temp: float = 4.0,
                 order_delay_hours: int = 24,
                 delivery_delay_hours: int = 48):
        
        super().__init__()
        
        # Environment parameters
        self.num_hospitals = num_hospitals
        self.num_blood_banks = num_blood_banks
        self.num_pods = num_pods
        self.total_nodes = num_hospitals + num_blood_banks + num_pods
        self.time_horizon = time_horizon
        self.max_inventory = max_inventory
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.optimal_temp = optimal_temp
        self.order_delay_hours = order_delay_hours
        self.delivery_delay_hours = delivery_delay_hours
        
        # Blood groups and their relative frequencies
        self.blood_groups = list(BloodGroup)
        self.num_blood_groups = len(self.blood_groups)
        self.blood_group_frequencies = {
            BloodGroup.A_POSITIVE: 0.30,
            BloodGroup.A_NEGATIVE: 0.08,
            BloodGroup.B_POSITIVE: 0.09,
            BloodGroup.B_NEGATIVE: 0.02,
            BloodGroup.AB_POSITIVE: 0.04,
            BloodGroup.AB_NEGATIVE: 0.01,
            BloodGroup.O_POSITIVE: 0.38,
            BloodGroup.O_NEGATIVE: 0.08
        }
        
        # Node types and their characteristics
        self.node_types = []
        for i in range(num_hospitals):
            self.node_types.append(NodeType.HOSPITAL)
        for i in range(num_blood_banks):
            self.node_types.append(NodeType.BLOOD_BANK)
        for i in range(num_pods):
            self.node_types.append(NodeType.POD)
        
        # Action space: discrete actions for inventory management
        self.action_space = gym.spaces.Discrete(len(InventoryAction))
        
        # Enhanced observation space
        obs_dim = self._calculate_observation_dim()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Environment state
        self.reset()
        
    def _calculate_observation_dim(self) -> int:
        """Calculate observation space dimension"""
        dim = 0
        
        # Inventory levels by blood group and node
        dim += self.total_nodes * self.num_blood_groups
        
        # Age distribution (bins: 0-7, 8-14, 15-21, 22-28, 29-35, 36-42, 43+ days)
        dim += self.total_nodes * self.num_blood_groups * 7
        
        # Temperature readings for each node
        dim += self.total_nodes
        
        # Temperature violations count
        dim += self.total_nodes
        
        # Pending orders (by blood group and node)
        dim += self.total_nodes * self.num_blood_groups
        
        # Delivery times for pending orders
        dim += self.total_nodes * self.num_blood_groups
        
        # Demand forecast (by blood group and node)
        dim += self.total_nodes * self.num_blood_groups
        
        # Recent demand trends (last 24 hours)
        dim += self.total_nodes * self.num_blood_groups
        
        # Time features (hour, day, week, month, season)
        dim += 5
        
        # Node type indicators
        dim += 3  # hospital, blood_bank, pod
        
        return dim
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize blood inventory with age tracking
        self.blood_inventory = {}  # node -> blood_group -> list of BloodUnit
        self._initialize_blood_inventory()
        
        # Initialize temperatures
        self.temperatures = np.random.normal(
            loc=self.optimal_temp, scale=0.5, size=self.total_nodes
        )
        self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
        
        # Temperature violation counters
        self.temperature_violations = np.zeros(self.total_nodes, dtype=int)
        
        # Pending orders and delivery times
        self.pending_orders = np.zeros((self.total_nodes, self.num_blood_groups), dtype=int)
        self.delivery_times = np.zeros((self.total_nodes, self.num_blood_groups), dtype=int)
        
        # Demand tracking
        self.demand_forecast = self._generate_demand_forecast()
        self.recent_demand = np.zeros((self.total_nodes, self.num_blood_groups), dtype=int)
        
        # Time tracking
        self.current_time = 0
        self.current_datetime = datetime.now()
        
        # Performance metrics
        self.total_reward = 0
        self.inventory_shortages = 0
        self.successful_deliveries = 0
        self.expired_units = 0
        self.temperature_excursions = 0
        self.ordering_costs = 0
        self.holding_costs = 0
        
        return self._get_observation(), self._get_info()
    
    def _initialize_blood_inventory(self):
        """Initialize blood inventory with realistic age distribution"""
        for node in range(self.total_nodes):
            self.blood_inventory[node] = {}
            for bg in self.blood_groups:
                self.blood_inventory[node][bg] = []
                
                # Generate initial inventory with varying ages
                num_units = np.random.randint(10, self.max_inventory // 2)
                for _ in range(num_units):
                    # Random age between 0 and 35 days (avoid expired units initially)
                    age_days = np.random.randint(0, 35)
                    donation_date = self.current_datetime - timedelta(days=age_days)
                    quantity = np.random.randint(1, 3)  # 1-2 units per donation
                    
                    blood_unit = BloodUnit(bg, donation_date, quantity, f"node_{node}")
                    self.blood_inventory[node][bg].append(blood_unit)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Store previous state for reward calculation
        prev_inventory_levels = self._get_inventory_levels()
        prev_temperatures = self.temperatures.copy()
        
        # Execute action
        action_reward = self._execute_action(action)
        
        # Update environment state
        self._update_blood_units()
        self._update_demand()
        self._update_temperatures()
        self._update_orders()
        self._update_inventory()
        
        # Calculate comprehensive reward
        step_reward = self._calculate_reward(prev_inventory_levels, prev_temperatures)
        total_reward = action_reward + step_reward
        
        # Update metrics
        self.total_reward += total_reward
        self.current_time += 1
        
        # Check if episode is done
        done = self.current_time >= self.time_horizon
        
        return self._get_observation(), total_reward, done, False, self._get_info()
    
    def _execute_action(self, action: int) -> float:
        """Execute the selected action and return immediate reward"""
        action_enum = InventoryAction(action)
        reward = 0
        
        if action_enum == InventoryAction.MAINTAIN_TEMP:
            # Small positive reward for maintaining temperature
            reward += 0.1
            
        elif action_enum == InventoryAction.ADJUST_TEMP_UP:
            # Adjust temperature up for all nodes
            adjustment = np.random.normal(0.3, 0.1, self.total_nodes)
            self.temperatures += adjustment
            self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
            reward -= 0.2  # Cost of adjustment
            
        elif action_enum == InventoryAction.ADJUST_TEMP_DOWN:
            # Adjust temperature down for all nodes
            adjustment = np.random.normal(0.3, 0.1, self.total_nodes)
            self.temperatures -= adjustment
            self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
            reward -= 0.2  # Cost of adjustment
            
        elif action_enum == InventoryAction.ORDER_SMALL:
            # Place small orders for low inventory
            reward += self._place_orders(10, 20)
            
        elif action_enum == InventoryAction.ORDER_MEDIUM:
            # Place medium orders
            reward += self._place_orders(20, 40)
            
        elif action_enum == InventoryAction.ORDER_LARGE:
            # Place large orders
            reward += self._place_orders(40, 60)
            
        elif action_enum == InventoryAction.REDISTRIBUTE_LOCAL:
            # Redistribute within same facility type
            reward += self._redistribute_inventory(local=True)
            
        elif action_enum == InventoryAction.REDISTRIBUTE_REGIONAL:
            # Redistribute between facilities
            reward += self._redistribute_inventory(local=False)
            
        elif action_enum == InventoryAction.EMERGENCY_ORDER:
            # Emergency order (faster delivery but higher cost)
            reward += self._place_emergency_orders()
            
        elif action_enum == InventoryAction.ACTIVATE_BACKUP:
            # Activate backup storage systems
            reward += self._activate_backup_systems()
            
        return reward
    
    def _place_orders(self, min_quantity: int, max_quantity: int) -> float:
        """Place orders for low inventory blood groups"""
        reward = 0
        order_cost = 0
        
        for node in range(self.total_nodes):
            for bg_idx, bg in enumerate(self.blood_groups):
                current_level = len(self.blood_inventory[node][bg])
                
                # Order if inventory is low
                if current_level < self.max_inventory // 4:
                    quantity = np.random.randint(min_quantity, max_quantity + 1)
                    self.pending_orders[node, bg_idx] += quantity
                    self.delivery_times[node, bg_idx] = self.delivery_delay_hours
                    
                    order_cost += quantity * 0.1  # Cost per unit
                    reward -= 0.1  # Small penalty for ordering
        
        self.ordering_costs += order_cost
        return reward
    
    def _place_emergency_orders(self) -> float:
        """Place emergency orders with faster delivery"""
        reward = 0
        
        for node in range(self.total_nodes):
            for bg_idx, bg in enumerate(self.blood_groups):
                current_level = len(self.blood_inventory[node][bg])
                
                if current_level < self.max_inventory // 8:  # Very low inventory
                    quantity = np.random.randint(20, 40)
                    self.pending_orders[node, bg_idx] += quantity
                    self.delivery_times[node, bg_idx] = self.delivery_delay_hours // 2  # Faster delivery
                    
                    reward -= 0.3  # Higher cost for emergency orders
        
        return reward
    
    def _redistribute_inventory(self, local: bool = True) -> float:
        """Redistribute inventory between nodes"""
        reward = 0
        
        for bg in self.blood_groups:
            # Find nodes with high and low inventory
            inventory_levels = [len(self.blood_inventory[node][bg]) for node in range(self.total_nodes)]
            high_nodes = [i for i, level in enumerate(inventory_levels) if level > self.max_inventory // 2]
            low_nodes = [i for i, level in enumerate(inventory_levels) if level < self.max_inventory // 4]
            
            if local:
                # Only redistribute within same node type
                for high_node in high_nodes:
                    for low_node in low_nodes:
                        if (self.node_types[high_node] == self.node_types[low_node] and 
                            high_node != low_node):
                            reward += self._transfer_blood_units(high_node, low_node, bg, 5)
            else:
                # Redistribute between different node types
                for high_node in high_nodes:
                    for low_node in low_nodes:
                        if high_node != low_node:
                            reward += self._transfer_blood_units(high_node, low_node, bg, 3)
        
        return reward
    
    def _transfer_blood_units(self, from_node: int, to_node: int, 
                             blood_group: BloodGroup, max_units: int) -> float:
        """Transfer blood units between nodes"""
        if len(self.blood_inventory[from_node][blood_group]) == 0:
            return 0
        
        # Transfer oldest units first
        units_to_transfer = min(max_units, len(self.blood_inventory[from_node][blood_group]))
        transferred_units = self.blood_inventory[from_node][blood_group][:units_to_transfer]
        
        # Remove from source
        self.blood_inventory[from_node][blood_group] = self.blood_inventory[from_node][blood_group][units_to_transfer:]
        
        # Add to destination
        self.blood_inventory[to_node][blood_group].extend(transferred_units)
        
        return 0.1  # Small reward for successful transfer
    
    def _activate_backup_systems(self) -> float:
        """Activate backup storage systems"""
        reward = 0
        
        # Improve temperature stability
        temp_improvement = np.random.normal(0.2, 0.05, self.total_nodes)
        self.temperatures = np.clip(self.temperatures - temp_improvement, self.min_temp, self.max_temp)
        
        # Reduce temperature violations
        self.temperature_violations = np.maximum(0, self.temperature_violations - 1)
        
        reward += 0.2  # Reward for activating backup systems
        reward -= 0.1  # Cost of backup system activation
        
        return reward
    
    def _update_blood_units(self):
        """Update blood units (age, expiry, temperature)"""
        for node in range(self.total_nodes):
            for bg in self.blood_groups:
                # Update each blood unit
                expired_units = []
                for unit in self.blood_inventory[node][bg]:
                    # Update age
                    age = unit.update_age(self.current_datetime)
                    
                    # Add temperature reading
                    unit.add_temperature_reading(self.temperatures[node], self.current_datetime)
                    
                    # Check for expiry
                    if unit.is_expired:
                        expired_units.append(unit)
                        self.expired_units += unit.quantity
                
                # Remove expired units
                for unit in expired_units:
                    self.blood_inventory[node][bg].remove(unit)
    
    def _update_demand(self):
        """Update demand forecast and recent demand"""
        # Update recent demand (shift and add new demand)
        self.recent_demand = np.roll(self.recent_demand, 1, axis=0)
        
        # Generate new demand
        new_demand = self._generate_demand_forecast()
        self.recent_demand[0] = new_demand
        
        # Update forecast
        self.demand_forecast = new_demand
        
        # Update datetime
        self.current_datetime += timedelta(hours=1)
    
    def _update_temperatures(self):
        """Update temperatures with natural drift and external factors"""
        # Natural temperature drift
        drift = np.random.normal(0, 0.1, self.total_nodes)
        self.temperatures += drift
        
        # External factors (time of day, equipment wear, etc.)
        hour = self.current_datetime.hour
        if 2 <= hour <= 6:  # Early morning - equipment might be less efficient
            self.temperatures += np.random.normal(0.1, 0.05, self.total_nodes)
        
        # Clip to valid range
        self.temperatures = np.clip(self.temperatures, self.min_temp, self.max_temp)
        
        # Update temperature violations
        for node in range(self.total_nodes):
            if self.temperatures[node] < self.min_temp or self.temperatures[node] > self.max_temp:
                self.temperature_violations[node] += 1
                self.temperature_excursions += 1
    
    def _update_orders(self):
        """Update pending orders and delivery times"""
        for node in range(self.total_nodes):
            for bg_idx in range(self.num_blood_groups):
                if self.pending_orders[node, bg_idx] > 0:
                    self.delivery_times[node, bg_idx] -= 1
                    
                    # Deliver if time is up
                    if self.delivery_times[node, bg_idx] <= 0:
                        bg = self.blood_groups[bg_idx]
                        quantity = self.pending_orders[node, bg_idx]
                        
                        # Add delivered units to inventory
                        for _ in range(quantity):
                            donation_date = self.current_datetime - timedelta(hours=np.random.randint(0, 24))
                            blood_unit = BloodUnit(bg, donation_date, 1, f"node_{node}")
                            self.blood_inventory[node][bg].append(blood_unit)
                        
                        self.pending_orders[node, bg_idx] = 0
                        self.delivery_times[node, bg_idx] = 0
    
    def _update_inventory(self):
        """Update inventory based on demand fulfillment"""
        # Simulate demand fulfillment
        for node in range(self.total_nodes):
            for bg_idx, bg in enumerate(self.blood_groups):
                demand = int(self.demand_forecast[node, bg_idx])
                available_units = self.blood_inventory[node][bg]
                
                if demand > 0 and available_units:
                    # Fulfill demand using oldest units first
                    fulfilled = min(demand, len(available_units))
                    
                    # Remove fulfilled units
                    self.blood_inventory[node][bg] = available_units[fulfilled:]
                    self.successful_deliveries += fulfilled
                    
                    if demand > fulfilled:
                        self.inventory_shortages += (demand - fulfilled)
    
    def _calculate_reward(self, prev_inventory_levels: np.ndarray, 
                         prev_temperatures: np.ndarray) -> float:
        """Calculate comprehensive reward based on multiple factors"""
        reward = 0
        
        # Inventory management rewards
        current_levels = self._get_inventory_levels()
        for node in range(self.total_nodes):
            for bg_idx in range(self.num_blood_groups):
                level = current_levels[node, bg_idx]
                
                # Reward for maintaining adequate inventory (20-80% of max)
                if 0.2 <= level <= 0.8:
                    reward += 0.2
                elif level < 0.1:  # Critical shortage
                    reward -= 0.5
                elif level > 0.9:  # Overstocking
                    reward -= 0.1
                
                # Penalty for inventory decrease
                if level < prev_inventory_levels[node, bg_idx]:
                    reward -= 0.1
        
        # Temperature control rewards
        for node in range(self.total_nodes):
            temp = self.temperatures[node]
            if self.min_temp <= temp <= self.max_temp:
                # Reward for optimal temperature
                if abs(temp - self.optimal_temp) < 0.5:
                    reward += 0.3
                else:
                    reward += 0.1
            else:
                reward -= 0.8  # Severe penalty for temperature violations
        
        # Demand fulfillment rewards
        if self.successful_deliveries > 0:
            reward += 0.4
        
        # Penalties for issues
        if self.inventory_shortages > 0:
            reward -= 0.6
        
        if self.expired_units > 0:
            reward -= 0.3 * self.expired_units
        
        if self.temperature_excursions > 0:
            reward -= 0.2 * self.temperature_excursions
        
        # Cost considerations
        reward -= self.ordering_costs * 0.01
        reward -= self.holding_costs * 0.01
        
        return reward
    
    def _get_inventory_levels(self) -> np.ndarray:
        """Get current inventory levels normalized by max inventory"""
        levels = np.zeros((self.total_nodes, self.num_blood_groups))
        for node in range(self.total_nodes):
            for bg_idx, bg in enumerate(self.blood_groups):
                levels[node, bg_idx] = len(self.blood_inventory[node][bg]) / self.max_inventory
        return levels
    
    def _generate_demand_forecast(self) -> np.ndarray:
        """Generate realistic demand forecast with seasonal patterns"""
        forecast = np.zeros((self.total_nodes, self.num_blood_groups))
        
        # Base demand varies by blood group
        base_demand = np.array([0.3, 0.08, 0.09, 0.02, 0.04, 0.01, 0.38, 0.08])
        
        # Time-based demand patterns
        hour = self.current_datetime.hour
        day_of_week = self.current_datetime.weekday()
        month = self.current_datetime.month
        
        # Higher demand during business hours and weekdays
        time_multiplier = 1.0
        if 8 <= hour <= 18:  # Business hours
            time_multiplier = 1.5
        if day_of_week < 5:  # Weekdays
            time_multiplier *= 1.2
        
        # Seasonal patterns (higher in winter months)
        if month in [12, 1, 2]:  # Winter
            time_multiplier *= 1.1
        
        # Node-specific demand patterns
        for node in range(self.total_nodes):
            node_multiplier = 1.0
            if self.node_types[node] == NodeType.HOSPITAL:
                node_multiplier = 1.5  # Hospitals have higher demand
            elif self.node_types[node] == NodeType.BLOOD_BANK:
                node_multiplier = 0.8  # Blood banks have lower demand
            
            for bg_idx in range(self.num_blood_groups):
                # Base demand with randomness
                demand = base_demand[bg_idx] * time_multiplier * node_multiplier
                demand *= np.random.normal(1.0, 0.3)  # Add randomness
                demand = max(0, demand)  # Ensure non-negative
                forecast[node, bg_idx] = demand
        
        return forecast
    
    def _get_observation(self) -> np.ndarray:
        """Get comprehensive observation as numpy array"""
        obs = []
        
        # Inventory levels by blood group and node
        inventory_levels = self._get_inventory_levels()
        obs.extend(inventory_levels.flatten())
        
        # Age distribution (7 bins: 0-7, 8-14, 15-21, 22-28, 29-35, 36-42, 43+ days)
        age_distribution = self._get_age_distribution()
        obs.extend(age_distribution.flatten())
        
        # Temperature readings (normalized)
        obs.extend((self.temperatures - self.min_temp) / (self.max_temp - self.min_temp))
        
        # Temperature violations (normalized)
        max_violations = np.max(self.temperature_violations) if np.max(self.temperature_violations) > 0 else 1
        obs.extend(self.temperature_violations / max_violations)
        
        # Pending orders (normalized)
        max_orders = np.max(self.pending_orders) if np.max(self.pending_orders) > 0 else 1
        obs.extend(self.pending_orders.flatten() / max_orders)
        
        # Delivery times (normalized)
        max_delivery_time = np.max(self.delivery_times) if np.max(self.delivery_times) > 0 else 1
        obs.extend(self.delivery_times.flatten() / max_delivery_time)
        
        # Demand forecast (normalized)
        max_demand = np.max(self.demand_forecast) if np.max(self.demand_forecast) > 0 else 1
        obs.extend(self.demand_forecast.flatten() / max_demand)
        
        # Recent demand trends (normalized)
        max_recent_demand = np.max(self.recent_demand) if np.max(self.recent_demand) > 0 else 1
        obs.extend(self.recent_demand.flatten() / max_recent_demand)
        
        # Time features
        hour = self.current_datetime.hour / 24.0
        day = self.current_datetime.weekday() / 7.0
        week = self.current_datetime.isocalendar()[1] / 53.0
        month = self.current_datetime.month / 12.0
        season = (self.current_datetime.month % 12) // 3 / 4.0  # 0-3 for seasons
        
        obs.extend([hour, day, week, month, season])
        
        # Node type indicators (one-hot encoding)
        hospital_count = sum(1 for nt in self.node_types if nt == NodeType.HOSPITAL)
        blood_bank_count = sum(1 for nt in self.node_types if nt == NodeType.BLOOD_BANK)
        pod_count = sum(1 for nt in self.node_types if nt == NodeType.POD)
        
        obs.extend([hospital_count / self.total_nodes, 
                   blood_bank_count / self.total_nodes, 
                   pod_count / self.total_nodes])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_age_distribution(self) -> np.ndarray:
        """Get age distribution for each node and blood group"""
        age_bins = 7  # 0-7, 8-14, 15-21, 22-28, 29-35, 36-42, 43+ days
        distribution = np.zeros((self.total_nodes, self.num_blood_groups, age_bins))
        
        for node in range(self.total_nodes):
            for bg_idx, bg in enumerate(self.blood_groups):
                for unit in self.blood_inventory[node][bg]:
                    age = (self.current_datetime - unit.donation_date).days
                    bin_idx = min(age // 7, age_bins - 1)  # 7-day bins
                    distribution[node, bg_idx, bin_idx] += unit.quantity
        
        # Normalize by max inventory
        distribution = distribution / self.max_inventory
        return distribution
    
    def _get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the environment state"""
        return {
            'current_time': self.current_time,
            'current_datetime': self.current_datetime,
            'total_reward': self.total_reward,
            'inventory_shortages': self.inventory_shortages,
            'temperature_violations': int(np.sum(self.temperature_violations)),
            'successful_deliveries': self.successful_deliveries,
            'expired_units': self.expired_units,
            'temperature_excursions': self.temperature_excursions,
            'ordering_costs': self.ordering_costs,
            'holding_costs': self.holding_costs,
            'avg_inventory_level': np.mean(self._get_inventory_levels()),
            'avg_temperature': np.mean(self.temperatures),
            'min_inventory': np.min(self._get_inventory_levels()),
            'max_inventory': np.max(self._get_inventory_levels()),
            'total_blood_units': sum(len(units) for node in self.blood_inventory.values() 
                                   for units in node.values()),
            'pending_orders_total': int(np.sum(self.pending_orders)),
        }
    
    def render(self):
        """Render the current state (for debugging)"""
        print(f"Time: {self.current_time}/{self.time_horizon}")
        print(f"DateTime: {self.current_datetime}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Average Temperature: {np.mean(self.temperatures):.2f}°C")
        print(f"Average Inventory Level: {np.mean(self._get_inventory_levels()):.3f}")
        print(f"Shortages: {self.inventory_shortages}")
        print(f"Temperature Violations: {int(np.sum(self.temperature_violations))}")
        print(f"Successful Deliveries: {self.successful_deliveries}")
        print(f"Expired Units: {self.expired_units}")
        print(f"Temperature Excursions: {self.temperature_excursions}")
        print(f"Total Blood Units: {sum(len(units) for node in self.blood_inventory.values() for units in node.values())}")
        print(f"Pending Orders: {int(np.sum(self.pending_orders))}")
        print("-" * 50)
