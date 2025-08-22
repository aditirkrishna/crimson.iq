"""
Phase 1.1: Data Collection and Preprocessing
Handles blood inventory and sensor temperature data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BloodGroup(Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

@dataclass
class DataSchema:
    """Data schema for blood inventory and sensor temperature logs"""
    
    # Blood Inventory Schema
    inventory_fields = {
        'timestamp': 'datetime64[ns]',
        'pod_id': 'str',
        'blood_unit_id': 'str',
        'blood_group': 'category',
        'quantity': 'int32',
        'temperature': 'float32',
        'expiry_date': 'datetime64[ns]',
        'donation_date': 'datetime64[ns]',
        'status': 'category'  # available, reserved, expired, used
    }
    
    # Temperature Sensor Schema
    sensor_fields = {
        'timestamp': 'datetime64[ns]',
        'pod_id': 'str',
        'sensor_id': 'str',
        'temperature': 'float32',
        'humidity': 'float32',
        'alert_level': 'category'  # normal, warning, critical
    }
    
    # Demand Schema
    demand_fields = {
        'timestamp': 'datetime64[ns]',
        'pod_id': 'str',
        'blood_group': 'category',
        'demand_quantity': 'int32',
        'fulfilled_quantity': 'int32',
        'request_type': 'category'  # emergency, scheduled, routine
    }

class DataPreprocessor:
    """Handles data preprocessing for blood inventory management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.schema = DataSchema()
        
    def generate_sample_data(self, days: int = 30, num_pods: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate sample datasets for development and testing"""
        
        logger.info(f"Generating sample data for {days} days with {num_pods} pods")
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate inventory data
        inventory_data = self._generate_inventory_data(timestamps, num_pods)
        
        # Generate sensor data
        sensor_data = self._generate_sensor_data(timestamps, num_pods)
        
        # Generate demand data
        demand_data = self._generate_demand_data(timestamps, num_pods)
        
        return inventory_data, sensor_data, demand_data
    
    def _generate_inventory_data(self, timestamps: pd.DatetimeIndex, num_pods: int) -> pd.DataFrame:
        """Generate sample inventory data"""
        data = []
        blood_groups = [bg.value for bg in BloodGroup]
        
        for timestamp in timestamps:
            for pod_id in range(num_pods):
                for blood_group in blood_groups:
                    # Base inventory with some randomness
                    base_quantity = np.random.randint(10, 50)
                    
                    # Add time-based patterns (lower inventory at night)
                    hour = timestamp.hour
                    if 22 <= hour or hour <= 6:
                        base_quantity = int(base_quantity * 0.7)
                    
                    # Add some missing values (5% chance)
                    if np.random.random() < 0.05:
                        continue
                    
                    # Generate blood unit data
                    num_units = np.random.randint(1, 10)
                    for unit_id in range(num_units):
                        donation_date = timestamp - timedelta(days=np.random.randint(1, 30))
                        expiry_date = donation_date + timedelta(days=42)  # 42-day shelf life
                        
                        data.append({
                            'timestamp': timestamp,
                            'pod_id': f'pod_{pod_id}',
                            'blood_unit_id': f'unit_{pod_id}_{blood_group}_{unit_id}_{timestamp.strftime("%Y%m%d%H")}',
                            'blood_group': blood_group,
                            'quantity': np.random.randint(1, 5),
                            'temperature': np.random.normal(4.0, 0.5),
                            'expiry_date': expiry_date,
                            'donation_date': donation_date,
                            'status': np.random.choice(['available', 'reserved', 'used'], p=[0.8, 0.15, 0.05])
                        })
        
        df = pd.DataFrame(data)
        return self._apply_schema(df, self.schema.inventory_fields)
    
    def _generate_sensor_data(self, timestamps: pd.DatetimeIndex, num_pods: int) -> pd.DataFrame:
        """Generate sample sensor data"""
        data = []
        
        for timestamp in timestamps:
            for pod_id in range(num_pods):
                # Generate temperature with realistic patterns
                base_temp = 4.0
                
                # Add daily cycle (warmer during day)
                hour = timestamp.hour
                daily_cycle = 0.5 * np.sin(2 * np.pi * hour / 24)
                
                # Add weekly cycle (weekend effects)
                day_of_week = timestamp.weekday()
                weekly_cycle = 0.2 if day_of_week >= 5 else 0
                
                temperature = base_temp + daily_cycle + weekly_cycle + np.random.normal(0, 0.3)
                
                # Determine alert level
                if temperature < 2.0 or temperature > 6.0:
                    alert_level = 'critical'
                elif temperature < 2.5 or temperature > 5.5:
                    alert_level = 'warning'
                else:
                    alert_level = 'normal'
                
                data.append({
                    'timestamp': timestamp,
                    'pod_id': f'pod_{pod_id}',
                    'sensor_id': f'sensor_{pod_id}_01',
                    'temperature': temperature,
                    'humidity': np.random.normal(60, 10),
                    'alert_level': alert_level
                })
        
        df = pd.DataFrame(data)
        return self._apply_schema(df, self.schema.sensor_fields)
    
    def _generate_demand_data(self, timestamps: pd.DatetimeIndex, num_pods: int) -> pd.DataFrame:
        """Generate sample demand data"""
        data = []
        blood_groups = [bg.value for bg in BloodGroup]
        
        for timestamp in timestamps:
            for pod_id in range(num_pods):
                for blood_group in blood_groups:
                    # Skip some timestamps to simulate realistic demand patterns
                    if np.random.random() < 0.7:  # 30% chance of demand
                        continue
                    
                    # Base demand varies by blood group
                    base_demand = {
                        'O+': 0.4, 'A+': 0.3, 'B+': 0.2, 'AB+': 0.05,
                        'O-': 0.03, 'A-': 0.1, 'B-': 0.05, 'AB-': 0.02
                    }
                    
                    demand_prob = base_demand.get(blood_group, 0.1)
                    
                    # Add time-based patterns
                    hour = timestamp.hour
                    if 8 <= hour <= 18:  # Business hours
                        demand_prob *= 2.0
                    
                    if np.random.random() < demand_prob:
                        demand_quantity = np.random.randint(1, 8)
                        fulfilled_quantity = np.random.randint(0, demand_quantity + 1)
                        
                        # Emergency requests are more likely during certain hours
                        if 0 <= hour <= 6:
                            request_type = 'emergency'
                        else:
                            request_type = np.random.choice(['emergency', 'scheduled', 'routine'], 
                                                          p=[0.1, 0.3, 0.6])
                        
                        data.append({
                            'timestamp': timestamp,
                            'pod_id': f'pod_{pod_id}',
                            'blood_group': blood_group,
                            'demand_quantity': demand_quantity,
                            'fulfilled_quantity': fulfilled_quantity,
                            'request_type': request_type
                        })
        
        df = pd.DataFrame(data)
        return self._apply_schema(df, self.schema.demand_fields)
    
    def _apply_schema(self, df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        """Apply data types to DataFrame"""
        for column, dtype in schema.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert column {column} to {dtype}: {e}")
        return df
    
    def clean_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Clean and preprocess data"""
        logger.info(f"Cleaning {data_type} data with {len(df)} rows")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        if data_type == 'inventory':
            # For inventory, drop rows with missing critical fields
            critical_fields = ['timestamp', 'pod_id', 'blood_group', 'quantity']
            df = df.dropna(subset=critical_fields)
            
            # Fill missing temperatures with pod average
            if 'temperature' in df.columns:
                pod_avg_temp = df.groupby('pod_id')['temperature'].transform('mean')
                df['temperature'] = df['temperature'].fillna(pod_avg_temp)
        
        elif data_type == 'sensor':
            # For sensor data, interpolate missing temperatures
            if 'temperature' in df.columns:
                df = df.sort_values(['pod_id', 'timestamp'])
                df['temperature'] = df.groupby('pod_id')['temperature'].transform(
                    lambda x: x.interpolate(method='linear')
                )
        
        elif data_type == 'demand':
            # For demand data, fill missing fulfilled quantities with 0
            if 'fulfilled_quantity' in df.columns:
                df['fulfilled_quantity'] = df['fulfilled_quantity'].fillna(0)
        
        # Handle outliers
        df = self._handle_outliers(df, data_type)
        
        logger.info(f"Cleaned data has {len(df)} rows")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle outliers in the data"""
        
        if data_type == 'inventory':
            # Remove unrealistic quantities
            if 'quantity' in df.columns:
                df = df[df['quantity'] > 0]
                df = df[df['quantity'] <= 100]  # Max reasonable quantity
            
            # Remove unrealistic temperatures
            if 'temperature' in df.columns:
                df = df[df['temperature'] >= -10]  # Minimum reasonable temp
                df = df[df['temperature'] <= 20]   # Maximum reasonable temp
        
        elif data_type == 'sensor':
            # Remove unrealistic temperatures
            if 'temperature' in df.columns:
                df = df[df['temperature'] >= -20]
                df = df[df['temperature'] <= 30]
            
            # Remove unrealistic humidity
            if 'humidity' in df.columns:
                df = df[df['humidity'] >= 0]
                df = df[df['humidity'] <= 100]
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Engineer features for machine learning"""
        logger.info(f"Engineering features for {data_type} data")
        
        # Add time-based features
        df = self._add_time_features(df)
        
        if data_type == 'inventory':
            df = self._engineer_inventory_features(df)
        elif data_type == 'sensor':
            df = self._engineer_sensor_features(df)
        elif data_type == 'demand':
            df = self._engineer_demand_features(df)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        return df
    
    def _engineer_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specific to inventory data"""
        
        # Days until expiry
        if 'expiry_date' in df.columns and 'timestamp' in df.columns:
            df['days_until_expiry'] = (df['expiry_date'] - df['timestamp']).dt.days
        
        # Age of blood units
        if 'donation_date' in df.columns and 'timestamp' in df.columns:
            df['blood_age_days'] = (df['timestamp'] - df['donation_date']).dt.days
        
        # Rolling averages by pod and blood group
        if 'quantity' in df.columns:
            df = df.sort_values(['pod_id', 'blood_group', 'timestamp'])
            df['quantity_rolling_avg_24h'] = df.groupby(['pod_id', 'blood_group'])['quantity'].transform(
                lambda x: x.rolling(window=24, min_periods=1).mean()
            )
        
        return df
    
    def _engineer_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specific to sensor data"""
        
        # Temperature rolling statistics
        if 'temperature' in df.columns:
            df = df.sort_values(['pod_id', 'timestamp'])
            df['temp_rolling_avg_6h'] = df.groupby('pod_id')['temperature'].transform(
                lambda x: x.rolling(window=6, min_periods=1).mean()
            )
            df['temp_rolling_std_6h'] = df.groupby('pod_id')['temperature'].transform(
                lambda x: x.rolling(window=6, min_periods=1).std()
            )
            
            # Temperature change rate
            df['temp_change_rate'] = df.groupby('pod_id')['temperature'].diff()
        
        return df
    
    def _engineer_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specific to demand data"""
        
        # Demand fulfillment rate
        if 'demand_quantity' in df.columns and 'fulfilled_quantity' in df.columns:
            df['fulfillment_rate'] = df['fulfilled_quantity'] / df['demand_quantity'].replace(0, 1)
        
        # Rolling demand statistics
        if 'demand_quantity' in df.columns:
            df = df.sort_values(['pod_id', 'blood_group', 'timestamp'])
            df['demand_rolling_avg_24h'] = df.groupby(['pod_id', 'blood_group'])['demand_quantity'].transform(
                lambda x: x.rolling(window=24, min_periods=1).mean()
            )
        
        return df
    
    def resample_and_align(self, inventory_df: pd.DataFrame, sensor_df: pd.DataFrame, 
                          demand_df: pd.DataFrame, freq: str = 'H') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Resample and align time series data"""
        logger.info(f"Resampling data to {freq} frequency")
        
        # Resample inventory data (aggregate by pod, blood group, and time)
        inventory_resampled = self._resample_inventory(inventory_df, freq)
        
        # Resample sensor data (average by pod and time)
        sensor_resampled = self._resample_sensor(sensor_df, freq)
        
        # Resample demand data (sum by pod, blood group, and time)
        demand_resampled = self._resample_demand(demand_df, freq)
        
        return inventory_resampled, sensor_resampled, demand_resampled
    
    def _resample_inventory(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample inventory data"""
        if df.empty:
            return df
        
        # Group by pod, blood group, and resample time
        df_resampled = df.set_index('timestamp').groupby(['pod_id', 'blood_group']).resample(freq).agg({
            'quantity': 'sum',
            'temperature': 'mean',
            'status': lambda x: x.mode().iloc[0] if len(x) > 0 else 'available'
        }).reset_index()
        
        return df_resampled
    
    def _resample_sensor(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample sensor data"""
        if df.empty:
            return df
        
        # Group by pod and resample time
        df_resampled = df.set_index('timestamp').groupby('pod_id').resample(freq).agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'alert_level': lambda x: x.mode().iloc[0] if len(x) > 0 else 'normal'
        }).reset_index()
        
        return df_resampled
    
    def _resample_demand(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample demand data"""
        if df.empty:
            return df
        
        # Group by pod, blood group, and resample time
        df_resampled = df.set_index('timestamp').groupby(['pod_id', 'blood_group']).resample(freq).agg({
            'demand_quantity': 'sum',
            'fulfilled_quantity': 'sum',
            'request_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'routine'
        }).reset_index()
        
        return df_resampled
    
    def save_processed_data(self, inventory_df: pd.DataFrame, sensor_df: pd.DataFrame, 
                           demand_df: pd.DataFrame, output_dir: str = "processed_data"):
        """Save processed data to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to Parquet format for efficient storage
        inventory_df.to_parquet(f"{output_dir}/inventory_processed.parquet", index=False)
        sensor_df.to_parquet(f"{output_dir}/sensor_processed.parquet", index=False)
        demand_df.to_parquet(f"{output_dir}/demand_processed.parquet", index=False)
        
        # Save schema information
        schema_info = {
            'inventory_schema': dict(inventory_df.dtypes),
            'sensor_schema': dict(sensor_df.dtypes),
            'demand_schema': dict(demand_df.dtypes),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/schema_info.json", 'w') as f:
            json.dump(schema_info, f, indent=2, default=str)
        
        logger.info(f"Processed data saved to {output_dir}")
        
        return {
            'inventory_path': f"{output_dir}/inventory_processed.parquet",
            'sensor_path': f"{output_dir}/sensor_processed.parquet",
            'demand_path': f"{output_dir}/demand_processed.parquet",
            'schema_path': f"{output_dir}/schema_info.json"
        }

def main():
    """Main function for data preprocessing pipeline"""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Generate sample data
    logger.info("Generating sample data...")
    inventory_data, sensor_data, demand_data = preprocessor.generate_sample_data(days=30, num_pods=5)
    
    # Clean data
    logger.info("Cleaning data...")
    inventory_clean = preprocessor.clean_data(inventory_data, 'inventory')
    sensor_clean = preprocessor.clean_data(sensor_data, 'sensor')
    demand_clean = preprocessor.clean_data(demand_data, 'demand')
    
    # Engineer features
    logger.info("Engineering features...")
    inventory_features = preprocessor.engineer_features(inventory_clean, 'inventory')
    sensor_features = preprocessor.engineer_features(sensor_clean, 'sensor')
    demand_features = preprocessor.engineer_features(demand_clean, 'demand')
    
    # Resample and align
    logger.info("Resampling and aligning data...")
    inventory_resampled, sensor_resampled, demand_resampled = preprocessor.resample_and_align(
        inventory_features, sensor_features, demand_features, freq='H'
    )
    
    # Save processed data
    logger.info("Saving processed data...")
    output_paths = preprocessor.save_processed_data(
        inventory_resampled, sensor_resampled, demand_resampled
    )
    
    logger.info("Data preprocessing pipeline completed successfully!")
    logger.info(f"Output files: {output_paths}")
    
    return output_paths

if __name__ == "__main__":
    main()
