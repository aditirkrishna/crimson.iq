# Data loaders and preprocessors

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BloodInventoryDataProcessor:
    """Data processor for blood inventory time series data"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.scaler = None
        self.feature_columns = []
        self.target_column = None
        
    def _get_default_config(self) -> Dict:
        """Default configuration for data processing"""
        return {
            'target_column': 'demand',
            'time_column': 'timestamp',
            'group_columns': ['blood_group', 'pod_id'],
            'rolling_windows': [3, 7, 14, 30],
            'lag_features': [1, 2, 3, 7, 14],
            'min_data_points': 50,
            'max_missing_ratio': 0.3,
            'outlier_threshold': 3.0
        }
    
    def load_sample_data(self, data_path: str) -> pd.DataFrame:
        """Load and parse sample dataset from the provided files"""
        try:
            # Load the sample dataset
            df = pd.read_csv(data_path, sep='\t', skiprows=1)
            
            # Parse the data structure
            if 'Demand' in df.columns:
                # This is demand data
                df = self._parse_demand_data(df)
            else:
                # This is vehicle/capacity data
                df = self._parse_vehicle_data(df)
                
            logger.info(f"Loaded {len(df)} records from {data_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            # Generate synthetic data as fallback
            return self._generate_synthetic_data()
    
    def _parse_demand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse demand dataset structure"""
        # Rename columns for consistency
        column_mapping = {
            'Cust No': 'customer_id',
            'Longitude': 'longitude',
            'Latitude': 'latitude', 
            'Demand': 'demand',
            'Ready Time': 'ready_time',
            'Due Time': 'due_time',
            'Service Time': 'service_time'
        }
        df = df.rename(columns=column_mapping)
        
        # Add synthetic timestamps and blood inventory features
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)),
            periods=len(df),
            freq='H'
        )
        
        # Add blood group and pod information
        blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        df['blood_group'] = np.random.choice(blood_groups, len(df))
        df['pod_id'] = np.random.choice(['GZ_POD_1', 'GZ_POD_2', 'SZ_POD_1'], len(df))
        
        # Add temperature and humidity data
        df['temperature_celsius'] = np.random.normal(4.0, 1.0, len(df))  # Refrigeration temp
        df['humidity_percent'] = np.random.normal(60.0, 10.0, len(df))
        
        # Add volume information
        df['volume_ml'] = np.random.choice([450, 500, 550], len(df))
        
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic blood inventory data for testing"""
        logger.info("Generating synthetic blood inventory data")
        
        # Generate 6 months of hourly data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=180),
            end=datetime.now(),
            freq='H'
        )
        
        n_samples = len(dates)
        
        # Base demand patterns
        base_demand = np.random.poisson(15, n_samples)  # Base demand ~15 units/hour
        
        # Add weekly seasonality
        weekly_pattern = np.sin(2 * np.pi * dates.dayofweek / 7) * 5
        base_demand += weekly_pattern
        
        # Add daily seasonality (higher demand during day)
        daily_pattern = np.sin(2 * np.pi * dates.hour / 24) * 3
        base_demand += daily_pattern
        
        # Add some trend
        trend = np.linspace(0, 2, n_samples)
        base_demand += trend
        
        # Add noise
        noise = np.random.normal(0, 2, n_samples)
        base_demand += noise
        
        # Ensure non-negative
        base_demand = np.maximum(0, base_demand)
        
        data = {
            'timestamp': dates,
            'demand': base_demand,
            'blood_group': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_samples),
            'pod_id': np.random.choice(['GZ_POD_1', 'GZ_POD_2', 'SZ_POD_1'], n_samples),
            'temperature_celsius': np.random.normal(4.0, 1.0, n_samples),
            'humidity_percent': np.random.normal(60.0, 10.0, n_samples),
            'volume_ml': np.random.choice([450, 500, 550], n_samples),
            'customer_id': np.random.randint(1, 1000, n_samples),
            'longitude': np.random.uniform(113.0, 114.0, n_samples),
            'latitude': np.random.uniform(22.0, 24.0, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        logger.info("Starting data cleaning process")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        df_clean = self._handle_missing_values(df_clean)
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        
        # 2. Remove outliers
        outliers_before = len(df_clean)
        df_clean = self._remove_outliers(df_clean)
        outliers_after = len(df_clean)
        logger.info(f"Removed {outliers_before - outliers_after} outliers")
        
        # 3. Validate data types and ranges
        df_clean = self._validate_data_types(df_clean)
        
        # 4. Sort by timestamp
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df_clean)} records")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns, use forward fill then backward fill
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(method='ffill').fillna(method='bfill')
        
        # For categorical columns, use mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        # For timestamp, drop rows with missing timestamps
        df = df.dropna(subset=['timestamp'])
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col == 'customer_id':  # Skip ID columns
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.config['outlier_threshold'] * IQR
            upper_bound = Q3 + self.config['outlier_threshold'] * IQR
            
            # Only remove extreme outliers, keep moderate ones
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure demand is numeric and non-negative
        df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
        df['demand'] = df['demand'].clip(lower=0)
        
        # Ensure temperature is within reasonable range
        df['temperature_celsius'] = pd.to_numeric(df['temperature_celsius'], errors='coerce')
        df['temperature_celsius'] = df['temperature_celsius'].clip(-50, 50)
        
        # Ensure humidity is within range
        if 'humidity_percent' in df.columns:
            df['humidity_percent'] = pd.to_numeric(df['humidity_percent'], errors='coerce')
            df['humidity_percent'] = df['humidity_percent'].clip(0, 100)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for time series forecasting"""
        logger.info("Engineering features for time series forecasting")
        
        df_features = df.copy()
        
        # 1. Time-based features
        df_features = self._add_time_features(df_features)
        
        # 2. Rolling statistics
        df_features = self._add_rolling_features(df_features)
        
        # 3. Lag features
        df_features = self._add_lag_features(df_features)
        
        # 4. Blood-specific features
        df_features = self._add_blood_features(df_features)
        
        # 5. Temperature and environmental features
        df_features = self._add_environmental_features(df_features)
        
        logger.info(f"Feature engineering completed. Final features: {len(df_features.columns)}")
        return df_features
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        # Group by blood group and pod for rolling calculations
        group_cols = self.config['group_columns']
        
        for window in self.config['rolling_windows']:
            # Rolling demand statistics
            df[f'demand_rolling_mean_{window}'] = df.groupby(group_cols)['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'demand_rolling_std_{window}'] = df.groupby(group_cols)['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'demand_rolling_max_{window}'] = df.groupby(group_cols)['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Rolling temperature statistics
            if 'temperature_celsius' in df.columns:
                df[f'temp_rolling_mean_{window}'] = df.groupby(group_cols)['temperature_celsius'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'temp_rolling_std_{window}'] = df.groupby(group_cols)['temperature_celsius'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features"""
        group_cols = self.config['group_columns']
        
        for lag in self.config['lag_features']:
            df[f'demand_lag_{lag}'] = df.groupby(group_cols)['demand'].shift(lag)
            
            if 'temperature_celsius' in df.columns:
                df[f'temp_lag_{lag}'] = df.groupby(group_cols)['temperature_celsius'].shift(lag)
        
        return df
    
    def _add_blood_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add blood group specific features"""
        # Blood group encoding
        blood_group_encoding = {
            'A+': [1, 0, 0, 0, 0, 0, 0, 0],
            'A-': [0, 1, 0, 0, 0, 0, 0, 0],
            'B+': [0, 0, 1, 0, 0, 0, 0, 0],
            'B-': [0, 0, 0, 1, 0, 0, 0, 0],
            'AB+': [0, 0, 0, 0, 1, 0, 0, 0],
            'AB-': [0, 0, 0, 0, 0, 1, 0, 0],
            'O+': [0, 0, 0, 0, 0, 0, 1, 0],
            'O-': [0, 0, 0, 0, 0, 0, 0, 1]
        }
        
        for i, (group, encoding) in enumerate(blood_group_encoding.items()):
            df[f'blood_group_{i}'] = (df['blood_group'] == group).astype(int)
        
        # Pod encoding
        unique_pods = df['pod_id'].unique()
        for i, pod in enumerate(unique_pods):
            df[f'pod_{i}'] = (df['pod_id'] == pod).astype(int)
        
        return df
    
    def _add_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add environmental and temperature features"""
        if 'temperature_celsius' in df.columns:
            # Temperature deviation from optimal (4°C for blood storage)
            df['temp_deviation'] = abs(df['temperature_celsius'] - 4.0)
            df['temp_status'] = pd.cut(
                df['temperature_celsius'],
                bins=[-np.inf, 2, 6, np.inf],
                labels=['cold', 'normal', 'warm']
            )
            
            # Temperature status encoding
            df['temp_cold'] = (df['temp_status'] == 'cold').astype(int)
            df['temp_normal'] = (df['temp_status'] == 'normal').astype(int)
            df['temp_warm'] = (df['temp_status'] == 'warm').astype(int)
        
        if 'humidity_percent' in df.columns:
            # Humidity features
            df['humidity_deviation'] = abs(df['humidity_percent'] - 60.0)
            df['humidity_status'] = pd.cut(
                df['humidity_percent'],
                bins=[-np.inf, 40, 80, np.inf],
                labels=['low', 'normal', 'high']
            )
            
            # Humidity status encoding
            df['humidity_low'] = (df['humidity_status'] == 'low').astype(int)
            df['humidity_normal'] = (df['humidity_status'] == 'normal').astype(int)
            df['humidity_high'] = (df['humidity_status'] == 'high').astype(int)
        
        return df
    
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                target_column: str = 'demand',
                                sequence_length: int = 24,
                                forecast_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for time series forecasting"""
        logger.info(f"Preparing time series data with sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'customer_id', 'longitude', 'latitude', 'ready_time', 'due_time', 'service_time']
        feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_column]
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(df_sorted) - sequence_length - forecast_horizon + 1):
            # Input sequence
            X.append(df_sorted[feature_cols].iloc[i:i+sequence_length].values)
            # Target sequence
            y.append(df_sorted[target_column].iloc[i+sequence_length:i+sequence_length+forecast_horizon].values)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/validation/test
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"Time series data prepared: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        logger.info(f"Processed data saved to {output_path}")
        
        # Save metadata
        metadata = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Example usage of the data processor"""
    # Initialize processor
    processor = BloodInventoryDataProcessor()
    
    # Load sample data (using synthetic data for now)
    df = processor.load_sample_data("")
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Engineer features
    df_features = processor.engineer_features(df_clean)
    
    # Prepare time series data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = processor.prepare_time_series_data(
        df_features, sequence_length=24, forecast_horizon=7
    )
    
    # Save processed data
    processor.save_processed_data(df_features, "processed_blood_inventory_data.parquet")
    
    print("Data processing completed successfully!")
    print(f"Final dataset shape: {df_features.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")

if __name__ == "__main__":
    main()
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BloodInventoryDataProcessor:
    """Data processor for blood inventory time series data"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.scaler = None
        self.feature_columns = []
        self.target_column = None
        
    def _get_default_config(self) -> Dict:
        """Default configuration for data processing"""
        return {
            'target_column': 'demand',
            'time_column': 'timestamp',
            'group_columns': ['blood_group', 'pod_id'],
            'feature_columns': [
                'temperature_avg', 'temperature_std', 'humidity_avg',
                'day_of_week', 'month', 'hour', 'is_weekend',
                'days_since_collection', 'volume_ml'
            ],
            'rolling_windows': [3, 7, 14, 30],
            'lag_features': [1, 2, 3, 7, 14],
            'min_data_points': 50,
            'max_missing_ratio': 0.3,
            'outlier_threshold': 3.0
        }
    
    def load_sample_data(self, data_path: str) -> pd.DataFrame:
        """Load and parse sample dataset from the provided files"""
        try:
            # Load the sample dataset
            df = pd.read_csv(data_path, sep='\t', skiprows=1)
            
            # Parse the data structure
            if 'Demand' in df.columns:
                # This is demand data
                df = self._parse_demand_data(df)
            else:
                # This is vehicle/capacity data
                df = self._parse_vehicle_data(df)
                
            logger.info(f"Loaded {len(df)} records from {data_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            # Generate synthetic data as fallback
            return self._generate_synthetic_data()
    
    def _parse_demand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse demand dataset structure"""
        # Rename columns for consistency
        column_mapping = {
            'Cust No': 'customer_id',
            'Longitude': 'longitude',
            'Latitude': 'latitude', 
            'Demand': 'demand',
            'Ready Time': 'ready_time',
            'Due Time': 'due_time',
            'Service Time': 'service_time'
        }
        df = df.rename(columns=column_mapping)
        
        # Add synthetic timestamps and blood inventory features
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)),
            periods=len(df),
            freq='H'
        )
        
        # Add blood group and pod information
        blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        df['blood_group'] = np.random.choice(blood_groups, len(df))
        df['pod_id'] = np.random.choice(['GZ_POD_1', 'GZ_POD_2', 'SZ_POD_1'], len(df))
        
        # Add temperature and humidity data
        df['temperature_celsius'] = np.random.normal(4.0, 1.0, len(df))  # Refrigeration temp
        df['humidity_percent'] = np.random.normal(60.0, 10.0, len(df))
        
        # Add volume information
        df['volume_ml'] = np.random.choice([450, 500, 550], len(df))
        
        return df
    
    def _parse_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse vehicle/capacity dataset structure"""
        # This is vehicle capacity data, convert to demand-like format
        df = df.rename(columns={'VEHICLE NUMBER': 'vehicle_id', 'CAPACITY': 'capacity'})
        
        # Generate synthetic demand data based on capacity
        df['demand'] = np.random.poisson(df['capacity'] * 0.3)
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)),
            periods=len(df),
            freq='H'
        )
        
        # Add other required columns
        df['blood_group'] = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], len(df))
        df['pod_id'] = np.random.choice(['GZ_POD_1', 'GZ_POD_2', 'SZ_POD_1'], len(df))
        df['temperature_celsius'] = np.random.normal(4.0, 1.0, len(df))
        df['humidity_percent'] = np.random.normal(60.0, 10.0, len(df))
        df['volume_ml'] = np.random.choice([450, 500, 550], len(df))
        
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic blood inventory data for testing"""
        logger.info("Generating synthetic blood inventory data")
        
        # Generate 6 months of hourly data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=180),
            end=datetime.now(),
            freq='H'
        )
        
        n_samples = len(dates)
        
        # Base demand patterns
        base_demand = np.random.poisson(15, n_samples)  # Base demand ~15 units/hour
        
        # Add weekly seasonality
        weekly_pattern = np.sin(2 * np.pi * dates.dayofweek / 7) * 5
        base_demand += weekly_pattern
        
        # Add daily seasonality (higher demand during day)
        daily_pattern = np.sin(2 * np.pi * dates.hour / 24) * 3
        base_demand += daily_pattern
        
        # Add some trend
        trend = np.linspace(0, 2, n_samples)
        base_demand += trend
        
        # Add noise
        noise = np.random.normal(0, 2, n_samples)
        base_demand += noise
        
        # Ensure non-negative
        base_demand = np.maximum(0, base_demand)
        
        data = {
            'timestamp': dates,
            'demand': base_demand,
            'blood_group': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_samples),
            'pod_id': np.random.choice(['GZ_POD_1', 'GZ_POD_2', 'SZ_POD_1'], n_samples),
            'temperature_celsius': np.random.normal(4.0, 1.0, n_samples),
            'humidity_percent': np.random.normal(60.0, 10.0, n_samples),
            'volume_ml': np.random.choice([450, 500, 550], n_samples),
            'customer_id': np.random.randint(1, 1000, n_samples),
            'longitude': np.random.uniform(113.0, 114.0, n_samples),
            'latitude': np.random.uniform(22.0, 24.0, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        logger.info("Starting data cleaning process")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        df_clean = self._handle_missing_values(df_clean)
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        
        # 2. Remove outliers
        outliers_before = len(df_clean)
        df_clean = self._remove_outliers(df_clean)
        outliers_after = len(df_clean)
        logger.info(f"Removed {outliers_before - outliers_after} outliers")
        
        # 3. Validate data types and ranges
        df_clean = self._validate_data_types(df_clean)
        
        # 4. Sort by timestamp
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Final dataset: {len(df_clean)} records")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns, use forward fill then backward fill
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(method='ffill').fillna(method='bfill')
        
        # For categorical columns, use mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        # For timestamp, drop rows with missing timestamps
        df = df.dropna(subset=['timestamp'])
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col == 'customer_id':  # Skip ID columns
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.config['outlier_threshold'] * IQR
            upper_bound = Q3 + self.config['outlier_threshold'] * IQR
            
            # Only remove extreme outliers, keep moderate ones
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure demand is numeric and non-negative
        df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
        df['demand'] = df['demand'].clip(lower=0)
        
        # Ensure temperature is within reasonable range
        df['temperature_celsius'] = pd.to_numeric(df['temperature_celsius'], errors='coerce')
        df['temperature_celsius'] = df['temperature_celsius'].clip(-50, 50)
        
        # Ensure humidity is within range
        if 'humidity_percent' in df.columns:
            df['humidity_percent'] = pd.to_numeric(df['humidity_percent'], errors='coerce')
            df['humidity_percent'] = df['humidity_percent'].clip(0, 100)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for time series forecasting"""
        logger.info("Engineering features for time series forecasting")
        
        df_features = df.copy()
        
        # 1. Time-based features
        df_features = self._add_time_features(df_features)
        
        # 2. Rolling statistics
        df_features = self._add_rolling_features(df_features)
        
        # 3. Lag features
        df_features = self._add_lag_features(df_features)
        
        # 4. Blood-specific features
        df_features = self._add_blood_features(df_features)
        
        # 5. Temperature and environmental features
        df_features = self._add_environmental_features(df_features)
        
        logger.info(f"Feature engineering completed. Final features: {len(df_features.columns)}")
        return df_features
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        # Group by blood group and pod for rolling calculations
        group_cols = self.config['group_columns']
        
        for window in self.config['rolling_windows']:
            # Rolling demand statistics
            df[f'demand_rolling_mean_{window}'] = df.groupby(group_cols)['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'demand_rolling_std_{window}'] = df.groupby(group_cols)['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'demand_rolling_max_{window}'] = df.groupby(group_cols)['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Rolling temperature statistics
            if 'temperature_celsius' in df.columns:
                df[f'temp_rolling_mean_{window}'] = df.groupby(group_cols)['temperature_celsius'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'temp_rolling_std_{window}'] = df.groupby(group_cols)['temperature_celsius'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features"""
        group_cols = self.config['group_columns']
        
        for lag in self.config['lag_features']:
            df[f'demand_lag_{lag}'] = df.groupby(group_cols)['demand'].shift(lag)
            
            if 'temperature_celsius' in df.columns:
                df[f'temp_lag_{lag}'] = df.groupby(group_cols)['temperature_celsius'].shift(lag)
        
        return df
    
    def _add_blood_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add blood group specific features"""
        # Blood group encoding
        blood_group_encoding = {
            'A+': [1, 0, 0, 0, 0, 0, 0, 0],
            'A-': [0, 1, 0, 0, 0, 0, 0, 0],
            'B+': [0, 0, 1, 0, 0, 0, 0, 0],
            'B-': [0, 0, 0, 1, 0, 0, 0, 0],
            'AB+': [0, 0, 0, 0, 1, 0, 0, 0],
            'AB-': [0, 0, 0, 0, 0, 1, 0, 0],
            'O+': [0, 0, 0, 0, 0, 0, 1, 0],
            'O-': [0, 0, 0, 0, 0, 0, 0, 1]
        }
        
        for i, (group, encoding) in enumerate(blood_group_encoding.items()):
            df[f'blood_group_{i}'] = (df['blood_group'] == group).astype(int)
        
        # Pod encoding
        unique_pods = df['pod_id'].unique()
        for i, pod in enumerate(unique_pods):
            df[f'pod_{i}'] = (df['pod_id'] == pod).astype(int)
        
        return df
    
    def _add_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add environmental and temperature features"""
        if 'temperature_celsius' in df.columns:
            # Temperature deviation from optimal (4°C for blood storage)
            df['temp_deviation'] = abs(df['temperature_celsius'] - 4.0)
            df['temp_status'] = pd.cut(
                df['temperature_celsius'],
                bins=[-np.inf, 2, 6, np.inf],
                labels=['cold', 'normal', 'warm']
            )
            
            # Temperature status encoding
            df['temp_cold'] = (df['temp_status'] == 'cold').astype(int)
            df['temp_normal'] = (df['temp_status'] == 'normal').astype(int)
            df['temp_warm'] = (df['temp_status'] == 'warm').astype(int)
        
        if 'humidity_percent' in df.columns:
            # Humidity features
            df['humidity_deviation'] = abs(df['humidity_percent'] - 60.0)
            df['humidity_status'] = pd.cut(
                df['humidity_percent'],
                bins=[-np.inf, 40, 80, np.inf],
                labels=['low', 'normal', 'high']
            )
            
            # Humidity status encoding
            df['humidity_low'] = (df['humidity_status'] == 'low').astype(int)
            df['humidity_normal'] = (df['humidity_status'] == 'normal').astype(int)
            df['humidity_high'] = (df['humidity_status'] == 'high').astype(int)
        
        return df
    
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                target_column: str = 'demand',
                                sequence_length: int = 24,
                                forecast_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for time series forecasting"""
        logger.info(f"Preparing time series data with sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'customer_id', 'longitude', 'latitude', 'ready_time', 'due_time', 'service_time']
        feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_column]
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(df_sorted) - sequence_length - forecast_horizon + 1):
            # Input sequence
            X.append(df_sorted[feature_cols].iloc[i:i+sequence_length].values)
            # Target sequence
            y.append(df_sorted[target_column].iloc[i+sequence_length:i+sequence_length+forecast_horizon].values)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/validation/test
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"Time series data prepared: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        logger.info(f"Processed data saved to {output_path}")
        
        # Save metadata
        metadata = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Example usage of the data processor"""
    # Initialize processor
    processor = BloodInventoryDataProcessor()
    
    # Load sample data (using synthetic data for now)
    df = processor.load_sample_data("")
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Engineer features
    df_features = processor.engineer_features(df_clean)
    
    # Prepare time series data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = processor.prepare_time_series_data(
        df_features, sequence_length=24, forecast_horizon=7
    )
    
    # Save processed data
    processor.save_processed_data(df_features, "processed_blood_inventory_data.parquet")
    
    print("Data processing completed successfully!")
    print(f"Final dataset shape: {df_features.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")

if __name__ == "__main__":
    main()
