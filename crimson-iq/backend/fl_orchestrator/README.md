# Crimson IQ Federated Learning Orchestrator

This module implements a comprehensive federated learning system for the Crimson IQ blood inventory management platform, enabling collaborative model training across multiple decentralized nodes (hospitals, blood banks) while preserving data privacy.

## Features

- **Privacy-Preserving Training**: Implements differential privacy, encryption, and secure aggregation
- **Multiple Aggregation Strategies**: FedAvg, FedProx, FedNova, and robust aggregation with outlier detection
- **Healthcare-Optimized**: Designed specifically for blood inventory management use cases
- **Scalable Architecture**: Supports multiple clients with health monitoring and trust scoring
- **Model Compatibility**: Works with survival analysis, time series forecasting, and reinforcement learning models

## Architecture

```
fl_orchestrator/
├── main.py              # Main FL server with Flower integration
├── config.py            # Configuration management
├── security.py          # Privacy and security utilities
├── client_manager.py    # Client registration and health tracking
├── aggregator.py        # Aggregation strategies
└── README.md           # This file

ml_models/federated_training/
├── train_fed.py         # Training orchestration scripts
├── utils.py            # FL workflow utilities
└── models.py           # Federated-compatible model definitions

inventory/
└── federated_client.py  # FL client for inventory system
```

## Quick Start

### 1. Start the Federated Learning Server

```bash
# Start with default configuration
python backend/fl_orchestrator/main.py

# Start with custom parameters
python backend/fl_orchestrator/main.py \
    --host 0.0.0.0 \
    --port 8080 \
    --rounds 15 \
    --min-clients 2 \
    --strategy fedavg
```

### 2. Start Federated Learning Clients

```bash
# Start a hospital client
python backend/inventory/federated_client.py \
    --server localhost:8080 \
    --node-name "General Hospital Alpha" \
    --node-type hospital \
    --model-type survival_analysis

# Start a blood bank client
python backend/inventory/federated_client.py \
    --server localhost:8080 \
    --node-name "Central Blood Bank" \
    --node-type blood_bank \
    --model-type survival_analysis
```

### 3. Run Federated Training Jobs

```bash
# Run survival analysis training
python backend/ml_models/federated_training/train_fed.py \
    --model-type survival_analysis \
    --rounds 15 \
    --clients 3 \
    --strategy fedavg \
    --enable-dp
```

## Configuration

### Environment Variables

```bash
# Server Configuration
export FL_SERVER_HOST="0.0.0.0"
export FL_SERVER_PORT="8080"
export FL_NUM_ROUNDS="15"
export FL_MIN_FIT_CLIENTS="2"
export FL_MIN_EVALUATE_CLIENTS="2"
export FL_MIN_AVAILABLE_CLIENTS="2"

# Privacy Settings
export FL_ENABLE_DP="true"
export FL_DP_EPSILON="1.0"
export FL_DP_DELTA="1e-5"
export FL_DP_NOISE_MULTIPLIER="1.1"

# Model Settings
export FL_MODEL_TYPE="survival_analysis"
export FL_AGGREGATION_STRATEGY="fedavg"

# Security Settings
export FL_ENABLE_ENCRYPTION="true"
export FL_LOG_LEVEL="INFO"
```

### Configuration File

Create a `fl_config.json` file:

```json
{
    "server_host": "0.0.0.0",
    "server_port": 8080,
    "num_rounds": 15,
    "min_fit_clients": 2,
    "min_evaluate_clients": 2,
    "min_available_clients": 2,
    "enable_differential_privacy": true,
    "dp_epsilon": 1.0,
    "dp_delta": 1e-5,
    "dp_noise_multiplier": 1.1,
    "model_type": "survival_analysis",
    "aggregation_strategy": "fedavg",
    "enable_encryption": true,
    "log_level": "INFO",
    "max_concurrent_clients": 10,
    "batch_size": 32,
    "learning_rate": 0.01
}
```

## Aggregation Strategies

### 1. Federated Averaging (FedAvg)
- **Use Case**: Standard federated learning scenarios
- **Advantages**: Simple, widely used, good convergence
- **Configuration**: `--strategy fedavg`

### 2. Federated Proximal (FedProx)
- **Use Case**: Heterogeneous data distributions
- **Advantages**: Better convergence with non-IID data
- **Configuration**: `--strategy fedprox`

### 3. Federated Normalized Averaging (FedNova)
- **Use Case**: Clients with different local epochs
- **Advantages**: Handles varying client participation
- **Configuration**: `--strategy fednova`

### 4. Robust Aggregation
- **Use Case**: Byzantine-robust federated learning
- **Advantages**: Resistant to malicious clients
- **Configuration**: `--strategy robust`

## Privacy Features

### Differential Privacy
- **Epsilon**: Privacy budget (lower = more private)
- **Delta**: Failure probability
- **Noise Multiplier**: Controls noise addition

```python
# Example configuration
config = FLConfig(
    enable_differential_privacy=True,
    dp_epsilon=1.0,        # Privacy budget
    dp_delta=1e-5,         # Failure probability
    dp_noise_multiplier=1.1 # Noise scale
)
```

### Encryption
- **AES-256**: Model weight encryption
- **Secure Communication**: Encrypted client-server communication
- **Key Management**: Automatic key generation and rotation

## API Endpoints

### Server Status
```python
# Get server status and metrics
status = orchestrator.get_server_status()
print(f"Server running: {status['is_running']}")
print(f"Active clients: {status['client_stats']['active_clients']}")
```

### Client Management
```python
# Register a new client
client_id = orchestrator.register_client(
    node_name="New Hospital",
    node_type="hospital",
    ip_address="192.168.1.100",
    port=8081,
    capabilities=["survival_analysis", "time_series"],
    model_types=["survival_analysis"],
    data_size=5000
)

# Get client statistics
stats = orchestrator.client_manager.get_client_stats()
```

### Training Control
```python
# Set aggregation strategy
orchestrator.set_aggregation_strategy("fedprox")

# Export client data
client_data = orchestrator.export_client_data()

# Import client data
orchestrator.import_client_data(client_data_json)
```

## Model Types

### 1. Survival Analysis
- **Purpose**: Blood unit viability prediction
- **Features**: Blood group, temperature, quantity, time-based features
- **Target**: Survival duration and event prediction

### 2. Time Series Forecasting
- **Purpose**: Demand and supply forecasting
- **Features**: Historical demand, supply, temperature, seasonal patterns
- **Target**: Future demand prediction

### 3. Reinforcement Learning
- **Purpose**: Optimal inventory management policies
- **Features**: State representation of inventory system
- **Target**: Action selection for inventory decisions

## Monitoring and Metrics

### Training Metrics
- **Loss**: Model performance across rounds
- **Accuracy**: Prediction accuracy
- **Participation Rate**: Client engagement
- **Response Time**: Communication efficiency

### Privacy Metrics
- **Epsilon**: Privacy budget consumption
- **Membership Inference Risk**: Privacy leakage assessment
- **Cumulative Privacy**: Total privacy cost

### Communication Metrics
- **Total Messages**: Communication overhead
- **Data Transferred**: Bandwidth usage
- **Communication Efficiency**: Client-to-message ratio

## Security Considerations

### Threat Model
- **Honest-but-curious**: Server and clients follow protocol but may try to infer information
- **Byzantine**: Some clients may be malicious
- **Data Poisoning**: Clients may submit incorrect updates

### Mitigation Strategies
- **Differential Privacy**: Protects against membership inference
- **Robust Aggregation**: Detects and removes outliers
- **Encryption**: Protects model weights in transit
- **Trust Scoring**: Tracks client reliability

## Troubleshooting

### Common Issues

1. **Client Connection Failures**
   ```bash
   # Check server is running
   netstat -tulpn | grep 8080
   
   # Check firewall settings
   sudo ufw allow 8080
   ```

2. **Insufficient Clients**
   ```bash
   # Reduce minimum client requirements
   python main.py --min-clients 1
   ```

3. **Training Convergence Issues**
   ```bash
   # Try different aggregation strategy
   python main.py --strategy fedprox
   
   # Adjust learning rate
   export FL_LEARNING_RATE="0.001"
   ```

### Logging
- **Server Logs**: `fl_orchestrator.log`
- **Client Logs**: Check individual client output
- **Training Logs**: Saved in federated training directory

## Performance Optimization

### Server Optimization
- **Connection Pooling**: Reuse connections
- **Async Processing**: Non-blocking operations
- **Memory Management**: Efficient weight storage

### Client Optimization
- **Local Caching**: Cache model weights
- **Batch Processing**: Efficient local training
- **Compression**: Reduce communication overhead

## Integration with Crimson IQ

### Backend Integration
```python
from fl_orchestrator.main import FederatedLearningOrchestrator
from fl_orchestrator.config import FLConfig

# Initialize orchestrator
config = FLConfig(model_type="survival_analysis")
orchestrator = FederatedLearningOrchestrator(config)

# Start server
orchestrator.start_server()
```

### Model Integration
```python
from ml_models.survival_analysis.models import CoxProportionalHazardsModel
from inventory.federated_client import CrimsonFLClient

# Create federated client
client = CrimsonFLClient(
    node_name="Hospital Alpha",
    node_type="hospital",
    model_type="survival_analysis"
)
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure privacy and security compliance

## License

This module is part of the Crimson IQ project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Consult the API documentation
4. Contact the development team
