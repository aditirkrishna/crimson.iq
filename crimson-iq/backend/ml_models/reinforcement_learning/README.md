# Crimson IQ Reinforcement Learning System

This module implements a comprehensive reinforcement learning system for blood inventory and cold chain optimization, enabling intelligent decision-making for inventory management across multiple healthcare facilities.

## Features

- **Enhanced Environment**: Multi-node simulation with age/expiry tracking, temperature monitoring, and supply chain dynamics
- **Multiple RL Algorithms**: DQN, PPO, and A2C with hyperparameter optimization
- **Intelligent Decision Support**: Real-time inventory decisions with confidence scoring
- **Ensemble Methods**: Multi-algorithm decision aggregation for improved reliability
- **Fallback Mechanisms**: Robust fallback policies when RL models have low confidence
- **Comprehensive Monitoring**: Performance tracking and system health monitoring
- **Baseline Comparison**: Evaluation against rule-based and heuristic policies

## Architecture

```
reinforcement_learning/
├── env.py                 # Enhanced RL environment with age tracking
├── train.py              # Training manager with multiple algorithms
├── inference.py          # Inference engine and decision support
├── test_rl_system.py     # Comprehensive test and demo script
└── README.md            # This file
```

## Quick Start

### 1. Test the RL System

```bash
cd crimson-iq/backend/ml_models/reinforcement_learning
python test_rl_system.py
```

### 2. Train RL Models

```bash
# Train all algorithms
python train.py --algorithm all --timesteps 100000

# Train specific algorithm with hyperparameter optimization
python train.py --algorithm ppo --timesteps 50000 --optimize --n-trials 30

# Train with custom configuration
python train.py --algorithm dqn --timesteps 200000 --eval-episodes 200
```

### 3. Use Inference for Decision Support

```python
from inference import create_rl_decision_support

# Create decision support system
decision_support = create_rl_decision_support("./models")

# Get optimal decision
current_state = {
    "inventory_levels": {
        "0": {"A+": 0.25, "A-": 0.08, "B+": 0.15, "B-": 0.03, 
              "AB+": 0.04, "AB-": 0.01, "O+": 0.3, "O-": 0.05},
        "1": {"A+": 0.2, "A-": 0.06, "B+": 0.12, "B-": 0.02, 
              "AB+": 0.03, "AB-": 0.01, "O+": 0.25, "O-": 0.04}
    },
    "temperatures": [4.2, 3.9, 4.1, 4.0, 4.3, 4.0],
    "shortages": 3,
    "temperature_violations": 1
}

# Get ensemble decision
decision = decision_support.get_optimal_decision(current_state, ensemble=True)
print(f"Action: {decision['action']}")
print(f"Confidence: {decision['confidence']:.3f}")
print(f"Reasoning: {decision['reasoning']}")
```

## Environment Features

### State Space Design

The environment includes comprehensive state representation:

- **Inventory Levels**: By blood group and location (hospitals, blood banks, pods)
- **Age Distribution**: 7-day bins tracking blood unit age (0-7, 8-14, 15-21, 22-28, 29-35, 36-42, 43+ days)
- **Temperature Monitoring**: Current temperatures and violation counts
- **Supply Chain**: Pending orders and delivery times
- **Demand Patterns**: Current demand and recent trends
- **Time Features**: Hour, day, week, month, and seasonal patterns

### Action Space

11 discrete actions for inventory management:

1. **MAINTAIN_TEMP**: Maintain current temperature settings
2. **ADJUST_TEMP_UP**: Increase temperature to prevent freezing
3. **ADJUST_TEMP_DOWN**: Decrease temperature to prevent spoilage
4. **ORDER_SMALL**: Place small order (10-20 units)
5. **ORDER_MEDIUM**: Place medium order (20-40 units)
6. **ORDER_LARGE**: Place large order (40-60 units)
7. **REDISTRIBUTE_LOCAL**: Redistribute within same facility
8. **REDISTRIBUTE_REGIONAL**: Redistribute between facilities
9. **EMERGENCY_ORDER**: Place emergency order (faster delivery)
10. **ACTIVATE_BACKUP**: Activate backup storage systems
11. **NO_ACTION**: No action required

### Reward Function

Comprehensive reward system considering:

- **Inventory Management**: Rewards for maintaining adequate inventory levels
- **Temperature Control**: Rewards for optimal temperature maintenance
- **Demand Fulfillment**: Rewards for successful deliveries
- **Cost Optimization**: Penalties for ordering and holding costs
- **Risk Management**: Penalties for shortages, expired units, and temperature violations

## RL Algorithms

### 1. Deep Q-Network (DQN)

- **Use Case**: Discrete action selection with experience replay
- **Advantages**: Stable learning, good for discrete decisions
- **Configuration**: Configurable network architecture and exploration

### 2. Proximal Policy Optimization (PPO)

- **Use Case**: Policy gradient optimization with clipping
- **Advantages**: Stable training, good sample efficiency
- **Configuration**: Adjustable clipping range and epochs

### 3. Advantage Actor-Critic (A2C)

- **Use Case**: On-policy learning with advantage estimation
- **Advantages**: Good convergence, suitable for continuous control
- **Configuration**: Configurable learning rates and network architecture

## Training Features

### Hyperparameter Optimization

Automatic optimization using Optuna:

```bash
python train.py --algorithm ppo --optimize --n-trials 50
```

Optimized parameters include:
- Learning rates
- Network architectures
- Batch sizes
- Exploration parameters
- Training frequencies

### Evaluation Metrics

Comprehensive evaluation including:
- **Reward Performance**: Mean and standard deviation of episode rewards
- **Inventory Metrics**: Shortages, expired units, successful deliveries
- **Temperature Compliance**: Violations and excursions
- **Cost Efficiency**: Ordering and holding costs

### Baseline Comparison

Automatic comparison against:
- **Random Policy**: Random action selection
- **Conservative Policy**: Always maintain temperature, order when low
- **Aggressive Policy**: Always place medium orders
- **Temperature-Focused Policy**: Prioritize temperature control

## Inference System

### Decision Support Features

- **Confidence Scoring**: Estimate decision confidence for each algorithm
- **Ensemble Methods**: Weighted voting from multiple algorithms
- **Fallback Mechanisms**: Heuristic-based decisions when RL fails
- **Performance Monitoring**: Track decision success rates and system health

### Integration Points

```python
# Backend integration
from ml_models.reinforcement_learning.inference import RLDecisionSupport

# Create decision support
decision_support = RLDecisionSupport({
    "dqn": "./models/dqn_best/best_model",
    "ppo": "./models/ppo_best/best_model",
    "a2c": "./models/a2c_best/best_model"
})

# Get decision for current state
decision = decision_support.get_optimal_decision(current_state, ensemble=True)

# Monitor system health
health = decision_support.get_system_health()
```

## Configuration

### Environment Configuration

```python
config = {
    "num_hospitals": 3,
    "num_blood_banks": 2,
    "num_pods": 5,
    "time_horizon": 168,  # 1 week in hours
    "max_inventory": 200,
    "min_temp": 2.0,
    "max_temp": 6.0,
    "optimal_temp": 4.0,
    "order_delay_hours": 24,
    "delivery_delay_hours": 48
}
```

### Training Configuration

```python
# DQN parameters
dqn_params = {
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "batch_size": 32,
    "gamma": 0.99,
    "exploration_fraction": 0.1,
    "policy_kwargs": {"net_arch": [64, 64]}
}

# PPO parameters
ppo_params = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "clip_range": 0.2
}
```

## Monitoring and Metrics

### Performance Tracking

- **Decision Success Rate**: Percentage of high-confidence decisions
- **Average Confidence**: Mean confidence across all decisions
- **System Health**: Overall performance and engine status
- **Decision History**: Recent decisions with timestamps and reasoning

### System Health Monitoring

```python
health = decision_support.get_system_health()
print(f"Success Rate: {health['overall_performance']['success_rate']:.2%}")
print(f"Average Confidence: {health['overall_performance']['avg_confidence']:.3f}")
print(f"Active Engines: {health['active_engines']}/{health['total_engines']}")
```

## Integration with Crimson IQ

### Backend Integration

The RL system integrates seamlessly with the existing Crimson IQ backend:

1. **Inventory Management**: RL decisions feed into inventory ordering systems
2. **Temperature Control**: RL recommendations for cold chain optimization
3. **Supply Chain**: RL-driven redistribution and emergency ordering
4. **Monitoring**: Integration with existing monitoring and alerting systems

### API Endpoints

```python
# Example API integration
@app.post("/api/rl/decision")
def get_rl_decision(current_state: Dict):
    decision = decision_support.get_optimal_decision(current_state)
    return {
        "action": decision["action"],
        "confidence": decision["confidence"],
        "reasoning": decision["reasoning"],
        "expected_impact": decision["expected_impact"]
    }

@app.get("/api/rl/health")
def get_rl_health():
    return decision_support.get_system_health()
```

## Deployment

### Production Setup

1. **Train Models**: Use production data to train RL models
2. **Deploy Models**: Save trained models to production directory
3. **Initialize Decision Support**: Load models in production environment
4. **Monitor Performance**: Set up monitoring and alerting
5. **Continuous Learning**: Periodically retrain models with new data

### Docker Deployment

```dockerfile
# Example Dockerfile for RL system
FROM python:3.9-slim

WORKDIR /app
COPY requirements/reinforcement_learning.txt .
RUN pip install -r reinforcement_learning.txt

COPY backend/ml_models/reinforcement_learning/ ./rl/
COPY models/ ./models/

CMD ["python", "rl/inference.py"]
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model files exist
   ls -la models/*/best_model.zip
   
   # Verify model compatibility
   python -c "from stable_baselines3 import DQN; DQN.load('models/dqn_best/best_model')"
   ```

2. **Training Convergence Issues**
   ```bash
   # Try different hyperparameters
   python train.py --algorithm ppo --optimize --n-trials 100
   
   # Increase training time
   python train.py --algorithm dqn --timesteps 500000
   ```

3. **Low Decision Confidence**
   ```python
   # Check system health
   health = decision_support.get_system_health()
   print(health['overall_performance'])
   
   # Retrain models with more data
   python train.py --algorithm all --timesteps 1000000
   ```

### Performance Optimization

- **GPU Acceleration**: Use CUDA-enabled PyTorch for faster training
- **Parallel Environments**: Increase number of parallel environments for faster training
- **Model Compression**: Use smaller network architectures for faster inference
- **Caching**: Cache frequently used decisions for faster response times

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure compatibility with existing ML models
5. Test with realistic blood inventory scenarios

## License

This module is part of the Crimson IQ project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test scripts for examples
3. Consult the API documentation
4. Contact the development team
