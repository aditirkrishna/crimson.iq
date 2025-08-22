"""
Reinforcement Learning Training Script for Blood Inventory Management
Implements multiple RL algorithms and training pipelines
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
from optuna.integration import OptunaSearchCV

from env import BloodInventoryEnv, NodeType, BloodGroup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLTrainingManager:
    """Manages RL training for blood inventory management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.training_results = {}
        self.evaluation_results = {}
        
    def create_environment(self, env_id: str = "blood_inventory") -> BloodInventoryEnv:
        """Create and register the blood inventory environment"""
        env = BloodInventoryEnv(
            num_hospitals=self.config.get("num_hospitals", 3),
            num_blood_banks=self.config.get("num_blood_banks", 2),
            num_pods=self.config.get("num_pods", 5),
            time_horizon=self.config.get("time_horizon", 168),
            max_inventory=self.config.get("max_inventory", 200),
            min_temp=self.config.get("min_temp", 2.0),
            max_temp=self.config.get("max_temp", 6.0),
            optimal_temp=self.config.get("optimal_temp", 4.0),
            order_delay_hours=self.config.get("order_delay_hours", 24),
            delivery_delay_hours=self.config.get("delivery_delay_hours", 48)
        )
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        return env
    
    def create_vectorized_env(self, num_envs: int = 4) -> VecNormalize:
        """Create vectorized environment for parallel training"""
        envs = [self.create_environment() for _ in range(num_envs)]
        vec_env = DummyVecEnv([lambda: env for env in envs])
        
        # Normalize observations and rewards
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        return vec_env
    
    def train_dqn(self, total_timesteps: int = 100000, **kwargs) -> DQN:
        """Train DQN agent"""
        logger.info("Training DQN agent...")
        
        # Create environment
        env = self.create_vectorized_env()
        eval_env = self.create_vectorized_env(1)
        
        # DQN parameters
        dqn_params = {
            "learning_rate": kwargs.get("learning_rate", 1e-4),
            "buffer_size": kwargs.get("buffer_size", 100000),
            "learning_starts": kwargs.get("learning_starts", 1000),
            "batch_size": kwargs.get("batch_size", 32),
            "tau": kwargs.get("tau", 1.0),
            "gamma": kwargs.get("gamma", 0.99),
            "train_freq": kwargs.get("train_freq", 4),
            "gradient_steps": kwargs.get("gradient_steps", 1),
            "target_update_interval": kwargs.get("target_update_interval", 1000),
            "exploration_fraction": kwargs.get("exploration_fraction", 0.1),
            "exploration_initial_eps": kwargs.get("exploration_initial_eps", 1.0),
            "exploration_final_eps": kwargs.get("exploration_final_eps", 0.05),
            "policy_kwargs": kwargs.get("policy_kwargs", {"net_arch": [64, 64]})
        }
        
        # Create model
        model = DQN("MlpPolicy", env, verbose=1, **dqn_params)
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/dqn_best/",
            log_path="./logs/dqn/",
            eval_freq=max(total_timesteps // 10, 1),
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, 1),
            save_path="./models/dqn_checkpoints/",
            name_prefix="dqn_model"
        )
        
        # Train model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Store results
        self.models["dqn"] = model
        self.training_results["dqn"] = {
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "parameters": dqn_params
        }
        
        logger.info(f"DQN training completed in {training_time:.2f} seconds")
        return model
    
    def train_ppo(self, total_timesteps: int = 100000, **kwargs) -> PPO:
        """Train PPO agent"""
        logger.info("Training PPO agent...")
        
        # Create environment
        env = self.create_vectorized_env()
        eval_env = self.create_vectorized_env(1)
        
        # PPO parameters
        ppo_params = {
            "learning_rate": kwargs.get("learning_rate", 3e-4),
            "n_steps": kwargs.get("n_steps", 2048),
            "batch_size": kwargs.get("batch_size", 64),
            "n_epochs": kwargs.get("n_epochs", 10),
            "gamma": kwargs.get("gamma", 0.99),
            "gae_lambda": kwargs.get("gae_lambda", 0.95),
            "clip_range": kwargs.get("clip_range", 0.2),
            "clip_range_vf": kwargs.get("clip_range_vf", None),
            "normalize_advantage": kwargs.get("normalize_advantage", True),
            "ent_coef": kwargs.get("ent_coef", 0.0),
            "vf_coef": kwargs.get("vf_coef", 0.5),
            "max_grad_norm": kwargs.get("max_grad_norm", 0.5),
            "use_sde": kwargs.get("use_sde", False),
            "sde_sample_freq": kwargs.get("sde_sample_freq", -1),
            "target_kl": kwargs.get("target_kl", None),
            "policy_kwargs": kwargs.get("policy_kwargs", {"net_arch": [64, 64]})
        }
        
        # Create model
        model = PPO("MlpPolicy", env, verbose=1, **ppo_params)
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/ppo_best/",
            log_path="./logs/ppo/",
            eval_freq=max(total_timesteps // 10, 1),
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, 1),
            save_path="./models/ppo_checkpoints/",
            name_prefix="ppo_model"
        )
        
        # Train model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Store results
        self.models["ppo"] = model
        self.training_results["ppo"] = {
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "parameters": ppo_params
        }
        
        logger.info(f"PPO training completed in {training_time:.2f} seconds")
        return model
    
    def train_a2c(self, total_timesteps: int = 100000, **kwargs) -> A2C:
        """Train A2C agent"""
        logger.info("Training A2C agent...")
        
        # Create environment
        env = self.create_vectorized_env()
        eval_env = self.create_vectorized_env(1)
        
        # A2C parameters
        a2c_params = {
            "learning_rate": kwargs.get("learning_rate", 7e-4),
            "n_steps": kwargs.get("n_steps", 5),
            "gamma": kwargs.get("gamma", 0.99),
            "gae_lambda": kwargs.get("gae_lambda", 1.0),
            "ent_coef": kwargs.get("ent_coef", 0.0),
            "vf_coef": kwargs.get("vf_coef", 0.25),
            "max_grad_norm": kwargs.get("max_grad_norm", 0.5),
            "rms_prop_eps": kwargs.get("rms_prop_eps", 1e-5),
            "use_sde": kwargs.get("use_sde", False),
            "sde_sample_freq": kwargs.get("sde_sample_freq", -1),
            "use_rms_prop": kwargs.get("use_rms_prop", True),
            "policy_kwargs": kwargs.get("policy_kwargs", {"net_arch": [64, 64]})
        }
        
        # Create model
        model = A2C("MlpPolicy", env, verbose=1, **a2c_params)
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/a2c_best/",
            log_path="./logs/a2c/",
            eval_freq=max(total_timesteps // 10, 1),
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, 1),
            save_path="./models/a2c_checkpoints/",
            name_prefix="a2c_model"
        )
        
        # Train model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Store results
        self.models["a2c"] = model
        self.training_results["a2c"] = {
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "parameters": a2c_params
        }
        
        logger.info(f"A2C training completed in {training_time:.2f} seconds")
        return model
    
    def optimize_hyperparameters(self, algorithm: str, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {algorithm}...")
        
        def objective(trial):
            # Define hyperparameter search space
            if algorithm == "dqn":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                    "buffer_size": trial.suggest_int("buffer_size", 50000, 200000),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                    "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                    "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
                    "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
                    "policy_kwargs": {
                        "net_arch": trial.suggest_categorical("net_arch", 
                            [[32, 32], [64, 64], [128, 128], [64, 64, 64]])
                    }
                }
                model = DQN("MlpPolicy", self.create_vectorized_env(1), verbose=0, **params)
                
            elif algorithm == "ppo":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                    "n_steps": trial.suggest_int("n_steps", 1024, 4096),
                    "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                    "n_epochs": trial.suggest_int("n_epochs", 5, 20),
                    "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                    "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
                    "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
                    "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
                    "policy_kwargs": {
                        "net_arch": trial.suggest_categorical("net_arch", 
                            [[32, 32], [64, 64], [128, 128], [64, 64, 64]])
                    }
                }
                model = PPO("MlpPolicy", self.create_vectorized_env(1), verbose=0, **params)
                
            elif algorithm == "a2c":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                    "n_steps": trial.suggest_int("n_steps", 3, 10),
                    "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                    "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
                    "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
                    "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.5),
                    "policy_kwargs": {
                        "net_arch": trial.suggest_categorical("net_arch", 
                            [[32, 32], [64, 64], [128, 128], [64, 64, 64]])
                    }
                }
                model = A2C("MlpPolicy", self.create_vectorized_env(1), verbose=0, **params)
            
            # Train for a shorter period for optimization
            model.learn(total_timesteps=10000, progress_bar=False)
            
            # Evaluate
            eval_env = self.create_vectorized_env(1)
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
            
            return mean_reward
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        logger.info(f"Best {algorithm} parameters: {best_params}")
        logger.info(f"Best {algorithm} score: {study.best_value}")
        
        return best_params
    
    def evaluate_model(self, model, algorithm: str, n_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate trained model"""
        logger.info(f"Evaluating {algorithm} model...")
        
        env = self.create_environment()
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_metrics.append(info)
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        # Aggregate metrics
        avg_shortages = np.mean([m.get("inventory_shortages", 0) for m in episode_metrics])
        avg_temperature_violations = np.mean([m.get("temperature_violations", 0) for m in episode_metrics])
        avg_deliveries = np.mean([m.get("successful_deliveries", 0) for m in episode_metrics])
        avg_expired_units = np.mean([m.get("expired_units", 0) for m in episode_metrics])
        
        results = {
            "algorithm": algorithm,
            "n_episodes": n_episodes,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "avg_shortages": avg_shortages,
            "avg_temperature_violations": avg_temperature_violations,
            "avg_deliveries": avg_deliveries,
            "avg_expired_units": avg_expired_units,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
        
        self.evaluation_results[algorithm] = results
        
        logger.info(f"{algorithm} evaluation results:")
        logger.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Mean episode length: {mean_length:.1f}")
        logger.info(f"  Avg shortages: {avg_shortages:.1f}")
        logger.info(f"  Avg temperature violations: {avg_temperature_violations:.1f}")
        logger.info(f"  Avg successful deliveries: {avg_deliveries:.1f}")
        logger.info(f"  Avg expired units: {avg_expired_units:.1f}")
        
        return results
    
    def compare_with_baselines(self) -> Dict[str, Any]:
        """Compare RL policies with baseline policies"""
        logger.info("Comparing with baseline policies...")
        
        baseline_results = {}
        
        # Random policy baseline
        env = self.create_environment()
        random_rewards = []
        random_metrics = []
        
        for episode in range(50):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = env.action_space.sample()  # Random action
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            random_rewards.append(total_reward)
            random_metrics.append(info)
        
        baseline_results["random"] = {
            "mean_reward": np.mean(random_rewards),
            "std_reward": np.std(random_rewards),
            "avg_shortages": np.mean([m.get("inventory_shortages", 0) for m in random_metrics]),
            "avg_temperature_violations": np.mean([m.get("temperature_violations", 0) for m in random_metrics]),
            "avg_deliveries": np.mean([m.get("successful_deliveries", 0) for m in random_metrics]),
            "avg_expired_units": np.mean([m.get("expired_units", 0) for m in random_metrics])
        }
        
        # Conservative policy baseline (always maintain temperature, order when low)
        env = self.create_environment()
        conservative_rewards = []
        conservative_metrics = []
        
        for episode in range(50):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Simple heuristic: maintain temperature and order when inventory is low
                inventory_levels = obs[:env.total_nodes * env.num_blood_groups]
                avg_inventory = np.mean(inventory_levels)
                
                if avg_inventory < 0.2:  # Low inventory
                    action = 4  # ORDER_MEDIUM
                else:
                    action = 0  # MAINTAIN_TEMP
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            conservative_rewards.append(total_reward)
            conservative_metrics.append(info)
        
        baseline_results["conservative"] = {
            "mean_reward": np.mean(conservative_rewards),
            "std_reward": np.std(conservative_rewards),
            "avg_shortages": np.mean([m.get("inventory_shortages", 0) for m in conservative_metrics]),
            "avg_temperature_violations": np.mean([m.get("temperature_violations", 0) for m in conservative_metrics]),
            "avg_deliveries": np.mean([m.get("successful_deliveries", 0) for m in conservative_metrics]),
            "avg_expired_units": np.mean([m.get("expired_units", 0) for m in conservative_metrics])
        }
        
        # Compare with RL results
        comparison = {
            "baselines": baseline_results,
            "rl_models": {k: {
                "mean_reward": v["mean_reward"],
                "avg_shortages": v["avg_shortages"],
                "avg_temperature_violations": v["avg_temperature_violations"],
                "avg_deliveries": v["avg_deliveries"],
                "avg_expired_units": v["avg_expired_units"]
            } for k, v in self.evaluation_results.items()}
        }
        
        logger.info("Policy comparison:")
        logger.info(f"Random policy: {baseline_results['random']['mean_reward']:.2f}")
        logger.info(f"Conservative policy: {baseline_results['conservative']['mean_reward']:.2f}")
        for alg, results in self.evaluation_results.items():
            logger.info(f"{alg.upper()}: {results['mean_reward']:.2f}")
        
        return comparison
    
    def save_results(self, output_dir: str = "./results"):
        """Save training and evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for algorithm, model in self.models.items():
            model_path = os.path.join(output_dir, f"{algorithm}_model")
            model.save(model_path)
            logger.info(f"Saved {algorithm} model to {model_path}")
        
        # Save results
        results = {
            "training_results": self.training_results,
            "evaluation_results": self.evaluation_results,
            "timestamp": datetime.now().isoformat()
        }
        
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {results_path}")
    
    def load_model(self, model_path: str, algorithm: str):
        """Load a trained model"""
        if algorithm == "dqn":
            model = DQN.load(model_path)
        elif algorithm == "ppo":
            model = PPO.load(model_path)
        elif algorithm == "a2c":
            model = A2C.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.models[algorithm] = model
        logger.info(f"Loaded {algorithm} model from {model_path}")
        return model

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="RL Training for Blood Inventory Management")
    parser.add_argument("--algorithm", type=str, default="all", 
                       choices=["dqn", "ppo", "a2c", "all"],
                       help="RL algorithm to train")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Number of training timesteps")
    parser.add_argument("--optimize", action="store_true",
                       help="Optimize hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "num_hospitals": 3,
        "num_blood_banks": 2,
        "num_pods": 5,
        "time_horizon": 168,
        "max_inventory": 200,
        "min_temp": 2.0,
        "max_temp": 6.0,
        "optimal_temp": 4.0,
        "order_delay_hours": 24,
        "delivery_delay_hours": 48
    }
    
    # Create training manager
    manager = RLTrainingManager(config)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    algorithms = ["dqn", "ppo", "a2c"] if args.algorithm == "all" else [args.algorithm]
    
    for algorithm in algorithms:
        logger.info(f"Training {algorithm.upper()}...")
        
        # Optimize hyperparameters if requested
        if args.optimize:
            best_params = manager.optimize_hyperparameters(algorithm, args.n_trials)
        else:
            best_params = {}
        
        # Train model
        if algorithm == "dqn":
            model = manager.train_dqn(args.timesteps, **best_params)
        elif algorithm == "ppo":
            model = manager.train_ppo(args.timesteps, **best_params)
        elif algorithm == "a2c":
            model = manager.train_a2c(args.timesteps, **best_params)
        
        # Evaluate model
        manager.evaluate_model(model, algorithm, args.eval_episodes)
    
    # Compare with baselines
    comparison = manager.compare_with_baselines()
    
    # Save results
    manager.save_results(args.output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
