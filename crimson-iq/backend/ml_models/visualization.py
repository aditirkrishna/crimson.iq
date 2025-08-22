"""
Phase 1.5: Visualization and Dashboard Integration
Creates visualization scripts and generates JSON outputs for frontend dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Configure logging and plotting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DashboardVisualizer:
    """Creates visualizations for the blood inventory dashboard"""
    
    def __init__(self, output_dir: str = "dashboard_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_demand_forecast_visualization(self, 
                                          actual_data: pd.DataFrame,
                                          forecast_data: Dict[str, Any],
                                          save_path: str = None) -> Dict[str, Any]:
        """Create demand forecast visualization"""
        
        # Create time series plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Blood Demand Forecast', 'Forecast Confidence Intervals'),
            vertical_spacing=0.1
        )
        
        # Plot actual data
        if not actual_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual_data.index,
                    y=actual_data['demand_quantity'],
                    mode='lines+markers',
                    name='Actual Demand',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # Plot forecast
        forecast_times = list(range(len(forecast_data['predictions'])))
        fig.add_trace(
            go.Scatter(
                x=forecast_times,
                y=forecast_data['predictions'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Plot confidence intervals
        if 'confidence_interval' in forecast_data:
            ci = forecast_data['confidence_interval']
            fig.add_trace(
                go.Scatter(
                    x=forecast_times,
                    y=ci['upper_bound'],
                    mode='lines',
                    name='Upper CI',
                    line=dict(color='lightgray', width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_times,
                    y=ci['lower_bound'],
                    mode='lines',
                    fill='tonexty',
                    name='Confidence Interval',
                    line=dict(color='lightgray', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_times,
                    y=forecast_data['predictions'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Blood Demand Forecast - {forecast_data.get('blood_group', 'All Groups')}",
            xaxis_title="Time (hours)",
            yaxis_title="Demand Quantity",
            height=600,
            showlegend=True
        )
        
        # Save plot
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Demand forecast visualization saved to {save_path}")
        
        # Create JSON output for frontend
        json_output = {
            'chart_type': 'demand_forecast',
            'title': f"Blood Demand Forecast - {forecast_data.get('blood_group', 'All Groups')}",
            'data': {
                'actual': {
                    'x': actual_data.index.tolist() if not actual_data.empty else [],
                    'y': actual_data['demand_quantity'].tolist() if not actual_data.empty else []
                },
                'forecast': {
                    'x': forecast_times,
                    'y': forecast_data['predictions']
                },
                'confidence_interval': forecast_data.get('confidence_interval', {})
            },
            'metadata': {
                'model_name': forecast_data.get('model_name', 'Unknown'),
                'prediction_steps': forecast_data.get('prediction_steps', 24),
                'timestamp': forecast_data.get('timestamp', datetime.now().isoformat())
            }
        }
        
        return json_output
        
    def create_survival_analysis_visualization(self, 
                                             survival_data: Dict[str, Any],
                                             save_path: str = None) -> Dict[str, Any]:
        """Create survival analysis visualization"""
        
        # Create survival curves plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Survival Function', 'Hazard Function', 
                          'Risk Scores', 'Survival by Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot survival function
        prediction_times = survival_data['prediction_times']
        survival_probs = survival_data['survival_probabilities']
        
        if len(survival_probs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=prediction_times,
                    y=survival_probs[0],  # First blood unit
                    mode='lines+markers',
                    name='Survival Probability',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
        
        # Plot hazard function
        hazard_rates = survival_data['hazard_rates']
        if len(hazard_rates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=prediction_times,
                    y=hazard_rates[0],  # First blood unit
                    mode='lines+markers',
                    name='Hazard Rate',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        # Plot risk scores
        risk_scores = survival_data['risk_scores']
        risk_metrics = list(risk_scores.keys())
        risk_values = list(risk_scores.values())
        
        fig.add_trace(
            go.Bar(
                x=risk_metrics,
                y=risk_values,
                name='Risk Scores',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Plot survival by time
        if len(survival_probs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=prediction_times,
                    y=survival_probs[0],
                    mode='lines+markers',
                    name='Survival Over Time',
                    line=dict(color='blue', width=2),
                    fill='tonexty'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Blood Unit Survival Analysis",
            height=800,
            showlegend=True
        )
        
        # Save plot
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Survival analysis visualization saved to {save_path}")
        
        # Create JSON output for frontend
        json_output = {
            'chart_type': 'survival_analysis',
            'title': 'Blood Unit Survival Analysis',
            'data': {
                'survival_function': {
                    'x': prediction_times,
                    'y': survival_probs[0] if len(survival_probs) > 0 else []
                },
                'hazard_function': {
                    'x': prediction_times,
                    'y': hazard_rates[0] if len(hazard_rates) > 0 else []
                },
                'risk_scores': risk_scores,
                'survival_by_time': {
                    'x': prediction_times,
                    'y': survival_probs[0] if len(survival_probs) > 0 else []
                }
            },
            'metadata': {
                'model_name': survival_data.get('model_name', 'Unknown'),
                'blood_units_analyzed': survival_data.get('blood_units_analyzed', 0),
                'timestamp': survival_data.get('timestamp', datetime.now().isoformat())
            }
        }
        
        return json_output
        
    def create_inventory_dashboard(self, 
                                 inventory_data: pd.DataFrame,
                                 save_path: str = None) -> Dict[str, Any]:
        """Create comprehensive inventory dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Inventory by Blood Group', 'Temperature Distribution',
                          'Inventory by Pod', 'Inventory Status',
                          'Temperature Trends', 'Inventory Age Distribution'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # 1. Inventory by Blood Group
        blood_group_counts = inventory_data['blood_group'].value_counts()
        fig.add_trace(
            go.Bar(
                x=blood_group_counts.index,
                y=blood_group_counts.values,
                name='Inventory by Blood Group',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Temperature Distribution
        fig.add_trace(
            go.Histogram(
                x=inventory_data['temperature'],
                name='Temperature Distribution',
                nbinsx=20,
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # 3. Inventory by Pod
        pod_counts = inventory_data['pod_id'].value_counts()
        fig.add_trace(
            go.Bar(
                x=pod_counts.index,
                y=pod_counts.values,
                name='Inventory by Pod',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        # 4. Inventory Status
        status_counts = inventory_data['status'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name='Inventory Status'
            ),
            row=2, col=2
        )
        
        # 5. Temperature Trends (if timestamp available)
        if 'timestamp' in inventory_data.columns:
            # Group by timestamp and calculate mean temperature
            temp_trends = inventory_data.groupby('timestamp')['temperature'].mean()
            fig.add_trace(
                go.Scatter(
                    x=temp_trends.index,
                    y=temp_trends.values,
                    mode='lines',
                    name='Temperature Trends',
                    line=dict(color='orange', width=2)
                ),
                row=3, col=1
            )
        
        # 6. Inventory Age Distribution (if donation_date available)
        if 'donation_date' in inventory_data.columns:
            # Calculate age in days
            inventory_data['age_days'] = (
                pd.to_datetime(inventory_data['timestamp']) - 
                pd.to_datetime(inventory_data['donation_date'])
            ).dt.days
            
            fig.add_trace(
                go.Histogram(
                    x=inventory_data['age_days'],
                    name='Inventory Age Distribution',
                    nbinsx=20,
                    marker_color='lightpink'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Blood Inventory Dashboard",
            height=1200,
            showlegend=False
        )
        
        # Save plot
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Inventory dashboard saved to {save_path}")
        
        # Create JSON output for frontend
        json_output = {
            'chart_type': 'inventory_dashboard',
            'title': 'Blood Inventory Dashboard',
            'data': {
                'blood_group_inventory': {
                    'labels': blood_group_counts.index.tolist(),
                    'values': blood_group_counts.values.tolist()
                },
                'temperature_distribution': {
                    'values': inventory_data['temperature'].tolist()
                },
                'pod_inventory': {
                    'labels': pod_counts.index.tolist(),
                    'values': pod_counts.values.tolist()
                },
                'status_distribution': {
                    'labels': status_counts.index.tolist(),
                    'values': status_counts.values.tolist()
                }
            },
            'metadata': {
                'total_units': len(inventory_data),
                'unique_blood_groups': len(blood_group_counts),
                'unique_pods': len(pod_counts),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return json_output
        
    def create_model_comparison_visualization(self, 
                                            model_results: Dict[str, Dict[str, float]],
                                            save_path: str = None) -> Dict[str, Any]:
        """Create model comparison visualization"""
        
        # Extract metrics
        models = list(model_results.keys())
        metrics = ['mae', 'rmse', 'mape']
        
        # Create subplots for each metric
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 'Mean Absolute Percentage Error')
        )
        
        for i, metric in enumerate(metrics):
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.upper(),
                    marker_color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][i % 4]
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title="Model Performance Comparison",
            height=500,
            showlegend=False
        )
        
        # Save plot
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Model comparison visualization saved to {save_path}")
        
        # Create JSON output for frontend
        json_output = {
            'chart_type': 'model_comparison',
            'title': 'Model Performance Comparison',
            'data': {
                'models': models,
                'metrics': {
                    'mae': [model_results[model].get('mae', 0) for model in models],
                    'rmse': [model_results[model].get('rmse', 0) for model in models],
                    'mape': [model_results[model].get('mape', 0) for model in models]
                }
            },
            'metadata': {
                'num_models': len(models),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return json_output
        
    def create_alert_visualization(self, 
                                 alert_data: List[Dict[str, Any]],
                                 save_path: str = None) -> Dict[str, Any]:
        """Create alert visualization"""
        
        if not alert_data:
            return {'chart_type': 'alerts', 'data': [], 'message': 'No alerts'}
        
        # Convert to DataFrame
        df = pd.DataFrame(alert_data)
        
        # Create alert summary
        alert_counts = df['alert_type'].value_counts()
        alert_severity = df['severity'].value_counts()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Alerts by Type', 'Alerts by Severity',
                          'Alerts Over Time', 'Alerts by Pod'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Alerts by Type
        fig.add_trace(
            go.Bar(
                x=alert_counts.index,
                y=alert_counts.values,
                name='Alerts by Type',
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # 2. Alerts by Severity
        fig.add_trace(
            go.Bar(
                x=alert_severity.index,
                y=alert_severity.values,
                name='Alerts by Severity',
                marker_color='orange'
            ),
            row=1, col=2
        )
        
        # 3. Alerts Over Time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_counts = df.groupby(df['timestamp'].dt.date).size()
            
            fig.add_trace(
                go.Scatter(
                    x=time_counts.index,
                    y=time_counts.values,
                    mode='lines+markers',
                    name='Alerts Over Time',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
        
        # 4. Alerts by Pod
        if 'pod_id' in df.columns:
            pod_counts = df['pod_id'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=pod_counts.index,
                    y=pod_counts.values,
                    name='Alerts by Pod',
                    marker_color='brown'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="System Alerts Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save plot
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Alert visualization saved to {save_path}")
        
        # Create JSON output for frontend
        json_output = {
            'chart_type': 'alerts',
            'title': 'System Alerts Dashboard',
            'data': {
                'alerts_by_type': {
                    'labels': alert_counts.index.tolist(),
                    'values': alert_counts.values.tolist()
                },
                'alerts_by_severity': {
                    'labels': alert_severity.index.tolist(),
                    'values': alert_severity.values.tolist()
                },
                'total_alerts': len(alert_data),
                'recent_alerts': alert_data[-10:] if len(alert_data) > 10 else alert_data
            },
            'metadata': {
                'total_alerts': len(alert_data),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return json_output
        
    def save_json_output(self, json_data: Dict[str, Any], filename: str):
        """Save JSON output for frontend consumption"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        logger.info(f"JSON output saved to {output_path}")
        
    def create_dashboard_summary(self, 
                               inventory_data: pd.DataFrame,
                               forecast_data: Dict[str, Any] = None,
                               survival_data: Dict[str, Any] = None,
                               model_results: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """Create comprehensive dashboard summary"""
        
        # Calculate key metrics
        total_units = len(inventory_data)
        unique_blood_groups = inventory_data['blood_group'].nunique()
        unique_pods = inventory_data['pod_id'].nunique()
        
        # Temperature statistics
        temp_stats = {
            'mean': float(inventory_data['temperature'].mean()),
            'std': float(inventory_data['temperature'].std()),
            'min': float(inventory_data['temperature'].min()),
            'max': float(inventory_data['temperature'].max())
        }
        
        # Status distribution
        status_dist = inventory_data['status'].value_counts().to_dict()
        
        # Create summary
        summary = {
            'dashboard_type': 'blood_inventory_summary',
            'timestamp': datetime.now().isoformat(),
            'key_metrics': {
                'total_blood_units': total_units,
                'unique_blood_groups': unique_blood_groups,
                'unique_pods': unique_pods,
                'average_temperature': temp_stats['mean']
            },
            'temperature_stats': temp_stats,
            'status_distribution': status_dist,
            'blood_group_distribution': inventory_data['blood_group'].value_counts().to_dict(),
            'pod_distribution': inventory_data['pod_id'].value_counts().to_dict()
        }
        
        # Add forecast data if available
        if forecast_data:
            summary['forecast'] = {
                'model_name': forecast_data.get('model_name'),
                'prediction_steps': forecast_data.get('prediction_steps'),
                'next_24h_demand': sum(forecast_data.get('predictions', [])[:24])
            }
        
        # Add survival data if available
        if survival_data:
            summary['survival_analysis'] = {
                'model_name': survival_data.get('model_name'),
                'blood_units_analyzed': survival_data.get('blood_units_analyzed'),
                'average_risk_score': survival_data.get('risk_scores', {}).get('average_risk', 0)
            }
        
        # Add model results if available
        if model_results:
            best_model = min(model_results.keys(), 
                           key=lambda x: model_results[x].get('rmse', float('inf')))
            summary['model_performance'] = {
                'best_model': best_model,
                'best_rmse': model_results[best_model].get('rmse', 0),
                'all_results': model_results
            }
        
        return summary

def main():
    """Main function for visualization pipeline"""
    
    # Initialize visualizer
    visualizer = DashboardVisualizer()
    
    # Load sample data
    try:
        inventory_data = pd.read_parquet("processed_data/inventory_processed.parquet")
        demand_data = pd.read_parquet("processed_data/demand_processed.parquet")
        logger.info("Loaded processed data for visualization")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run the data preprocessing pipeline first.")
        return
    
    # Create sample forecast data
    sample_forecast = {
        'model_name': 'RandomForest',
        'blood_group': 'O+',
        'prediction_steps': 24,
        'predictions': np.random.poisson(15, 24).tolist(),
        'confidence_interval': {
            'lower_bound': np.random.poisson(10, 24).tolist(),
            'upper_bound': np.random.poisson(20, 24).tolist(),
            'confidence_level': 0.95
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Create sample survival data
    sample_survival = {
        'model_name': 'CoxPH',
        'prediction_times': [7, 14, 21, 30, 35, 42],
        'survival_probabilities': [np.random.uniform(0.8, 1.0, 6).tolist()],
        'hazard_rates': [np.random.uniform(0.01, 0.05, 6).tolist()],
        'risk_scores': {
            'immediate_risk_7d': 0.15,
            'short_term_risk_14d': 0.25,
            'medium_term_risk_30d': 0.40,
            'expiry_risk_42d': 0.60,
            'average_risk': 0.35,
            'max_risk': 0.60
        },
        'blood_units_analyzed': 100,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create sample model results
    sample_model_results = {
        'ARIMA': {'mae': 2.5, 'rmse': 3.2, 'mape': 15.5},
        'ExponentialSmoothing': {'mae': 2.8, 'rmse': 3.5, 'mape': 17.2},
        'RandomForest': {'mae': 2.1, 'rmse': 2.8, 'mape': 13.8}
    }
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Demand forecast visualization
    forecast_json = visualizer.create_demand_forecast_visualization(
        demand_data, sample_forecast, 
        save_path="dashboard_outputs/demand_forecast.html"
    )
    visualizer.save_json_output(forecast_json, "demand_forecast.json")
    
    # 2. Survival analysis visualization
    survival_json = visualizer.create_survival_analysis_visualization(
        sample_survival,
        save_path="dashboard_outputs/survival_analysis.html"
    )
    visualizer.save_json_output(survival_json, "survival_analysis.json")
    
    # 3. Inventory dashboard
    inventory_json = visualizer.create_inventory_dashboard(
        inventory_data,
        save_path="dashboard_outputs/inventory_dashboard.html"
    )
    visualizer.save_json_output(inventory_json, "inventory_dashboard.json")
    
    # 4. Model comparison visualization
    model_comparison_json = visualizer.create_model_comparison_visualization(
        sample_model_results,
        save_path="dashboard_outputs/model_comparison.html"
    )
    visualizer.save_json_output(model_comparison_json, "model_comparison.json")
    
    # 5. Create dashboard summary
    summary_json = visualizer.create_dashboard_summary(
        inventory_data, sample_forecast, sample_survival, sample_model_results
    )
    visualizer.save_json_output(summary_json, "dashboard_summary.json")
    
    logger.info("All visualizations created successfully!")
    logger.info(f"Output files saved to: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
