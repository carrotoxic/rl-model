import os
import sys
import numpy as np
import torch
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.trading_env import TradingEnvironment
from agents.deep_trader_agent import DeepTraderAgent
from utils.data_loader import DataLoader
from utils.metrics import (
    evaluate_portfolio,
    print_performance_report,
    plot_portfolio_performance,
    plot_drawdowns
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained DeepTrader model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (default: determined from model path)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides detection)')
    parser.add_argument('--data_split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and show performance plots')
    parser.add_argument('--save_weights', action='store_true',
                        help='Save portfolio weights over time')
    return parser.parse_args()


def main():
    """Main evaluation script."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.dirname(args.model_path))
        if not os.path.exists(args.output_dir):
            args.output_dir = 'evaluation_results'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(
        data_dir='data',
        market_name=config['training']['market']
    )
    
    data_dict = data_loader.load_data(
        window_size=config['training']['window_size'],
        relation_file=config['model']['relation_file']
    )
    
    print(f"Loaded {data_dict['num_stocks']} stocks with {data_dict['num_days']} days of data")
    
    # Create environment
    env = TradingEnvironment(
        stock_data=data_dict['stock_data'],
        market_data=data_dict['market_data'],
        returns_data=data_dict['returns_data'],
        window_size=config['training']['window_size'],
        num_assets_select=config['model']['portfolio']['num_assets_select'],
        reward_type=config['environment']['reward_type'],
        transaction_fee=config['environment']['transaction_fee'],
        initial_balance=config['environment']['initial_balance']
    )
    
    # Create agent
    agent = DeepTraderAgent(
        stock_feature_dim=data_dict['stock_features'],
        market_feature_dim=data_dict['market_features'],
        num_assets=data_dict['num_stocks'],
        window_size=config['training']['window_size'],
        hidden_dim=config['model']['asu']['hidden_dim'],
        use_market_unit=config['model']['use_market_unit'],
        use_spatial_attention=config['model']['use_spatial_attention'],
        use_gcn=config['model']['use_gcn'],
        relation_matrix=data_dict['relation_matrix'],
        use_adaptive_adj=config['model']['use_adaptive_adj'],
        num_assets_select=config['model']['portfolio']['num_assets_select'],
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        batch_size=config['training']['batch_size'],
        device=device
    )
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    agent.load(args.model_path)
    
    # Choose data split to evaluate on
    if args.data_split == 'train':
        evaluation_indices = data_dict['train_indices']
        split_name = 'Training'
    elif args.data_split == 'val':
        evaluation_indices = data_dict['val_indices']
        split_name = 'Validation'
    else:  # test
        evaluation_indices = data_dict['test_indices']
        split_name = 'Test'
    
    print(f"Evaluating on {split_name} set with {len(evaluation_indices)} samples...")
    
    # Evaluate model
    portfolio_values = []
    portfolio_weights_history = []
    daily_returns = []
    action_history = []
    
    initial_balance = config['environment']['initial_balance']
    
    # Run evaluation
    for idx in tqdm(evaluation_indices, desc=f"Evaluating on {split_name} set"):
        # Reset environment to current index
        env.current_step = idx - config['training']['window_size']
        env.portfolio_value = initial_balance
        env.portfolio_weights = np.zeros(data_dict['num_stocks'])
        env.portfolio_history = [env.portfolio_value]
        obs = env._get_observation()
        
        # Select action (no exploration)
        action = agent.select_action(obs, epsilon=0.0)
        action_history.append(action)
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Track performance
        portfolio_values.append(info['portfolio_value'])
        portfolio_weights_history.append(env.portfolio_weights.copy())
        
        # Calculate daily return
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
            daily_returns.append(daily_return)
        
        obs = next_obs
    
    # Convert to numpy arrays
    portfolio_values = np.array(portfolio_values)
    portfolio_weights_history = np.array(portfolio_weights_history)
    daily_returns = np.array(daily_returns) if daily_returns else np.array([])
    
    # Create benchmark (equal weight portfolio)
    benchmark_weights = np.zeros(data_dict['num_stocks'])
    long_indices = np.arange(config['model']['portfolio']['num_assets_select'])
    short_indices = np.arange(config['model']['portfolio']['num_assets_select'])
    benchmark_weights[long_indices] = 1.0 / config['model']['portfolio']['num_assets_select']
    benchmark_weights[short_indices] = -1.0 / config['model']['portfolio']['num_assets_select']
    
    benchmark_portfolio_values = [initial_balance]
    for idx in evaluation_indices:
        returns = data_dict['returns_data'][:, idx]
        portfolio_return = np.sum(benchmark_weights * returns)
        benchmark_portfolio_values.append(benchmark_portfolio_values[-1] * (1 + portfolio_return))
    
    benchmark_portfolio_values = np.array(benchmark_portfolio_values[1:])  # Skip initial value
    
    # Calculate and print performance metrics
    print(f"\n{split_name} Set Performance:")
    print_performance_report(portfolio_values, benchmark_portfolio_values)
    
    # Save results
    results = evaluate_portfolio(portfolio_values)
    benchmark_results = evaluate_portfolio(benchmark_portfolio_values)
    
    results_dict = {
        'model_performance': {
            'total_return': float(results['total_return']),
            'annualized_return': float(results['annualized_return']),
            'sharpe_ratio': float(results['sharpe_ratio']),
            'sortino_ratio': float(results['sortino_ratio']),
            'max_drawdown': float(results['max_drawdown']),
            'calmar_ratio': float(results['calmar_ratio'])
        },
        'benchmark_performance': {
            'total_return': float(benchmark_results['total_return']),
            'annualized_return': float(benchmark_results['annualized_return']),
            'sharpe_ratio': float(benchmark_results['sharpe_ratio']),
            'sortino_ratio': float(benchmark_results['sortino_ratio']),
            'max_drawdown': float(benchmark_results['max_drawdown']),
            'calmar_ratio': float(benchmark_results['calmar_ratio'])
        },
        'comparison': {
            'return_difference': float(results['total_return'] - benchmark_results['total_return']),
            'sharpe_difference': float(results['sharpe_ratio'] - benchmark_results['sharpe_ratio'])
        }
    }
    
    # Save results to file
    results_path = os.path.join(args.output_dir, f"{args.data_split}_results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results_dict, f)
    
    print(f"Results saved to {results_path}")
    
    # Save portfolio weights if requested
    if args.save_weights:
        weights_path = os.path.join(args.output_dir, f"{args.data_split}_weights.npy")
        np.save(weights_path, portfolio_weights_history)
        print(f"Portfolio weights saved to {weights_path}")
    
    # Generate plots
    # Performance plot
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_values, label="DeepTrader", linewidth=2)
    plt.plot(benchmark_portfolio_values, label="Benchmark", linewidth=2, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"{split_name} Set Performance")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    
    # Save performance plot
    plot_path = os.path.join(args.output_dir, f"{args.data_split}_performance.png")
    plt.savefig(plot_path)
    if args.plot:
        plt.show()
    plt.close()
    
    # Drawdown plot
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    
    plt.figure(figsize=(12, 8))
    plt.plot(drawdown, color='red', linewidth=2)
    plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title(f"{split_name} Set Drawdowns")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown")
    
    # Add maximum drawdown to the plot
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    plt.axhline(y=max_dd, color='black', linestyle='--', alpha=0.5)
    plt.text(max_dd_idx, max_dd, f" Max Drawdown: {max_dd:.2%}", verticalalignment='bottom')
    
    # Save drawdown plot
    drawdown_path = os.path.join(args.output_dir, f"{args.data_split}_drawdowns.png")
    plt.savefig(drawdown_path)
    if args.plot:
        plt.show()
    plt.close()
    
    # Generate daily returns distribution plot
    if len(daily_returns) > 0:
        plt.figure(figsize=(12, 8))
        plt.hist(daily_returns, bins=50, alpha=0.75)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.axvline(x=np.mean(daily_returns), color='green', linestyle='-', label=f"Mean: {np.mean(daily_returns):.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"{split_name} Set Daily Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        
        # Save returns distribution plot
        returns_path = os.path.join(args.output_dir, f"{args.data_split}_returns_dist.png")
        plt.savefig(returns_path)
        if args.plot:
            plt.show()
        plt.close()
    
    print(f"Plots saved to {args.output_dir}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 