import os
import sys
import numpy as np
import torch
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.portfolio_manager import PortfolioManager
from environments.trading_env import TradingEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the Portfolio Manager model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda)')
    return parser.parse_args()


def evaluate_portfolio_performance(portfolio_values, initial_value=10000):
    """
    Calculate performance metrics for a portfolio.
    
    Args:
        portfolio_values: List of portfolio values over time
        initial_value: Initial portfolio value
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    num_days = len(portfolio_values)
    annualized_return = (1 + total_return) ** (252 / num_days) - 1
    
    # Calculate volatility
    daily_volatility = np.std(returns)
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Calculate maximum drawdown
    peak = portfolio_values[0]
    drawdowns = []
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns)
    
    # Calculate Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate win rate
    num_positive_days = np.sum(returns > 0)
    win_rate = num_positive_days / len(returns) if len(returns) > 0 else 0
    
    # Return metrics dictionary
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'num_days': num_days
    }


def evaluate_model(
    data_dict,
    model_path,
    config,
    output_dir,
    device="cpu"
):
    """
    Evaluate the portfolio manager model on test data.
    
    Args:
        data_dict: Dictionary containing data
        model_path: Path to model weights
        config: Configuration dictionary
        output_dir: Output directory
        device: Device to use for evaluation
        
    Returns:
        Performance metrics and portfolio history
    """
    # Extract parameters from config
    model_config = config['model']
    backtest_config = config['backtest']
    
    # Extract test indices
    test_indices = data_dict['test_indices']
    
    # Create environment
    env = TradingEnvironment(
        stock_data=data_dict['stock_data'],
        window_size=model_config['window_size'],
        initial_balance=backtest_config['initial_balance'],
        transaction_fee=backtest_config['transaction_fee'],
        holding_period=backtest_config.get('holding_period', 5),
        reward_type="return",
        num_assets_select=model_config['num_assets_select'],
        use_randomized_starts=False  # We want deterministic start for evaluation
    )
    
    # Create model
    manager = PortfolioManager(
        stock_feature_dim=data_dict['stock_features'],
        num_assets=data_dict['num_stocks'],
        window_size=model_config['window_size'],
        hidden_dim=model_config['hidden_dim'],
        use_spatial_attention=model_config['use_spatial_attention'],
        use_gcn=model_config['use_gcn'],
        use_adaptive_adj=model_config.get('use_adaptive_adj', True),
        num_assets_select=model_config['num_assets_select'],
        device=device
    )
    
    # Load model weights
    print(f"Loading model from {model_path}")
    manager.load(model_path)
    
    # Set model to evaluation mode
    manager.asu.eval()
    
    # Run evaluation on each test episode
    all_portfolio_values = []
    all_portfolio_returns = []
    all_portfolio_weights = []
    all_asset_scores = []
    
    # Sort test indices for chronological evaluation
    test_indices.sort()
    
    # Set start index to the earliest test index
    start_idx = test_indices[0]
    state = env.reset_to_index(start_idx)
    
    done = False
    step = 0
    
    # Initialize tracking variables
    portfolio_values = [env.portfolio_value]
    portfolio_returns = []
    portfolio_weights = []
    asset_scores = []
    
    print(f"Starting evaluation from index {start_idx}...")
    
    # Run until done
    while not done:
        # Get asset scores from portfolio manager
        with torch.no_grad():
            scores = manager.predict_scores(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(scores)
        
        # Track performance
        portfolio_values.append(env.portfolio_value)
        portfolio_returns.append(info['portfolio_return'])
        portfolio_weights.append(env.portfolio_weights.copy())
        asset_scores.append(scores)
        
        # Update state
        state = next_state
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}: Portfolio Value = ${env.portfolio_value:.2f}")
        
        step += 1
    
    # Calculate performance metrics
    metrics = evaluate_portfolio_performance(
        portfolio_values, 
        initial_value=backtest_config['initial_balance']
    )
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trading Days: {metrics['num_days']}")
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'return': [0] + portfolio_returns,  # Add 0 for initial return
    })
    
    # Add portfolio weights
    weights_df = pd.DataFrame(
        portfolio_weights, 
        columns=[f'weight_{ticker}' for ticker in data_dict['tickers']]
    )
    
    # Prepend an initial row of zeros for weights (initial state)
    initial_weights = pd.DataFrame(
        [np.zeros(data_dict['num_stocks'])], 
        columns=weights_df.columns
    )
    weights_df = pd.concat([initial_weights, weights_df])
    
    # Combine results
    results = pd.concat([results, weights_df], axis=1)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(portfolio_values)
    plt.title('Portfolio Value')
    plt.xlabel('Trading Days')
    plt.ylabel('Value ($)')
    plt.grid(True)
    
    # Portfolio returns
    plt.subplot(3, 1, 2)
    plt.plot(portfolio_returns)
    plt.title('Portfolio Returns')
    plt.xlabel('Trading Days')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    # Asset weights over time (stacked area chart for top assets)
    plt.subplot(3, 1, 3)
    
    # Get top N assets by average weight
    top_n = min(10, data_dict['num_stocks'])
    avg_weights = np.mean(portfolio_weights, axis=0)
    top_indices = np.argsort(-avg_weights)[:top_n]
    
    # Extract weights for top assets
    top_weights = np.array(portfolio_weights)[:, top_indices]
    
    # Create labels for top assets
    top_tickers = [data_dict['tickers'][i] for i in top_indices]
    
    # Plot stacked area chart
    plt.stackplot(
        range(len(portfolio_weights)),
        top_weights.T,
        labels=top_tickers,
        alpha=0.7
    )
    plt.title('Portfolio Composition (Top Assets)')
    plt.xlabel('Trading Days')
    plt.ylabel('Weight')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'))
    
    # Save detailed results to CSV
    results.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    return metrics, results


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/evaluation_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load data
    data_loader = DataLoader(data_dir=args.data_dir)
    data_dict = data_loader.load_data(window_size=config['model']['window_size'])
    
    # Evaluate model
    metrics, results = evaluate_model(
        data_dict=data_dict,
        model_path=args.model_path,
        config=config,
        output_dir=output_dir,
        device=device
    )
    
    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 