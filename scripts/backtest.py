import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import argparse
import json
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.portfolio_manager import PortfolioManager
from environments.trading_env import TradingEnvironment

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_data(data_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load preprocessed stock data for backtesting.
    
    Returns:
        stock_data: Stock data with shape [num_stocks, num_days, features]
        tickers: List of stock tickers
    """
    # Load stock data
    stock_data_path = os.path.join(data_dir, "stock_data.npy")
    if not os.path.exists(stock_data_path):
        raise FileNotFoundError(f"Stock data file {stock_data_path} not found")
    
    stock_data = np.load(stock_data_path)  # [num_stocks, num_days, features]
    
    # Load ticker list
    ticker_path = os.path.join(data_dir, "tickers.json")
    if not os.path.exists(ticker_path):
        raise FileNotFoundError(f"Ticker file {ticker_path} not found")
    
    with open(ticker_path, 'r') as f:
        tickers = json.load(f)
    
    return stock_data, tickers

def backtest(
    stock_data: np.ndarray,
    tickers: List[str],
    window_size: int = 13,
    model_path: Optional[str] = None,
    hidden_dim: int = 64,
    use_spatial_attention: bool = True,
    use_gcn: bool = True,
    num_assets_select: int = 4,
    initial_balance: float = 10000,
    transaction_fee: float = 0.001
) -> pd.DataFrame:
    """
    Run backtest of portfolio manager on historical data.
    
    Args:
        stock_data: Stock data with shape [num_stocks, num_days, features]
        tickers: List of stock tickers
        window_size: Size of observation window
        model_path: Path to pretrained model weights (if None, uses untrained model)
        hidden_dim: Dimension of hidden layers
        use_spatial_attention: Whether to use spatial attention
        use_gcn: Whether to use graph convolutional network
        num_assets_select: Number of assets to select for portfolio
        initial_balance: Initial portfolio value
        transaction_fee: Fee for each transaction as percentage
        
    Returns:
        Performance metrics and portfolio history as DataFrame
    """
    num_stocks, num_days, num_features = stock_data.shape
    print(f"Running backtest on {num_stocks} stocks over {num_days} days")
    print(f"Feature dimensions: {num_features}")
    
    # Create environment
    env = TradingEnvironment(
        stock_data=stock_data,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        holding_period=5,  # Weekly rebalance
        reward_type="return",
        num_assets_select=num_assets_select,
        use_randomized_starts=False  # We want deterministic start for backtest
    )
    
    # Create portfolio manager
    manager = PortfolioManager(
        stock_feature_dim=num_features,
        num_assets=num_stocks,
        window_size=window_size,
        hidden_dim=hidden_dim,
        use_spatial_attention=use_spatial_attention,
        use_gcn=use_gcn,
        num_assets_select=num_assets_select
    )
    
    # Load model weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        manager.load(model_path)
    else:
        print("Using untrained model")
    
    # Run backtest
    state = env.reset()
    done = False
    
    # Track portfolio performance
    portfolio_values = [env.portfolio_value]
    portfolio_returns = []
    portfolio_weights_history = []
    timestamps = []
    
    step = 0
    while not done:
        # Get asset scores from portfolio manager
        scores = manager.predict_scores(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(scores)
        
        # Track performance
        portfolio_values.append(env.portfolio_value)
        portfolio_returns.append(info['portfolio_return'])
        portfolio_weights_history.append(env.portfolio_weights.copy())
        timestamps.append(env.current_step)
        
        # Update state
        state = next_state
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}: Portfolio Value = ${env.portfolio_value:.2f}")
        
        step += 1
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Calculate drawdowns
    peak = portfolio_values[0]
    drawdowns = []
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns)
    
    # Print performance summary
    print("\nBacktest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Volatility (Annualized): {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'timestamp': timestamps,
        'portfolio_value': portfolio_values[1:],  # Skip initial value
        'return': portfolio_returns,
    })
    
    # Add portfolio weights
    weights_df = pd.DataFrame(
        portfolio_weights_history, 
        columns=[f'weight_{ticker}' for ticker in tickers]
    )
    results = pd.concat([results, weights_df], axis=1)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title('Portfolio Value')
    plt.xlabel('Trading Days')
    plt.ylabel('Value ($)')
    plt.grid(True)
    
    # Portfolio returns
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_returns)
    plt.title('Portfolio Returns')
    plt.xlabel('Trading Days')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'backtest_results_{timestamp}.png')
    
    return results

def main():
    """Main function for running backtest."""
    parser = argparse.ArgumentParser(description='Run portfolio backtest')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to preprocessed data directory')
    parser.add_argument('--model_path', type=str, help='Path to model weights file (optional)')
    parser.add_argument('--output_file', type=str, help='Path to save results (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    stock_data, tickers = load_data(args.data_dir)
    
    # Run backtest
    results = backtest(
        stock_data=stock_data,
        tickers=tickers,
        window_size=config['model']['window_size'],
        model_path=args.model_path,
        hidden_dim=config['model']['hidden_dim'],
        use_spatial_attention=config['model']['use_spatial_attention'],
        use_gcn=config['model']['use_gcn'],
        num_assets_select=config['model']['num_assets_select'],
        initial_balance=config['backtest']['initial_balance'],
        transaction_fee=config['backtest']['transaction_fee']
    )
    
    # Save results if output file specified
    if args.output_file:
        results.to_csv(args.output_file)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 