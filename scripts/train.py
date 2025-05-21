import os
import sys
import numpy as np
import torch
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.trading_env import TradingEnvironment
from agents.deep_trader_agent import DeepTraderAgent
from utils.data_loader import DataLoader
from utils.metrics import evaluate_portfolio, print_performance_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the DeepTrader model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides detection)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to load checkpoint from')
    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed = args.seed if args.seed is not None else config['training']['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    market_name = config['training']['market']
    output_dir = f"outputs/{market_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up tensorboard
    if config['training']['tensorboard_log']:
        log_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(
        data_dir='data',
        market_name=market_name
    )
    
    data_dict = data_loader.load_data(
        window_size=config['training']['window_size'],
        relation_file=config['model']['relation_file']
    )
    
    print(f"Loaded {data_dict['num_stocks']} stocks with {data_dict['num_days']} days of data")
    print(f"Stock features: {data_dict['stock_features']}, Market features: {data_dict['market_features']}")
    print(f"Train samples: {len(data_dict['train_indices'])}, "
          f"Val samples: {len(data_dict['val_indices'])}, "
          f"Test samples: {len(data_dict['test_indices'])}")
    
    # Create training environment
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
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Training loop
    print("Starting training...")
    total_episodes = config['training']['epochs']
    save_interval = config['training']['save_interval']
    
    # Keep track of best validation performance
    best_val_return = -np.inf
    
    for episode in range(1, total_episodes + 1):
        # Training phase
        train_portfolio_values = []
        train_rewards = []
        
        # Train using all training indices
        for idx in tqdm(data_dict['train_indices'], desc=f"Episode {episode}/{total_episodes} (Train)"):
            # Reset environment to current index
            env.current_step = idx - config['training']['window_size']
            env.portfolio_value = config['environment']['initial_balance']
            env.portfolio_weights = np.zeros(data_dict['num_stocks'])
            env.portfolio_history = [env.portfolio_value]
            obs = env._get_observation()
            
            # Epsilon-greedy exploration (decaying epsilon)
            epsilon = max(0.1, 1.0 - episode / (total_episodes / 2))
            
            # Select action
            action = agent.select_action(obs, epsilon)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Train agent
            loss = agent.train()
            
            # Track performance
            train_portfolio_values.append(info['portfolio_value'])
            train_rewards.append(reward)
            
            obs = next_obs
        
        # Calculate training metrics
        train_return = (train_portfolio_values[-1] / train_portfolio_values[0]) - 1
        train_metrics = evaluate_portfolio(np.array(train_portfolio_values))
        
        # Validation phase
        val_portfolio_values = []
        val_rewards = []
        
        # Evaluate on validation set without exploration
        for idx in tqdm(data_dict['val_indices'], desc=f"Episode {episode}/{total_episodes} (Val)"):
            # Reset environment to current index
            env.current_step = idx - config['training']['window_size']
            env.portfolio_value = config['environment']['initial_balance']
            env.portfolio_weights = np.zeros(data_dict['num_stocks'])
            env.portfolio_history = [env.portfolio_value]
            obs = env._get_observation()
            
            # Select action (no exploration)
            action = agent.select_action(obs, epsilon=0.0)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Track performance
            val_portfolio_values.append(info['portfolio_value'])
            val_rewards.append(reward)
            
            obs = next_obs
        
        # Calculate validation metrics
        val_return = (val_portfolio_values[-1] / val_portfolio_values[0]) - 1
        val_metrics = evaluate_portfolio(np.array(val_portfolio_values))
        
        # Print episode summary
        print(f"Episode {episode}/{total_episodes} Summary:")
        print(f"  Train Return: {train_return:.2%}, Sharpe: {train_metrics['sharpe_ratio']:.4f}, "
              f"Max DD: {train_metrics['max_drawdown']:.2%}")
        print(f"  Val Return: {val_return:.2%}, Sharpe: {val_metrics['sharpe_ratio']:.4f}, "
              f"Max DD: {val_metrics['max_drawdown']:.2%}")
        
        # Save checkpoint
        if episode % save_interval == 0 or episode == total_episodes:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode}.pth")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model based on validation return
        if val_return > best_val_return:
            best_val_return = val_return
            best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
            agent.save(best_model_path)
            print(f"New best model with val return {val_return:.2%}")
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Train/Return', train_return, episode)
            writer.add_scalar('Train/Sharpe', train_metrics['sharpe_ratio'], episode)
            writer.add_scalar('Train/MaxDrawdown', train_metrics['max_drawdown'], episode)
            writer.add_scalar('Train/AvgReward', np.mean(train_rewards), episode)
            
            writer.add_scalar('Val/Return', val_return, episode)
            writer.add_scalar('Val/Sharpe', val_metrics['sharpe_ratio'], episode)
            writer.add_scalar('Val/MaxDrawdown', val_metrics['max_drawdown'], episode)
            writer.add_scalar('Val/AvgReward', np.mean(val_rewards), episode)
            
            writer.add_scalar('Train/Epsilon', epsilon, episode)
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    
    # Load best model
    agent.load(best_model_path)
    
    test_portfolio_values = []
    test_weights_history = []
    
    # Evaluate on test set
    for idx in tqdm(data_dict['test_indices'], desc="Testing"):
        # Reset environment to current index
        env.current_step = idx - config['training']['window_size']
        env.portfolio_value = config['environment']['initial_balance']
        env.portfolio_weights = np.zeros(data_dict['num_stocks'])
        env.portfolio_history = [env.portfolio_value]
        obs = env._get_observation()
        
        # Select action (no exploration)
        action = agent.select_action(obs, epsilon=0.0)
        
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Track performance
        test_portfolio_values.append(info['portfolio_value'])
        test_weights_history.append(env.portfolio_weights.copy())
        
        obs = next_obs
    
    # Calculate test metrics
    test_return = (test_portfolio_values[-1] / test_portfolio_values[0]) - 1
    test_metrics = evaluate_portfolio(np.array(test_portfolio_values))
    
    # Create benchmark (equal weight portfolio)
    benchmark_weights = np.zeros(data_dict['num_stocks'])
    long_indices = np.arange(config['model']['portfolio']['num_assets_select'])
    short_indices = np.arange(config['model']['portfolio']['num_assets_select'])
    benchmark_weights[long_indices] = 1.0 / config['model']['portfolio']['num_assets_select']
    benchmark_weights[short_indices] = -1.0 / config['model']['portfolio']['num_assets_select']
    
    benchmark_portfolio_values = [config['environment']['initial_balance']]
    for idx in data_dict['test_indices']:
        returns = data_dict['returns_data'][:, idx]
        portfolio_return = np.sum(benchmark_weights * returns)
        benchmark_portfolio_values.append(benchmark_portfolio_values[-1] * (1 + portfolio_return))
    
    # Print test performance report
    print("\nTest Performance:")
    print_performance_report(
        np.array(test_portfolio_values),
        np.array(benchmark_portfolio_values[1:])  # Skip initial value
    )
    
    # Save test results
    results = {
        'test_return': test_return,
        'test_sharpe': test_metrics['sharpe_ratio'],
        'test_sortino': test_metrics['sortino_ratio'],
        'test_max_drawdown': test_metrics['max_drawdown'],
        'test_calmar': test_metrics['calmar_ratio'],
        'benchmark_return': (benchmark_portfolio_values[-1] / benchmark_portfolio_values[0]) - 1
    }
    
    # Save results to file
    results_path = os.path.join(output_dir, "results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    print(f"Results saved to {results_path}")
    
    # Plot test performance
    plt.figure(figsize=(12, 8))
    plt.plot(test_portfolio_values, label="DeepTrader", linewidth=2)
    plt.plot(benchmark_portfolio_values[1:], label="Benchmark", linewidth=2, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Test Performance")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    
    # Save plot
    plot_path = os.path.join(output_dir, "test_performance.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()