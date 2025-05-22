import os
import sys
import numpy as np
import torch
import yaml
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.portfolio_manager import PortfolioManager
from environments.trading_env import TradingEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the Portfolio Manager model')
    parser.add_argument('--config', type=str, default='../configs/default_config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda)')
    parser.add_argument('--relation_file', type=str, default=None, help='Path to relation matrix file')
    parser.add_argument('--market_data', type=str, default=None, help='Path to market data file')
    return parser.parse_args()


def load_relation_matrix(file_path, num_stocks):
    """
    Load relation matrix for Graph Convolutional Network.
    
    Args:
        file_path: Path to relation matrix file (.npy)
        num_stocks: Number of stocks in the portfolio
    
    Returns:
        Relationship matrix of shape [num_stocks, num_stocks]
    """
    if not os.path.exists(file_path):
        print(f"Relation matrix file not found at {file_path}, using identity matrix")
        # If file doesn't exist, return identity matrix
        return np.eye(num_stocks)
    
    try:
        relation_matrix = np.load(file_path)
        
        # Verify shape matches expected number of stocks
        if relation_matrix.shape != (num_stocks, num_stocks):
            print(f"Warning: Relation matrix shape {relation_matrix.shape} doesn't match expected shape ({num_stocks}, {num_stocks})")
            return np.eye(num_stocks)
        
        print(f"Loaded relation matrix from {file_path} with shape {relation_matrix.shape}")
        return relation_matrix
    
    except Exception as e:
        print(f"Error loading relation matrix: {e}")
        return np.eye(num_stocks)


def collect_batch_experiences(env, batch_size, device):
    """
    Collect a batch of experiences for batch training.
    
    Args:
        env: Trading environment
        batch_size: Number of experiences to collect
        device: Device to use for tensor operations
    
    Returns:
        Dictionary of batched experiences
    """
    # Reset environment with random start (returns batch)
    state = env.reset()
    # Take a random step to get returns (returns batch)
    _, reward, _, info = env.step(np.random.random(env.num_stocks))
    # Extract asset returns from info (info is a list of dictionaries)
    asset_returns = np.array([i.get('asset_returns', np.zeros(env.num_stocks)) for i in info])
    # Store experiences
    states_tensor = torch.FloatTensor(np.array(state['stocks'])).to(device)
    returns_tensor = torch.FloatTensor(asset_returns).to(device)
    rewards_tensor = torch.FloatTensor(reward).to(device)
    return {
        'states': states_tensor,
        'asset_returns': returns_tensor,
        'rewards': rewards_tensor
    }


def train_portfolio_manager(
    stock_data,
    tickers,
    config,
    output_dir,
    device="cpu",
    relation_matrix=None
):
    """
    Train the portfolio manager using the trading environment.
    
    Args:
        stock_data: Stock data with shape [num_stocks, num_days, features]
        tickers: List of stock tickers
        config: Configuration dictionary
        output_dir: Output directory
        device: Device to use for training
        relation_matrix: Relationship matrix for GCN [num_stocks, num_stocks]
    
    Returns:
        Trained model and training metrics
    """
    # Extract parameters from config
    model_config = config['model']
    training_config = config['training']
    backtest_config = config['backtest']
    
    # Create environment
    env = TradingEnvironment(
        stock_data=stock_data,
        window_size=model_config['window_size'],
        initial_balance=backtest_config['initial_balance'],
        transaction_fee=backtest_config['transaction_fee'],
        holding_period=backtest_config.get('holding_period', 5),
        reward_type="return",
        num_assets_select=model_config['num_assets_select'],
        use_randomized_starts=True,
        batch_size=training_config['batch_size'],
        train_split=training_config.get('train_split', 0.8),
        validation_split=training_config.get('validation_split', 0.1),
        test_split=training_config.get('test_split', 0.1)
    )
    
    # Get dimensions
    num_stocks, num_days, num_features = stock_data.shape
    
    # Create model with relationship matrix
    manager = PortfolioManager(
        stock_feature_dim=num_features,
        num_assets=num_stocks,
        window_size=model_config['window_size'],
        hidden_dim=model_config['hidden_dim'],
        use_spatial_attention=model_config.get('use_spatial_attention', True),
        use_gcn=model_config.get('use_gcn', True),
        relation_matrix=relation_matrix,
        use_adaptive_adj=model_config.get('use_adaptive_adj', True),
        num_assets_select=model_config['num_assets_select'],
        device=device
    )
    
    # Extract training parameters
    batch_size = training_config['batch_size']
    learning_rate = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    num_epochs = training_config['num_epochs']
    episodes_per_epoch = training_config.get('episodes_per_epoch', 10)
    
    # Create optimizer with gradient clipping
    optimizer = torch.optim.Adam(
        manager.asu.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )
    
    # Initialize tracking variables
    train_rewards = []
    val_rewards = []
    best_val_reward = float('-inf')
    train_losses = []
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")
    for epoch in range(num_epochs):
        # Training phase
        manager.asu.train()
        epoch_rewards = []
        epoch_losses = []
        
        # Set environment to training mode
        env.set_mode('train')
        
        for _ in range(episodes_per_epoch):
            # Collect batch of experiences
            batch_data = collect_batch_experiences(env, batch_size, device)
            
            # Forward pass with batch
            optimizer.zero_grad()
            predicted_scores = manager.asu(batch_data['states'])
            
            # Calculate portfolio weights using softmax
            weights = F.softmax(predicted_scores, dim=-1)
            
            # Calculate portfolio returns
            portfolio_returns = torch.sum(weights * batch_data['asset_returns'], dim=-1)
            
            # Calculate loss (negative Sharpe ratio)
            returns_mean = torch.mean(portfolio_returns)
            returns_std = torch.std(portfolio_returns) + 1e-6  # Add small epsilon for stability
            sharpe_ratio = returns_mean / returns_std
            loss = -sharpe_ratio  # Negative because we want to maximize Sharpe ratio
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(manager.asu.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track batch rewards and losses
            epoch_rewards.extend(batch_data['rewards'].cpu().numpy())
            epoch_losses.append(loss.item())
        
        # Calculate average training metrics
        avg_train_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
        train_rewards.append(avg_train_reward)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        manager.asu.eval()
        val_episode_rewards = []
        
        # Set environment to validation mode
        env.set_mode('validation')
        
        with torch.no_grad():
            # Run multiple validation episodes
            for _ in range(10):  # Run 10 validation episodes
                # Collect batch of validation experiences
                val_batch = collect_batch_experiences(env, batch_size, device)
                
                # Add rewards to validation tracking
                val_episode_rewards.extend(val_batch['rewards'].cpu().numpy())
        
        # Calculate average validation return
        avg_val_reward = np.mean(val_episode_rewards) if val_episode_rewards else 0
        val_rewards.append(avg_val_reward)
        
        # Update learning rate based on validation performance
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_reward)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Return={avg_train_reward:.6f}, "
              f"Train Loss={avg_train_loss:.6f}, "
              f"Val Return={avg_val_reward:.6f}")
        
        # Save best model
        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            manager.save(os.path.join(output_dir, "best_model.pt"))
            print(f"Saved best model with validation return {best_val_reward:.6f}")
    
    # Save final model
    manager.save(os.path.join(output_dir, "final_model.pt"))
    
    # Evaluate on test set
    manager.asu.eval()
    test_rewards = []
    
    # Set environment to test mode
    env.set_mode('test')
    
    with torch.no_grad():
        # Run multiple test episodes
        for _ in range(10):  # Run 10 test episodes
            # Collect batch of test experiences
            test_batch = collect_batch_experiences(env, batch_size, device)
            
            # Add rewards to test tracking
            test_rewards.extend(test_batch['rewards'].cpu().numpy())
    
    # Calculate average test return
    avg_test_reward = np.mean(test_rewards) if test_rewards else 0
    print(f"Test Return: {avg_test_reward:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(train_rewards, label='Training Return')
    plt.plot(val_rewards, label='Validation Return')
    plt.axhline(y=avg_test_reward, color='r', linestyle='--', label=f'Test Return: {avg_test_reward:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    plt.title('Training, Validation, and Test Returns')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 1, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Save training metrics
    metrics = {
        'train_rewards': train_rewards,
        'val_rewards': val_rewards,
        'train_losses': train_losses,
        'best_val_reward': best_val_reward,
        'test_reward': avg_test_reward
    }
    
    # After training loop, print summary
    print("\nTraining Summary:")
    print(f"Best validation return: {best_val_reward:.6f}")
    print(f"Test return: {avg_test_reward:.6f}")
    print(f"Training metrics saved to {output_dir}/metrics.json")
    print(f"Training curves saved to {output_dir}/training_curves.png")
    
    return manager, metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/training_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load stock data
    stock_data_path = os.path.join(args.data_dir, "stock_data.npy")
    if not os.path.exists(stock_data_path):
        raise FileNotFoundError(f"Stock data file {stock_data_path} not found")
    
    # Load the stock data
    stock_data = np.load(stock_data_path)
    
    # Load ticker list
    tickers = []
    ticker_path = os.path.join(args.data_dir, "tickers.json")
    if os.path.exists(ticker_path):
        with open(ticker_path, 'r') as f:
            tickers = json.load(f)
    else:
        # If no ticker file, generate placeholder names
        num_stocks = stock_data.shape[0]
        tickers = [f"STOCK_{i}" for i in range(num_stocks)]
    
    # Load relation matrix for GCN if specified
    relation_file = args.relation_file
    if relation_file is None and 'relation_file' in config['model']:
        relation_file = os.path.join(args.data_dir, config['model']['relation_file'])
    
    if relation_file:
        relation_matrix = load_relation_matrix(relation_file, stock_data.shape[0])
    else:
        relation_matrix = None
    
    # Train model
    model, metrics = train_portfolio_manager(
        stock_data=stock_data,
        tickers=tickers,
        config=config,
        output_dir=output_dir,
        device=device,
        relation_matrix=relation_matrix
    )
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            'train_rewards': [float(x) for x in metrics['train_rewards']],
            'val_rewards': [float(x) for x in metrics['val_rewards']],
            'train_losses': [float(x) for x in metrics['train_losses']],
            'best_val_reward': float(metrics['best_val_reward']),
            'test_reward': float(metrics['test_reward'])
        }
        json.dump(metrics_json, f, indent=4)
    
    print(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 