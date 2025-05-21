# RL-Trader: Deep Reinforcement Learning for Portfolio Management

RL-Trader is a deep reinforcement learning framework for portfolio management with a focus on balancing risk and return through market condition embeddings.

## Features

- **Deep Reinforcement Learning**: Uses DRL to optimize investment policies
- **Market Condition Embedding**: Dynamically adjusts portfolio based on market trends
- **Asset Evaluation**: Learns patterns from historical data to evaluate individual assets
- **Graph-based Asset Relationships**: Captures dependencies between assets using graph neural networks
- **Risk-Return Balance**: Optimizes for both returns and risk metrics like maximum drawdown

## Project Structure

- `agents/`: RL agents implementation
- `configs/`: Configuration files for different experiments
- `data/`: Data storage (not included - you'll need to provide your own market data)
- `environments/`: Trading environment simulators
- `models/`: Neural network model implementations
- `scripts/`: Utility scripts for training and evaluation
- `utils/`: Helper functions and utilities

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Getting Started

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place asset data in `data/{MARKET_NAME}/stocks_data.npy`
   - Place market data in `data/{MARKET_NAME}/market_data.npy`
   - Place rate of return data in `data/{MARKET_NAME}/ror.npy`
   - Place asset relationship matrix in `data/{MARKET_NAME}/relation_matrix.npy`

3. Configure parameters:
   - Edit configuration in `configs/default_config.yaml`

4. Train the model:
   ```
   python scripts/train.py --config configs/default_config.yaml
   ```

5. Evaluate performance:
   ```
   python scripts/evaluate.py --model_path checkpoints/best_model.pth
   ```

## Data Format

The following data files are needed:

| File name | Shape | Description |
|-----------|-------|-------------|
| stocks_data.npy | [num_stocks, num_days, num_features] | Features for individual assets |
| market_data.npy | [num_days, num_features] | Market indicators |
| ror.npy | [num_stocks, num_days] | Rate of return for calculating rewards |
| relation_matrix.npy | [num_stocks, num_stocks] | Relationship matrix for GNN | 