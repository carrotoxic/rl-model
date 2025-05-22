import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class TradingEnvironment(gym.Env):
    """
    A trading environment for portfolio management using reinforcement learning.
    
    This environment simulates portfolio allocation across multiple assets,
    allowing for both long and short positions.
    """
    
    # Define feature indices for clarity
    CLOSE_IDX = 0
    MAX_IDX = 1
    MIN_IDX = 2
    VOLUME_IDX = 3
    
    def __init__(
        self,
        stock_data: np.ndarray,
        window_size: int = 13 * 5,
        initial_balance: float = 10000,
        transaction_fee: float = 0.001,
        holding_period: int = 5 * 5,
        reward_type: str = "sharpe",
        num_assets_select: int = 4,
        use_randomized_starts: bool = True,  # Whether to randomize starting indices
        batch_size: int = 32,
        train_split: float = 0.8,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ):
        """
        Initialize the trading environment.
        
        Args:
            stock_data: Historical data for stocks [num_stocks, num_days, num_features]
                        Features are expected to be [close, max, min, volume]
            window_size: Number of days to use as observation window
            initial_balance: Initial portfolio value
            transaction_fee: Fee for each transaction as percentage
            holding_period: Number of days to hold before rebalancing
            reward_type: Type of reward function to use
            num_assets_select: Number of assets to select for long/short positions
            use_randomized_starts: Whether to randomize starting indices
            batch_size: Number of samples to use for each batch
            train_split: Proportion of data to use for training
            validation_split: Proportion of data to use for validation
            test_split: Proportion of data to use for testing
        """
        super(TradingEnvironment, self).__init__()
        
        self.stock_data = stock_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.holding_period = holding_period
        self.reward_type = reward_type
        self.num_assets_select = num_assets_select
        self.use_randomized_starts = use_randomized_starts
        self.batch_size = batch_size
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Verify we have the expected number of features
        self.num_stocks, self.num_days, self.num_features = stock_data.shape
        if self.num_features != 4:
            raise ValueError(f"Expected 4 features [close, max, min, volume], but got {self.num_features}")
        
        # Calculate returns internally from prices: [num_stocks, num_days]
        self.returns_data = self._calculate_returns_from_prices()
        
        # Calculate weekly features once for efficiency
        self.weekly_features = self._calculate_weekly_features()
        
        # Define action space: asset weights [0, 1] for each asset
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_stocks,), dtype=np.float32
        )
        
        # Define observation space with weekly features
        weekly_feature_count = 4  # Weekly avg close, weekly max, weekly min, weekly avg volume
        num_weeks = self.window_size // 5
        stock_obs_shape = (num_weeks, self.num_stocks, weekly_feature_count)
        portfolio_obs_shape = (self.num_stocks,)  # Current portfolio weights
        
        self.observation_space = spaces.Dict({
            'stocks': spaces.Box(low=-np.inf, high=np.inf, shape=stock_obs_shape),
            'portfolio': spaces.Box(low=0, high=1, shape=portfolio_obs_shape)
        })
        
        # Valid starting indices (ensuring we have enough history and future data)
        valid_indices = list(range(
            self.window_size, 
            self.num_days - self.holding_period
        ))
        
        # Split indices into train, validation, and test sets
        self._create_data_splits(valid_indices)
        
        # Default to training mode
        self.mode = 'train'
        self.valid_start_indices = self.train_indices
        
        self.reset()
    
    def _create_data_splits(self, valid_indices):
        """
        Split valid indices into train, validation, and test sets.
        
        Args:
            valid_indices: List of valid starting indices
        """
        # Shuffle indices
        indices = valid_indices.copy()
        np.random.shuffle(indices)
        
        # Calculate split sizes
        total_size = len(indices)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.validation_split)
        
        # Split indices
        self.train_indices = indices[:train_size]
        self.validation_indices = indices[train_size:train_size + val_size]
        self.test_indices = indices[train_size + val_size:]
        
        print(f"Data split: Train={len(self.train_indices)}, "
              f"Validation={len(self.validation_indices)}, "
              f"Test={len(self.test_indices)}")
    
    def set_mode(self, mode: str):
        """
        Set the environment mode to control which data split is used.
        
        Args:
            mode: One of 'train', 'validation', or 'test'
        """
        if mode not in ['train', 'validation', 'test']:
            raise ValueError(f"Mode must be one of 'train', 'validation', or 'test', got {mode}")
        
        self.mode = mode
        
        # Set valid indices based on mode
        if mode == 'train':
            self.valid_start_indices = self.train_indices
        elif mode == 'validation':
            self.valid_start_indices = self.validation_indices
        else:  # test
            self.valid_start_indices = self.test_indices
            
        print(f"Environment mode set to '{mode}' with {len(self.valid_start_indices)} valid indices")
        
        return self.reset()
    
    def _calculate_returns_from_prices(self) -> np.ndarray:
        """
        Calculate daily returns from close price data.
        
        Returns:
            Returns data array [num_stocks, num_days]
        """
        # Extract close prices for all stocks across all days
        prices = self.stock_data[:, :, self.CLOSE_IDX]  # [num_stocks, num_days]
        
        # Calculate percentage returns: (price_t - price_{t-1}) / price_{t-1}
        returns = np.zeros_like(prices)
        
        # Vectorized calculation of returns
        returns[:, 1:] = np.where(
            prices[:, :-1] > 0,  # Check for valid previous prices
            (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1],
            0  # Use 0 for invalid prices
        )
        
        # First day has no return
        returns[:, 0] = 0
        
        return returns
    
    def _calculate_weekly_features(self) -> np.ndarray:
        """
        Calculate weekly features for all stocks efficiently.
        
        Returns:
            Weekly features array [num_stocks, num_weeks, 4]
            Features: 
                0: close_t / close_(t-1) - Momentum / price trend
                1: max(high) / close - Upper volatility
                2: min(low) / close - Lower volatility
                3: sum(volume) - Raw trading activity
        """
        # Calculate number of full weeks
        num_weeks = self.num_days // 5
        
        # Reshape data to group by weeks - this makes calculations more efficient
        # Create a view where possible to avoid copying data
        days_to_use = num_weeks * 5  # Only use complete weeks
        
        # Extract features
        close_prices = self.stock_data[:, :days_to_use, self.CLOSE_IDX]
        max_prices = self.stock_data[:, :days_to_use, self.MAX_IDX]
        min_prices = self.stock_data[:, :days_to_use, self.MIN_IDX]
        volumes = self.stock_data[:, :days_to_use, self.VOLUME_IDX]
        
        # Reshape to [num_stocks, num_weeks, 5] for each feature
        close_weekly = close_prices.reshape(self.num_stocks, num_weeks, 5)
        max_weekly = max_prices.reshape(self.num_stocks, num_weeks, 5)
        min_weekly = min_prices.reshape(self.num_stocks, num_weeks, 5)
        volume_weekly = volumes.reshape(self.num_stocks, num_weeks, 5)
        
        # Initialize features array
        weekly_features = np.zeros((self.num_stocks, num_weeks, 4))
        
        # 1. Momentum / price trend: close_t / close_(t-1)
        # Get last day close of each week
        last_day_close = close_weekly[:, :, -1]  # [num_stocks, num_weeks]
        
        # Initialize first week with 1 (no previous week)
        weekly_features[:, 0, 0] = 1.0
        
        # Calculate ratio for subsequent weeks
        for w in range(1, num_weeks):
            prev_week_close = last_day_close[:, w-1]
            curr_week_close = last_day_close[:, w]
            # Avoid division by zero
            valid_stocks = prev_week_close > 0
            weekly_features[valid_stocks, w, 0] = curr_week_close[valid_stocks] / prev_week_close[valid_stocks]
        
        # 2. Upper volatility: max(high) / close
        weekly_max = np.max(max_weekly, axis=2)  # Maximum high price in the week
        
        # Calculate ratios for each stock and week, avoiding division by zero
        for s in range(self.num_stocks):
            for w in range(num_weeks):
                if last_day_close[s, w] > 0:
                    weekly_features[s, w, 1] = weekly_max[s, w] / last_day_close[s, w]
                else:
                    weekly_features[s, w, 1] = 1.0  # Default to 1.0 for invalid data
        
        # 3. Lower volatility: min(low) / close
        weekly_min = np.min(min_weekly, axis=2)  # Minimum low price in the week
        for s in range(self.num_stocks):
            for w in range(num_weeks):
                if last_day_close[s, w] > 0:
                    weekly_features[s, w, 2] = weekly_min[s, w] / last_day_close[s, w]
                else:
                    weekly_features[s, w, 2] = 1.0  # Default to 1.0 for invalid data
        
        # 4. Raw trading activity: sum(volume)
        weekly_features[:, :, 3] = np.sum(volume_weekly, axis=2)
        
        return weekly_features
    
    def reset(self):
        """Reset the environment to initial state with optional random start."""
        # Select random start indices for batch
        if self.use_randomized_starts and len(self.valid_start_indices) > 0:
            self.current_steps = np.random.choice(
                self.valid_start_indices, 
                size=self.batch_size
            )
        else:
            # Default to first valid index if no randomization
            self.current_steps = np.ones(self.batch_size, dtype=int) * self.window_size
        
        # Initialize portfolio values and weights for batch
        self.portfolio_values = np.ones(self.batch_size) * self.initial_balance
        self.portfolio_weights = np.zeros((self.batch_size, self.num_stocks))
        
        # Initialize portfolio history for each environment in batch
        self.portfolio_histories = [[value] for value in self.portfolio_values]
        
        # Get batched observations
        return self._get_observation()
    
    def reset_to_index(self, index: int):
        """Reset to a specific index for reproducible testing."""
        if index < self.window_size or index >= self.num_days - self.holding_period:
            raise ValueError(f"Index {index} is out of valid range [{self.window_size}, {self.num_days - self.holding_period - 1}]")
            
        # Reset all environments in batch to the same index
        self.current_steps = np.ones(self.batch_size, dtype=int) * index
        
        # Initialize portfolio values and weights
        self.portfolio_values = np.ones(self.batch_size) * self.initial_balance
        self.portfolio_weights = np.zeros((self.batch_size, self.num_stocks))
        
        # Initialize portfolio history for each environment in batch
        self.portfolio_histories = [[value] for value in self.portfolio_values]
        
        return self._get_observation()
    
    def step(self, scores: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, List[Dict]]:
        """
        Take a step in the environment using the specified action.
        
        Args:
            scores: Model output scores for assets, either a single array [num_stocks]
                   or a batch of scores [batch_size, num_stocks]
        
        Returns:
            observation: Batched observations
            reward: Batch of rewards [batch_size]
            done: Batch of done flags [batch_size]
            info: List of info dictionaries for each environment in batch
        """
        # Check if scores is for a single environment or batch
        if scores.ndim == 1:
            # Expand to batch dimension
            scores = np.tile(scores, (self.batch_size, 1))
        
        # Initialize arrays for tracking
        rewards = np.zeros(self.batch_size)
        dones = np.zeros(self.batch_size, dtype=bool)
        infos = []
        
        # Process each environment in the batch
        for b in range(self.batch_size):
            # Get current portfolio weights
            current_weights = self.portfolio_weights[b]
            
            # Convert scores to weights by selecting top assets
            target_weights = self._select_assets(scores[b])
            
            # Calculate transaction costs
            transaction_cost = self._calculate_transaction_cost(current_weights, target_weights)
            
            # Update portfolio weights
            self.portfolio_weights[b] = target_weights
            
            # Move to the next period (skip forward by holding period)
            self.current_steps[b] += self.holding_period
            
            # Check if done
            done = self.current_steps[b] >= self.num_days - self.holding_period
            dones[b] = done
            
            # Calculate returns based on price change over holding period
            start_step = self.current_steps[b] - self.holding_period
            end_step = self.current_steps[b]
            
            # Get starting and ending prices
            start_price = self.stock_data[:, start_step, self.CLOSE_IDX]
            end_price = self.stock_data[:, end_step, self.CLOSE_IDX]
            
            # Calculate individual asset returns
            valid_mask = start_price != 0
            individual_returns = np.zeros_like(start_price)
            individual_returns[valid_mask] = (end_price[valid_mask] / start_price[valid_mask]) - 1
            
            # Calculate portfolio return
            portfolio_return = np.dot(target_weights, individual_returns) - transaction_cost
            
            # Update portfolio value
            self.portfolio_values[b] *= (1 + portfolio_return)
            
            # Update portfolio history
            self.portfolio_histories[b].append(self.portfolio_values[b])
            
            # Calculate reward
            rewards[b] = self._calculate_reward(b)
            
            # Store info
            info = {
                'portfolio_value': self.portfolio_values[b],
                'portfolio_return': portfolio_return,
                'transaction_cost': transaction_cost,
                'asset_returns': individual_returns
            }
            infos.append(info)
        
        # Get next observation
        obs = self._get_observation()
        
        return obs, rewards, dones, infos
    
    def _get_observation(self) -> Dict:
        """Get the current observation with weekly features for the batch."""
        # Initialize batch observations
        batch_obs = []
        
        # Process each environment in the batch
        for b in range(self.batch_size):
            # Calculate which weeks are in our window
            start_day = self.current_steps[b] - self.window_size
            end_day = self.current_steps[b]
            
            start_week = start_day // 5
            end_week = (end_day + 4) // 5  # Add 4 to include the current week
            
            # Make sure we don't go beyond our calculated weekly features
            max_week = self.weekly_features.shape[1]
            end_week = min(end_week, max_week)
            
            # Calculate number of weeks to extract
            num_weeks = end_week - start_week
            
            # Ensure we have a consistent window size for all environments
            if num_weeks < self.window_size // 5:
                # Pad with zeros at the beginning if needed
                padding_weeks = (self.window_size // 5) - num_weeks
                
                # Create padded observation
                padded_obs = np.zeros((self.window_size // 5, self.num_stocks, 4))
                
                # Extract available weekly features
                available_obs = self.weekly_features[:, start_week:end_week, :]
                available_obs = np.transpose(available_obs, (1, 0, 2))
                
                # Place available features at the end of the padding
                padded_obs[-num_weeks:] = available_obs
                
                weekly_obs = padded_obs
            else:
                # Extract latest window_size // 5 weeks of features to ensure consistent shape
                effective_start_week = end_week - (self.window_size // 5)
                effective_start_week = max(start_week, effective_start_week)
                
                # Extract weekly features for our window
                weekly_obs = self.weekly_features[:, effective_start_week:end_week, :]
                
                # Convert to the expected shape [weeks, stocks, features]
                weekly_obs = np.transpose(weekly_obs, (1, 0, 2))
            
            batch_obs.append(weekly_obs)
        
        # Stack observations along batch dimension
        stacked_obs = np.stack(batch_obs) #shape: (batch_size, num_weeks, num_stocks, num_features)
        
        return {
            'stocks': stacked_obs,
            'portfolio': self.portfolio_weights
        }
    
    def _select_assets(self, score: np.ndarray) -> np.ndarray:
        """
        Select top assets based on scores and allocate equal weights.
        
        Args:
            score: Asset scores [num_stocks]
            
        Returns:
            Portfolio weights [num_stocks]
        """
        # Sort indices by score (descending)
        sorted_indices = np.argsort(-score)
        
        selected_indices = sorted_indices[:self.num_assets_select]
        
        # Extract top-k scores
        top_scores = score[selected_indices]
        
        # Apply softmax only to top-k scores
        exp_scores = np.exp(top_scores - np.max(top_scores))  # stability trick
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Create full weight vector
        weights = np.zeros_like(score)
        weights[selected_indices] = softmax_weights
        
        return weights
    
    def _calculate_transaction_cost(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Calculate transaction cost for rebalancing."""
        turnover = np.sum(np.abs(new_weights - old_weights))
        return turnover * self.transaction_fee
    
    def _calculate_reward(self, batch_idx: int) -> float:
        """
        Calculate reward for a specific environment in the batch.
        
        Args:
            batch_idx: Index in the batch
        
        Returns:
            Reward value
        """
        history = self.portfolio_histories[batch_idx]
        
        if len(history) < 2:
            return 0.0
        
        if self.reward_type == "return":
            # Simple return
            return (history[-1] / history[-2]) - 1
            
        elif self.reward_type == "sharpe":
            # Sharpe ratio (using only recent history to approximate)
            if len(history) < 30:
                return 0.0
            
            recent_returns = np.diff(history[-30:]) / history[-31:-1]
            if np.std(recent_returns) == 0:
                return 0.0
            return np.mean(recent_returns) / np.std(recent_returns)
            
        elif self.reward_type == "sortino":
            # Sortino ratio (using only recent history to approximate)
            if len(history) < 30:
                return 0.0
            
            recent_returns = np.diff(history[-30:]) / history[-31:-1]
            negative_returns = recent_returns[recent_returns < 0]
            if len(negative_returns) == 0 or np.std(negative_returns) == 0:
                return np.mean(recent_returns) * 10  # Reward positive returns with no downside
            return np.mean(recent_returns) / np.std(negative_returns)
            
        elif self.reward_type == "max_drawdown":
            # Negative maximum drawdown
            if len(history) < 5:
                return 0.0
            
            max_value = max(history)
            current_value = history[-1]
            drawdown = (max_value - current_value) / max_value
            
            # We negate drawdown so that minimizing drawdown means maximizing reward
            return -drawdown
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def render(self, mode='human'):
        """Render the environment."""
        pass  # Implement visualization if needed
    
    
if __name__ == "__main__":
    '''
    This is a test script to test the TradingEnvironment class.
    It generates synthetic data and runs a few random trades to test the environment.
    '''
    
    # Generate synthetic data for testing
    print("Generating synthetic data for testing...")
    num_stocks = 10
    num_days = 252  # One trading year
    num_features = 4  # [close, max, min, volume]
    batch_size = 4
    
    # Create random stock data
    np.random.seed(42)
    stock_data = np.zeros((num_stocks, num_days, num_features))
    
    # Generate price data with random walk
    for i in range(num_stocks):
        # Start with base price
        base_price = np.random.uniform(50, 200)
        
        for d in range(num_days):
            if d == 0:
                # First day
                close_price = base_price
                max_price = close_price * (1 + np.random.uniform(0, 0.02))  # Up to 2% higher
                min_price = close_price * (1 - np.random.uniform(0, 0.02))  # Up to 2% lower
            else:
                # Subsequent days - random walk with some correlation to previous day
                prev_close = stock_data[i, d-1, 0]
                daily_return = (np.random.random() - 0.5) * 0.04  # -2% to +2%
                close_price = prev_close * (1 + daily_return)
                
                # High and low prices
                daily_volatility = np.random.uniform(0.005, 0.02)
                max_price = close_price * (1 + daily_volatility)
                min_price = close_price * (1 - daily_volatility)
                
                # Ensure high >= close >= low
                max_price = max(max_price, close_price)
                min_price = min(min_price, close_price)
            
            # Set prices
            stock_data[i, d, 0] = close_price  # Close
            stock_data[i, d, 1] = max_price    # Max
            stock_data[i, d, 2] = min_price    # Min
            
            # Volume - correlate with volatility and price
            vol_base = np.random.uniform(10000, 100000)
            vol_factor = 1 + 5 * abs(max_price - min_price) / close_price  # Higher volatility -> higher volume
            stock_data[i, d, 3] = vol_base * vol_factor
    
    # Create environment with batch size of 4
    print("\nCreating trading environment with batch size of 4...")
    env = TradingEnvironment(
        stock_data=stock_data,
        window_size=20,  # 20 days (4 weeks)
        reward_type="return",
        initial_balance=10000,
        transaction_fee=0.001,
        num_assets_select=3,
        holding_period=5,  # Weekly rebalance
        batch_size=batch_size
    )
    
    print(f"Environment created with {env.num_stocks} stocks and {env.num_days} days")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Weekly features shape: {env.weekly_features.shape}")
    
    # Test reset
    print("\nTesting batch reset...")
    obs = env.reset()
    print(f"Observation shapes:")
    print(f"  stocks: {obs['stocks'].shape}")
    print(f"  portfolio: {obs['portfolio'].shape}")
    print(f"Current steps (should be different for random starts): {env.current_steps}")
    
    # Test step with batch
    print("\nTesting batch step...")
    # Create a random action
    scores = np.random.random(env.num_stocks)
    
    # Take step
    next_obs, rewards, dones, infos = env.step(scores)
    
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
    print(f"Number of info dictionaries: {len(infos)}")
    
    print("\nRewards for each environment in batch:")
    for i, reward in enumerate(rewards):
        print(f"  Env {i}: Reward = {reward:.4f}, Portfolio Value = ${infos[i]['portfolio_value']:.2f}")
    
    # Test reset_to_index
    print("\nTesting reset_to_index (all to same index)...")
    fixed_idx = env.window_size + 10
    obs = env.reset_to_index(fixed_idx)
    print(f"All current steps should be {fixed_idx}: {env.current_steps}")
    
    print("\nTest completed successfully!")
    
    
