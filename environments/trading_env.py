import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional


class TradingEnvironment(gym.Env):
    """
    A trading environment for portfolio management using reinforcement learning.
    
    This environment simulates portfolio allocation across multiple assets,
    allowing for both long and short positions.
    """
    
    def __init__(
        self,
        stock_data: np.ndarray,
        market_data: np.ndarray,
        returns_data: np.ndarray,
        window_size: int = 13,
        initial_balance: float = 10000,
        transaction_fee: float = 0.001,
        reward_type: str = "sharpe",
        num_assets_select: int = 4,
    ):
        """
        Initialize the trading environment.
        
        Args:
            stock_data: Historical data for stocks [num_stocks, num_days, num_features]
            market_data: Market indicators data [num_days, num_features]
            returns_data: Rate of return data [num_stocks, num_days]
            window_size: Number of days to use as observation window
            initial_balance: Initial portfolio value
            transaction_fee: Fee for each transaction as percentage
            reward_type: Type of reward function to use
            num_assets_select: Number of assets to select for long/short positions
        """
        super(TradingEnvironment, self).__init__()
        
        self.stock_data = stock_data
        self.market_data = market_data
        self.returns_data = returns_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_type = reward_type
        self.num_assets_select = num_assets_select
        
        self.num_stocks = stock_data.shape[0]
        self.num_days = stock_data.shape[1]
        
        # Define action space: asset weights [-1, 1] for each asset
        # -1 represents max short, 1 represents max long
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32
        )
        
        # Define observation space
        stock_obs_shape = (self.window_size, self.num_stocks, stock_data.shape[2])
        market_obs_shape = (self.window_size, market_data.shape[1])
        portfolio_obs_shape = (self.num_stocks,)  # Current portfolio weights
        
        self.observation_space = spaces.Dict({
            'stocks': spaces.Box(low=-np.inf, high=np.inf, shape=stock_obs_shape),
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=market_obs_shape),
            'portfolio': spaces.Box(low=-1, high=1, shape=portfolio_obs_shape)
        })
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.portfolio_weights = np.zeros(self.num_stocks)
        self.portfolio_history = [self.portfolio_value]
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment using the specified action.
        
        Args:
            action: Portfolio weights for each asset [-1, 1]
            
        Returns:
            observation: Current observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Normalize weights to ensure they sum to 0 (market neutral)
        # This ensures equal allocation to long and short
        target_weights = self._normalize_weights(action)
        
        # Calculate transaction costs
        transaction_cost = self._calculate_transaction_cost(self.portfolio_weights, target_weights)
        
        # Update portfolio weights
        self.portfolio_weights = target_weights
        
        # Move to the next day
        self.current_step += 1
        
        # Calculate returns based on the weights and actual returns
        returns = self.returns_data[:, self.current_step]
        portfolio_return = np.sum(self.portfolio_weights * returns) - transaction_cost
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.current_step >= self.num_days - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> Dict:
        """Get the current observation."""
        # Extract window of data
        stock_obs = self.stock_data[:, self.current_step-self.window_size:self.current_step, :]
        stock_obs = np.transpose(stock_obs, (1, 0, 2))  # [window, stocks, features]
        
        market_obs = self.market_data[self.current_step-self.window_size:self.current_step, :]
        
        return {
            'stocks': stock_obs,
            'market': market_obs,
            'portfolio': self.portfolio_weights
        }
    
    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize weights to ensure market neutrality and limited selection.
        
        This selects top and bottom assets based on the action values 
        and allocates equal weights to create a balanced long/short portfolio.
        """
        # Create a copy to avoid modifying the original action
        weights = action.copy()
        
        # Sort indices by action values
        sorted_indices = np.argsort(weights)
        
        # Select top and bottom assets
        short_indices = sorted_indices[:self.num_assets_select]
        long_indices = sorted_indices[-self.num_assets_select:]
        
        # Zero out all weights
        weights = np.zeros_like(weights)
        
        # Assign equal weights to selected assets
        weights[long_indices] = 1.0 / self.num_assets_select
        weights[short_indices] = -1.0 / self.num_assets_select
        
        return weights
    
    def _calculate_transaction_cost(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Calculate transaction cost for rebalancing."""
        turnover = np.sum(np.abs(new_weights - old_weights))
        return turnover * self.transaction_fee
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on specified reward type."""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        if self.reward_type == "return":
            # Simple return
            return (self.portfolio_history[-1] / self.portfolio_history[-2]) - 1
            
        elif self.reward_type == "sharpe":
            # Sharpe ratio (using only recent history to approximate)
            if len(self.portfolio_history) < 30:
                return 0.0
            
            recent_returns = np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-31:-1]
            if np.std(recent_returns) == 0:
                return 0.0
            return np.mean(recent_returns) / np.std(recent_returns)
            
        elif self.reward_type == "sortino":
            # Sortino ratio (using only recent history to approximate)
            if len(self.portfolio_history) < 30:
                return 0.0
            
            recent_returns = np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-31:-1]
            negative_returns = recent_returns[recent_returns < 0]
            if len(negative_returns) == 0 or np.std(negative_returns) == 0:
                return np.mean(recent_returns) * 10  # Reward positive returns with no downside
            return np.mean(recent_returns) / np.std(negative_returns)
            
        elif self.reward_type == "max_drawdown":
            # Negative maximum drawdown
            if len(self.portfolio_history) < 5:
                return 0.0
            
            max_value = max(self.portfolio_history)
            current_value = self.portfolio_history[-1]
            drawdown = (max_value - current_value) / max_value
            
            # We negate drawdown so that minimizing drawdown means maximizing reward
            return -drawdown
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def render(self, mode='human'):
        """Render the environment."""
        pass  # Implement visualization if needed 