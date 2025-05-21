import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from collections import deque
import random

from models.deep_trader import DeepTrader


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)


class DeepTraderAgent:
    """
    Deep Trader agent using Deep Q-Network for portfolio management.
    """
    def __init__(
        self,
        stock_feature_dim: int,
        market_feature_dim: int,
        num_assets: int,
        window_size: int = 13,
        hidden_dim: int = 64,
        use_market_unit: bool = True,
        use_spatial_attention: bool = True,
        use_gcn: bool = True,
        relation_matrix: Optional[np.ndarray] = None,
        use_adaptive_adj: bool = True,
        num_assets_select: int = 4,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        replay_buffer_size: int = 10000,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Deep Trader agent.
        
        Args:
            stock_feature_dim: Number of features for each stock
            market_feature_dim: Number of market indicators
            num_assets: Number of assets in the portfolio
            window_size: Size of the input window
            hidden_dim: Dimension of hidden layers
            use_market_unit: Whether to use Market Scoring Unit
            use_spatial_attention: Whether to use spatial attention in ASU
            use_gcn: Whether to use graph convolutional network in ASU
            relation_matrix: Relationship matrix between stocks
            use_adaptive_adj: Whether to use adaptive adjacency matrix in GCN
            num_assets_select: Number of assets to select for long/short positions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for RL
            replay_buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            device: Device to run the model on
        """
        self.stock_feature_dim = stock_feature_dim
        self.market_feature_dim = market_feature_dim
        self.num_assets = num_assets
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_assets_select = num_assets_select
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        
        # Create policy network
        self.policy_net = DeepTrader(
            stock_feature_dim=stock_feature_dim,
            market_feature_dim=market_feature_dim,
            num_assets=num_assets,
            window_size=window_size,
            hidden_dim=hidden_dim,
            use_market_unit=use_market_unit,
            use_spatial_attention=use_spatial_attention,
            use_gcn=use_gcn,
            relation_matrix=relation_matrix,
            use_adaptive_adj=use_adaptive_adj,
            num_assets_select=num_assets_select
        ).to(device)
        
        # Create target network (for stable training)
        self.target_net = DeepTrader(
            stock_feature_dim=stock_feature_dim,
            market_feature_dim=market_feature_dim,
            num_assets=num_assets,
            window_size=window_size,
            hidden_dim=hidden_dim,
            use_market_unit=use_market_unit,
            use_spatial_attention=use_spatial_attention,
            use_gcn=use_gcn,
            relation_matrix=relation_matrix,
            use_adaptive_adj=use_adaptive_adj,
            num_assets_select=num_assets_select
        ).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set to evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Track training iterations
        self.training_iterations = 0
        
    def select_action(self, state: Dict[str, np.ndarray], epsilon: float = 0.0) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            epsilon: Probability of selecting a random action
            
        Returns:
            Selected action (portfolio weights)
        """
        # Convert state to tensors and move to device
        state_tensors = {}
        for key, value in state.items():
            state_tensors[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Random action: randomly assign +1 to top k assets and -1 to bottom k assets
            action_values = np.random.randn(self.num_assets)
        else:
            # Greedy action: use policy network
            with torch.no_grad():
                action_values = self.policy_net(state_tensors).cpu().numpy()[0]
        
        return action_values
    
    def train(self, target_update_freq: int = 10) -> float:
        """
        Train the agent using experiences from the replay buffer.
        
        Args:
            target_update_freq: Frequency of target network updates
            
        Returns:
            Average loss for this training batch
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples for training
        
        # Sample batch from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # Unpack transitions
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*transitions)
        
        # Process states and next_states
        state_batch = {}
        next_state_batch = {}
        
        # Process each key in the state dictionary
        for key in batch_states[0].keys():
            # Stack arrays for current states
            state_batch[key] = torch.FloatTensor(
                np.array([s[key] for s in batch_states])
            ).to(self.device)
            
            # Stack arrays for next states
            next_state_batch[key] = torch.FloatTensor(
                np.array([s[key] for s in batch_next_states])
            ).to(self.device)
        
        # Convert other data to tensors
        action_batch = torch.FloatTensor(np.array(batch_actions)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch_dones)).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(state_batch)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            
        # Compute expected Q values
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values.max(1)[0]
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Update target network periodically
        self.training_iterations += 1
        if self.training_iterations % target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_iterations': self.training_iterations
        }, path)
        
    def load(self, path: str):
        """Load model weights from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_iterations = checkpoint['training_iterations'] 