import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time steps in the market data.
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        
        # Attention mechanism
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor for dot product attention
        self.scaling_factor = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Register scaling factor to the device of input tensor
        self.scaling_factor = self.scaling_factor.to(x.device)
        
        # Compute query, key and value projections
        q = self.query(x)  # [batch, seq, hidden]
        k = self.key(x)  # [batch, seq, hidden]
        v = self.value(x)  # [batch, seq, hidden]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scaling_factor  # [batch, seq, seq]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, seq, seq]
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)  # [batch, seq, hidden]
        
        return context, attention_weights


class MarketScoringUnit(nn.Module):
    """
    Market Scoring Unit for analyzing market conditions to adjust portfolio allocation.
    """
    def __init__(
        self,
        market_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 13
    ):
        """
        Initialize the Market Scoring Unit.
        
        Args:
            market_feature_dim: Number of market indicators
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            window_size: Size of the input window
        """
        super(MarketScoringUnit, self).__init__()
        
        self.market_feature_dim = market_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size
        
        # Feature extraction for market data
        self.feature_extractor = nn.Sequential(
            nn.Linear(market_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for capturing temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Market regime prediction (binary classification: bull or bear)
        self.regime_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Market features [batch_size, window_size, market_feature_dim]
            
        Returns:
            Market score (between 0 and 1) [batch_size]
        """
        batch_size = x.size(0)
        
        # Feature extraction for each time step
        features = self.feature_extractor(x)  # [batch, window, hidden]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # [batch, window, hidden]
        
        # Apply temporal attention
        context, _ = self.temporal_attention(lstm_out)  # [batch, window, hidden]
        
        # Take the last time step's context vector
        context = context[:, -1, :]  # [batch, hidden]
        
        # Predict market regime
        market_score = self.regime_predictor(context).squeeze(-1)  # [batch]
        
        return market_score 