import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from models.asset_scoring_unit import AssetScoringUnit
from models.market_scoring_unit import MarketScoringUnit


class DeepTrader(nn.Module):
    """
    DeepTrader: A Deep Reinforcement Learning Approach to Risk-Return Balanced
    Portfolio Management with Market Conditions Embedding.
    
    This model combines an Asset Scoring Unit (ASU) and a Market Scoring Unit (MSU)
    to create a portfolio that adapts to market conditions while balancing risk and return.
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
        num_assets_select: int = 4
    ):
        """
        Initialize the DeepTrader model.
        
        Args:
            stock_feature_dim: Number of features for each stock
            market_feature_dim: Number of market indicators
            num_assets: Number of assets in the portfolio
            window_size: Size of the input window
            hidden_dim: Dimension of hidden layers
            use_market_unit: Whether to use the Market Scoring Unit
            use_spatial_attention: Whether to use spatial attention in ASU
            use_gcn: Whether to use graph convolutional network in ASU
            relation_matrix: Relationship matrix between stocks
            use_adaptive_adj: Whether to use adaptive adjacency matrix in GCN
            num_assets_select: Number of assets to select for long/short positions
        """
        super(DeepTrader, self).__init__()
        
        self.stock_feature_dim = stock_feature_dim
        self.market_feature_dim = market_feature_dim
        self.num_assets = num_assets
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.use_market_unit = use_market_unit
        self.num_assets_select = num_assets_select
        
        # Asset Scoring Unit
        self.asu = AssetScoringUnit(
            stock_feature_dim=stock_feature_dim,
            hidden_dim=hidden_dim,
            num_assets=num_assets,
            window_size=window_size,
            use_spatial_attention=use_spatial_attention,
            use_gcn=use_gcn,
            relation_matrix=relation_matrix,
            use_adaptive_adj=use_adaptive_adj
        )
        
        # Market Scoring Unit (optional)
        if use_market_unit:
            self.msu = MarketScoringUnit(
                market_feature_dim=market_feature_dim,
                hidden_dim=hidden_dim,
                window_size=window_size
            )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: Dictionary containing:
                - 'stocks': Stock features [batch_size, window_size, num_assets, stock_feature_dim]
                - 'market': Market features [batch_size, window_size, market_feature_dim]
                - 'portfolio': Current portfolio weights [batch_size, num_assets]
                
        Returns:
            Action values for each asset [batch_size, num_assets]
        """
        batch_size = observations['stocks'].size(0)
        
        # Get asset scores from Asset Scoring Unit
        asset_scores = self.asu(observations['stocks'])  # [batch, assets]
        
        # Get market score from Market Scoring Unit (if enabled)
        if self.use_market_unit:
            market_score = self.msu(observations['market'])  # [batch]
            
            # Adjust asset scores based on market score
            # Market score close to 1 means bullish market (favor long positions)
            # Market score close to 0 means bearish market (favor short positions)
            market_bias = (2 * market_score - 1).unsqueeze(-1)  # Scale to [-1, 1] and expand
            
            # Apply market bias to asset scores
            # In bullish market, keep asset scores as they are (favor long)
            # In bearish market, invert asset scores (favor short)
            adjusted_scores = asset_scores * market_bias
        else:
            adjusted_scores = asset_scores
        
        return adjusted_scores
    
    def get_portfolio_weights(self, action_values: torch.Tensor) -> torch.Tensor:
        """
        Convert action values to portfolio weights.
        
        Args:
            action_values: Action values for each asset [batch_size, num_assets]
            
        Returns:
            Portfolio weights [batch_size, num_assets]
        """
        batch_size = action_values.size(0)
        
        # Sort assets by their action values
        sorted_indices = torch.argsort(action_values, dim=1)
        
        # Initialize portfolio weights
        weights = torch.zeros_like(action_values)
        
        # Get indices for long and short positions
        short_indices = sorted_indices[:, :self.num_assets_select]  # Bottom assets for short
        long_indices = sorted_indices[:, -self.num_assets_select:]  # Top assets for long
        
        # Assign equal weights to selected assets
        for b in range(batch_size):
            weights[b, long_indices[b]] = 1.0 / self.num_assets_select
            weights[b, short_indices[b]] = -1.0 / self.num_assets_select
        
        return weights 