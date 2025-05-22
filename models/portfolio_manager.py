import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import os


class NodeGraphConvolution(nn.Module):
    """
    Node-wise Graph Convolution Layer.

    Performs spatial graph convolution on the node dimension.
    """
    def __init__(self):
        super(NodeGraphConvolution, self).__init__()

    def forward(self, input: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph convolution.

        Args:
            input (Tensor): Input tensor of shape (batch_size, num_nodes, features)
            adjacency_matrix (Tensor): Adjacency matrix of shape (num_nodes, num_nodes)

        Returns:
            Tensor: Output tensor of shape (batch_size, num_nodes, features)
        """
        # Apply graph convolution: X' = A * X
        # For each node, aggregate features from its neighbors according to adjacency matrix
        return torch.bmm(adjacency_matrix.unsqueeze(0).expand(input.size(0), -1, -1), input)


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolutional Network layer for learning from relationships between assets.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(GraphConvolutionLayer, self).__init__()
        
        # input
        self.in_features = in_features
        self.out_features = out_features
        
        # layers
        self.nconv = NodeGraphConvolution()
        self.mlp = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        # parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset weights using Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, input: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Node features [batch_size, num_nodes, in_features]
            adjacency_matrix: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Apply graph convolution
        support = self.nconv(input, adjacency_matrix)
        
        # Apply linear transformation and activation
        output = self.mlp(support)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to learn relationships between assets.
    """
    def __init__(self, hidden_dim: int):
        super(SpatialAttention, self).__init__()
        
        # Attention mechanism using a simple feed-forward network
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, num_nodes, hidden_dim]
            
        Returns:
            Tuple of (updated features, attention matrix)
        """
        batch_size, num_nodes, hidden_dim = x.size()
        
        # Create all pairs of nodes for attention calculation
        nodes_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, nodes, nodes, hidden]
        nodes_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, nodes, nodes, hidden]
        
        # Concatenate features from both nodes
        node_pairs = torch.cat([nodes_i, nodes_j], dim=-1)  # [batch, nodes, nodes, 2*hidden]
        
        # Calculate attention scores
        attention_logits = self.attention(node_pairs).squeeze(-1)  # [batch, nodes, nodes]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention to get updated node features
        updated_features = torch.bmm(attention_weights, x)  # [batch, nodes, hidden]
        
        return updated_features, attention_weights


class TemporalConvModule(nn.Module):
    """
    Temporal Convolutional Network module for capturing time-series patterns.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConvModule, self).__init__()
        
        # Causal padding to ensure output length matches input length
        padding = (kernel_size - 1)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, sequence_length, in_channels]
            
        Returns:
            Processed features [batch_size, sequence_length, out_channels]
        """
        # Transpose for 1D convolution (batch, channels, sequence)
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv(x)
        
        # Transpose back to (batch, sequence, channels)
        x = x.transpose(1, 2)
        
        # Remove the extra padding at the end to maintain original sequence length
        x = x[:, :-2, :]
        
        return x


class AssetScoringUnit(nn.Module):
    """
    Asset Scoring Unit for evaluating individual assets based on historical data
    and their relationships.
    """
    def __init__(
        self,
        stock_feature_dim: int,
        hidden_dim: int = 64,
        num_assets: int = 30,
        window_size: int = 13,
        use_spatial_attention: bool = True,
        use_gcn: bool = True,
        relation_matrix: Optional[np.ndarray] = None,
        use_adaptive_adj: bool = True,
    ):
        """
        Initialize the Asset Scoring Unit.
        
        Args:
            stock_feature_dim: Number of features for each stock
            hidden_dim: Dimension of hidden layers
            num_assets: Number of assets in the portfolio
            window_size: Size of the input window
            use_spatial_attention: Whether to use spatial attention
            use_gcn: Whether to use graph convolutional network
            relation_matrix: Relationship matrix between stocks
            use_adaptive_adj: Whether to use adaptive adjacency matrix
        """
        super(AssetScoringUnit, self).__init__()
        
        self.stock_feature_dim = stock_feature_dim
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        self.window_size = window_size
        self.use_spatial_attention = use_spatial_attention
        self.use_gcn = use_gcn
        self.use_adaptive_adj = use_adaptive_adj
        
        # Feature extraction for stock data
        self.feature_extractor = nn.Sequential(
            nn.Linear(stock_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal module to capture time-series patterns
        self.temporal_module = TemporalConvModule(hidden_dim, hidden_dim)
        
        # Spatial attention module
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(hidden_dim)
        
        # Graph Convolutional Network
        if use_gcn:
            self.gcn = GraphConvolutionLayer(hidden_dim, hidden_dim)
            
            # Initialize adjacency matrix from relation matrix if provided
            if relation_matrix is not None:
                adj_matrix = torch.FloatTensor(relation_matrix)
                # Add self-loops (diagonal = 1)
                adj_matrix = adj_matrix + torch.eye(num_assets)
                # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
                rowsum = adj_matrix.sum(1)
                d_inv_sqrt = torch.pow(rowsum, -0.5)
                d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
                adj_matrix = adj_matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
                
                self.register_buffer('adj_matrix', adj_matrix)
            else:
                # If no relation matrix provided, use identity matrix as adjacency
                self.register_buffer('adj_matrix', torch.eye(num_assets))
        
        # Create adaptive adjacency matrix if specified
        if use_adaptive_adj:
            self.adaptive_adj = nn.Parameter(torch.zeros(num_assets, num_assets))
            nn.init.xavier_uniform_(self.adaptive_adj)
            
        # Final MLP for asset scoring
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, window_size, num_assets, stock_feature_dim]
            
        Returns:
            Asset scores [batch_size, num_assets]
        """
        batch_size, window_size, num_assets, _ = x.shape
        
        # Reshape for feature extraction
        x_reshaped = x.reshape(batch_size * window_size * num_assets, -1)
        
        # Extract features
        features = self.feature_extractor(x_reshaped)
        
        # Reshape back
        features = features.reshape(batch_size, window_size, num_assets, -1)
        
        # Transpose to [batch, assets, window, features] for asset-centric processing
        features = features.permute(0, 2, 1, 3)
        
        # Reshape for temporal processing
        features_temporal = features.reshape(batch_size * num_assets, window_size, -1)
        
        # Apply temporal module
        features_temporal = self.temporal_module(features_temporal)
        
        # Reshape back
        features = features_temporal.reshape(batch_size, num_assets, -1, self.hidden_dim)
        
        # Take the last temporal output as the asset representation
        asset_features = features[:, :, -1, :]  # [batch, assets, hidden]
        
        # Apply spatial attention if specified
        if self.use_spatial_attention:
            asset_features, _ = self.spatial_attention(asset_features)
        
        # Apply GCN if specified
        if self.use_gcn:
            # Combine predefined adjacency matrix with adaptive one if needed
            if self.use_adaptive_adj:
                # Create symmetric adjacency by averaging with transpose
                adaptive_adj = 0.5 * (self.adaptive_adj + self.adaptive_adj.transpose(0, 1))
                # Apply softmax to make it a valid transition matrix
                adaptive_adj = F.softmax(adaptive_adj, dim=1)
                # Combine with predefined adjacency
                combined_adj = self.adj_matrix + adaptive_adj
            else:
                combined_adj = self.adj_matrix
                
            # Apply GCN
            asset_features = self.gcn(asset_features, combined_adj)
        
        # Score each asset
        scores = self.score_mlp(asset_features).squeeze(-1)  # [batch, assets]
        
        return scores


class PortfolioManager:
    """
    Portfolio manager using Asset Scoring Unit for asset selection and portfolio construction.
    """
    def __init__(
        self,
        stock_feature_dim: int,
        num_assets: int,
        window_size: int = 13,
        hidden_dim: int = 64,
        use_spatial_attention: bool = True,
        use_gcn: bool = True,
        relation_matrix: Optional[np.ndarray] = None,
        use_adaptive_adj: bool = True,
        num_assets_select: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the portfolio manager.
        
        Args:
            stock_feature_dim: Number of features for each stock
            num_assets: Number of assets in the portfolio
            window_size: Size of the input window
            hidden_dim: Dimension of hidden layers
            use_spatial_attention: Whether to use spatial attention in ASU
            use_gcn: Whether to use graph convolutional network in ASU
            relation_matrix: Relationship matrix between stocks
            use_adaptive_adj: Whether to use adaptive adjacency matrix in GCN
            num_assets_select: Number of assets to select for long positions
            device: Device to run the model on
        """
        self.stock_feature_dim = stock_feature_dim
        self.num_assets = num_assets
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_assets_select = num_assets_select
        self.device = device
        
        # Create Asset Scoring Unit
        self.asu = AssetScoringUnit(
            stock_feature_dim=stock_feature_dim,
            hidden_dim=hidden_dim,
            num_assets=num_assets,
            window_size=window_size,
            use_spatial_attention=use_spatial_attention,
            use_gcn=use_gcn,
            relation_matrix=relation_matrix,
            use_adaptive_adj=use_adaptive_adj
        ).to(device)
        
    def predict_scores(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict asset scores using the Asset Scoring Unit.
        
        Args:
            state: Dictionary containing 'stocks' key with stock data
                  [window_size, num_assets, stock_feature_dim]
            
        Returns:
            Asset scores [num_assets]
        """
        # Convert state to tensor and move to device
        stock_data = torch.FloatTensor(state['stocks']).unsqueeze(0).to(self.device)
        
        # Get scores from ASU
        with torch.no_grad():
            scores = self.asu(stock_data).cpu().numpy()[0]
        
        return scores
    
    def get_portfolio_weights(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert asset scores to portfolio weights.
        
        Args:
            scores: Asset scores [num_assets]
            
        Returns:
            Portfolio weights [num_assets]
        """
        # Sort assets by scores
        sorted_indices = np.argsort(-scores)
        selected_indices = sorted_indices[:self.num_assets_select]
        
        # Extract top-k scores
        top_scores = scores[selected_indices]
        
        # Apply softmax only to top-k scores
        exp_scores = np.exp(top_scores - np.max(top_scores))  # stability trick
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Create full weight vector
        weights = np.zeros_like(scores)
        weights[selected_indices] = softmax_weights
        
        return weights
        
    
    def save(self, path: str):
        """Save model weights to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.asu.state_dict(), path)
        
    def load(self, path: str):
        """Load model weights from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
            
        self.asu.load_state_dict(torch.load(path, map_location=self.device))


def test_portfolio_manager():
    """
    Test the portfolio manager with synthetic data.
    """
    print("Testing PortfolioManager with synthetic data...")
    
    # Parameters
    num_assets = 10
    window_size = 13
    stock_feature_dim = 4
    num_assets_select = 3
    
    # Generate synthetic data - shape: [window_size, num_assets, stock_feature_dim]
    np.random.seed(42)
    stock_data = np.random.randn(window_size, num_assets, stock_feature_dim)
    
    # Create portfolio manager
    manager = PortfolioManager(
        stock_feature_dim=stock_feature_dim,
        num_assets=num_assets,
        window_size=window_size,
        num_assets_select=num_assets_select,
        device="cpu",
        use_gcn=False,  # Disable GCN for simple testing
        use_spatial_attention=False  # Disable spatial attention for simple testing
    )
    
    # Create state
    state = {'stocks': stock_data}
    
    # Get scores
    scores = manager.predict_scores(state)
    print(f"Asset scores: {scores}")
    
    # Get portfolio weights
    weights = manager.get_portfolio_weights(scores)
    print(f"Portfolio weights: {weights}")
    
    # Verify weights sum to 1
    weight_sum = np.sum(weights)
    print(f"Sum of weights: {weight_sum}")
    
    # Verify correct number of assets selected
    num_selected = np.sum(weights > 0)
    print(f"Number of assets selected: {num_selected}")
    
    # Visualize scores and weights
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(num_assets), scores)
    plt.title('Asset Scores')
    plt.xlabel('Asset')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(num_assets), weights)
    plt.title('Portfolio Weights')
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    
    plt.tight_layout()
    plt.savefig('portfolio_manager_test.png')
    print("Test results saved to portfolio_manager_test.png")
    
    return scores, weights


if __name__ == "__main__":
    test_portfolio_manager() 