import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolutional Network layer for learning from relationships between assets.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weights and bias
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset weights using Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Expand adjacency matrix for batch processing if needed
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Transform features
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        
        # Graph convolution operation
        output = torch.bmm(adj, support)  # [batch_size, num_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
            
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
        # For each node i, we calculate attention with every other node j
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
        # For a kernel of size k, we need padding of (k-1) to maintain sequence length
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
                rowsum = adj_matrix.sum(dim=1)
                d_inv_sqrt = torch.pow(rowsum, -0.5)
                d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
                adj_matrix = torch.mm(torch.mm(d_mat_inv_sqrt, adj_matrix), d_mat_inv_sqrt)
                self.register_buffer('base_adj', adj_matrix)
            else:
                self.register_buffer('base_adj', torch.eye(num_assets))
            
            # Adaptive adjacency matrix (learnable)
            if use_adaptive_adj:
                self.adaptive_adj = nn.Parameter(torch.FloatTensor(num_assets, num_assets))
                nn.init.constant_(self.adaptive_adj, 0.1)
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Stock features [batch_size, window_size, num_assets, stock_feature_dim]
            
        Returns:
            Asset scores [batch_size, num_assets]
        """
        batch_size = x.size(0)
        
        # Process each time step
        processed_features = []
        for t in range(self.window_size):
            # Get features at time t
            x_t = x[:, t, :, :]  # [batch, assets, features]
            
            # Feature extraction
            features = self.feature_extractor(x_t)  # [batch, assets, hidden]
            
            processed_features.append(features)
            
        # Stack processed features along time dimension
        features = torch.stack(processed_features, dim=1)  # [batch, time, assets, hidden]
        
        # Reshape for temporal convolution
        features = features.reshape(batch_size * self.num_assets, self.window_size, self.hidden_dim)
        
        # Apply temporal convolution
        features = self.temporal_module(features)  # [batch*assets, time, hidden]
        
        # Take the last time step
        features = features[:, -1, :]  # [batch*assets, hidden]
        
        # Reshape back to [batch, assets, hidden]
        features = features.reshape(batch_size, self.num_assets, self.hidden_dim)
        
        # Apply spatial attention if enabled
        attention_weights = None
        if self.use_spatial_attention:
            features, attention_weights = self.spatial_attention(features)
        
        # Apply GCN if enabled
        if self.use_gcn:
            # Compute adjacency matrix
            if self.use_adaptive_adj:
                # Adaptive adjacency = learned matrix + base matrix
                adj = F.relu(self.adaptive_adj) + self.base_adj
                
                # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
                rowsum = adj.sum(dim=1)
                d_inv_sqrt = torch.pow(rowsum, -0.5)
                d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
                adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
            else:
                adj = self.base_adj
                
            # Apply GCN
            features = self.gcn(features, adj)
        
        # Final prediction (score for each asset)
        scores = self.predictor(features).squeeze(-1)  # [batch, assets]
        
        return scores 