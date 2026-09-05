"""UrbanMARL Graph Neural Network (GNN) Message-Passing Policy Module.

Provides cooperative inter-agent message-passing embeddings for multi-UAV swarms:

- Dynamically computes distance-thresholded inter-UAV communication graphs:
  A_ij = I(||p_i - p_j|| <= R_comm).
- Vectorized edge-feature aggregation:
  m_ij = MLP([h_j, p_j - p_i, v_j - v_i]).
- Permutation-invariant node feature updates:
  h_i' = MLP([h_i, sum_{j in N_i} m_ij]).
- Compatible with decentralized actor-critic architectures in CTDE MARL.
"""

from typing import Optional

import torch
import torch.nn as nn


class VectorizedGNNMessagePassing(nn.Module):
    """Vectorized multi-agent Graph Neural Network message-passing module.

    Attributes:
        node_dim (int): Dimension of individual agent observation vectors.
        hidden_dim (int): Hidden dimension for message and update networks.
        out_dim (int): Output embedding dimension per agent.
        r_comm (float): Inter-UAV communication radius in meters.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        r_comm: float = 120.0,
    ) -> None:
        """Initializes the GNN message-passing module.

        Args:
            node_dim (int): Input agent feature dimension.
            hidden_dim (int): Hidden layer size. Defaults to 64.
            out_dim (int): Output feature embedding size. Defaults to 64.
            r_comm (float): Maximum inter-UAV communication range in meters. Defaults to 120.0.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.r_comm = r_comm

        # Message network: takes [h_j, relative_position (3), relative_velocity (3)]
        edge_input_dim = node_dim + 3 + 3
        self.msg_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Node update network: takes [h_i, aggregated_messages]
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes vectorized graph message-passing across agents.

        Args:
            node_features (torch.Tensor): Agent features of shape (B, N, node_dim).
            positions (torch.Tensor): Agent 3D coordinates of shape (B, N, 3).
            velocities (Optional[torch.Tensor]): Agent 3D velocities of shape (B, N, 3).

        Returns:
            torch.Tensor: Updated cooperative agent embeddings of shape (B, N, out_dim).
        """
        B, N, _ = node_features.shape
        device = node_features.device

        if velocities is None:
            velocities = torch.zeros_like(positions)

        # Pairwise relative displacements: shape (B, N, N, 3)
        # diff[b, i, j] = pos[b, j] - pos[b, i]
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(2)
        rel_vel = velocities.unsqueeze(1) - velocities.unsqueeze(2)

        # Pairwise distances: shape (B, N, N)
        dists = torch.norm(rel_pos, dim=-1)

        # Adjacency matrix: connected if within R_comm and not self-loop
        adj = (dists <= self.r_comm) & (
            ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        )
        adj_weight = adj.float().unsqueeze(-1)  # (B, N, N, 1)

        # Expand node features of neighbor j: shape (B, N, N, node_dim)
        # h_j_exp[b, i, j] = node_features[b, j]
        h_j_exp = node_features.unsqueeze(1).expand(B, N, N, self.node_dim)

        # Edge features: [h_j, rel_pos, rel_vel] -> (B, N, N, edge_input_dim)
        edge_feat = torch.cat([h_j_exp, rel_pos, rel_vel], dim=-1)

        # Compute raw messages: (B, N, N, hidden_dim)
        raw_msgs = self.msg_mlp(edge_feat)

        # Masked message aggregation: sum over neighbor dimension j
        masked_msgs = raw_msgs * adj_weight
        agg_msgs = masked_msgs.sum(dim=2)  # (B, N, hidden_dim)

        # Update node representations: [h_i, agg_msgs] -> (B, N, out_dim)
        combined = torch.cat([node_features, agg_msgs], dim=-1)
        updated_nodes = self.update_mlp(combined)

        return updated_nodes
