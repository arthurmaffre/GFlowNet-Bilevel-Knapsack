from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GFlowNet(nn.Module):
    """
    Actor GFlowNet with dynamic budget state embedding.
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 150,
        hidden_dim: int = 360,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        # embeddings
        self.embed_sel = nn.Linear(num_items, embedding_dim)
        self.embed_B   = nn.Linear(1, embedding_dim)
        self.embed_u   = nn.Linear(num_items, embedding_dim)
        self.embed_t   = nn.Linear(num_items, embedding_dim)
        # actor MLP
        dims = [4 * embedding_dim, hidden_dim,
                hidden_dim // 2, hidden_dim // 4, hidden_dim // 8]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()]
        self.mlp_stack = nn.ModuleList(layers)
        self.head      = nn.Linear(dims[-1], 1)
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        selected: Tensor,    # (B, n)
        remaining_B: Tensor, # (B, 1)
        u: Tensor,           # (B, n)
        t: Tensor,           # (B, n)
    ) -> Tensor:
        """Return Bernoulli probabilities for next action."""
        sel_emb = F.relu(self.embed_sel(selected))
        B_emb   = F.relu(self.embed_B(remaining_B))
        u_emb   = F.relu(self.embed_u(u))
        t_emb   = F.relu(self.embed_t(t))
        h = torch.cat((sel_emb, B_emb, u_emb, t_emb), dim=1)
        for layer in self.mlp_stack:
            h = layer(h)
        logits = self.head(h)
        return torch.sigmoid(logits).squeeze(-1)

    def generate_trajectories(
        self,
        B_init: Tensor,
        u: Tensor,
        t: Tensor,
        batch_size: int,
        num_items: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample a batch of trajectories and return (sequence_logp, selected).
        """
        # fixed embeddings

        u_emb = F.relu(self.embed_u(u))
        t_emb = F.relu(self.embed_t(t))
        selected = torch.full((batch_size, num_items), -1.0, device=device)
        logp_acc = torch.zeros(batch_size, device=device)
        remaining_B = B_init.clone()
        for i in range(num_items):
            # forward pass
            sel_emb = F.relu(self.embed_sel(selected))
            B_emb   = F.relu(self.embed_B(remaining_B))
            h = torch.cat((sel_emb, B_emb, u_emb, t_emb), dim=1)
            for layer in self.mlp_stack:
                h = layer(h)
            probs = torch.sigmoid(self.head(h)).squeeze(-1)
            # apply budget mask
            feasible = remaining_B.squeeze(1) >= t[:, i]
            probs = probs * feasible.float()
            dist = torch.distributions.Bernoulli(probs=probs.clamp(1e-8,1-1e-8))
            act = dist.sample()
            logp_acc += dist.log_prob(act)
            # update selected and budget
            new_sel = selected.clone()
            new_sel[:, i] = act * 2 - 1
            selected = new_sel
            remaining_B = remaining_B - (act * t[:, i]).unsqueeze(1)


        return logp_acc, selected


class Critic(nn.Module):
    """
    Critic estimating log Z given the same inputs as actor forward.
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 150,
        hidden_dim: int = 360,
    ) -> None:
        super().__init__()
        # reuse embeddings
        self.embed_B   = nn.Linear(1, embedding_dim)
        self.embed_u   = nn.Linear(num_items, embedding_dim)
        self.embed_t   = nn.Linear(num_items, embedding_dim)
        # critic MLP
        dims = [3 * embedding_dim, hidden_dim,
                hidden_dim // 2, hidden_dim // 4, hidden_dim // 8]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()]
        self.mlp_stack = nn.ModuleList(layers)
        self.head      = nn.Linear(dims[-1], 1)
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        B: Tensor, # (B)
        u: Tensor,           # (num_items)
        t: Tensor,           # (num_items)
    ) -> Tensor:
        """Predict log Z scalar for each sample."""
        B_emb   = F.relu(self.embed_B(B))
        u_emb   = F.relu(self.embed_u(u))
        t_emb   = F.relu(self.embed_t(t))
        h = torch.cat((B_emb, u_emb, t_emb), dim=1)
        for layer in self.mlp_stack:
            h = layer(h)
        return self.head(h).squeeze(-1)


def compute_loss(
    sequence_logp: Tensor,
    reward: Tensor,
    logZ_pred: Tensor,
) -> Tensor:
    """
    TB-loss using externally provided logZ values.

    Args:
        sequence_logp: log probability of trajectories (B,)
        reward: reward values (B,)
        logZ_pred: predicted log Z (B,)
    Returns:
        loss scalar
    """
    log_r = torch.log(reward.clamp_min(1e-6))
    diff = torch.clamp(sequence_logp + logZ_pred - log_r, min=-100.0, max=100.0)
    return diff.square().mean()

