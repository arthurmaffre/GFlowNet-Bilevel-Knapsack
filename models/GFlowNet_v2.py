from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


###############################################################################
# Vectorised GFlowNet for 0‑1 Knapsack                                         #
# – Unifies the per‑item MLP into a single forward pass                       #
# – Removes the costly Python loop over items inside each time‑step           #
###############################################################################

class GFlowNetVector(nn.Module):
    """Actor GFlowNet vectorisé : une passe MLP sur *tous* les items par pas.
    Arguments
    ---------
    num_items : int
        Nombre total d'items du problème (taille du sac à dos).
    embedding_dim : int, optional (default=150)
        Dimension des embeddings internes.
    hidden_dim : int, optional (default=360)
        Taille de la première couche cachée du MLP.
    """

    def __init__(self, num_items: int, embedding_dim_ac_sel: int = 150, embedding_dim_ac_B: int = 150, embedding_dim_ac_u: int = 150, embedding_dim_ac_t: int = 150, hidden_dim_ac: int = 360) -> None:
        super().__init__()
        self.num_items = num_items

        # Embeddings : on traite chaque scalaire (1) -> embedding_dim
        self.embed_sel = nn.Linear(num_items, embedding_dim_ac_sel)
        self.embed_B = nn.Linear(1, embedding_dim_ac_B)  # sera broadcasté sur les items
        self.embed_u = nn.Linear(num_items, embedding_dim_ac_u)
        self.embed_t = nn.Linear(num_items, embedding_dim_ac_t)

        # MLP partagé appliqué à chaque item (via nn.Sequential → dernière dim)
        dims = [(embedding_dim_ac_t + embedding_dim_ac_u + embedding_dim_ac_B + embedding_dim_ac_sel), hidden_dim_ac,
                hidden_dim_ac // 2, hidden_dim_ac // 4, hidden_dim_ac // 8]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)  # (B, n, d) -> (B, n, 1)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------
    # Forward (vectorisé) : renvoie les *logits* de Bernoulli pour chaque item
    # ---------------------------------------------------------------------
    def forward_all(
        self,
        selected: Tensor,     # (B, n)  valeurs -1 (non choisi) / +1 (choisi) / 0 (encore indécis)
        remaining_B: Tensor,  # (B, 1)
        u: Tensor,            # (B, n)
        t: Tensor,            # (B, n)
    ) -> Tensor:

        # Empile (B, n, 1) puis Linear(1 -> E)
        e_sel = F.relu(self.embed_sel(selected))
        e_B = F.relu(self.embed_B(remaining_B))
        e_u = F.relu(self.embed_u(u))
        e_t = F.relu(self.embed_t(t))

        h = torch.cat([e_sel, e_B, e_u, e_t], dim=-1)  # (B, n, 4E)
        h = self.mlp(h)
        logits = self.head(h)
        return logits.squeeze(-1)

    # ---------------------------------------------------------------------
    # Trajectory sampler (séquence)                                         
    # ---------------------------------------------------------------------

    def generate_trajectories(
        self,
        remaining_B: Tensor,
        u: Tensor,
        t: Tensor,
        batch_size: int,
        num_items: int,
        device,
    ) -> tuple[Tensor, Tensor]:
        """
        Génère une trajectoire par batch en suivant la politique actuelle.
        Retourne :
            logp_acc  : log‑probabilité cumulée de la trajectoire    (B,)
            selected  : matrice -1 / +1 finale indiquant les choix   (B, n)
        """
        selected = torch.full((batch_size, num_items), -1.0, device=device, requires_grad=False)
        logp_acc = torch.zeros(batch_size, device=device)

        for i in range(num_items):
            # Calcul des logits pour l’item i uniquement
            logits = self.forward_all(selected, remaining_B, u, t)  # (batch_size, 1)
            probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)              # (batch_size, 1)
            # apply budget mask
            feasible = remaining_B.squeeze(1) >= t[:, i]
            probs = probs * feasible.float()
            dist = torch.distributions.Bernoulli(probs=probs)
            act = dist.sample()
            logp_acc += dist.log_prob(act)
            # update selected and budget
            new_sel = selected.clone()
            new_sel[:, i] = act * 2 - 1
            selected = new_sel
            remaining_B = remaining_B - (act * t[:, i]).unsqueeze(1)

        return logp_acc, selected


###############################################################################
# Critic identique à l'ancienne implémentation                               #
###############################################################################

class Critic(nn.Module):
    """
    Critic estimating log Z given the same inputs as actor forward.
    """
    def __init__(self, num_items: int, embedding_dim_cr_sel: int = 150, embedding_dim_cr_B: int = 150, embedding_dim_cr_u: int = 150, embedding_dim_cr_t: int = 150, hidden_dim_cr: int = 360) -> None:
        super().__init__()

        # Embeddings : on traite chaque scalaire (1) -> embedding_dim
        self.embed_sel = nn.Linear(num_items, embedding_dim_cr_sel)
        self.embed_B = nn.Linear(1, embedding_dim_cr_B)  # sera broadcasté sur les items
        self.embed_u = nn.Linear(num_items, embedding_dim_cr_u)
        self.embed_t = nn.Linear(num_items, embedding_dim_cr_t)
        # reuse embeddings

        self.embed_B   = nn.Linear(1, embedding_dim_cr_B)
        self.embed_u   = nn.Linear(num_items, embedding_dim_cr_u)
        self.embed_t   = nn.Linear(num_items, embedding_dim_cr_t)
        # critic MLP
        dims = [(embedding_dim_cr_t + embedding_dim_cr_B + embedding_dim_cr_u), hidden_dim_cr,
                hidden_dim_cr // 2, hidden_dim_cr // 4, hidden_dim_cr // 8]
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


###############################################################################
# Trajectory Balance Loss                                                     #
###############################################################################

def compute_tb_loss(sequence_logp: Tensor, reward: Tensor, logZ_pred: Tensor) -> Tensor:
    """Loss TB classique.
    Args
    ----
    sequence_logp : (B,)
    reward        : (B,)
    logZ_pred     : (B,)
    """
    log_r = torch.log(reward.clamp_min(1e-6))
    #diff = torch.clamp(sequence_logp + logZ_pred - log_r, min=-100.0, max=100.0)
    return torch.nn.MSELoss(sequence_logp + logZ_pred, log_r)


# ---------------------------------------------------------------------------
# Astuce performance :
#   model = torch.compile(GFlowNetVector(num_items))  # PyTorch ≥ 2.1
# ---------------------------------------------------------------------------
