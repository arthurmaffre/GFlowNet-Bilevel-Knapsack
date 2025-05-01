"""
train_bilevel.py – Bi-level Knapsack GFlowNet Trainer with Benders Cuts
===============================================================

🚀 ONE-MINUTE PITCH
──────────────────
We tackle the 0-1 knapsack bilevel pricing problem: a leader (Amazon) sets prices,
followers (consumers) choose item bundles under budget constraints.  Instead of
solving followers’ NP‑hard subproblems repeatedly, a single GFlowNet learns to
sample optimal bundles, guided by a critic that estimates the partition function
(log Z), and dynamically generated Benders cuts enforce follower rationality.

0. AUDIENCE GUIDE
────────────────
• 🎓 Economists
  – Think of the leader-follower game as iterative price setting and demand
    estimation.  GFlowNet approximates demand distributions under each pricing.
• 🛠 Operations Research
  – We integrate Benders decomposition: master problem over prices, cuts from
    consumer subproblems.  GFlowNet accelerates subproblem solves via sampling.
• 🤖 ML Engineers
  – Actor-Critic GFlowNet: actor generates bundles, critic predicts logZ.  TB loss
    uses critic output.  Adam optimizer updates both end-to-end.

QUICK-START
──────────
```bash
python train_bilevel.py --data_path data/data.pickle \
    --num_epochs 300 --batch_size 128 \
    --lr 1e-3 --penalty_weight 1.0 --cut_interval 50
```
"""








import argparse
import pickle
from pathlib import Path
import itertools
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb


# ────────────────────────────────────────────────────────────────────────────────
# Local imports – keep them grouped for clarity
# ────────────────────────────────────────────────────────────────────────────────

from models.GFlowNet_v1 import Critic, compute_loss, GFlowNet as GFlownetv3 
from reward.reward import compute_reward

# Mapping from CLI flag to concrete class
MODEL_MAP = {
    "v1": GFlownetv3,
}

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Return the *best* available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def prepare_tensors(data: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Expand 1‑D data arrays to batched tensors on the chosen *device*."""
    u = torch.tensor(data["u"], dtype=torch.float32, device=device).expand(batch_size, -1)
    t = torch.tensor(data["t"], dtype=torch.float32, device=device).expand(batch_size, -1)
    B = torch.tensor(data["B"], dtype=torch.float32, device=device).view(1, 1).expand(batch_size, 1)
    return u.detach(), t.detach(), B.detach(), u.size(1)

# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────

def train(cfg):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare batch data (repeat instance)
    batch_size = cfg.batch_size
    u = torch.tensor(u_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
    t = torch.tensor(t_vals, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
    B = torch.tensor([[B_val]], dtype=torch.float32, device=device).expand(batch_size, 1)
    num_items = u.size(1)

    # instantiate models once
    ModelCls = MODEL_MAP[cfg.model_version]
    actor = ModelCls(num_items=num_items,
                     embedding_dim=cfg.embedding_dim,
                     hidden_dim=cfg.hidden_dim).to(device)
    
    critic = Critic(num_items=num_items,
                    embedding_dim=cfg.embedding_dim,
                    hidden_dim=cfg.hidden_dim).to(device)

    # optimizers
    optimizer_ac = torch.optim.SGD(actor.parameters(), lr=cfg.lr_ac)
    optimizer_cr = torch.optim.SGD(critic.parameters(), lr=cfg.lr_cr)

    # initialize Benders cuts
    cuts: list[tuple[Tensor, float]] = []
    first_cut = benders_master(u_vals, t_vals, B_val, cuts)
    if first_cut:
        cuts.append(first_cut)

    # training loop
    global_step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        actor.train(); critic.train()
        optimizer.zero_grad()

        # generate trajectories
        seq_logp, selected = actor.generate_trajectories(B, u, t,
                                                          cfg.batch_size,
                                                          num_items,
                                                          device)
        reward = compute_reward(selected, u, t, B)

        # predict logZ
        logZ_pred = critic(selected, B, u, t)

        # TB loss
        tb_loss = compute_loss(seq_logp, reward, logZ_pred)

        # Benders penalty
        penalty = torch.tensor(0.0, device=device)
        util = ((u - t) * ((selected + 1) / 2)).sum(dim=1)
        for alpha, beta in cuts:
            violation = torch.relu(util - beta)
            penalty = penalty + violation.mean()
        total_loss = tb_loss + cfg.penalty_weight * penalty

        # backward & step
        total_loss.backward()
        optimizer.step()

        # logging
        if epoch % cfg.log_interval == 0:
            avg_reward = reward.mean().item()
            writer.add_scalar('Loss/TB', tb_loss.item(), epoch)
            writer.add_scalar('Loss/Total', total_loss.item(), epoch)
            writer.add_scalar('Reward/Avg', avg_reward, epoch)
            writer.add_scalar('Penalty', penalty.item(), epoch)

        # generate new cut periodically
        if cfg.cut_interval and epoch % cfg.cut_interval == 0:
            new_cut = benders_master(u_vals, t_vals, B_val, cuts)
            if new_cut:
                cuts.append(new_cut)

    writer.close()
    return actor, critic, cuts


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bi-level GFlowNet Training")
    parser.add_argument('--data_path', type=str, default='data/data.pickle')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--embedding_dim', type=int, default=150)
    parser.add_argument('--hidden_dim', type=int, default=360)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--penalty_weight', type=float, default=1.0)
    parser.add_argument('--cut_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--log_dir', type=str, default='runs')
    cfg = parser.parse_args()

    # load instance
    if not Path(cfg.data_path).exists():
        raise FileNotFoundError(f"Instance file not found: {cfg.data_path}")
    with open(cfg.data_path, 'rb') as f:
        data = pickle.load(f)
    u_vals, t_vals, B_val = data['u'], data['t'], data['B']

    train(u_vals, t_vals, B_val, cfg)
