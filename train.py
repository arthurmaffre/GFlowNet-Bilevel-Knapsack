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
from tqdm import tqdm
import wandb
import numpy as np
import pulp as pl


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

# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────

def solve(u_vals, t_vals, B_val, cfg):
    # device
    device = select_device()

    u = np.array(data["u"], dtype=np.float32).flatten()
    t = np.array(data["t"], dtype=np.float32).flatten()
    B = np.array(data["B"], dtype=np.float32).item()
    num_items = u.shape[0]

    # prepare batch data (repeat instance)
    batch_size = cfg.batch_size

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

    # Initialisation du master problem (sans contraintes (4c))
    iteration = 0
    master = pl.LpProblem('CPP_value_function', pl.LpMaximize)

    indices = list(range(num_items))
    #Variables du master
    x = pl.LpVariable.dicts('x', indices, cat='Binary')
    s = pl.LpVariable.dicts('s', indices, lowBound=0)
    L = pl.LpVariable('L')

    M = u.copy()

    t_ma = {}
    for i in indices:
        if t[i] == 5:
            t_ma[i] = pl.LpVariable(f't_{i}', lowBound=0, upBound=M[i])
        else:
            t_ma[i] = t[i]

    # McCormick envelope
    for i in indices:
        master += s[i] <= t[i]
        master += s[i] <= M[i] * x[i]
        master += t[i] - s[i] <= M[i] * (1 - x[i])

    # Objective
    master += pl.lpSum(s[i] for i in indices), 'Leader_Revenue'

    # Budget constraint
    master += pl.lpSum(s[i] for i in indices) <= B

    #contrainte 4c
    master += pl.lpSum(u[i] * x[i] - s[i] for i in indices) == L

    #first cut
    master += L >= 0

    while True:
        iteration += 1
        master.solve()
        
        # Récupère solution actuelle
        t_sol = np.array([t[i].varValue if isinstance(t[i], pl.LpVariable) else t[i] for i in indices])
        x_sol = np.array([x[i].varValue for i in indices])
        L_sol = L.varValue
        
        # Résout follower problem via solver_with_traj
        follower_val_t, x_hat_sol = train_model(u_vals=u, t_vals=t_sol, B_val=B, epochs=cfg.num_epochs, batch_size=batch_size, actor=actor, critic=critic, device=device)
        exit()
        
        # Calcul des valeurs
        master_val = np.sum((u - t_sol) * x_sol)
        
        follower_val = np.sum((u - t_sol) * x_hat_sol)
        print("-------------")
        print(follower_val)
        print(follower_val_t)
        print(master_val)
        print(L.varValue)

        print("-------------")
        print(x_hat_sol)
        print(x_sol)

        print("-------------")
        print(t_sol)
        print(u)




        print(f"Iteration {iteration} → Master: {master_val:.4f}, Follower: {follower_val:.4f}")
        
        # Vérifie faisabilité bilevel
        if follower_val > master_val:
            print("→ Adding new constraint from follower.")
            master += L >= follower_val
        else:
            print("→ Bilevel feasible solution found.")
            break




def train_model(u_vals, t_vals, B_val, epochs, batch_size, actor, critic, device):
    print(f"🚀 Starting training on device: {device} for {epochs} epochs\n")
    # training loop
    global_step = 0

    u_tensor, t_tensor, B_tensor, num_items = prepare_tensors(u_vals, t_vals, B_val, batch_size, device)
    
    for epoch in range(1, epochs + 1):

        logp_cand, selected_cand = actor.generate_trajectories(B_tensor, u_tensor, t_tensor, batch_size, num_items, device)
        print(logp_cand.shape, logp_cand)
        print(selected_cand.shape, selected_cand)
        exit()
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


def prepare_tensors(u: np.ndarray, t: np.ndarray, B: np.ndarray, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Expand 1‑D data arrays to batched tensors on the chosen *device*."""
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device).expand(batch_size, -1)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(batch_size, -1)
    B_tensor = torch.tensor(B, dtype=torch.float32, device=device).view(1, 1).expand(batch_size, 1)
    return u_tensor.detach(), t_tensor.detach(), B_tensor.detach(), u_tensor.size(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bi-level GFlowNet Training")
    parser.add_argument('--data_path', type=str, default='data/data.pickle')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--embedding_dim', type=int, default=150)
    parser.add_argument('--hidden_dim', type=int, default=360)
    
    parser.add_argument('--lr_ac', type=float, default=1e-3)
    parser.add_argument('--lr_cr', type=float, default=1e-3)

    parser.add_argument('--penalty_weight', type=float, default=1.0)
    parser.add_argument('--cut_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--model_version', type=str, default='v1')
    cfg = parser.parse_args()

    # load instance
    if not Path(cfg.data_path).exists():
        raise FileNotFoundError(f"Instance file not found: {cfg.data_path}")
    with open(cfg.data_path, 'rb') as f:
        data = pickle.load(f)
    u_vals, t_vals, B_val = data['u'], data['t'], data['B']

    solve(u_vals, t_vals, B_val, cfg)
