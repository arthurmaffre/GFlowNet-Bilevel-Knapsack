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
from tqdm.auto import trange
import wandb
import numpy as np
import pulp as pl
import time


# ────────────────────────────────────────────────────────────────────────────────
# Local imports – keep them grouped for clarity
# ────────────────────────────────────────────────────────────────────────────────

from models.GFlowNet_v1 import compute_loss, GFlowNet as GFlownet_v1
from models.GFlowNet_v2 import Critic, GFlowNetVector as GFlownet_v2
from reward.reward import compute_reward
from solver.solver_with_traj import compute_analytical_reward

# Mapping from CLI flag to concrete class
MODEL_MAP = {
    "v1": GFlownet_v1,
    "v2": GFlownet_v2,
}
# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Return the *best* available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.1)
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
    #device = torch.device("cpu")

    print(device)

    u = np.array(data["u"], dtype=np.float32).flatten()
    t = np.array(data["t"], dtype=np.float32).flatten()
    B = np.array(data["B"], dtype=np.float32).item()
    num_items = u.shape[0]

    # prepare batch data (repeat instance)
    batch_size = cfg.batch_size

    # instantiate models once
    ModelCls = MODEL_MAP[cfg.model_version]
    actor = ModelCls(
        num_items=num_items,
        embedding_dim_ac_sel=cfg.embedding_dim_ac_sel,
        embedding_dim_ac_B  =cfg.embedding_dim_ac_B,
        embedding_dim_ac_u  =cfg.embedding_dim_ac_u,
        embedding_dim_ac_t  =cfg.embedding_dim_ac_t,
        hidden_dim_ac       =cfg.hidden_dim_ac,
    ).to(device)
    
    critic = Critic(
        num_items=num_items,
        embedding_dim_cr_sel=cfg.embedding_dim_cr_sel,
        embedding_dim_cr_B  =cfg.embedding_dim_cr_B,
        embedding_dim_cr_u  =cfg.embedding_dim_cr_u,
        embedding_dim_cr_t  =cfg.embedding_dim_cr_t,
        hidden_dim_cr       =cfg.hidden_dim_cr,
    ).to(device)
    
    #actor  = torch.compile(actor)
    #critic = torch.compile(critic)


    # optimizers
    optimizer_ac = torch.optim.SGD(actor.parameters(), lr=cfg.lr_ac, momentum=cfg.mom_ac)
    optimizer_cr = torch.optim.SGD(critic.parameters(), lr=cfg.lr_cr, momentum=cfg.mom_cr)

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
        master += s[i] <= t_ma[i]
        master += s[i] <= M[i] * x[i]
        master += t_ma[i] - s[i] <= M[i] * (1 - x[i])

    # Objective
    master += pl.lpSum(s[i] for i in indices), 'Leader_Revenue'

    # Budget constraint
    master += pl.lpSum(s[i] for i in indices) <= B

    #contrainte 4c
    master += pl.lpSum(u[i] * x[i] - s[i] for i in indices) == L

    #first cut
    master += L >= 0
    tolerance = 1e-4
    no_cut_streak = 0        # ← NEW
    start_time = time.time()
    while True:
        iteration += 1
        master.solve()
        
        # Récupère solution actuelle
        t_sol = np.array([t_ma[i].varValue if isinstance(t_ma[i], pl.LpVariable) else t_ma[i] for i in indices])
        x_sol = np.array([x[i].varValue for i in indices])
        L_sol = L.varValue
        
        # Résout follower problem via solver_with_traj
        follower_val_t, x_hat_sol = train_model(u_vals=u, t_vals=t_sol, B_val=B, epochs=cfg.num_epochs, batch_size=batch_size, actor=actor, critic=critic, optimizer_ac=optimizer_ac, optimizer_cr=optimizer_cr, device=device)
        #follower_val_t_an, x_hat_sol_an = compute_analytical_reward(u_vals=u, t_vals=t_sol, B_val=B)

        # Calcul des valeurs
        master_val = np.sum((u - t_sol) * x_sol)
        
        follower_val = np.sum((u - t_sol) * x_hat_sol)
        




       # ----- log d’itération synthétique -----
        gap = follower_val - master_val
        
        wandb.log({
             "iteration": iteration,
             "master_val": master_val,
             "follower_val": follower_val,
             "gap": gap,
         })
        
        print(
            f"Iteration {iteration} → "
            f"Master: {master_val:.6f}, "
            f"Follower(seq): {follower_val:.6f}, "
            #f"Follower(analytic): {follower_val_t_an:.6f}, "
            f"Gap: {gap:+.6f}"
        )
        cut_added = False
        # Vérifie faisabilité bilevel
        if follower_val > master_val + tolerance:
            print("→ Adding new constraint from follower.")
            master += pl.lpSum(u[i] * x[i] - s[i] for i in indices) >= follower_val
            cut_added = True          # ← NEW
            no_cut_streak = 0         # ← NEW (reset)

        elif abs(follower_val - master_val) <= tolerance:
            print("→ Convergence reached (follower == master within tolerance).")
            break

        if not cut_added:
            no_cut_streak += 1

        if no_cut_streak >= 5:
            msg = f"❌ 5 iterations sans nouvelle coupe – run marqué invalide."
            print(msg)

            wandb.run.summary["valid"] = False     # tag dans le résumé
            wandb.run.summary["reason"] = "no_new_cuts_15"

            # termine le run avec exit_code ≠ 0 pour que le sweep l’ignore
            wandb.finish(exit_code=1)

            # lève une exception pour stopper proprement le script
            raise RuntimeError(msg)


    return master_val, follower_val, iteration, start_time




def train_model(u_vals, t_vals, B_val, epochs, batch_size, actor, critic, optimizer_ac, optimizer_cr, device):
    print(f"🚀 Starting training on device: {device} for {epochs} epochs\n")
    # training loop

    u_tensor, t_tensor, B_tensor, num_items = prepare_tensors(u_vals, t_vals, B_val, batch_size, device) # tensor size ([num_items])

    best_reward = -float("inf")
    best_sequence = None

# Boucle principale avec barre de progression
    progress = trange(1, epochs + 1, desc="Training", unit="epoch")
    for epoch in progress:
        optimizer_ac.zero_grad(set_to_none=True)
        optimizer_cr.zero_grad(set_to_none=True)

        log_Z_cand = critic(B_tensor, u_tensor, t_tensor)

        # Génération de trajectoires
        logp_cand, selected_cand = actor.generate_trajectories(
            B_tensor.expand(batch_size, 1).detach(),
            u_tensor.expand(batch_size, -1).detach(),
            t_tensor.expand(batch_size, -1).detach(),
            batch_size,
            num_items,
            device,
        )

        reward = compute_reward(selected_cand, u_tensor, t_tensor, B_tensor, num_items)
        loss = compute_loss(logp_cand, reward, log_Z_cand)

        loss.backward()

        optimizer_ac.step()
        optimizer_cr.step()

        # Stats pour la barre de progression
        with torch.no_grad():
            batch_max_reward, max_idx = reward.max(dim=0)
            if batch_max_reward.item() > best_reward:
                best_reward = batch_max_reward.item()
                # convertit : -1 → 0, 1 → 1
                best_sequence = (selected_cand[max_idx]
                                .detach()
                                .cpu()
                                .clone())
                best_sequence[best_sequence == -1] = 0  # remap

            progress.set_postfix(
                loss=loss.item(),
                batch_max_reward=batch_max_reward.item(),
                best_reward=best_reward,
            )

            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "batch_max_reward": batch_max_reward.item(),
                "best_reward_running": best_reward,
                "z": log_Z_cand.item(),
            })

    print("\n🏁 Training finished")
    print(f"✨ Best reward: {best_reward}")
    print(f"🧩 Best sequence: {best_sequence.tolist()}")

    # --- conversion en NumPy -----------------------------
    best_reward_np   = np.asarray(best_reward, dtype=np.float32)   # shape ()
    best_sequence_np = best_sequence.numpy().astype(np.int8)       # shape (num_items,)
    # ------------------------------------------------------

    return best_reward_np, best_sequence_np



def prepare_tensors(u: np.ndarray, t: np.ndarray, B: np.ndarray, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Expand 1‑D data arrays to batched tensors on the chosen *device*."""
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    B_tensor = torch.tensor(B, dtype=torch.float32, device=device).view(1, 1).contiguous()
    return u_tensor.detach(), t_tensor.detach(), B_tensor.detach(), u_tensor.shape[1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bi-level GFlowNet Training")
    parser.add_argument('--wandb_project', type=str, default='Bi-level GFlowNet Training')

    parser.add_argument('--data_path', type=str, default='data/data.pickle')
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--num_epochs', type=int, default=50)
    
    parser.add_argument('--hidden_dim_ac', type=int, default=150)
    parser.add_argument('--hidden_dim_cr', type=int, default=150)

    parser.add_argument('--embedding_dim_ac_sel', type=int, default=150)
    parser.add_argument('--embedding_dim_cr_sel', type=int, default=150)

    parser.add_argument('--embedding_dim_ac_B', type=int, default=150)
    parser.add_argument('--embedding_dim_cr_B', type=int, default=150)

    parser.add_argument('--embedding_dim_ac_u', type=int, default=150)
    parser.add_argument('--embedding_dim_cr_u', type=int, default=150)

    parser.add_argument('--embedding_dim_ac_t', type=int, default=150)
    parser.add_argument('--embedding_dim_cr_t', type=int, default=150)


    parser.add_argument('--lr_ac', type=float, default=1e-3)
    parser.add_argument('--lr_cr', type=float, default=1e-3)

    parser.add_argument('--mom_ac', type=float, default=0.8)
    parser.add_argument('--mom_cr', type=float, default=0.5)

    parser.add_argument('--penalty_weight', type=float, default=1.0)
    parser.add_argument('--cut_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)

    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--model_version', type=str, default='v2')
    cfg = parser.parse_args()

    # Init W&B ---------------------------------------------------------
    wandb.init(project=cfg.wandb_project, config=vars(cfg))
    
    # load instance
    if not Path(cfg.data_path).exists():
        raise FileNotFoundError(f"Instance file not found: {cfg.data_path}")
    with open(cfg.data_path, 'rb') as f:
        data = pickle.load(f)
    u_vals, t_vals, B_val = data['u'], data['t'], data['B']

    master_val, follower_val, iteration, start_time = solve(u_vals, t_vals, B_val, cfg)

    print("\n✅ Experiment finished → "
           f"master_val={master_val:.4f}, follower_val={follower_val:.4f}, iterations={iteration}")
    elapsed = time.time() - start_time   # ← NEW
    wandb.log({"master_final": master_val, "follower_final": follower_val, "iterations": iteration, "runtime_sec": elapsed})
    wandb.finish()

