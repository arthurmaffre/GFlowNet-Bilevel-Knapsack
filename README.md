# 📦 GFlowNet–Bilevel–Knapsack 

A research prototype combining **Generative Flow Networks (GFlowNets)** with a **cutting‑plane MILP** to solve the *combinatorial pricing problem* (CPP) and its 0‑1 knapsack variant.

👉 **One‑minute pitch**
- *Leader–follower game*: a leader sets prices `t`, each follower chooses a basket `x` that maximises their net utility under budget.
- Exact MILP (value‑function) solved with **dynamic Benders cuts**.
- A **Meta‑Flow‑Net** (actor + critic) learns the follower distribution once and amortises thousands of lower‑level solves.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `models/` | `remaining_budget_v3.py` (actor) · `critic.py` |
| `train_bilevel.py` | End‑to‑end training (GFlowNet + critic + Benders) |
| `cpp_pulp_dynamic.py` | Reference MILP with PuLP + dynamic cuts |
| `cpp_bruteforce.py` | Tiny brute‑force baseline / unit tests |
| `data/instance_gen.py` | Generator for synthetic knapsack instances |
| `reward/`, `metrics/` | Reward computation & W&B diagnostics |
| `runs/` | TensorBoard logs (auto‑created) |

---

## Quick start

```bash
# 1️⃣ Create a Python 3.10 env
conda create -n gflownet-blevel python=3.10 pytorch torchvision cpuonly -c pytorch
conda activate gflownet-blevel
pip install -r requirements.txt   # wandb, pulp, tqdm, ...

# 2️⃣ Generate a toy instance (15 items)
python data/instance_gen.py --n_items 15 --out data/data.pickle

# 3️⃣ Train the Meta‑Flow‑Net (GPU ⇡ if available)
python train_bilevel.py \
    --data_path data/data.pickle \
    --num_epochs 300 --batch_size 128 \
    --cut_interval 50 --penalty_weight 1.0

# 4️⃣ Inspect learning curves
tensorboard --logdir runs
```

For small instances (`n ≤ 12`) you can verify optimality with the MILP:
```bash
python cpp_pulp_dynamic.py --data_path data/data.pickle
```

---

## Reproducibility checklist

- **Deterministic seeds** (`--seed`) for PyTorch & NumPy.
- All hyper‑parameters logged via **Weights & Biases**; set `WANDB_MODE=offline` for air‑gapped runs.
- Dockerfile provided (`docker build -t gfn .`) for exact environment.

---

## Citations & background

- Bui Q.M., Carvalho M., Neto J. (2023) *Solving Combinatorial Pricing Problems using Embedded Dynamic Programming Models*.
- Bengio Y. et al. (2021) *Flow Network Based Generative Models for Non‑Iterative Diverse Candidate Generation*.
- Lozano L. et al. (2021) *Decision diagrams for network interdiction*.

```bibtex
@article{bui2023cpp,
  title={Solving Combinatorial Pricing Problems using Embedded Dynamic Programming Models},
  author={Bui, Quang Minh and Carvalho, Margarida and Neto, Jos{\'e}},
  year={2023}
}
```

---

## License

MIT © 2025 — feel free to fork / cite! 😊
