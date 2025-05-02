import itertools
import numpy as np
from typing import Sequence, Union, Tuple

def compute_analytical_reward(
    u_vals: Sequence[Union[float, int]],
    t_vals: Sequence[Union[float, int]],
    B_val: Union[float, int]
) -> Tuple[float, np.ndarray]:
    """
    Calcul exact du knapsack 0-1 par brute-force.

    Args:
        u_vals: utilités des items, peut être list, np.ndarray ou torch.Tensor
        t_vals: coûts des items, même format que u_vals
        B_val : budget total, scalaire ou tableau unidimensionnel

    Returns:
        max_reward: float, récompense max = max_{\sum t_i x_i <= B} ∑ (u_i - t_i) x_i
        x_star: np.ndarray, vecteur binaire des items choisis (1 = choisi, 0 = pas choisi)
    """
    # Convertir en numpy et aplatir
    u_arr = np.asarray(u_vals).flatten()
    t_arr = np.asarray(t_vals).flatten()
    B = float(np.asarray(B_val).flatten()[0])

    # vecteur des gains marginaux
    marg = u_arr - t_arr
    n = len(marg)
    max_r = float("-inf")
    best_subset = []

    # Parcours de toutes les combinaisons
    for k in range(n + 1):
        for subset in itertools.combinations(range(n), k):
            cost = t_arr[list(subset)].sum()
            if cost <= B:
                reward = marg[list(subset)].sum()
                if reward > max_r:
                    max_r = reward
                    best_subset = subset

    # Créer la séquence binaire x^*
    x_star = np.zeros(n, dtype=int)
    x_star[list(best_subset)] = 1

    return max_r, x_star



if __name__ == "__main__":
    # Petit test direct : charge le pickle et affiche la valeur max trouvée et la séquence choisie
    
    import pickle

    # Charge le pickle
    with open("data/data.pickle", "rb") as f:
        data = pickle.load(f)

    u = data['u'].flatten()
    t = data['t'].flatten()
    B = float(data['B'].flatten()[0])

    # Appelle compute_analytical_reward
    reward, x_star = compute_analytical_reward(u, t, B)

    # Affiche le résultat
    print(f"Max reward from analytical knapsack: {reward:.4f}")
    print(f"Best sequence (x*): {x_star}")

    # === Mini test automatique ===
    total_cost = np.sum(t * x_star)
    total_reward = np.sum((u - t) * x_star)

    print(f"Total cost of x*: {total_cost:.4f} (Budget B = {B:.4f})")
    print(f"Total reward computed from x*: {total_reward:.4f}")

    if total_cost > B + 1e-6:
        print("❌ ERROR: Selected sequence exceeds budget!")
    elif abs(total_reward - reward) > 1e-4:
        print("❌ ERROR: Reward does not match reported maximum!")
        print(f"Expected (recomputed) reward: {total_reward:.4f}")
        print(f"Reported reward from function: {reward:.4f}")
    else:
        print("✅ Test passed: sequence and reward are consistent.")