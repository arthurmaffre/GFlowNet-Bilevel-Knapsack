import itertools
import numpy as np
from typing import Sequence, Union

def compute_analytical_reward(
    u_vals: Sequence[Union[float, int]],
    t_vals: Sequence[Union[float, int]],
    B_val:  Union[float, int]
) -> float:
    """
    Calcul exact du knapsack 0-1 par brute‑force.

    Args:
        u_vals: utilités des items, peut être list, np.ndarray ou torch.Tensor
        t_vals: coûts des items, même format que u_vals
        B_val : budget total, scalaire ou tableau unidimensionnel

    Returns:
        max_reward: float, récompense max = max_{\sum t_i x_i <= B} ∑ (u_i - t_i) x_i
    """
    # Convertir en numpy et aplatir
    u_arr = np.asarray(u_vals).flatten()
    t_arr = np.asarray(t_vals).flatten()
    # Budget comme scalaire float
    B = float(np.asarray(B_val).flatten()[0])

    # vecteur des gains marginaux
    marg = u_arr - t_arr
    n = len(marg)
    max_r = float("-inf")

    # Parcours de toutes les combinaisons
    for k in range(n + 1):
        for subset in itertools.combinations(range(n), k):
            cost = t_arr[list(subset)].sum()
            if cost <= B:
                reward = marg[list(subset)].sum()
                if reward > max_r:
                    max_r = reward

    return max_r