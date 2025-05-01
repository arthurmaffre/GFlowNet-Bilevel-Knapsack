import torch

def compute_reward(selected, u, t, B, num_items):
    """
    Calcule la récompense : sum((u - t) * x) si sum(t * x) <= B, sinon 0.
    - selected : (batch_size, num_items) - vecteur de sélection (-1 ou 1)
    - u : (num_items,) - utilités
    - t : (num_items,) - prix
    - B : budget (scalaire)
    """
    B_flat = B.view(-1)
    # Convertir selected en binaire : -1 -> 0, 1 -> 1
    
    selected_binary = (selected > 0).to(torch.float32)

    # Calculer le coût total pour chaque trajectoire
    total_cost = (selected_binary * t).sum(dim=1)  # (batch_size,)

    # Calculer la récompense potentielle
    marginal = u-t
    potential_reward = (selected_binary * marginal).sum(dim=1)


    # Vérifier le budget : si total_cost > B, reward = 0
    reward = torch.where(total_cost <= B_flat, potential_reward, torch.zeros_like(potential_reward))
    
    return reward