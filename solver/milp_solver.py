def benders_master(u_vals, t_vals, B_val, cuts):
    """
    Enumerate subproblem to produce a Benders cut (alpha, beta).
    Returns None if duplicate.
    """
    n = len(u_vals)
    best_val = float('-inf')
    best_x = None
    for bits in itertools.product([0, 1], repeat=n):
        cost = sum(val * bits[i] for i, val in enumerate(t_vals))
        if cost <= B_val:
            val = sum((u_vals[i] - t_vals[i]) * bits[i] for i in range(n))
            if val > best_val:
                best_val = val
                best_x = bits
    alpha = torch.tensor(best_x, dtype=torch.float32)
    beta = best_val
    for a_old, b_old in cuts:
        if torch.allclose(a_old, alpha) and abs(b_old - beta) < 1e-6:
            return None
    return (alpha, beta)