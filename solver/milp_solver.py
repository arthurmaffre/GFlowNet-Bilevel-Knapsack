import pulp as pl
import pickle
import numpy as np

from solver_with_traj import compute_analytical_reward

with open("data/data.pickle", "rb") as f:
    data = pickle.load(f)

u = data['u'].flatten()
t_init = data['t'].flatten()
B = data['B'].item()

n = len(u)
indices = list(range(n))

M = u.copy()







# Initialisation du master problem (sans contraintes (4c))
iteration = 0
master = pl.LpProblem('CPP_value_function', pl.LpMaximize)

#Variables du master
x = pl.LpVariable.dicts('x', indices, cat='Binary')
s = pl.LpVariable.dicts('s', indices, lowBound=0)
L = pl.LpVariable('L')



t = {}
for i in indices:
    if t_init[i] == 5:
        t[i] = pl.LpVariable(f't_{i}', lowBound=0, upBound=M[i])
    else:
        t[i] = t_init[i]

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

# Boucle dynamique

while True:
    iteration += 1
    master.solve()
    
    # Récupère solution actuelle
    t_sol = np.array([t[i].varValue if isinstance(t[i], pl.LpVariable) else t[i] for i in indices])
    x_sol = np.array([x[i].varValue for i in indices])
    L_sol = L.varValue
    
    # Résout follower problem via solver_with_traj
    follower_val_t, x_hat_sol = compute_analytical_reward(u_vals=u, t_vals=t_sol, B_val=B)

    
    # Calcul des valeurs
    master_val = np.sum((u - t_sol) * x_sol)
    
    follower_val = np.sum((u - t_sol) * x_hat_sol)

    print(f"Iteration {iteration} → Master: {master_val:.4f}, Follower: {follower_val:.4f}")
    
    # Vérifie faisabilité bilevel
    if follower_val > master_val:
        print("→ Adding new constraint from follower.")
        master += pl.lpSum(u[i] * x[i] - s[i] for i in indices) >= follower_val
    else:
        print("→ Bilevel feasible solution found.")
        break

# Affiche les résultats
print("\nFinal solution:")
print(f"{'Index':>5} {'t':>10} {'x':>10} {'s':>10} {'u':>10}")
for i in indices:
    ti_val = t[i].varValue if isinstance(t[i], pl.LpVariable) else t[i]
    xi_val = x[i].varValue if x[i].varValue is not None else 0
    si_val = s[i].varValue if s[i].varValue is not None else 0
    ui_val = u[i]
    print(f"{i:5d} {ti_val:10.4f} {xi_val:10.4f} {si_val:10.4f} {ui_val:10.4f}")

if L.varValue is not None:
    print(f"\nL = {L.varValue:.4f}")
else:
    print("\nL = No solution / not set")