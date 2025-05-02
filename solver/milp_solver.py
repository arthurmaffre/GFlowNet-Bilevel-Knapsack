import pulp as pl

# variables
n = len(u)
t = pl.LpVariable.dicts('t', range(n), lowBound=0)
x = pl.LpVariable.dicts('x', range(n), cat='Binary')
L = pl.LpVariable('L')

prob = pl.LpProblem('CPP_value_function', pl.LpMaximize)

# objectif (4a)
prob += pl.lpSum(t[i]*x[i] for i in range(n))

# contraintes knapsack ou autres, ex. weights w, capacity C
prob += pl.lpSum(w[i]*x[i] for i in range(n)) <= C

# coupes dynamiques (4c)  —  on les ajoutera dans une boucle
cuts = []          # stock des x̂ ajoutés

def add_cut(x_hat):
    prob += L >= pl.lpSum((u[i]-t[i])*x_hat[i] for i in range(n))

# dualité forte (4d)
prob += pl.lpSum((u[i]-t[i])*x[i] for i in range(n)) == L

# McCormick (6)  pour chaque produit bilinéaire t_i * x_i
for i in range(n):
    M = u[i]            # bound from remark 1
    s = pl.LpVariable(f's_{i}', lowBound=0)
    prob += s <= M * x[i]
    prob += t[i] - s <= M * (1 - x[i])
    # remplace t_i * x_i par s dans l'objectif
    # (ici l'exemple reste simple ; sinon crée une var revenue)

# --------- boucle coupe -------------
for k in range(max_iter):
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    t_sol = [t[i].value() for i in range(n)]
    x_sol = [int(round(x[i].value())) for i in range(n)]

    # follower best response (exact knapsack bruteforce ou autre solveur)
    x_hat = follower_best_response(u, t_sol, w, C)
    if sum((u[i]-t_sol[i])*x_sol[i] for i in range(n)) \
       + 1e-6 < sum((u[i]-t_sol[i])*x_hat[i] for i in range(n)):
        add_cut(x_hat)           # ajoute la contrainte (4c)
    else:
        break

print('t* =', t_sol)
print('x* =', x_sol)