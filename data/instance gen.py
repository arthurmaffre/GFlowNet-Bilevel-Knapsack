import numpy as np
import threading
from queue import Queue  # Pour récupérer les résultats des threads
import pickle
from pathlib import Path

import numpy as np

class generate_knapsack_instance_numpy():
    """
    Classe pour générer une instance du Knapsack Pricing Problem adaptée à une formulation simplifiée,
    avec une structure ordonnée pour optimiser les calculs futurs.
    """

    def __init__(self, difficulty, n_consumers=1):
        """
        Initialise la classe avec les paramètres de base.

        Paramètres:
          - difficulty : entier indiquant le niveau de difficulté.
          - n_consumers : nombre de consommateurs (par défaut 1).
        """
        self.difficulty = difficulty
        self.n_consumers = n_consumers
        self.budget_init_A = 5  # Budget initial pour les produits contrôlés

    def generate(self):
        """
        Génère une instance du Knapsack Pricing Problem.

        Renvoie un dictionnaire contenant :
          - "num_items"    : nombre total de produits (|J|)
          - "n_controlled" : nombre de produits contrôlés par Amazon (séparateur entre J_A et J_O)
          - "u"            : matrice d'utilités (n_consumers x num_items)
          - "t"            : vecteur complet des prix, ordonné (contrôlés puis exogènes)
          - "B"            : budget de chaque consommateur (variation ±20%)
          - "n_consumers"  : nombre de consommateurs
        """
        # Nombre d'items en fonction de la difficulté
        num_items = {1: 5, 2: 10, 3: 15}.get(self.difficulty, 5 * self.difficulty)

        # Détermination du nombre de produits exogènes (~20%)
        n_controlled = max(1, int(0.2 * num_items))

        # Création des prix : contrôlés (fixe à 5), puis exogènes (entre 6 et 10)
        t = np.empty(num_items, dtype=np.float16)
        t[:n_controlled] = 5
        t[n_controlled:] = np.random.uniform(6, 10, size=num_items - n_controlled).astype(np.float16)

        # Génération des utilités pour chaque consommateur et produit (entre 10 et 100)
        u = np.random.randint(10, 101, size=(self.n_consumers, num_items), dtype=np.uint8)


        # Budget par consommateur : 40% de la somme totale des prix ±20%
        total_price = self.budget_init_A * n_controlled + np.sum(t[n_controlled:])
        base_budget = total_price * 0.4
        B = np.random.uniform(0.8 * base_budget, 1.2 * base_budget, size=self.n_consumers).astype(np.uint16)

        # Construction de l'instance finale simplifiée
        instance = {
        "num_items": num_items,
        "n_controlled": n_controlled,
        "u": u,
        "t": t,
        "B": B,
        "n_consumers": self.n_consumers  # Ajouter cette ligne
        }

        return instance



if __name__ == '__main__':
    print("Début de la procédure de test pour generate_knapsack_instance_numpy avec multithreading\n")

    # Liste des niveaux de difficulté à tester
    difficulties = 3
    n_consumers = 1  # Par exemple, 3 consommateurs

    instance = generate_knapsack_instance_numpy(difficulties, n_consumers)
    instance = instance.generate()

    # Save instance in <project_root>/data/data.pickle regardless of CWD
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / 'data' / 'data.pickle'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump(instance, f, pickle.HIGHEST_PROTOCOL)
    print(f"Instance saved to {out_path}")
    