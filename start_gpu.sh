#!/bin/bash

# Vérifie si la commande W&B est fournie comme paramètre
if [ -z "$1" ]; then
    echo "Erreur : Veuillez fournir la commande de l'agent W&B (ex. 'wandb agent <sweep_id>')"
    exit 1
fi

# Commande de l'agent W&B
WANDB_COMMAND="$1"

# Nombre maximum d'agents en parallèle
MAX_AGENTS=10

# Compteur pour suivre les agents lancés
COUNTER=0

echo "Lancement de $MAX_AGENTS agents W&B en parallèle..."

# Boucle pour lancer les agents
for ((i=1; i<=MAX_AGENTS; i++)); do
    echo "Lancement de l'agent $i/$MAX_AGENTS"
    # Exécute la commande W&B en arrière-plan
    # Ajoute CUDA_VISIBLE_DEVICES pour s'assurer que tous utilisent le même GPU
    CUDA_VISIBLE_DEVICES=0 $WANDB_COMMAND &
    ((COUNTER++))
    
    # Si le nombre maximum d'agents est atteint, attendre qu'ils finissent
    if [ $COUNTER -eq $MAX_AGENTS ]; then
        echo "Attente de la fin des $MAX_AGENTS agents..."
        wait
        COUNTER=0
    fi
done

# Attendre la fin des derniers agents
wait

echo "✅ Toutes les expériences sont terminées !"