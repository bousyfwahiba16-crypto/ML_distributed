# TP - Entraînement distribué d’un réseau de neurones simple

## Objectif
Mettre en œuvre un entraînement distribué par parallélisme de données avec PyTorch.

## Prérequis
- Python 3.8+
- pip install -r requirements.txt

## Structure du projet
- `src/model.py` : Définition du CNN
- `src/data_utils.py` : Chargement distribué de MNIST
- `src/train.py` : Logique d'entraînement DDP
- `run_distributed.py` : Point d'entrée pour l'entraînement distribué
- `run_single.py` : Entraînement mono-processus pour comparaison

## Exécution
### Entraînement distribué (2 workers)
```bash
python run_distributed.py