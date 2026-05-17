# TP - Entraînement distribué d’un réseau de neurones simple

## Objectif
Mettre en œuvre un entraînement distribué par parallélisme de données avec PyTorch.

## Prérequis
- Python 3.8+
- pip install -r requirements.txt

## Activez l'environnement 
- venv\Scripts\activate.bat
- Vous verrez (venv) apparaître. Ensuite installez les dépendances avec ce cmd : pip install -r requirements.txt
- apres meli kantinstaller les packages et tt kadiri run les files for exemple : python run_single.py
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