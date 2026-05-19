import torch
import torch.multiprocessing as mp
from src.train_gpu import train_worker_gpu

if __name__ == "__main__":
    # Détection du nombre de GPUs disponibles
    n_gpus = torch.cuda.device_count()
    
    print(f"{n_gpus} GPUs détectés. Lancement de l'entraînement distribué...")
    
    world_size = n_gpus
    # Augmente le batch_size local pour que chaque GPU traite plus de données
    mp.spawn(train_worker_gpu, args=(world_size, 2, 128), nprocs=world_size, join=True)
