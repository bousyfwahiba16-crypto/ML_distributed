# import torch.multiprocessing as mp
# from src.train import train_worker

# if __name__ == "__main__":
#     world_size = 2          # Changez ici pour tester avec plus de processus
#     mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
#----------------------------------------------
import torch.multiprocessing as mp
from training.train import train_worker

if __name__ == "__main__":
    world_size = 2
    # On augmente le batch_size local pour que chaque worker fasse plus de calcul
    mp.spawn(train_worker, args=(world_size, 2, 128), nprocs=world_size, join=True)
    # Les arguments : (rank, world_size, epochs=2, batch_size=128)