import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

from src.models.model import SimpleCNN
from src.data.data_utils import get_dataloader


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, epochs=2, batch_size=64, data_root='./src/data'):
    
    setup(rank, world_size)

    train_loader, sampler = get_dataloader(rank, world_size, batch_size, data_root)
    model = SimpleCNN()                          # Reste sur CPU, pas de .to(rank)
    ddp_model = DDP(model, device_ids=None)      # device_ids=None pour CPU
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Les données sont déjà sur CPU, on ne les déplace pas
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"[Rank 0] Epoch {epoch} | Loss: {avg_loss:.4f}")
    if rank == 0:
        print(f"Temps d'entraînement (world_size={world_size}): {time.time()-start_time:.1f}s")
    cleanup()