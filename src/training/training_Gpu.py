import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
from src.model import SimpleCNN
from src.data_utils import get_dataloader

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker_gpu(rank, world_size, epochs=2, batch_size=64, data_root='./src/data'):
    setup(rank, world_size)
    
    # Device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # DataLoader distribué
    train_loader, sampler = get_dataloader(rank, world_size, batch_size, data_root)
    
    # Modèle sur GPU
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Transfert des données vers le GPU
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"[Rank 0 - GPU:{rank}] Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    if rank == 0:
        elapsed = time.time() - start_time
        print(f"Temps d'entraînement GPU distribué ({world_size} GPUs): {elapsed:.1f}s")
    
    cleanup()


def train_single_gpu(epochs=2, batch_size=256, data_root='./src/data'):
   
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print("NCCL: ", torch.distributed.is_nccl_available())

    # Device
    device = torch.device('cuda')
    
    # DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Modèle
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"Temps d'entraînement GPU (single): {elapsed:.1f}s")
