import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

def get_dataloader(rank, world_size, batch_size=64, data_root='./src/data'):
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=data_root, train=True,
                             download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=sampler, num_workers=2, pin_memory=True)
    return loader, sampler