import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.model import SimpleCNN
import time

def train_single(epochs=2, batch_size=256, data_root='./src/data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
    print(f"Temps d'entraînement (single): {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    train_single()