import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)        # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)       # 26x26 -> 24x24
        self.fc1 = nn.Linear(9216, 128)             # 64 * 12 * 12 = 9216 après maxpool 2x2
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)                     # 24x24 -> 12x12
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#------------------------------------------------------------------------

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 64, 3, 1)      # 28->26
#         self.conv2 = nn.Conv2d(64, 128, 3, 1)    # 26->24
#         self.conv3 = nn.Conv2d(128, 256, 3, 1)   # 24->22
#         self.conv4 = nn.Conv2d(256, 256, 3, 1)   # 22->20
#         self.pool = nn.AdaptiveAvgPool2d((4, 4)) # force la sortie en 4x4 peu importe l'entrée
#         self.fc1 = nn.Linear(256 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)