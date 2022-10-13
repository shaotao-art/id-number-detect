import torch
from torch import nn
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        x = x.view(-1, 28*28)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _init_params(self):
        pass

    def __str__(self) -> str:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad == True)
        return f'\nModel\n\tnum params: {num_params}\n'