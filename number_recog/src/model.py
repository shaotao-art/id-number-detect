import torch
from torch import nn
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(self) -> None:
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 11)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def _init_params(self):
        pass

    def __str__(self) -> str:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad == True)
        return f'\nModel\n\tnum params: {num_params}\n'