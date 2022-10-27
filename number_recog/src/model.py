from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k_s, s, p) -> None:
        super().__init__()
        self.block = nn.Sequential(*[
            nn.Conv2d(in_channel, out_channel, k_s, s, p),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ])
    
    def forward(self, x):
        return self.block(x)


class NNet(nn.Module):
    def __init__(self) -> None:
        super(NNet, self).__init__()
        self.conv1 = ConvBlock(1, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = ConvBlock(16, 32, 3, 2, 2)
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(*[
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

    def _init_params(self):
        pass

    def __str__(self) -> str:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad == True)
        return f'\nModel\n\tnum params: {num_params}\n'

if __name__ == "__main__":
    block = ConvBlock(1, 10, 3, 1, 1)
    x = torch.randn(8, 1, 28, 28)
    out = block(x)
    assert out.shape == (8, 10, 28, 28)

    model = NNet()
    out = model(x)
    assert out.shape == (8, 11)