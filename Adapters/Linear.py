import torch
import torch.nn as nn

class LinearWrapper(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.device = torch.device("cpu")

    def forward(self, x):
        return self.linear(x)

    def to(self, device):
        self.device = torch.device(device)
        return super().to(device)