import torch
import torch.nn as nn
import torch.nn.functional as F

class GOLCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        kernel = torch.tensor([[2, 2, 2],
                               [2, 1, 2],
                               [2, 2, 2]])
        kernel = kernel.float().unsqueeze(0).unsqueeze(0)
        self.conv.weight = torch.nn.Parameter(kernel)

    def activation(self, x):
        return torch.heaviside(x - 4.5, torch.tensor(1.0)) - torch.heaviside(x - 7.5, torch.tensor(1.0))

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='circular')
        return self.activation(self.conv(x))