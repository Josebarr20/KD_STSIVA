import torch
import torch.nn as nn
import numpy as np
from utils import BinaryQuantize_1

def hadamard(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard(n // 2)
        return np.block([[h, h], [h, -h]])

class OpticsSPC(nn.Module):
    def __init__(self, input_size: tuple, snapshots: int, real: str, snr: int):
        super(OpticsSPC, self).__init__()
        _, self.M, self.N = input_size
        self.snapshots = snapshots
        self.real = real
        self.snr = snr

        if self.real == "hadamard":
            a = hadamard(self.M * self.N)
            a = a[: self.snapshots, :]
            ca = torch.tensor(a).float()
            ca = ca.view(self.snapshots, self.M, self.N)
            self.cas = nn.Parameter(ca, requires_grad=False)

        else:
            ca = torch.normal(0, 1, size=(self.snapshots, self.M, self.N))
            ca = ca / torch.sqrt(torch.tensor(self.M * self.N).float())
            self.cas = nn.Parameter(ca, requires_grad=True)

    def forward(self, x):
        y = self.forward_pass(x)
        x = self.transpose_pass(y)
        return x

    def forward_pass(self, x):
        ca = self.get_coded_aperture()
        y = x * ca
        y = torch.sum(y, dim=(-2, -1))
        y = y.unsqueeze(-1).unsqueeze(-1)

        if self.snr != 0:
            y = y + self.noise(y)

        return y

    def transpose_pass(self, y):
        ca = self.get_coded_aperture()
        x = y * ca
        x = torch.sum(x, dim=1)
        x = x.unsqueeze(1)
        x = x / torch.max(x)
        return x

    def get_coded_aperture(self):
        ca = self.cas.unsqueeze(0)
        if self.real == "False":
            ca = BinaryQuantize_1.apply(ca)
            return ca
        elif self.real == "True" or self.real == "hadamard":
            return ca

    def noise(self, y):
        sigma = torch.sum(torch.pow(y, 2)) / ((y.shape[0] * y.shape[1]) * 10 ** (self.snr / 10))
        noise = torch.normal(mean=0, std=torch.sqrt(sigma).item(), size=y.shape)
        noise = noise.to(y.device)
        return noise