import torch
import torch.nn as nn
import numpy as np
from utils import BinaryQuantize_1, hadamard

def forward_spc(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    r"""

    Forward propagation through the Single Pixel Camera (SPC) model.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging 10.1109/TIP.2020.2971150

    Args:
        x (torch.Tensor): Input image tensor of size (B, L, M, N).
        H (torch.Tensor): Measurement matrix of size (S, M*N).

    Returns:
        torch.Tensor: Output measurement tensor of size (B, S, L).
    """
    B, L, M, N = x.size()
    x = x.contiguous().view(B, L, M * N)
    x = x.permute(0, 2, 1)

    # measurement
    H = H.unsqueeze(0).repeat(B, 1, 1)
    y = torch.bmm(H, x)
    return y


def backward_spc(y: torch.Tensor, H: torch.Tensor, pinv= False) -> torch.Tensor:
    r"""

    Inverse operation to reconstruct the image from measurements.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging  10.1109/TIP.2020.2971150

    Args:
        y (torch.Tensor): Measurement tensor of size (B, S, L).
        H (torch.Tensor): Measurement matrix of size (S, M*N).
        pinv (bool): Boolean, if True the pseudo-inverse of H is used, otherwise the transpose of H is used, defaults to False.
    Returns:
        torch.Tensor: Reconstructed image tensor of size (B, L, M, N).
    """

    Hinv   = torch.pinverse(H) if pinv else torch.transpose(H, 0, 1)
    Hinv   = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

    x = torch.bmm(Hinv, y)
    x = x.permute(0, 2, 1)
    b, c, hw = x.size()
    h = int(np.sqrt(hw))
    x = x.reshape(b, c, h, h)
    x = x/x.max() # normalizetion step to avoid numerical issues
    return x

class SPC(nn.Module):
    def __init__(self, pinv:bool = False, num_measurements:int=100, img_size:tuple=(32,32), trainable:bool=True):
        super(SPC, self).__init__()
        self.pinv = pinv
        M, N = img_size
        H = torch.randn(num_measurements, M*N)
        self.H = nn.Parameter(H, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch:
        y = forward_spc(x, self.H)
        x_hat = backward_spc(y, self.H, self.pinv)
        return x_hat

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        return forward_spc(x, self.H)

    def backward_pass(self, y: torch.Tensor) -> torch.Tensor:
        return backward_spc(y, self.H, self.pinv)

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