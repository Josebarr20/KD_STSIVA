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
    
    
# RGB SPC Acquisition

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
    def __init__(self, pinv:bool=False, num_measurements:int=100, img_size:tuple=(32,32), trainable:bool=True, real: str = "True"):
        super(SPC, self).__init__()
        self.pinv = pinv
        M, N = img_size
        H = torch.randn(num_measurements, M*N)
        self.H = nn.Parameter(H, requires_grad=trainable)
        self.real = real

    def forward(self, x: torch.Tensor) -> torch:
        y = forward_spc(x, self.H if self.real == "True" else BinaryQuantize_1.apply(self.H))
        x_hat = backward_spc(y, self.H if self.real == "True" else BinaryQuantize_1.apply(self.H), self.pinv)
        return x_hat
    
# if __name__ == "__main__":
#     x = torch.randn(32, 3, 32, 32)
#     spc = SPC(pinv=False, num_measurements=64, img_size=(32, 32), trainable=True, real="True")
#     x_hat = spc(x)
#     print(x_hat.size())
