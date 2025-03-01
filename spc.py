import torch
import torch.nn as nn
import optics as op

class SPC(nn.Module):
    def __init__(self, pinv:bool = False, num_measurements:int=100, img_size:tuple=(32,32), trainable:bool=True):
        super(SPC, self).__init__()
        self.pinv = pinv
        M, N = img_size
        H = torch.randn(num_measurements, M*N)
        self.H = nn.Parameter(H, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch:
        y = op.forward_spc(x, self.H)
        x_hat = op.backward_spc(y, self.H, self.pinv)
        return x_hat

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        return op.forward_spc(x, self.H)

    def backward_pass(self, y: torch.Tensor) -> torch.Tensor:
        return op.backward_spc(y, self.H, self.pinv)