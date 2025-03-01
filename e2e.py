import torch
import torch.nn as nn
import spc
import model as m

class E2E(nn.Module):
    def __init__(self, pinv:bool=False, num_measurements:int=100, img_size:tuple=(32,32), trainable:bool=True, num_channels:int=1):
        super(E2E, self).__init__()
        self.spc = spc.SPC(pinv, num_measurements, img_size, trainable)
        self.unet = m.UNet(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.spc(x)
        x_hat = self.unet(x_hat)
        return x_hat