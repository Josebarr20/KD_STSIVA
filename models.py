import torch
import torch.nn as nn
import torch.nn.functional as F

class CI_model(nn.Module):
    def __init__(
        self,
        input_size: tuple,  # (channel, height, width)
        snapshots: int,
        n_stages: int,
        device: str,
        SystemLayer: str,
        real: str,
        sampling_pattern: str,
        base_channels: int,
        decoder: str,
        acc_factor: int,
        snr: int = 0,
    ):
        super(CI_model, self).__init__()

        if SystemLayer == "SPC":
            self.system_layer = OpticsSPC(input_size, snapshots, real, snr)
            multiplier = 4

        else:
            raise ValueError("Invalid System Layer")

        self.decoder = decoder

        if self.decoder == "unroll":
            channels = input_size[0]
            self.proximals = nn.ModuleList(
                [
                    Proximal_Mapping(channel=channels, device=device, multiplier=multiplier).to(
                        device
                    )
                    for _ in range(n_stages)
                ]
            )
            self.alphas = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True) * 0.01) for _ in range(n_stages)]
            )

            self.rhos = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True) * 0.01) for _ in range(n_stages)]
            )

            self.betas = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True) * 0.01) for _ in range(n_stages)]
            )
            self.n_stages = n_stages

        elif self.decoder == "unroll_unet":
            channels = input_size[0]
            self.proximals = nn.ModuleList(
                [
                    UNet(n_channels=channels, base_channel=base_channels).to(device)
                    for _ in range(n_stages)
                ]
            )
            self.alphas = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True) * 0.01) for _ in range(n_stages)]
            )

            self.rhos = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True) * 0.01) for _ in range(n_stages)]
            )

            self.betas = nn.ParameterList(
                [nn.Parameter(torch.ones(1, requires_grad=True) * 0.01) for _ in range(n_stages)]
            )
            self.n_stages = n_stages

        elif self.decoder == "unet":
            self.net = UNet(n_channels=input_size[0], base_channel=base_channels)

        else:
            raise ValueError("Invalid Decoder")

    def forward(self, x):
        if self.decoder == "unroll":
            return self.forward_unroll(x)
        elif self.decoder == "unroll_unet":
            return self.forward_unroll_unet(x)
        elif self.decoder == "unet":
            return self.forward_unet(x)
        elif self.decoder == "cbam":
            return self.forward_unet(x)

    def forward_unroll(self, x):

        y = self.system_layer.forward_pass(x)
        x = self.system_layer(x)
        u = torch.zeros_like(x)
        Xt = [x]
        Xs = []
        gradients = []
        for i in range(self.n_stages):

            h, xs = self.proximals[i](x + u)
            gradient = self.system_layer.transpose_pass(self.system_layer.forward_pass(x) - y)
            x = x - self.alphas[i] * (gradient + self.rhos[i] * (x - h + u))
            u = u + self.betas[i] * (x - h)

            Xt.append(x)
            Xs.append(xs)
            gradients.append(gradient)

        return Xt[-1], [gradients, xs], Xt[0]

    def forward_unroll_unet(self, x):

        y = self.system_layer.forward_pass(x)
        x = self.system_layer(x)
        u = torch.zeros_like(x)
        Xt = [x]
        feats = []
        for i in range(self.n_stages):

            h, feats_unet = self.proximals[i](x + u)
            gradient = self.system_layer.transpose_pass(self.system_layer.forward_pass(x) - y)
            x = x - self.alphas[i] * (gradient + self.rhos[i] * (x - h + u))
            u = u + self.betas[i] * (x - h)

            Xt.append(x)
            feats.append(feats_unet[3])

        return Xt[-1], feats, Xt[0]

    def forward_unet(self, x):
        y = self.system_layer.forward_pass(x)
        x_est = self.system_layer.transpose_pass(y)
        out, features = self.net(x_est)
        return out, features, x_est, y


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def _init_(self, n_channels, bilinear=False, divisor:int=4):
        super(UNet, self)._init_()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64//divisor))
        self.down1 = (Down(64//divisor, 128//divisor))
        self.down2 = (Down(128//divisor, 256//divisor))
        self.down3 = (Down(256//divisor, 512//divisor))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512//divisor, 1024//divisor))
        self.up1 = (Up(1024//divisor, 512//divisor, bilinear))
        self.up2 = (Up(512//divisor, 256//divisor, bilinear))
        self.up3 = (Up(256//divisor, 128//divisor, bilinear))
        self.up4 = (Up(128//divisor, 64//divisor, bilinear))
        self.outc = (OutConv(64//divisor, n_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def _init_(self, in_channels, out_channels, mid_channels=None):
        super()._init_()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def _init_(self, in_channels, out_channels):
        super()._init_()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def _init_(self, in_channels, out_channels, bilinear=True):
        super()._init_()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def _init_(self, in_channels, out_channels):
        super(OutConv, self)._init_()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)