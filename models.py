import torch
import torch.nn as nn
import torch.nn.functional as F
from optics import OpticsSPC

class Proximal_Mapping(nn.Module):
    def __init__(self, channel, device, multiplier: int = 1.0):
        super(Proximal_Mapping, self).__init__()

        self.conv1 = nn.Conv2d(channel, 8 * multiplier, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1)

        self.theta = nn.Parameter(torch.ones(1, requires_grad=True) * 0.01).to(device)

        self.conv5 = nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(8 * multiplier, channel, kernel_size=3, padding=1)

        self.Sp = nn.Softplus()

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Softhreshold
        xs = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.theta))

        # Decode
        x = F.relu(self.conv5(xs))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        return x, xs


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

# adapted from https://github.com/usuyama/pytorch-unet/tree/master
class UNet(nn.Module):

    def __init__(self, n_channels, base_channel):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, base_channel)
        self.dconv_down2 = double_conv(base_channel, base_channel * 2)
        self.dconv_down3 = double_conv(base_channel * 2, base_channel * 4)
        self.dconv_down4 = double_conv(base_channel * 4, base_channel * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dconv_up3 = double_conv(base_channel * 12, base_channel * 4)
        self.dconv_up2 = double_conv(base_channel * 6, base_channel * 2)
        self.dconv_up1 = double_conv(base_channel * 3, base_channel)

        self.conv_last = nn.Conv2d(base_channel, n_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 256x256

        x = self.maxpool(conv1)  # 128x128
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)  # 64x64
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)  # 32x32
        bootle = self.dconv_down4(x)

        x = self.upsample(bootle)  # 64x64
        x = torch.cat([x, conv3], dim=1)
        up1 = self.dconv_up3(x)

        x = self.upsample(up1)  # 128x128
        x = torch.cat([x, conv2], dim=1)
        up2 = self.dconv_up2(x)

        x = self.upsample(up2)  # 256x256
        x = torch.cat([x, conv1], dim=1)
        up3 = self.dconv_up1(x)

        out = self.conv_last(up3)

        return out, [conv1, conv2, conv3, bootle, up1, up2, up3]