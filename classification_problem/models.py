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

        if self.decoder == "resnet18":
            self.net = ResNet18(num_classes=10)
        else:
            raise ValueError("Invalid Decoder")

    def forward(self, x):
        if self.decoder == "resnet18":
            return self.forward_resnet18(x)

    def forward_resnet18(self, x):
        y = self.system_layer.forward_pass(x)
        x_est = self.system_layer.transpose_pass(y)
        out = self.net(x_est)
        return out, x_est, y

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=2, padding=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.layer5 = self._make_layer(512, 1024, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x