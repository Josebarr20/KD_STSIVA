import torch
import torch.nn as nn
import torch.nn.functional as F
from optics import *

class CI_model(nn.Module):
    def __init__(
        self,
        input_size: tuple,  # (channel, height, width)
        snapshots: int,
        real: str,
        dropout_rate: float
    ):
        super(CI_model, self).__init__()
        self.system_layer = SPC(pinv=False, num_measurements=snapshots, img_size=(input_size[1], input_size[2]), trainable=True, real=real)
        self.net = ResNet18(num_channels=input_size[0],num_classes=10, dropout_rate=dropout_rate)

    def forward(self, x):
        y, x_hat = self.system_layer(x)
        x_hat, resnet_features = self.net(x_hat)
        return y, x_hat, resnet_features

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        # add dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)    
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(out)  # apply dropout after the residual connection
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=True)
        # add dropout layer
        #self.dropout = nn.Dropout(p=dropout_rate)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate = dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate = dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate = dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_rate = dropout_rate)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out6 = F.avg_pool2d(out5, 4)
        out7 = out6.view(out6.size(0), -1)
        out8 = self.linear(out7)
        return out8, [out1, out2, out3, out4, out5, out6, out7]


def ResNet18(num_channels=3, num_classes=10, dropout_rate=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels, dropout_rate=dropout_rate)
    
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     img = torch.randn(32, 3, 32, 32).to(device)
#     model = CI_model(input_size=(3, 32, 32), snapshots=100, real="False").to(device)
#     out = model(img)
#     print(out.shape)