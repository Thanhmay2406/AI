import torch
import torch.nn as nn

class block(nn.Module):
    expansion = 1
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(output_channels)
                    )

    def forward(self, x):
        I = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += I
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block=block, out_channels=64,
                                       num_blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block=block, out_channels=128,
                                       num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, out_channels=256,
                                       num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, out_channels=512,
                                       num_blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layer = []
        layer.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(num_blocks):
            layer.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(num_classes):
    return ResNet(block, [2, 2, 2, 2], num_classes)
