import math
import torch
import torch.nn as nn


__all__ = ['ResNet', 'ResNet3x3']


__CONFIGS__ = {
    'ResNet26': [2, 2, 2, 2],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
}


class Bottleneck(nn.Module):

    expansion = 4  # this is hardcoded for every ResNet, and should not be changed!

    def __init__(self, in_channels, med_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        out_channels = med_channels * self.expansion

        # skip branch transform
        self.downsample = downsample

        # residual branch transform
        # channel compression
        self.conv1 = nn.Conv2d(in_channels, med_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # spatial convolutions
        self.conv2 = nn.Conv2d(med_channels, med_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # channel decompression
        self.conv3 = nn.Conv2d(med_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):

        # skip branch
        xs = x
        if self.downsample is not None:
            xs = self.downsample(xs)

        # residual branch
        xr = self.conv1(x)
        xr = self.bn1(xr)
        xr = self.relu1(xr)
        xr = self.conv2(xr)
        xr = self.bn2(xr)
        xr = self.relu2(xr)
        xr = self.conv3(xr)
        xr = self.bn3(xr)

        # join
        x = torch.add(xs, xr)
        x = self.relu3(x)

        return x


class ResNet(nn.Module):

    def __init__(self, config, num_classes=1000):

        self.in_channels = 64
        block = Bottleneck

        super(ResNet, self).__init__()

        # adapter
        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # feature extractor
        self.layer1 = self._make_layer(block, 64,  __CONFIGS__[config][0])
        self.layer2 = self._make_layer(block, 128, __CONFIGS__[config][1], stride=2)
        self.layer3 = self._make_layer(block, 256, __CONFIGS__[config][2], stride=2)
        self.layer4 = self._make_layer(block, 512, __CONFIGS__[config][3], stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialise_params()

    def _make_layer(self, block, med_channels, num_blocks, stride=1):
        out_channels = med_channels * block.expansion

        # first block
        if stride == 1 and self.in_channels == out_channels:
            downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        blocks = [block(self.in_channels, med_channels, stride, downsample)]
        self.in_channels = out_channels

        # remaining blocks
        for i in range(1, num_blocks):
            blocks.append(block(self.in_channels, med_channels))

        return nn.Sequential(*blocks)

    def _initialise_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.adapter(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet3x3(ResNet):
    """

    Basic ResNets use a single convolutional layer whose filters have 7x7
    spatial dimensions. This variant will use instead a stack of convolutional
    layers whose filters have 3x3 spatial dimensions.
    """

    def __init__(self, adapter_size, config, num_classes=1000):
        super(ResNet3x3, self).__init__(config, num_classes=num_classes)

        adapter = [
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        for i in range(1, adapter_size):
            adapter.extend([
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ])
        self.adapter = nn.Sequential(*adapter)

        self._initialise_params()
