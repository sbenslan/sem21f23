import torch
from torch import nn

__all__ = ['VGG7']

class VGG7(nn.Module):
    def __init__(self, capacity : int):
        super(VGG7, self).__init__()
        self.features = self._make_features(capacity)
        n_fc = 1024
        c3 = capacity * 128 * 4
        self.classifier = nn.Sequential(
            nn.Linear(c3*4*4, n_fc),
            nn.ReLU(inplace=True),
            nn.Linear(n_fc, n_fc),
            nn.ReLU(inplace=True),
            nn.Linear(n_fc, 10)
        )

    @staticmethod
    def _make_features(capacity : int):
        c0 = 3
        c1 = int(capacity * 128)
        c2 = int(capacity * 128 * 2)
        c3 = int(capacity * 128 * 4)

        in_ch = c0
        layers = []
        for c in (c1, c2, c3):
            layers.append(nn.Conv2d(in_ch, c, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

