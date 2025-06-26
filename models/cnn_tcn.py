import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNN_TCN(nn.Module):
    def __init__(self, num_classes, tcn_channels=[512, 256, 128], kernel_size=3):
        super().__init__()

        # CNN backbone
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: [B*T, 512, 1, 1]

        # Temporal Convolutional Network
        layers = []
        in_channels = 512
        for out_channels in tcn_channels:
            layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)

        self.classifier = nn.Linear(tcn_channels[-1], num_classes)

    def forward(self, x, lengths=None):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 3, 224, 224]
        x = self.cnn(x)  # [B*T, 512, H', W']
        x = self.pool(x).view(B, T, -1)  # [B, T, 512]
        x = x.transpose(1, 2)  # [B, 512, T]
        x = self.tcn(x)  # [B, hidden, T]
        x = x.mean(dim=2)  # [B, hidden]
        out = self.classifier(x)  # [B, num_classes]
        return out
