import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class DilatedResidualLayer(nn.Module):
    def __init__(self, num_filters, dilation):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            num_filters, num_filters, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out  # Residual connection


class TCNStage(nn.Module):
    def __init__(self, num_layers, num_filters):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(DilatedResidualLayer(num_filters, dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MS_TCN(nn.Module):
    def __init__(
        self,
        num_classes,
        precomputed_features=False,
        feature_dim=512,
        num_stages=4,
        num_layers=10,
        num_filters=64,
    ):
        super().__init__()

        self.precomputed_features = precomputed_features

        if not self.precomputed_features:
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = 512

        # Project features into TCN input space
        self.feature_projection = nn.Conv1d(feature_dim, num_filters, kernel_size=1)

        self.stage1 = TCNStage(num_layers, num_filters)
        self.stage1_classifier = nn.Conv1d(num_filters, num_classes, kernel_size=1)

        self.stages = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.refinement_input_proj = nn.ModuleList()

        for _ in range(num_stages - 1):
            self.refinement_input_proj.append(
                nn.Conv1d(num_classes, num_filters, kernel_size=1)
            )
            self.stages.append(TCNStage(num_layers, num_filters))
            self.classifiers.append(nn.Conv1d(num_filters, num_classes, kernel_size=1))

    def forward(self, x):
        """
        If raw input: x = (B, T, C, H, W)
        If features: x = (B, T, feature_dim)
        """
        if not self.precomputed_features:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.cnn(x)
            x = self.pool(x).view(B, T, -1)  # (B, T, 512)

        # For features mode: x already shape (B, T, feature_dim)
        if self.precomputed_features and x.ndim == 3:
            B, T, feature_dim = x.shape

        x = x.permute(0, 2, 1)  # (B, feature_dim, T)
        x = self.feature_projection(x)

        out = self.stage1(x)
        out = self.stage1_classifier(out)
        outputs = [out]

        for proj, stage, classifier in zip(
            self.refinement_input_proj, self.stages, self.classifiers
        ):
            out = proj(out)
            out = stage(out)
            out = classifier(out)
            outputs.append(out)

        return outputs
