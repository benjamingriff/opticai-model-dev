import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).view(B, T, -1)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(x)
        out = self.classifier(hn[-1])
        return out
