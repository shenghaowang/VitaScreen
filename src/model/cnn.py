import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Reduce to 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Reduce to 3x3
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Reduce to 1x1
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Global average pooling (outputs shape [B, 64])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.r1 = nn.Linear(64, 64)  # hidden dense layer
        self.sm = nn.Linear(64, 1)  # final output layer (binary classification)

    def forward(self, x):
        x = self.conv0(x)  # 15x15
        x = self.conv1(x)  # 7x7
        x = self.conv2(x)  # 3x3
        x = self.conv3(x)  # 1x1
        x = self.global_pool(x)  # [B, 64, 1, 1]
        x = torch.flatten(x, 1)  # [B, 64]
        x = F.relu(self.r1(x))  # hidden FC
        x = self.sm(x)  # logits for binary classification
        return x
