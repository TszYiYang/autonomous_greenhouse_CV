"""
model.py

This module defines the TomatoNet model for predicting plant traits from RGBD images.

Classes:
- TomatoNet: A neural network model that takes 4-channel RGBD images as input and predicts plant traits.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig


class TomatoNet(nn.Module):
    def __init__(self, num_traits):
        super(TomatoNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Adjust input channels to 4
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_traits)

    def forward(self, x):
        return self.resnet(x)


class TomatoViT(nn.Module):
    def __init__(self, num_traits):
        super(TomatoViT, self).__init__()
        self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.config.num_channels = 4  # Modify input channels to 4
        self.vit = ViTModel(self.config)
        self.fc = nn.Linear(self.vit.config.hidden_size, num_traits)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Take the [CLS] token output
        out = self.fc(cls_output)
        return out
