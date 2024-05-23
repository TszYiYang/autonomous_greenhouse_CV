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


class TomatoPCD(nn.Module):
    def __init__(self, input_dim=6, output_dim=256):
        super(TomatoPCD, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.fc_bn1 = nn.BatchNorm1d(64)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc_bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = torch.relu(self.fc_bn1(self.fc1(x)))
        x = torch.relu(self.fc_bn2(self.fc2(x)))
        x = self.fc_bn3(self.fc3(x))
        x = torch.max(x, 1, keepdim=True)[0]
        return x.squeeze(1)


class TomatoCombo(nn.Module):
    def __init__(self, num_traits):
        super(TomatoCombo, self).__init__()
        self.vit_model = TomatoViT(num_traits=num_traits)
        self.pcd_model = TomatoPCD(input_dim=6, output_dim=256)
        self.fc1 = nn.Linear(
            256 + 768, 512
        )  # Adjust the size based on concatenated features
        self.fc2 = nn.Linear(512, num_traits)

    def forward(self, original_rgbd_images, point_cloud):
        img_features = self.vit_model(original_rgbd_images)
        point_cloud_features = self.pcd_model(point_cloud)
        combined_features = torch.cat((img_features, point_cloud_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        outputs = self.fc2(x)
        return outputs
