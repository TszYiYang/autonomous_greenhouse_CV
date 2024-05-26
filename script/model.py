import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ViTModel, ViTConfig


class TomatoNet(nn.Module):
    """
    TomatoNet is a modified ResNet50 model that takes 4-channel input (RGBD) images
    and predicts plant traits.

    Attributes:
        resnet (nn.Module): The ResNet50 model with modified input and output layers.
    """

    def __init__(self, num_traits):
        super(TomatoNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Adjust input channels to 4
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_traits)

    def forward(self, x):
        out = self.resnet(x)
        # print(f"TomatoNet - Output Shape: {out.shape}")
        return out


class TomatoViT(nn.Module):
    """
    TomatoViT is a Vision Transformer (ViT) model that takes 4-channel input (RGBD) images
    and predicts plant traits.

    Attributes:
        vit (ViTModel): The Vision Transformer model.
        layer_norm (nn.LayerNorm): Layer normalization for the output.
        fc (nn.Linear): Fully connected layer to map to the desired output.
    """

    def __init__(self, num_traits):
        super(TomatoViT, self).__init__()
        self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.config.num_channels = 4  # Modify input channels to 4
        self.vit = ViTModel(self.config)
        self.layer_norm = nn.LayerNorm(
            self.vit.config.hidden_size
        )  # Layer normalization
        self.fc = nn.Linear(
            self.vit.config.hidden_size, num_traits
        )  # Modify output features to num_traits

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Take the [CLS] token output
        # print(f"TomatoViT - CLS Output Shape: {cls_output.shape}")
        cls_output = self.layer_norm(cls_output)  # Apply layer normalization
        out = self.fc(cls_output)
        # print(f"TomatoViT - Output Shape: {out.shape}")
        return out


class TomatoPCD(nn.Module):
    """
    TomatoPCD processes point cloud data to extract features useful for plant trait prediction.

    Attributes:
        fc1, fc2, fc3, fc_final (nn.Linear): Fully connected layers.
        bn1, bn2, bn3, bn_final (nn.BatchNorm1d): Batch normalization layers.
        pool (nn.AdaptiveMaxPool1d): Adaptive max pooling layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, input_dim=6, output_dim=256):
        super(TomatoPCD, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc_final = nn.Linear(256 * 10000, output_dim)  # Flattened dimension
        self.bn_final = nn.BatchNorm1d(output_dim)
        self.pool = nn.AdaptiveMaxPool1d(10000)  # Downsampling to 10,000 points
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print(f"Original Point Cloud Shape: {x.shape}")
        x = self.pool(x)
        x = x.transpose(1, 2)  # [B, 6, 10000] -> [B, 10000, 6]
        # print(f"After Adaptive Pooling: {x.shape}")

        x = torch.relu(self.fc1(x))  # [B, 10000, 6] -> [B, 10000, 64]
        x = x.transpose(1, 2)  # [B, 10000, 64] -> [B, 64, 10000]
        x = self.bn1(x)
        x = x.transpose(1, 2)  # [B, 64, 10000] -> [B, 10000, 64]
        # print(f"After bn1: {x.shape}")

        x = torch.relu(self.fc2(x))  # [B, 10000, 64] -> [B, 10000, 128]
        x = x.transpose(1, 2)  # [B, 10000, 128] -> [B, 128, 10000]
        x = self.bn2(x)
        x = x.transpose(1, 2)  # [B, 128, 10000] -> [B, 10000, 128]
        # print(f"After bn2: {x.shape}")

        x = torch.relu(self.fc3(x))  # [B, 10000, 128] -> [B, 10000, 256]
        x = x.transpose(1, 2)  # [B, 10000, 256] -> [B, 256, 10000]
        x = self.bn3(x)
        x = x.transpose(1, 2)  # [B, 256, 10000] -> [B, 10000, 256]
        # print(f"After bn3: {x.shape}")

        x = x.reshape(x.size(0), -1)  # Flatten to [B, 256 * 10000]
        # print(f"After Flatten: {x.shape}")

        x = self.dropout(x)
        x = torch.relu(self.bn_final(self.fc_final(x)))  # [B, 256]
        # print(f"After fc_final: {x.shape}")

        return x


class PointNet(nn.Module):
    """
    PointNet processes point cloud data and extracts features for plant trait prediction.

    Attributes:
        conv1, conv2, conv3 (nn.Conv1d): Convolutional layers.
        bn1, bn2, bn3 (nn.BatchNorm1d): Batch normalization layers.
        fc1, fc2, fc3 (nn.Linear): Fully connected layers.
        bn4, bn5 (nn.BatchNorm1d): Batch normalization layers.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, k=256):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, k, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(k)
        self.fc1 = nn.Linear(k, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # print(f"Original Point Cloud Shape: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 6, N] -> [B, 64, N]
        # print(f"After conv1: {x.shape}")
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, N] -> [B, 128, N]
        # print(f"After conv2: {x.shape}")
        x = self.bn3(self.conv3(x))  # [B, 128, N] -> [B, k, N]
        # print(f"After conv3: {x.shape}")
        x = torch.max(x, 2, keepdim=False)[0]  # [B, k, N] -> [B, k]
        # print(f"After max pooling: {x.shape}")
        x = F.relu(self.bn4(self.fc1(x)))  # [B, k] -> [B, 512]
        # print(f"After fc1: {x.shape}")
        x = F.relu(self.bn5(self.fc2(x)))  # [B, 512] -> [B, 256]
        # print(f"After fc2: {x.shape}")
        x = self.dropout(x)
        x = self.fc3(x)  # [B, 256]
        # print(f"PointNet Output Shape: {x.shape}")
        return x

    def load_pretrained_weights(self, weights_path):
        """
        Load pretrained weights into the PointNet model, ignoring mismatched keys.

        Args:
            weights_path (str): Path to the file containing the pretrained weights.
        """
        state_dict = torch.load(weights_path)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Pretrained weights loaded from {weights_path}")


class Attention(nn.Module):
    """
    Attention mechanism to combine image features and point cloud features.

    Attributes:
        query, key, value (nn.Linear): Linear layers for query, key, and value projections.
        softmax (nn.Softmax): Softmax layer for attention scores.
    """

    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = self.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        )
        context = torch.matmul(attention_scores, value)
        # print(f"Attention - Context Shape: {context.shape}")
        return context


class TomatoComboVitTomatoPCD(nn.Module):
    """
    TomatoComboVitTomatoPCD combines Vision Transformer (ViT) and TomatoPCD models
    to predict plant traits using both image and point cloud data.

    Attributes:
        vit_model (TomatoViT): Vision Transformer model for image data.
        pcd_model (TomatoPCD): Model for point cloud data.
        attention (Attention): Attention mechanism to combine features.
        fc1, fc2 (nn.Linear): Fully connected layers.
        bn1 (nn.BatchNorm1d): Batch normalization layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, num_traits):
        super(TomatoComboVitTomatoPCD, self).__init__()
        self.vit_model = TomatoViT(num_traits=256)
        self.pcd_model = TomatoPCD(input_dim=6, output_dim=256)
        self.attention = Attention(512, 512)
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_traits)
        self.dropout = nn.Dropout(0.5)

    def forward(self, original_rgbd_images, point_cloud):
        img_features = self.vit_model(original_rgbd_images)
        # print(f"Image Features Shape: {img_features.shape}")

        batch_size = point_cloud.size(0)
        downsampled_point_cloud = F.adaptive_max_pool1d(
            point_cloud.view(batch_size, 6, -1), output_size=10000
        )
        # print(f"Downsampled Point Cloud Shape: {downsampled_point_cloud.shape}")

        point_cloud_features = self.pcd_model(downsampled_point_cloud)
        # print(f"Point Cloud Features Shape: {point_cloud_features.shape}")

        combined_features = torch.cat((img_features, point_cloud_features), dim=1)
        # print(f"Combined Features Shape: {combined_features.shape}")

        attention_output = self.attention(combined_features)
        # print(f"Attention Output Shape: {attention_output.shape}")

        x = torch.relu(self.bn1(self.fc1(attention_output)))
        # print(f"After fc1: {x.shape}")
        x = self.dropout(x)
        outputs = self.fc2(x)
        # print(f"After fc2: {outputs.shape}")
        return outputs


class TomatoComboVitPointNet(nn.Module):
    """
    TomatoComboVitPointNet combines Vision Transformer (ViT) and PointNet models
    to predict plant traits using both image and point cloud data.

    Attributes:
        vit_model (TomatoViT): Vision Transformer model for image data.
        pcd_model (PointNet): Model for point cloud data.
        attention (Attention): Attention mechanism to combine features.
        fc1, fc2 (nn.Linear): Fully connected layers.
        bn1 (nn.BatchNorm1d): Batch normalization layer.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, num_traits):
        super(TomatoComboVitPointNet, self).__init__()
        self.vit_model = TomatoViT(num_traits=256)
        self.pcd_model = PointNet(k=256)
        self.attention = Attention(512, 512)
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_traits)
        self.dropout = nn.Dropout(0.5)

    def forward(self, original_rgbd_images, point_cloud):
        img_features = self.vit_model(original_rgbd_images)
        # print(f"Image Features Shape: {img_features.shape}")

        batch_size = point_cloud.size(0)
        downsampled_point_cloud = F.adaptive_max_pool1d(
            point_cloud.view(batch_size, 6, -1), output_size=10000
        )
        # print(f"Downsampled Point Cloud Shape: {downsampled_point_cloud.shape}")

        point_cloud_features = self.pcd_model(downsampled_point_cloud)
        # print(f"Point Cloud Features Shape: {point_cloud_features.shape}")

        combined_features = torch.cat((img_features, point_cloud_features), dim=1)
        # print(f"Combined Features Shape: {combined_features.shape}")

        attention_output = self.attention(combined_features)
        # print(f"Attention Output Shape: {attention_output.shape}")

        x = torch.relu(self.bn1(self.fc1(attention_output)))
        # print(f"After fc1: {x.shape}")
        x = self.dropout(x)
        outputs = self.fc2(x)
        # print(f"After fc2: {outputs.shape}")
        return outputs
