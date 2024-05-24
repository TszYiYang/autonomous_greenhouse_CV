import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        residual = x
        #print(f"ResidualBlock - input: {x.shape}")
        out = self.fc1(x)
        #print(f"ResidualBlock - after fc1: {out.shape}")
        out = out.transpose(1, 2)
        out = self.relu(self.bn1(out))
        out = out.transpose(1, 2)
        #print(f"ResidualBlock - after bn1 and relu: {out.shape}")
        out = self.fc2(out)
        #print(f"ResidualBlock - after fc2: {out.shape}")
        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)
        #print(f"ResidualBlock - after bn2: {out.shape}")
        out += residual
        out = self.relu(out)
        #print(f"ResidualBlock - after adding residual and relu: {out.shape}")
        return out


# TomatoNet Model
class TomatoNet(nn.Module):
    def __init__(self, num_traits):
        super(TomatoNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Adjust input channels to 4
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_traits)

    def forward(self, x):
        out = self.resnet(x)
        #print(f"TomatoNet - Output Shape: {out.shape}")
        return out

# TomatoViT Model
class TomatoViT(nn.Module):
    def __init__(self, num_traits):
        super(TomatoViT, self).__init__()
        self.config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.config.num_channels = 4  # Modify input channels to 4
        self.vit = ViTModel(self.config)
        self.layer_norm = nn.LayerNorm(self.vit.config.hidden_size)  # Layer normalization
        self.fc = nn.Linear(self.vit.config.hidden_size, 256)  # Modify output features to 256

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Take the [CLS] token output
        #print(f"TomatoViT - CLS Output Shape: {cls_output.shape}")
        cls_output = self.layer_norm(cls_output)  # Apply layer normalization
        out = self.fc(cls_output)
        #print(f"TomatoViT - Output Shape: {out.shape}")
        return out

# TomatoPCD Model
class TomatoPCD(nn.Module):
    def __init__(self, input_dim=6, output_dim=256):
        super(TomatoPCD, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.residual_block1 = ResidualBlock(64, 64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.residual_block2 = ResidualBlock(128, 128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc_final = nn.Linear(256 * 64, output_dim)  # Flattened dimension
        self.bn_final = nn.BatchNorm1d(output_dim)
        self.pool = nn.AdaptiveMaxPool1d(64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #print(f"Original Point Cloud Shape: {x.shape}")
        x = self.pool(x.transpose(2, 1))  # [B, 8294400, 6] -> [B, 6, 64]
        x = x.transpose(1, 2)  # [B, 6, 64] -> [B, 64, 6]
        #print(f"After Adaptive Pooling: {x.shape}")

        x = x.reshape(-1, x.size(-1))  # Flatten the last two dimensions: [B*64, 6]
        #print(f"Before fc1: {x.shape}")
        x = torch.relu(self.fc1(x))  # [B*64, 6] -> [B*64, 64]
        #print(f"After fc1: {x.shape}")
        x = x.reshape(-1, 64, 64)  # Reshape back: [B, 64, 64]
        x = self.bn1(x)
        #print(f"After bn1: {x.shape}")

        x = self.residual_block1(x)
        #print(f"After Residual Block 1: {x.shape}")

        x = x.reshape(-1, x.size(-1))  # Flatten the last two dimensions: [B*64, 64]
        #print(f"Before fc2: {x.shape}")
        x = torch.relu(self.fc2(x))  # [B*64, 64] -> [B*64, 128]
        #print(f"After fc2: {x.shape}")
        x = x.reshape(-1, 128, 64)  # Reshape back: [B, 128, 64]
        #print(f"Before bn2: {x.shape}")
        x = self.bn2(x)
        #print(f"After bn2: {x.shape}")

        x = x.transpose(1, 2)  # [B, 128, 64] -> [B, 64, 128] to match expected input for ResidualBlock
        #print(f"Before Residual Block 2: {x.shape}")
        x = self.residual_block2(x)
        #print(f"After Residual Block 2: {x.shape}")

        x = x.transpose(1, 2)  # Transpose back [B, 64, 128] -> [B, 128, 64]
        x = x.reshape(-1, 128)  # Flatten the last two dimensions: [B*64, 128]
        #print(f"Before fc3: {x.shape}")
        x = torch.relu(self.fc3(x))  # [B*64, 128] -> [B*64, 256]
        #print(f"After fc3: {x.shape}")
        x = x.reshape(-1, 256, 64)  # Reshape back: [B, 256, 64]
        x = self.bn3(x)
        #print(f"After bn3: {x.shape}")

        x = x.view(x.size(0), -1)  # Flatten to [B, 256 * 64]
        #print(f"After Flatten: {x.shape}")

        x = self.dropout(x)
        x = torch.relu(self.bn_final(self.fc_final(x)))  # [B, 256]
        #print(f"After fc_final: {x.shape}")

        return x


# Attention Mechanism
class Attention(nn.Module):
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
        attention_scores = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5))
        context = torch.matmul(attention_scores, value)
        #print(f"Attention - Context Shape: {context.shape}")
        return context

# TomatoCombo Model
class TomatoCombo(nn.Module):
    def __init__(self, num_traits):
        super(TomatoCombo, self).__init__()
        self.vit_model = TomatoViT(num_traits=num_traits)
        self.pcd_model = TomatoPCD(input_dim=6, output_dim=256)
        self.attention = Attention(512, 512)
        self.fc1 = nn.Linear(256 + 256, 512)  # Adjust the size based on concatenated features
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_traits)
        self.dropout = nn.Dropout(0.5)

    def forward(self, original_rgbd_images, point_cloud):
        img_features = self.vit_model(original_rgbd_images)
        #print(f"Image Features Shape: {img_features.shape}")
        point_cloud_features = self.pcd_model(point_cloud)
        #print(f"Point Cloud Features Shape: {point_cloud_features.shape}")
        
        # Apply attention mechanism
        combined_features = torch.cat((img_features, point_cloud_features), dim=1)
        attention_output = self.attention(combined_features)
        #print(f"Attention Output Shape: {attention_output.shape}")

        x = torch.relu(self.bn1(self.fc1(attention_output)))
        #print(f"After fc1: {x.shape}")
        x = self.dropout(x)
        outputs = self.fc2(x)
        #print(f"After fc2: {outputs.shape}")
        return outputs
