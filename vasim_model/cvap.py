import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderCNN(nn.Module):
    def __init__(self, in_channels=4, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
            nn.Flatten(),                 # [B, 64]
            nn.Linear(64, embedding_dim)  # [B, D]
        )

    def forward(self, x):
        return self.encoder(x)

class PoseEncoderMLP(nn.Module):
    def __init__(self, input_dim=7, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class ContrastiveModel(nn.Module):
    def __init__(self, image_channels=4, pose_dim=7, embedding_dim=128):
        super().__init__()
        self.image_encoder = ImageEncoderCNN(in_channels=image_channels, embedding_dim=embedding_dim)
        self.pose_encoder = PoseEncoderMLP(input_dim=pose_dim, embedding_dim=embedding_dim)

    def forward(self, image_feat, pose):
        """
        Inputs:
            image_feat: Tensor [B, C=4, H=16, W=16]
            pose:       Tensor [B, D=7]
        Returns:
            z_img:  [B, embedding_dim]
            z_pose: [B, embedding_dim]
        """
        z_img = self.image_encoder(image_feat)
        z_pose = self.pose_encoder(pose)
        return z_img, z_pose

    def nt_xent_loss(self, z_img, z_pose, temperature=0.07):
        """
        InfoNCE loss between image and pose embeddings
        Inputs:
            z_img:  [B, D]
            z_pose: [B, D]
        Returns:
            scalar tensor loss
        """
        z_img = F.normalize(z_img, dim=1)
        z_pose = F.normalize(z_pose, dim=1)

        logits = torch.matmul(z_img, z_pose.T)  # sim matrix [B, B]
        logits /= temperature

        labels = torch.arange(z_img.size(0), device=z_img.device)
        loss_i2p = F.cross_entropy(logits, labels)
        loss_p2i = F.cross_entropy(logits.T, labels)

        return (loss_i2p + loss_p2i) / 2