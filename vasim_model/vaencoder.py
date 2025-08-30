import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, input_channels=64, input_size=(8, 8), output_dim=128):
        """
        input_channels: C from (B*T, C, H, W)
        input_size: (H, W) from VAE feature map
        """
        super().__init__()
        C, (H, W) = input_channels, input_size
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(C * H * W, output_dim)

    def forward(self, x):
        x = self.flatten(x)  # Flatten (B*T, C, H, W) → (B*T, C*H*W)
        return self.linear(x)

class PoseEncoder(nn.Module):
    def __init__(self, input_dim=7, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class ContrastiveModel(nn.Module):
    def __init__(self, 
                 vae_feature_shape=(64, 8, 8),  # (C, H, W)
                 pose_input_dim=7, 
                 embedding_dim=128):
        super().__init__()
        C, H, W = vae_feature_shape
        self.image_encoder = ImageEncoder(input_channels=C, input_size=(H, W), output_dim=embedding_dim)
        self.pose_encoder = PoseEncoder(input_dim=pose_input_dim, output_dim=embedding_dim)

    def forward(self, image_feat, pose):
        z_img = self.image_encoder(image_feat)
        z_pose = self.pose_encoder(pose)
        return z_img, z_pose

    def nt_xent_loss(self, z_img, z_pose, temperature=0.07):
        z_img = F.normalize(z_img, dim=1)
        z_pose = F.normalize(z_pose, dim=1)

        sim_matrix = torch.matmul(z_img, z_pose.T)
        logits = sim_matrix / temperature

        labels = torch.arange(z_img.size(0)).to(z_img.device)
        loss_i2p = F.cross_entropy(logits, labels)
        loss_p2i = F.cross_entropy(logits.T, labels)

        return (loss_i2p + loss_p2i) / 2
    

from tqdm import tqdm
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, poses in tqdm(dataloader):
        images = images.to(device)
        poses = poses.to(device)

        optimizer.zero_grad()
        z_img, z_pose = model(images, poses)
        loss = model.nt_xent_loss(z_img, z_pose)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
