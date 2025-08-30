import torch
import torch.nn as nn
from einops import rearrange,repeat
import math


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        video_length=None, # using frame-wise version or not
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries 
        self.video_length = video_length

        ## <num_queries> queries for each frame
        if video_length is not None: 
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1) ## B (T L) C
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents) # B L C or B (T L) C

        return latents
    
class ModalityCompressor(nn.Module):
    def __init__(self, input_dim, output_dim, num_queries=8, method='resampler'):
        super().__init__()
        self.method = method

        if method == 'resampler':
            self.resampler = Resampler(
                dim=output_dim,
                depth=4,
                dim_head=64,
                heads=8,
                num_queries=num_queries,
                embedding_dim=input_dim,
                output_dim=output_dim,
            )
        else:
            # 其他方法，比如 mean pooling
            self.pool = lambda x: x.mean(dim=1, keepdim=True)
            self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.method == 'resampler':
            return self.resampler(x)  # (B, num_queries, output_dim)
        else:
            pooled = self.pool(x)
            return self.projector(pooled)

class TemporalTransformerCondition(nn.Module):
    def __init__(self, in_channels, frame_size, max_frames, hidden_size, patch_embed_dim, proj_dim, num_heads, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_embed_dim = patch_embed_dim
        self.proj_dim = proj_dim
        self.max_frames = max_frames
        self.frame_size = frame_size  # (H, W)
        
        C, H, W = in_channels, *frame_size

        # Frame embedding: flatten each frame and map to hidden_size
        self.frame_embed = nn.Sequential(
            nn.Linear(C * H * W, patch_embed_dim),
            nn.ReLU(),
            nn.Linear(patch_embed_dim, hidden_size)
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_size))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.time_embed = nn.Sequential(
            nn.Embedding(1000, hidden_size),  # 假设DDIM最大步数为1000
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, proj_dim)

    def forward(self, video_samples, time_step):
        """
        video_samples: (B, C, T, H, W)
        time_step: (B,) int tensor or scalar representing the DDIM step
        Returns: (B, 1, proj_dim)
        """
        B, C, T, H, W = video_samples.shape
        assert T <= self.max_frames, f"Input has {T} frames, but max_frames={self.max_frames}"

        x = rearrange(video_samples, 'b c t h w -> b t (c h w)')  # (B, T, C*H*W)
        x = self.frame_embed(x)  # (B, T, hidden_size)

        # Add position encoding
        pos = self.pos_embed[:, :T, :]
        x = x + pos

        # Get time embedding
        if isinstance(time_step, int):
            time_step = torch.tensor([time_step] * B, device=x.device)
        elif isinstance(time_step, torch.Tensor) and time_step.dim() == 0:
            time_step = time_step.expand(B).to(x.device)

        t_embed = self.time_embed(time_step)  # (B, hidden_size)
        t_embed = t_embed.unsqueeze(1)  # (B, 1, hidden_size)
        x = x + t_embed  # 将时间条件加到每一帧上

        x = self.temporal_transformer(x)  # (B, T, hidden_size)
        # x = x.mean(dim=1)  # (B, hidden_size)

        return self.output_proj(x) # (B, T, proj_dim)

    @classmethod
    def from_input_shape(cls, input_shape, proj_dim=256, target_model_size_mb=100):
        """
        Creates a model based on input shape: (C, T, H, W)
        Automatically adjusts hidden size and layers to match target size.
        """
        C, T, H, W = input_shape

        # Estimate hidden_size based on target size
        base_hidden = 512
        size_multiplier = int(math.sqrt(target_model_size_mb / 20))  # heuristic
        hidden_size = base_hidden * size_multiplier  # scale up/down
        hidden_size = min(hidden_size, 1024)  # cap to avoid overgrowth
        hidden_size = max(hidden_size, 256)

        patch_embed_dim = hidden_size // 2
        num_layers = max(2, min(8, target_model_size_mb // 20))
        num_heads = max(2, hidden_size // 128)
        max_frames = T

        return cls(
            in_channels=C,
            frame_size=(H, W),
            max_frames=max_frames,
            hidden_size=hidden_size,
            patch_embed_dim=patch_embed_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    


class LearnableTokenTemporalTransformerCondition(nn.Module):
    def __init__(self, in_channels, frame_size, max_frames, hidden_size, patch_embed_dim, proj_dim, num_heads, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_embed_dim = patch_embed_dim
        self.proj_dim = proj_dim
        self.max_frames = max_frames
        self.frame_size = frame_size  # (H, W)
        
        C, H, W = in_channels, *frame_size

        # Frame embedding
        self.frame_embed = nn.Sequential(
            nn.Linear(C * H * W, patch_embed_dim),
            nn.ReLU(),
            nn.Linear(patch_embed_dim, hidden_size)
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_size))

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(1000, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Learnable projection tokens
        self.proj_tokens = nn.Parameter(torch.randn(1, max_frames, proj_dim))

        # Cross-attention: Q = proj_tokens, K/V = transformer output
        self.cross_attn = nn.MultiheadAttention(embed_dim=proj_dim, kdim=hidden_size, vdim=hidden_size, num_heads=num_heads, batch_first=True)

        # Optional: project proj_tokens to match hidden_size before attention
        self.proj_token_proj = nn.Linear(proj_dim, proj_dim)

    def forward(self, video_samples, time_step):
        """
        video_samples: (B, C, T, H, W)
        time_step: (B,) int tensor or scalar representing the DDIM step
        Returns: (B, T, proj_dim)
        """
        B, C, T, H, W = video_samples.shape
        assert T <= self.max_frames, f"Input has {T} frames, but max_frames={self.max_frames}"

        # Frame embedding
        x = rearrange(video_samples, 'b c t h w -> b t (c h w)')
        x = self.frame_embed(x)  # (B, T, hidden_size)

        # Add position encoding
        pos = self.pos_embed[:, :T, :]
        x = x + pos

        # Time embedding
        if isinstance(time_step, int):
            time_step = torch.tensor([time_step] * B, device=x.device)
        elif isinstance(time_step, torch.Tensor) and time_step.dim() == 0:
            time_step = time_step.expand(B).to(x.device)
        t_embed = self.time_embed(time_step).unsqueeze(1)  # (B, 1, hidden_size)

        x = x + t_embed  # Add time embedding to frame features

        # Temporal Transformer
        x = self.temporal_transformer(x)  # (B, T, hidden_size)

        # Prepare projection tokens
        proj_tokens = repeat(self.proj_tokens[:, :T, :], '1 t d -> b t d', b=B)  # (B, T, proj_dim)
        proj_tokens = self.proj_token_proj(proj_tokens)  # optional projection

        # Cross-attention: proj_tokens attend to x
        out, _ = self.cross_attn(query=proj_tokens, key=x, value=x)  # (B, T, proj_dim)

        return out  # (B, T, proj_dim)

    @classmethod
    def from_input_shape(cls, input_shape, proj_dim=256, target_model_size_mb=100):
        C, T, H, W = input_shape

        base_hidden = 512
        size_multiplier = int(math.sqrt(target_model_size_mb / 20))
        hidden_size = base_hidden * size_multiplier
        hidden_size = min(hidden_size, 1024)
        hidden_size = max(hidden_size, 256)

        patch_embed_dim = hidden_size // 2
        num_layers = max(2, min(8, target_model_size_mb // 20))
        num_heads = max(2, hidden_size // 128)
        max_frames = T

        return cls(
            in_channels=C,
            frame_size=(H, W),
            max_frames=max_frames,
            hidden_size=hidden_size,
            patch_embed_dim=patch_embed_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    

class VideoTimeStepScheduler:
    def __init__(
        self,
        total_ddpm_steps: int = 1000,
        ddim_steps_list: list = [5, 10, 50],
        prefer_late_steps: bool = True,
        device: torch.device = torch.device("cuda"),
        final_phase_ratio: float = 0.9, 
    ):
        """
        Args:
            total_ddpm_steps: 总共 diffusion 步数（通常是1000）
            ddim_steps_list: 希望支持的推理 DDIM 步数（训练时用这些的时间步）
            prefer_late_steps: 是否偏向后期时间步（更稳定、噪声小）
            device: 所有 t_vid 的生成设备
        """
        self.total_ddpm_steps = total_ddpm_steps
        self.ddim_steps_list = ddim_steps_list
        self.device = device
        self.final_phase_ratio = final_phase_ratio

        # 构造所有 candidate 时间步
        self.candidate_ts = self.build_candidate_timestep_set()
        
        # 构造采样分布
        self.prefer_late_steps = prefer_late_steps
        self.probs = self.build_sampling_weights()

    def build_candidate_timestep_set(self):
        all_ts = []
        for steps in self.ddim_steps_list:
            ts = torch.linspace(0, self.total_ddpm_steps - 1, steps=steps).long()
            all_ts.append(ts)
        candidate_ts = torch.cat(all_ts).unique()
        return candidate_ts

    def build_sampling_weights(self):
        if not self.prefer_late_steps:
            return torch.ones(len(self.candidate_ts)) / len(self.candidate_ts)
        else:
            # 权重偏向后期时间步
            weights = torch.linspace(0.2, 1.0, steps=len(self.candidate_ts))
            return weights / weights.sum()

    def sample_train_timestep(self, batch_size: int):
        """
        训练时采样时间步
        """
        sampled_indices = torch.multinomial(self.probs, batch_size, replacement=True)
        return self.candidate_ts[sampled_indices].to(self.device)

    def get_ddim_schedule(self, ddim_steps: int):
        """
        推理时返回一个固定的时间步 schedule（例如 DDIM=10）
        """
        return torch.linspace(0, self.total_ddpm_steps - 1, steps=ddim_steps).long().to(self.device)

    def get_infer_timestep(self, batch_size: int, t_value: int):
        """
        推理时固定某个时间步
        """
        return torch.full((batch_size,), t_value, device=self.device, dtype=torch.long)
    
    def get_near_final_timestep(self, batch_size: int):
        """
        返回靠近最后阶段的时间步，比如最后10%的时间步。
        """
        start_idx = int(self.total_ddpm_steps * self.final_phase_ratio)
        t = torch.randint(low=start_idx, high=self.total_ddpm_steps, size=(batch_size,), device=self.device)
        return t
    
