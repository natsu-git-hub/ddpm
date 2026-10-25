import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import get_logger

logger = get_logger(__name__)

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


class UNet(nn.Module):
    """UNet for 28x28 inputs. Encoder: 28 -> 14 -> 7, bottleneck at 7x7."""

    def __init__(self, in_channels=1, enc_channels=(64, 128), bottleneck_channels=256, time_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder: conv blocks keep spatial size, pools halve it
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.time_projs_enc = nn.ModuleList()
        ch = in_channels
        for out_ch in enc_channels:
            self.enc_blocks.append(conv_block(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            self.time_projs_enc.append(nn.Linear(time_dim, out_ch))
            ch = out_ch

        self.bottleneck = conv_block(ch, bottleneck_channels)
        self.bottleneck_time_proj = nn.Linear(time_dim, bottleneck_channels)
        ch = bottleneck_channels

        # Decoder: upconv doubles spatial size, skip concat doubles channel count
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.time_projs_dec = nn.ModuleList()
        for out_ch in reversed(enc_channels):
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(conv_block(out_ch * 2, out_ch))
            self.time_projs_dec.append(nn.Linear(time_dim, out_ch))
            ch = out_ch

        self.head = nn.Conv2d(ch, self.out_channels, kernel_size=1)

    def forward(self, x, t):
        time_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        skips = []
        for block, pool, time_proj in zip(self.enc_blocks, self.pools, self.time_projs_enc):
            x = block(x)
            time_emb_proj = time_proj(time_emb)
            x = x + time_emb_proj[:, :, None, None]
            skips.append(x)
            x = pool(x)
        
        x = self.bottleneck(x)
        time_emb_proj = self.bottleneck_time_proj(time_emb)
        x = x + time_emb_proj[:, :, None, None]
        
        for upconv, block, time_proj in zip(self.upconvs, self.dec_blocks, self.time_projs_dec):
            x = upconv(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
            time_emb_proj = time_proj(time_emb)
            x = x + time_emb_proj[:, :, None, None]
        
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = UNet()
    input = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    logger.info(f"Input shape: {input.shape}")
    logger.info(f"Time shape: {t.shape}")
    output = model(input, t)
    logger.info(f"Output shape: {output.shape}")