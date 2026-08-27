import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """Bilinear upsample ×2 followed by Conv-BN-ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


class CDDecoder(nn.Module):
    """
    Lightweight change-detection decoder.

    Input : |F_T1 - F_T2|  shape (B, N, D)   N = grid*grid, D = feature dim
    Output: logit map       shape (B, 1, 256, 256)

    Channel progression: D → 256 → 128 → 64 → 32 → 1
    Spatial progression: 16 → 32 → 64 → 128 → 256
    """

    def __init__(self, in_dim: int = 768, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.up1  = UpBlock(in_dim, 256)
        self.up2  = UpBlock(256,    128)
        self.up3  = UpBlock(128,     64)
        self.up4  = UpBlock( 64,     32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        B, N, D = diff.shape
        x = diff.permute(0, 2, 1).reshape(B, D, self.grid, self.grid)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)
