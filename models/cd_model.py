import torch
import torch.nn as nn


class CDModel(nn.Module):
    """
    Siamese change-detection model.

    T1, T2  →  shared encoder  →  |F_T1 - F_T2|  →  decoder  →  logit map
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        f1   = self.encoder.forward_features(t1)['x_norm_patchtokens']
        f2   = self.encoder.forward_features(t2)['x_norm_patchtokens']
        diff = torch.abs(f1 - f2)
        return self.decoder(diff)


# ---------------------------------------------------------------------------
# CDLoRA
# ---------------------------------------------------------------------------

class CDLoRAModel(nn.Module):
    """
    CDLoRA: blocks outside [lora_start, lora_end) frozen, inside have LoRA whose scale is
    modulated by input-level diff statistics.

    f = [mean, std, sparsity, kurtosis] of |t1 - t2| (pixel-level, before transformer)
    γ_q = MLP_q(f) ∈ R^r,  γ_v = MLP_v(f) ∈ R^r
    For each deep block: delta = B(A(x) * γ)  [scale per rank component]
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32,
                 lora_start: int = 8, lora_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )

    @staticmethod
    def compute_diff_features(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Return (B, 4): [mean, std, sparsity, kurtosis] of |t1-t2|."""
        diff = (t1 - t2).abs().reshape(t1.shape[0], -1)
        mean     = diff.mean(dim=1)
        std      = diff.std(dim=1) + 1e-8
        sparsity = (diff < 0.1).float().mean(dim=1)
        centered = diff - mean.unsqueeze(1)
        kurt     = centered.pow(4).mean(dim=1) / std.pow(4) - 3.0
        return torch.stack([mean, std, sparsity, kurt], dim=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.compute_diff_features(t1, t2)

        gamma_q = self.scale_mlp_q(f)
        gamma_v = self.scale_mlp_v(f)

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        diff = (h1[:, 1:] - h2[:, 1:]).abs()
        return self.decoder(diff)


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

class CDLoRAScalarGammaModel(nn.Module):
    """
    Table 2b variant (b): CDLoRA with scalar γ (γ ∈ R¹, shared across all rank dims).
    Isolates the effect of rank-wise decomposition vs a single global scale.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32,
                 lora_start: int = 8, lora_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.rank       = rank
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = CDLoRAModel.compute_diff_features(t1, t2)

        # scalar → expand to (B, rank) so set_scale receives expected shape
        gamma_q = self.scale_mlp_q(f).expand(-1, self.rank).contiguous()
        gamma_v = self.scale_mlp_v(f).expand(-1, self.rank).contiguous()

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        diff = (h1[:, 1:] - h2[:, 1:]).abs()
        return self.decoder(diff)


class CDLoRA2DGammaModel(nn.Module):
    """
    Table 2b variant (c): CDLoRA with rank-wise γ but only 2D diff features (mean, std).
    Isolates the contribution of sparsity + kurtosis in the 4D feature set.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32,
                 lora_start: int = 8, lora_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )

    @staticmethod
    def compute_diff_features_2d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Return (B, 2): [mean, std] of |t1-t2|."""
        diff = (t1 - t2).abs().reshape(t1.shape[0], -1)
        mean = diff.mean(dim=1)
        std  = diff.std(dim=1) + 1e-8
        return torch.stack([mean, std], dim=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.compute_diff_features_2d(t1, t2)

        gamma_q = self.scale_mlp_q(f)
        gamma_v = self.scale_mlp_v(f)

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        diff = (h1[:, 1:] - h2[:, 1:]).abs()
        return self.decoder(diff)


# ---------------------------------------------------------------------------
# Bi-temporal Cross-Attention (BCA)
# ---------------------------------------------------------------------------

class BiTemporalCrossAttn(nn.Module):
    """
    单向轻量跨时相交叉注意力：x_q 从 x_kv 聚合上下文。

    W_O 零初始化 → 初始等价于恒等映射。
    alpha 初始化为 0.01，让模型自主决定跨时相交互强度。
    """

    def __init__(self, dim: int = 768, head_dim: int = 64):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.norm  = nn.LayerNorm(dim)
        self.W_Q   = nn.Linear(dim, head_dim, bias=False)
        self.W_K   = nn.Linear(dim, head_dim, bias=False)
        self.W_V   = nn.Linear(dim, head_dim, bias=False)
        self.W_O   = nn.Linear(head_dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.W_O.weight)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        Q    = self.W_Q(self.norm(x_q))
        K    = self.W_K(self.norm(x_kv))
        V    = self.W_V(self.norm(x_kv))
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        ctx  = self.W_O(attn @ V)
        return x_q + self.alpha * ctx


# ---------------------------------------------------------------------------
# TIDE: CDLoRA + BCA
# ---------------------------------------------------------------------------

class CDLoRATWModel(nn.Module):
    """
    Token-Wise CDLoRA: γ ∈ (B, N, r) — each patch gets its own scale vector.

    Diff features are computed per-patch from the input pixel difference,
    then fed through the same shared MLP to produce token-wise γ.
    Parameter count is identical to CDLoRAModel (MLP weights unchanged).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32, patch_size: int = 14,
                 lora_start: int = 8, lora_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.patch_size = patch_size
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )

    @staticmethod
    def compute_patch_diff_features(t1: torch.Tensor, t2: torch.Tensor,
                                    patch_size: int = 14) -> torch.Tensor:
        """
        Compute per-patch diff statistics.
        Input : t1, t2 ∈ (B, C, H, W)
        Output: (B, N, 4)  — N = (H/patch_size) * (W/patch_size)
        """
        B, C, H, W = t1.shape
        ph = pw = patch_size
        diff = (t1 - t2).abs()                                      # (B, C, H, W)
        diff = diff.reshape(B, C, H // ph, ph, W // pw, pw)
        diff = diff.permute(0, 2, 4, 1, 3, 5).contiguous()          # (B, nh, nw, C, ph, pw)
        diff = diff.reshape(B, (H // ph) * (W // pw), C * ph * pw)  # (B, N, C*ph*pw)

        mean     = diff.mean(dim=-1)
        std      = diff.std(dim=-1) + 1e-8
        sparsity = (diff < 0.1).float().mean(dim=-1)
        centered = diff - mean.unsqueeze(-1)
        kurt     = centered.pow(4).mean(dim=-1) / std.pow(4) - 3.0
        return torch.stack([mean, std, sparsity, kurt], dim=-1)      # (B, N, 4)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.compute_patch_diff_features(t1, t2, self.patch_size)  # (B, N, 4)

        gamma_q = self.scale_mlp_q(f)   # (B, N, r)
        gamma_v = self.scale_mlp_v(f)

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        diff = (h1[:, 1:] - h2[:, 1:]).abs()
        return self.decoder(diff)


class CDLoRABCAModel(nn.Module):
    """
    TIDE: CDLoRA + Bi-temporal Cross-Attention.

    CDLoRA injected into blocks [lora_start, lora_end).
    BCA applied after each block in [bca_start, bca_end).

    Trainable params (rank=8, mlp_hidden=32, bca_head_dim=64): ~2.65M
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32, bca_head_dim: int = 64,
                 lora_start: int = 8, lora_end: int = 12,
                 bca_start: int = 8, bca_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.bca_start  = bca_start
        self.bca_end    = bca_end

        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )

        n_bca = bca_end - bca_start
        self.bca = nn.ModuleList([
            BiTemporalCrossAttn(dim=768, head_dim=bca_head_dim)
            for _ in range(n_bca)
        ])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = CDLoRAModel.compute_diff_features(t1, t2)
        gamma_q = self.scale_mlp_q(f)
        gamma_v = self.scale_mlp_v(f)

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

            if self.bca_start <= i < self.bca_end:
                bca  = self.bca[i - self.bca_start]
                p1   = h1[:, 1:]
                p2   = h2[:, 1:]
                new_p1 = bca(p1, p2)
                new_p2 = bca(p2, p1)
                h1 = torch.cat([h1[:, :1], new_p1], dim=1)
                h2 = torch.cat([h2[:, :1], new_p2], dim=1)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        diff = (h1[:, 1:] - h2[:, 1:]).abs()
        return self.decoder(diff)


# ---------------------------------------------------------------------------
# CDLoRA2DBCA: rank-wise 2D γ (mean, std) + BCA
# ---------------------------------------------------------------------------

class CDLoRA2DBCAModel(nn.Module):
    """Rank-wise 2D γ (mean, std only) + BCA. Design ablation: simpler γ vs CDLoRA-TW."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32, bca_head_dim: int = 64,
                 lora_start: int = 8, lora_end: int = 12,
                 bca_start: int = 8, bca_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.bca_start  = bca_start
        self.bca_end    = bca_end
        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        n_bca = bca_end - bca_start
        self.bca = nn.ModuleList([
            BiTemporalCrossAttn(dim=768, head_dim=bca_head_dim)
            for _ in range(n_bca)
        ])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = CDLoRA2DGammaModel.compute_diff_features_2d(t1, t2)
        gamma_q = self.scale_mlp_q(f)
        gamma_v = self.scale_mlp_v(f)

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

            if self.bca_start <= i < self.bca_end:
                bca = self.bca[i - self.bca_start]
                p1, p2 = h1[:, 1:], h2[:, 1:]
                h1 = torch.cat([h1[:, :1], bca(p1, p2)], dim=1)
                h2 = torch.cat([h2[:, :1], bca(p2, p1)], dim=1)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        return self.decoder((h1[:, 1:] - h2[:, 1:]).abs())


# ---------------------------------------------------------------------------
# CDLoRAScalar2DBCA: scalar γ (R¹) from 2D image-level input + BCA
# Table 6 ablation: (b) Scalar γ — isolates rank-wise decomposition benefit
# ---------------------------------------------------------------------------

class CDLoRAScalar2DBCAModel(nn.Module):
    """Scalar γ ∈ R¹ (broadcast to rank) from 2D image-level [mean,std] + BCA."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32, bca_head_dim: int = 64,
                 lora_start: int = 8, lora_end: int = 12,
                 bca_start: int = 8, bca_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.rank       = rank
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.bca_start  = bca_start
        self.bca_end    = bca_end
        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        n_bca = bca_end - bca_start
        self.bca = nn.ModuleList([
            BiTemporalCrossAttn(dim=768, head_dim=bca_head_dim)
            for _ in range(n_bca)
        ])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = CDLoRA2DGammaModel.compute_diff_features_2d(t1, t2)
        gamma_q = self.scale_mlp_q(f).expand(-1, self.rank).contiguous()
        gamma_v = self.scale_mlp_v(f).expand(-1, self.rank).contiguous()

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

            if self.bca_start <= i < self.bca_end:
                bca = self.bca[i - self.bca_start]
                p1, p2 = h1[:, 1:], h2[:, 1:]
                h1 = torch.cat([h1[:, :1], bca(p1, p2)], dim=1)
                h2 = torch.cat([h2[:, :1], bca(p2, p1)], dim=1)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        return self.decoder((h1[:, 1:] - h2[:, 1:]).abs())


# ---------------------------------------------------------------------------
# DeepLoraBCA: plain LoRA (no γ) + BCA
# ---------------------------------------------------------------------------

class DeepLoraBCAModel(nn.Module):
    """Deep LoRA (blocks [lora_start, lora_end), no γ) + BCA."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 bca_head_dim: int = 64,
                 lora_start: int = 6, lora_end: int = 8,
                 bca_start: int = 6, bca_end: int = 8):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.bca_start = bca_start
        self.bca_end   = bca_end

        self.bca = nn.ModuleList([
            BiTemporalCrossAttn(dim=768, head_dim=bca_head_dim)
            for _ in range(bca_end - bca_start)
        ])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            h1 = block(h1)
            h2 = block(h2)

            if self.bca_start <= i < self.bca_end:
                bca    = self.bca[i - self.bca_start]
                p1, p2 = h1[:, 1:], h2[:, 1:]
                h1 = torch.cat([h1[:, :1], bca(p1, p2)], dim=1)
                h2 = torch.cat([h2[:, :1], bca(p2, p1)], dim=1)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        return self.decoder((h1[:, 1:] - h2[:, 1:]).abs())


# ---------------------------------------------------------------------------
# TIDE-TW: CDLoRA (token-wise γ) + BCA
# ---------------------------------------------------------------------------

class CDLoRABCATWModel(nn.Module):
    """
    TIDE-TW: token-wise CDLoRA + Bi-temporal Cross-Attention.

    Token-wise γ ∈ (B, N, r) — each patch gets its own scale vector.
    CDLoRA injected into blocks [lora_start, lora_end).
    BCA applied after each block in [bca_start, bca_end).
    Parameter count ≈ CDLoRABCAModel (MLP weights identical, BCA unchanged).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 rank: int = 8, mlp_hidden: int = 32, bca_head_dim: int = 64,
                 patch_size: int = 14,
                 lora_start: int = 8, lora_end: int = 12,
                 bca_start: int = 8, bca_end: int = 12):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.patch_size = patch_size
        self.lora_start = lora_start
        self.lora_end   = lora_end
        self.bca_start  = bca_start
        self.bca_end    = bca_end

        self.scale_mlp_q = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )
        self.scale_mlp_v = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, rank),
        )

        n_bca = bca_end - bca_start
        self.bca = nn.ModuleList([
            BiTemporalCrossAttn(dim=768, head_dim=bca_head_dim)
            for _ in range(n_bca)
        ])

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = CDLoRATWModel.compute_patch_diff_features(t1, t2, self.patch_size)  # (B, N, 4)

        gamma_q = self.scale_mlp_q(f)   # (B, N, r)
        gamma_v = self.scale_mlp_v(f)

        h1 = self.encoder.prepare_tokens_with_masks(t1)
        h2 = self.encoder.prepare_tokens_with_masks(t2)

        for i, block in enumerate(self.encoder.blocks):
            if self.lora_start <= i < self.lora_end:
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h1 = block(h1)
                block.attn.qkv.set_scale(gamma_q, gamma_v)
                h2 = block(h2)
            else:
                h1 = block(h1)
                h2 = block(h2)

            if self.bca_start <= i < self.bca_end:
                bca    = self.bca[i - self.bca_start]
                p1     = h1[:, 1:]
                p2     = h2[:, 1:]
                new_p1 = bca(p1, p2)
                new_p2 = bca(p2, p1)
                h1 = torch.cat([h1[:, :1], new_p1], dim=1)
                h2 = torch.cat([h2[:, :1], new_p2], dim=1)

        h1 = self.encoder.norm(h1)
        h2 = self.encoder.norm(h2)
        diff = (h1[:, 1:] - h2[:, 1:]).abs()
        return self.decoder(diff)
