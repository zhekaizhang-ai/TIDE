import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    """Bottleneck residual adapter: down-project → GELU → up-project → residual."""

    def __init__(self, dim: int = 768, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up   = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


class BlockWithAdapter(nn.Module):
    """Wraps a frozen DINOv2 block and appends an Adapter after its output."""

    def __init__(self, original_block: nn.Module, bottleneck: int = 32):
        super().__init__()
        self.block   = original_block
        self.adapter = Adapter(dim=768, bottleneck=bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(x))


def inject_adapter(model: nn.Module, bottleneck: int = 32) -> nn.Module:
    for i, block in enumerate(model.blocks):
        model.blocks[i] = BlockWithAdapter(block, bottleneck=bottleneck)
    return model


# ---------------------------------------------------------------------------
# Vanilla LoRA
# ---------------------------------------------------------------------------

class VanillaLoRABranch(nn.Module):
    """Single low-rank branch: x -> B(A(x)), standard LoRA init."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


class QKVWithVanillaLoRA(nn.Module):
    """
    Wraps a frozen qkv Linear(768 -> 2304) and injects independent
    LoRA branches into Q and V projections (K unchanged).

    Output layout: [ Q(768) | K(768) | V(768) ]
    """

    def __init__(self, original_qkv: nn.Linear, rank: int):
        super().__init__()
        self.original_qkv = original_qkv
        dim = 768
        self.lora_q = VanillaLoRABranch(dim, rank)
        self.lora_v = VanillaLoRABranch(dim, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out     = self.original_qkv(x)
        patches = x[:, 1:, :]

        out = out.clone()
        out[:, 1:, :768]  += self.lora_q(patches)
        out[:, 1:, 1536:] += self.lora_v(patches)
        return out


def inject_vanilla_lora(model: nn.Module, rank: int = 8,
                        start_block: int = 0, end_block: int = None) -> nn.Module:
    n = len(model.blocks)
    end = end_block if end_block is not None else n
    for i, block in enumerate(model.blocks):
        if start_block <= i < end:
            block.attn.qkv = QKVWithVanillaLoRA(original_qkv=block.attn.qkv, rank=rank)
    return model


# ---------------------------------------------------------------------------
# MetaPEFT-LoRA
# ---------------------------------------------------------------------------

class QKVWithMetaLoRA(nn.Module):
    """
    LoRA on Q/V with learnable module-wise gates.

    Each adapted block learns separate scalar gates for Q and V. The gates act as
    continuous layer/module selection and are initialized to metapeft_init_scale.
    """

    def __init__(self, original_qkv: nn.Linear, rank: int,
                 init_scale: float = 0.5):
        super().__init__()
        self.original_qkv = original_qkv
        dim = 768
        self.lora_q = VanillaLoRABranch(dim, rank)
        self.lora_v = VanillaLoRABranch(dim, rank)

        init_scale = min(max(init_scale, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_scale / (1.0 - init_scale))
        self.meta_scale_logit_q = nn.Parameter(torch.tensor(init_logit))
        self.meta_scale_logit_v = nn.Parameter(torch.tensor(init_logit))

    def gate_q(self) -> torch.Tensor:
        return torch.sigmoid(self.meta_scale_logit_q)

    def gate_v(self) -> torch.Tensor:
        return torch.sigmoid(self.meta_scale_logit_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out     = self.original_qkv(x)
        patches = x[:, 1:, :]

        out = out.clone()
        out[:, 1:, :768]  += self.gate_q() * self.lora_q(patches)
        out[:, 1:, 1536:] += self.gate_v() * self.lora_v(patches)
        return out


def inject_metapeft_lora(model: nn.Module, rank: int = 8,
                         init_scale: float = 0.5,
                         start_block: int = 0,
                         end_block: int = None) -> nn.Module:
    n = len(model.blocks)
    end = end_block if end_block is not None else n
    for i, block in enumerate(model.blocks):
        if start_block <= i < end:
            block.attn.qkv = QKVWithMetaLoRA(
                original_qkv=block.attn.qkv,
                rank=rank,
                init_scale=init_scale,
            )
    return model


def metapeft_gate_l1(model: nn.Module) -> torch.Tensor:
    gates = [
        torch.sigmoid(p)
        for name, p in model.named_parameters()
        if 'meta_scale_logit' in name
    ]
    if not gates:
        return next(model.parameters()).new_tensor(0.0)
    return torch.stack(gates).mean()


# ---------------------------------------------------------------------------
# LR-LoRA
# ---------------------------------------------------------------------------

class LearnableRankLoRABranch(nn.Module):
    """LoRA branch with a learnable gate for each rank component."""

    def __init__(self, dim: int, rank: int, init_scale: float = 0.5):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

        init_scale = min(max(init_scale, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_scale / (1.0 - init_scale))
        self.rank_logit = nn.Parameter(torch.full((rank,), init_logit))

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.rank_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = self.A(x) * self.gate()
        return self.B(mid)


class QKVWithLRLoRA(nn.Module):
    """
    LoRA on Q/V with learnable rank-component gates.

    The maximum rank is fixed by --lora_rank; each rank component is softly
    selected by sigmoid(rank_logit) during both training and inference.
    """

    def __init__(self, original_qkv: nn.Linear, rank: int,
                 init_scale: float = 0.5):
        super().__init__()
        self.original_qkv = original_qkv
        dim = 768
        self.lora_q = LearnableRankLoRABranch(dim, rank, init_scale)
        self.lora_v = LearnableRankLoRABranch(dim, rank, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out     = self.original_qkv(x)
        patches = x[:, 1:, :]

        out = out.clone()
        out[:, 1:, :768]  += self.lora_q(patches)
        out[:, 1:, 1536:] += self.lora_v(patches)
        return out


def inject_lrlora(model: nn.Module, rank: int = 8,
                  init_scale: float = 0.5,
                  start_block: int = 0,
                  end_block: int = None) -> nn.Module:
    n = len(model.blocks)
    end = end_block if end_block is not None else n
    for i, block in enumerate(model.blocks):
        if start_block <= i < end:
            block.attn.qkv = QKVWithLRLoRA(
                original_qkv=block.attn.qkv,
                rank=rank,
                init_scale=init_scale,
            )
    return model


def lrlora_gate_l1(model: nn.Module) -> torch.Tensor:
    gates = [
        torch.sigmoid(p).reshape(-1)
        for name, p in model.named_parameters()
        if 'rank_logit' in name
    ]
    if not gates:
        return next(model.parameters()).new_tensor(0.0)
    return torch.cat(gates).mean()


# ---------------------------------------------------------------------------
# DoRA
# ---------------------------------------------------------------------------

class DoRALinearBranch(nn.Module):
    """
    Weight-decomposed low-rank adaptation for one frozen Linear projection.

    DoRA learns a low-rank direction update and a per-output magnitude vector:
        W_eff = m * normalize(W_base + B @ A)
    """

    def __init__(self, base_weight: torch.Tensor, rank: int, eps: float = 1e-6):
        super().__init__()
        out_dim, in_dim = base_weight.shape
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        self.eps = eps

        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)
        self.magnitude = nn.Parameter(base_weight.detach().norm(dim=1))

    def effective_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        delta = self.B.weight @ self.A.weight
        direction = base_weight + delta
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return self.magnitude[:, None] * direction

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor,
                base_bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, self.effective_weight(base_weight), base_bias)


class QKVWithDoRA(nn.Module):
    """
    Wraps a frozen qkv Linear(768 -> 2304) and applies DoRA to Q and V only.

    Output layout: [ Q(768) | K(768) | V(768) ]
    """

    def __init__(self, original_qkv: nn.Linear, rank: int):
        super().__init__()
        self.original_qkv = original_qkv
        dim = 768
        weight = original_qkv.weight.detach()
        self.dora_q = DoRALinearBranch(weight[:dim], rank)
        self.dora_v = DoRALinearBranch(weight[2 * dim:], rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_qkv(x)
        patches = x[:, 1:, :]

        weight = self.original_qkv.weight
        bias = self.original_qkv.bias
        bias_q = bias[:768] if bias is not None else None
        bias_v = bias[1536:] if bias is not None else None

        out = out.clone()
        out[:, 1:, :768] = self.dora_q(patches, weight[:768], bias_q)
        out[:, 1:, 1536:] = self.dora_v(patches, weight[1536:], bias_v)
        return out


def inject_dora(model: nn.Module, rank: int = 8,
                start_block: int = 0, end_block: int = None) -> nn.Module:
    n = len(model.blocks)
    end = end_block if end_block is not None else n
    for i, block in enumerate(model.blocks):
        if start_block <= i < end:
            block.attn.qkv = QKVWithDoRA(original_qkv=block.attn.qkv, rank=rank)
    return model


# ---------------------------------------------------------------------------
# CDLoRA — diff-conditioned scale modulation, blocks 8-11 only
# ---------------------------------------------------------------------------

class QKVWithCDLoRA(nn.Module):
    """
    Frozen qkv + LoRA on Q and V with input-level diff-statistics scale modulation.

    CDLoRAModel calls set_scale(gamma_q, gamma_v) before each deep-block forward.
    gamma_q, gamma_v ∈ (B, r): scale the bottleneck intermediate A(x) * gamma → B.
    When set_scale is never called the behaviour is identical to QKVWithVanillaLoRA.
    """

    def __init__(self, original_qkv: nn.Linear, rank: int):
        super().__init__()
        self.original_qkv = original_qkv
        dim = 768
        self.lora_q = VanillaLoRABranch(dim, rank)
        self.lora_v = VanillaLoRABranch(dim, rank)
        self._scale_q: torch.Tensor | None = None
        self._scale_v: torch.Tensor | None = None

    def set_scale(self, gamma_q: torch.Tensor, gamma_v: torch.Tensor) -> None:
        self._scale_q = gamma_q   # (B, r)
        self._scale_v = gamma_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out     = self.original_qkv(x)
        patches = x[:, 1:]

        if self._scale_q is not None:
            sq = self._scale_q if self._scale_q.dim() == 3 else self._scale_q.unsqueeze(1)
            sv = self._scale_v if self._scale_v.dim() == 3 else self._scale_v.unsqueeze(1)
            mid_q = self.lora_q.A(patches) * sq
            mid_v = self.lora_v.A(patches) * sv
            delta_q = self.lora_q.B(mid_q)
            delta_v = self.lora_v.B(mid_v)
            self._scale_q = None
            self._scale_v = None
        else:
            delta_q = self.lora_q(patches)
            delta_v = self.lora_v(patches)

        out = out.clone()
        out[:, 1:, :768]  += delta_q
        out[:, 1:, 1536:] += delta_v
        return out


def inject_cdlora(model: nn.Module, rank: int = 8,
                  start_block: int = 8, end_block: int = None) -> nn.Module:
    """Inject QKVWithCDLoRA into blocks [start_block, end_block); earlier/later blocks untouched."""
    n = len(model.blocks)
    if end_block is None:
        end_block = n
    for i, block in enumerate(model.blocks):
        if start_block <= i < end_block:
            block.attn.qkv = QKVWithCDLoRA(original_qkv=block.attn.qkv, rank=rank)
    return model


def freeze_backbone(model: nn.Module, keep_keyword: str = 'lora') -> nn.Module:
    """Freeze all parameters whose name does not contain keep_keyword."""
    for name, p in model.named_parameters():
        if keep_keyword not in name:
            p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# VPT-Deep
# ---------------------------------------------------------------------------

class BlockWithVPT(nn.Module):
    """Wraps a frozen DINOv2 block and prepends learnable prompt tokens at each layer."""

    def __init__(self, block: nn.Module, num_prompts: int = 10, embed_dim: int = 768):
        super().__init__()
        self.block = block
        self.num_prompts = num_prompts
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        prompts = self.prompt_tokens.expand(B, -1, -1)
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
        x = self.block(x)
        return torch.cat([x[:, :1], x[:, 1 + self.num_prompts:]], dim=1)


def inject_vpt_deep(model: nn.Module, num_prompts: int = 10) -> nn.Module:
    embed_dim = model.embed_dim
    for i, block in enumerate(model.blocks):
        model.blocks[i] = BlockWithVPT(block, num_prompts=num_prompts, embed_dim=embed_dim)
    return model


# ---------------------------------------------------------------------------
# SSF (Scale & Shift Fine-tuning)
# ---------------------------------------------------------------------------

class LinearWithSSF(nn.Module):
    """Wraps a frozen Linear layer with learnable per-output scale and shift."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self.ssf_scale = nn.Parameter(torch.ones(linear.out_features))
        self.ssf_shift = nn.Parameter(torch.zeros(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.ssf_scale + self.ssf_shift


def inject_ssf(model: nn.Module) -> nn.Module:
    for block in model.blocks:
        block.attn.qkv  = LinearWithSSF(block.attn.qkv)
        block.attn.proj = LinearWithSSF(block.attn.proj)
        block.mlp.fc1   = LinearWithSSF(block.mlp.fc1)
        block.mlp.fc2   = LinearWithSSF(block.mlp.fc2)
    return model


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def build_dinov2_frozen() -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_dinov2_full_finetune() -> nn.Module:
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


def build_dinov2_with_vanilla_lora(rank: int = 8) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_vanilla_lora(model, rank=rank)
    model = freeze_backbone(model, keep_keyword='lora')
    return model


def build_dinov2_with_metapeft_lora(rank: int = 8,
                                    init_scale: float = 0.5,
                                    start_block: int = 0,
                                    end_block: int = None) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_metapeft_lora(
        model,
        rank=rank,
        init_scale=init_scale,
        start_block=start_block,
        end_block=end_block,
    )
    for name, p in model.named_parameters():
        if 'lora' not in name and 'meta_scale_logit' not in name:
            p.requires_grad = False
    return model


def build_dinov2_with_lrlora(rank: int = 8,
                             init_scale: float = 0.5,
                             start_block: int = 0,
                             end_block: int = None) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_lrlora(
        model,
        rank=rank,
        init_scale=init_scale,
        start_block=start_block,
        end_block=end_block,
    )
    for name, p in model.named_parameters():
        if 'lora' not in name and 'rank_logit' not in name:
            p.requires_grad = False
    return model


def build_dinov2_with_dora(rank: int = 8,
                           start_block: int = 0,
                           end_block: int = None) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_dora(
        model,
        rank=rank,
        start_block=start_block,
        end_block=end_block,
    )
    for name, p in model.named_parameters():
        if 'dora_' not in name:
            p.requires_grad = False
    return model


def build_dinov2_with_adapter(bottleneck: int = 32) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_adapter(model, bottleneck=bottleneck)
    for name, p in model.named_parameters():
        if 'adapter' not in name:
            p.requires_grad = False
    return model


def build_dinov2_with_deep_lora(rank: int = 8, start_block: int = 8,
                                end_block: int = None) -> nn.Module:
    """Vanilla LoRA injected only into blocks [start_block, end_block) (no conditioning)."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_vanilla_lora(model, rank=rank, start_block=start_block, end_block=end_block)
    model = freeze_backbone(model, keep_keyword='lora')
    return model


def build_dinov2_with_cdlora(rank: int = 8, start_block: int = 8,
                              end_block: int = None) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_cdlora(model, rank=rank, start_block=start_block, end_block=end_block)
    model = freeze_backbone(model, keep_keyword='lora')
    return model


def build_dinov2_with_cdlora_tw(rank: int = 8, start_block: int = 8,
                                 end_block: int = None) -> nn.Module:
    """CDLoRA (token-wise γ) injected into blocks [start_block, end_block); TW handled in model."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_cdlora(model, rank=rank, start_block=start_block, end_block=end_block)
    model = freeze_backbone(model, keep_keyword='lora')
    return model


def build_dinov2_with_bitfit() -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    return freeze_backbone(model, keep_keyword='bias')


def build_dinov2_with_vpt_deep(num_prompts: int = 10) -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_vpt_deep(model, num_prompts=num_prompts)
    return freeze_backbone(model, keep_keyword='prompt_tokens')


def build_dinov2_with_ssf() -> nn.Module:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = inject_ssf(model)
    return freeze_backbone(model, keep_keyword='ssf_')


def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total / 1e6:.2f}M")
    print(f"  Trainable params : {trainable / 1e6:.2f}M  ({100*trainable/total:.2f}%)")
