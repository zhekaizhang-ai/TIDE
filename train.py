import argparse
import datetime
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import DomainBalancedSampler
from data.levir_dataset import DOMAIN_ROOTS, LEVIRDataset, MultiSourceCDDataset
from models.backbone import (build_dinov2_frozen, build_dinov2_full_finetune,
                             build_dinov2_with_vanilla_lora,
                             build_dinov2_with_metapeft_lora,
                             build_dinov2_with_lrlora,
                             build_dinov2_with_dora,
                             build_dinov2_with_deep_lora,
                             build_dinov2_with_adapter,
                             build_dinov2_with_cdlora,
                             build_dinov2_with_cdlora_tw,
                             build_dinov2_with_bitfit,
                             build_dinov2_with_vpt_deep,
                             build_dinov2_with_ssf,
                             metapeft_gate_l1,
                             lrlora_gate_l1)
from models.cd_decoder import CDDecoder
from models.cd_model import (CDModel, CDLoRAModel, CDLoRABCAModel,
                              CDLoRAScalarGammaModel, CDLoRA2DGammaModel,
                              CDLoRATWModel, CDLoRABCATWModel,
                              DeepLoraBCAModel, CDLoRA2DBCAModel,
                              CDLoRAScalar2DBCAModel)
from utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='cdlora_bca',
                   choices=['frozen', 'full_finetune', 'vanilla_lora', 'adapter',
                            'metapeft_lora', 'lrlora', 'dora', 'deep_lora',
                            'cdlora', 'cdlora_scalar_gamma', 'cdlora_2d',
                            'cdlora_bca', 'cdlora_tw', 'cdlora_bca_tw',
                            'deep_lora_bca', 'cdlora_2d_bca',
                            'cdlora_scalar_2d_bca',
                            'bitfit', 'vpt_deep', 'ssf'])
    p.add_argument('--data_root',    default=DOMAIN_ROOTS['LEVIR'])
    p.add_argument('--train_sources', nargs='+', default=None,
                   help='Domain names for training, e.g. LEVIR GZ WHU. '
                        'If omitted, uses --data_root with the legacy single-source path.')
    p.add_argument('--eval_targets', nargs='+',
                   default=['LEVIR'],
                   help='Domain names to evaluate on at each epoch.')
    p.add_argument('--sampler', default='domain_balanced',
                   choices=['uniform', 'domain_balanced'])
    p.add_argument('--domain_caps', nargs='+', default=None, metavar='DOMAIN=N',
                   help='Cap train samples per domain, e.g. CDD=3000 GZ=1315.')
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--max_steps',    type=int, default=None,
                   help='Stop after this many optimizer steps (smoke-test mode).')
    p.add_argument('--batch_size',   type=int, default=48)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--img_size',     type=int, default=224)
    p.add_argument('--lora_rank',          type=int, default=8)
    p.add_argument('--adapter_bottleneck', type=int, default=32)
    p.add_argument('--exp_dir', default='.',
                   help='Root folder for checkpoints/ and logs/ (default: current dir).')
    p.add_argument('--run_name', default=None,
                   help='Override checkpoint/log name.')
    p.add_argument('--cdlora_mlp_hidden', type=int, default=32,
                   help='Hidden dim of scale MLP for CDLoRA.')
    p.add_argument('--cdlora_start', type=int, default=8,
                   help='First block to inject CDLoRA-TW (inclusive, default 8).')
    p.add_argument('--cdlora_end', type=int, default=12,
                   help='One-past-last block for CDLoRA-TW injection (exclusive, default 12).')
    p.add_argument('--bca_start', type=int, default=8,
                   help='First block to apply BCA (inclusive, default 8).')
    p.add_argument('--bca_end', type=int, default=12,
                   help='One-past-last block for BCA (exclusive, default 12).')
    p.add_argument('--bca_head_dim', type=int, default=64,
                   help='Head dim of BCA cross-attention for cdlora_bca.')
    p.add_argument('--vpt_num_prompts', type=int, default=10,
                   help='Number of learnable prompt tokens per block for VPT-Deep.')
    p.add_argument('--metapeft_init_scale', type=float, default=0.5,
                   help='Initial sigmoid gate value for MetaPEFT-LoRA.')
    p.add_argument('--metapeft_meta_lr', type=float, default=5e-4,
                   help='Learning rate for MetaPEFT gate logits.')
    p.add_argument('--metapeft_sparsity_lambda', type=float, default=0.0,
                   help='Optional L1 regularization weight on MetaPEFT gates.')
    p.add_argument('--lrlora_init_scale', type=float, default=0.5,
                   help='Initial sigmoid gate value for LR-LoRA rank components.')
    p.add_argument('--lrlora_gate_lr', type=float, default=1e-4,
                   help='Learning rate for LR-LoRA rank gate logits.')
    p.add_argument('--lrlora_sparsity_lambda', type=float, default=0.0,
                   help='Optional L1 regularization weight on LR-LoRA rank gates.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    pred   = torch.sigmoid(pred).reshape(-1)
    target = target.float().reshape(-1)
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * BCE + 0.5 * Dice"""
    bce  = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())
    dice = dice_loss(pred.squeeze(1), target)
    return 0.5 * bce + 0.5 * dice


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, device: torch.device):
    decoder = CDDecoder(in_dim=768, grid=16)
    mode    = args.mode

    if mode == 'frozen':
        encoder = build_dinov2_frozen()
        return CDModel(encoder, decoder).to(device)
    if mode == 'full_finetune':
        encoder = build_dinov2_full_finetune()
        return CDModel(encoder, decoder).to(device)
    if mode == 'vanilla_lora':
        encoder = build_dinov2_with_vanilla_lora(rank=args.lora_rank)
        return CDModel(encoder, decoder).to(device)
    if mode == 'metapeft_lora':
        encoder = build_dinov2_with_metapeft_lora(
            rank=args.lora_rank,
            init_scale=args.metapeft_init_scale,
        )
        return CDModel(encoder, decoder).to(device)
    if mode == 'lrlora':
        encoder = build_dinov2_with_lrlora(
            rank=args.lora_rank,
            init_scale=args.lrlora_init_scale,
        )
        return CDModel(encoder, decoder).to(device)
    if mode == 'dora':
        encoder = build_dinov2_with_dora(rank=args.lora_rank)
        return CDModel(encoder, decoder).to(device)
    if mode == 'adapter':
        encoder = build_dinov2_with_adapter(bottleneck=args.adapter_bottleneck)
        return CDModel(encoder, decoder).to(device)
    if mode == 'deep_lora':
        encoder = build_dinov2_with_deep_lora(rank=args.lora_rank,
                                               start_block=args.cdlora_start,
                                               end_block=args.cdlora_end)
        return CDModel(encoder, decoder).to(device)
    if mode == 'cdlora':
        encoder = build_dinov2_with_cdlora(rank=args.lora_rank,
                                            start_block=args.cdlora_start,
                                            end_block=args.cdlora_end)
        return CDLoRAModel(encoder, decoder,
                           rank=args.lora_rank,
                           mlp_hidden=args.cdlora_mlp_hidden,
                           lora_start=args.cdlora_start,
                           lora_end=args.cdlora_end).to(device)
    if mode == 'cdlora_scalar_gamma':
        encoder = build_dinov2_with_cdlora(rank=args.lora_rank,
                                            start_block=args.cdlora_start,
                                            end_block=args.cdlora_end)
        return CDLoRAScalarGammaModel(encoder, decoder,
                                      rank=args.lora_rank,
                                      mlp_hidden=args.cdlora_mlp_hidden,
                                      lora_start=args.cdlora_start,
                                      lora_end=args.cdlora_end).to(device)
    if mode == 'cdlora_2d':
        encoder = build_dinov2_with_cdlora(rank=args.lora_rank,
                                            start_block=args.cdlora_start,
                                            end_block=args.cdlora_end)
        return CDLoRA2DGammaModel(encoder, decoder,
                                  rank=args.lora_rank,
                                  mlp_hidden=args.cdlora_mlp_hidden,
                                  lora_start=args.cdlora_start,
                                  lora_end=args.cdlora_end).to(device)
    if mode == 'cdlora_bca':
        encoder = build_dinov2_with_cdlora(rank=args.lora_rank,
                                            start_block=args.cdlora_start,
                                            end_block=args.cdlora_end)
        return CDLoRABCAModel(encoder, decoder,
                              rank=args.lora_rank,
                              mlp_hidden=args.cdlora_mlp_hidden,
                              bca_head_dim=args.bca_head_dim,
                              lora_start=args.cdlora_start,
                              lora_end=args.cdlora_end,
                              bca_start=args.bca_start,
                              bca_end=args.bca_end).to(device)
    if mode == 'cdlora_tw':
        encoder = build_dinov2_with_cdlora_tw(rank=args.lora_rank,
                                               start_block=args.cdlora_start,
                                               end_block=args.cdlora_end)
        return CDLoRATWModel(encoder, decoder,
                             rank=args.lora_rank,
                             mlp_hidden=args.cdlora_mlp_hidden,
                             lora_start=args.cdlora_start,
                             lora_end=args.cdlora_end).to(device)
    if mode == 'deep_lora_bca':
        encoder = build_dinov2_with_deep_lora(rank=args.lora_rank,
                                               start_block=args.cdlora_start,
                                               end_block=args.cdlora_end)
        return DeepLoraBCAModel(encoder, decoder,
                                bca_head_dim=args.bca_head_dim,
                                lora_start=args.cdlora_start,
                                lora_end=args.cdlora_end,
                                bca_start=args.bca_start,
                                bca_end=args.bca_end).to(device)
    if mode == 'cdlora_bca_tw':
        encoder = build_dinov2_with_cdlora_tw(rank=args.lora_rank,
                                               start_block=args.cdlora_start,
                                               end_block=args.cdlora_end)
        return CDLoRABCATWModel(encoder, decoder,
                                rank=args.lora_rank,
                                mlp_hidden=args.cdlora_mlp_hidden,
                                bca_head_dim=args.bca_head_dim,
                                lora_start=args.cdlora_start,
                                lora_end=args.cdlora_end,
                                bca_start=args.bca_start,
                                bca_end=args.bca_end).to(device)
    if mode == 'cdlora_2d_bca':
        encoder = build_dinov2_with_cdlora(rank=args.lora_rank,
                                            start_block=args.cdlora_start,
                                            end_block=args.cdlora_end)
        return CDLoRA2DBCAModel(encoder, decoder,
                                rank=args.lora_rank,
                                mlp_hidden=args.cdlora_mlp_hidden,
                                bca_head_dim=args.bca_head_dim,
                                lora_start=args.cdlora_start,
                                lora_end=args.cdlora_end,
                                bca_start=args.bca_start,
                                bca_end=args.bca_end).to(device)
    if mode == 'cdlora_scalar_2d_bca':
        encoder = build_dinov2_with_cdlora(rank=args.lora_rank,
                                            start_block=args.cdlora_start,
                                            end_block=args.cdlora_end)
        return CDLoRAScalar2DBCAModel(encoder, decoder,
                                      rank=args.lora_rank,
                                      mlp_hidden=args.cdlora_mlp_hidden,
                                      bca_head_dim=args.bca_head_dim,
                                      lora_start=args.cdlora_start,
                                      lora_end=args.cdlora_end,
                                      bca_start=args.bca_start,
                                      bca_end=args.bca_end).to(device)
    if mode == 'bitfit':
        encoder = build_dinov2_with_bitfit()
        return CDModel(encoder, decoder).to(device)
    if mode == 'vpt_deep':
        encoder = build_dinov2_with_vpt_deep(num_prompts=args.vpt_num_prompts)
        return CDModel(encoder, decoder).to(device)
    if mode == 'ssf':
        encoder = build_dinov2_with_ssf()
        return CDModel(encoder, decoder).to(device)
    raise NotImplementedError(f"Unknown mode: {mode}")


def build_optimizer(model, args) -> AdamW:
    mode = args.mode
    if mode == 'full_finetune':
        return AdamW([
            {'params': model.encoder.parameters(), 'lr': 1e-5},
            {'params': model.decoder.parameters(), 'lr': 1e-4},
        ], weight_decay=1e-2)
    if mode == 'metapeft_lora':
        meta_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'meta_scale_logit' in name:
                meta_params.append(p)
            else:
                other_params.append(p)
        return AdamW([
            {'params': other_params, 'lr': 1e-4},
            {'params': meta_params, 'lr': args.metapeft_meta_lr},
        ], weight_decay=1e-2)
    if mode == 'lrlora':
        gate_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'rank_logit' in name:
                gate_params.append(p)
            else:
                other_params.append(p)
        return AdamW([
            {'params': other_params, 'lr': 1e-4},
            {'params': gate_params, 'lr': args.lrlora_gate_lr},
        ], weight_decay=1e-2)
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-2,
    )


def make_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    source_suffix = ''
    if args.train_sources:
        source_suffix = '_src_' + '_'.join(s.lower() for s in args.train_sources)
    if args.mode == 'vanilla_lora':
        return f"vanilla_lora_r{args.lora_rank}{source_suffix}"
    if args.mode == 'metapeft_lora':
        return f"metapeft_lora_r{args.lora_rank}{source_suffix}"
    if args.mode == 'lrlora':
        return f"lrlora_r{args.lora_rank}{source_suffix}"
    if args.mode == 'dora':
        return f"dora_r{args.lora_rank}{source_suffix}"
    if args.mode == 'adapter':
        return f"adapter_b{args.adapter_bottleneck}{source_suffix}"
    if args.mode == 'deep_lora':
        return f"deep_lora_r{args.lora_rank}_b{args.cdlora_start}-{args.cdlora_end}{source_suffix}"
    if args.mode == 'cdlora':
        return f"cdlora_r{args.lora_rank}_b{args.cdlora_start}-{args.cdlora_end}_h{args.cdlora_mlp_hidden}{source_suffix}"
    if args.mode == 'cdlora_scalar_gamma':
        return f"cdlora_scalar_r{args.lora_rank}_b{args.cdlora_start}-{args.cdlora_end}_h{args.cdlora_mlp_hidden}{source_suffix}"
    if args.mode == 'cdlora_2d':
        return f"cdlora_2d_r{args.lora_rank}_b{args.cdlora_start}-{args.cdlora_end}_h{args.cdlora_mlp_hidden}{source_suffix}"
    if args.mode == 'cdlora_bca':
        return f"cdlora_bca_r{args.lora_rank}_b{args.cdlora_start}_h{args.bca_head_dim}{source_suffix}"
    if args.mode == 'cdlora_tw':
        return f"cdlora_tw_c{args.cdlora_start}-{args.cdlora_end}{source_suffix}"
    if args.mode == 'deep_lora_bca':
        return (f"deep_lora_bca_b{args.cdlora_start}-{args.cdlora_end}"
                f"_bca{args.bca_start}-{args.bca_end}{source_suffix}")
    if args.mode == 'cdlora_bca_tw':
        return (f"cdlora_bca_tw_c{args.cdlora_start}-{args.cdlora_end}"
                f"_b{args.bca_start}-{args.bca_end}{source_suffix}")
    if args.mode == 'cdlora_2d_bca':
        return (f"cdlora_2d_bca_r{args.lora_rank}_b{args.cdlora_start}-{args.cdlora_end}"
                f"_bca{args.bca_start}-{args.bca_end}{source_suffix}")
    if args.mode == 'cdlora_scalar_2d_bca':
        return (f"cdlora_scalar_2d_bca_r{args.lora_rank}_b{args.cdlora_start}-{args.cdlora_end}"
                f"_bca{args.bca_start}-{args.bca_end}{source_suffix}")
    if args.mode == 'bitfit':
        return f"bitfit{source_suffix}"
    if args.mode == 'vpt_deep':
        return f"vpt_deep_p{args.vpt_num_prompts}{source_suffix}"
    if args.mode == 'ssf':
        return f"ssf{source_suffix}"
    return f"{args.mode}{source_suffix}"


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device,
                    metapeft_sparsity_lambda: float = 0.0,
                    lrlora_sparsity_lambda: float = 0.0,
                    max_steps: int = None, global_step: int = 0) -> tuple[float, int]:
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        if max_steps is not None and global_step + steps >= max_steps:
            break
        t1, t2, label = batch[:3]
        t1, t2, label = t1.to(device), t2.to(device), label.to(device)
        optimizer.zero_grad()
        loss = criterion(model(t1, t2), label)
        if metapeft_sparsity_lambda > 0:
            loss = loss + metapeft_sparsity_lambda * metapeft_gate_l1(model)
        if lrlora_sparsity_lambda > 0:
            loss = loss + lrlora_sparsity_lambda * lrlora_gate_l1(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        steps += 1
    avg_loss = total / steps if steps > 0 else 0.0
    return avg_loss, steps


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        t1, t2, label = batch[:3]
        t1, t2 = t1.to(device), t2.to(device)
        pred_bin = (torch.sigmoid(model(t1, t2).squeeze(1)) > 0.5).cpu()
        preds.append(pred_bin)
        labels.append(label)
    return compute_metrics(torch.cat(preds), torch.cat(labels))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _validate_domains(domains: list[str], arg_name: str) -> None:
    unknown = [d for d in domains if d not in DOMAIN_ROOTS]
    if unknown:
        raise ValueError(f"Unknown {arg_name}: {unknown}. "
                         f"Expected one of {sorted(DOMAIN_ROOTS)}")


def build_train_loader(args):
    if not args.train_sources:
        dataset = LEVIRDataset(args.data_root, 'train', args.img_size, augment=True)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
        return loader, None, {'legacy': len(dataset)}

    _validate_domains(args.train_sources, 'train_sources')
    if len(args.train_sources) == 1:
        source  = args.train_sources[0]
        dataset = LEVIRDataset(DOMAIN_ROOTS[source], 'train', args.img_size, augment=True)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
        return loader, args.train_sources, {source: len(dataset)}

    domain_caps = {}
    if args.domain_caps:
        for item in args.domain_caps:
            k, v = item.split('=')
            domain_caps[k] = int(v)

    dataset      = MultiSourceCDDataset(args.train_sources, 'train', args.img_size,
                                        augment=True, domain_caps=domain_caps or None)
    sample_counts = {src: len(dataset.domain_indices[i])
                     for i, src in enumerate(args.train_sources)}

    if args.sampler == 'domain_balanced':
        batch_sampler = DomainBalancedSampler(dataset, batch_size=args.batch_size,
                                              num_domains=len(args.train_sources))
        loader = DataLoader(dataset, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    return loader, args.train_sources, sample_counts


def build_eval_loaders(args):
    if not args.train_sources:
        dataset = LEVIRDataset(args.data_root, 'val', args.img_size)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        return {'legacy': loader}, {'legacy': len(dataset)}

    _validate_domains(args.eval_targets, 'eval_targets')
    loaders, counts = {}, {}
    for target in args.eval_targets:
        dataset          = LEVIRDataset(DOMAIN_ROOTS[target], 'val', args.img_size)
        loaders[target]  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
        counts[target]   = len(dataset)
    return loaders, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    run_name = make_run_name(args)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, train_sources, train_counts = build_train_loader(args)
    eval_loaders, eval_counts                 = build_eval_loaders(args)
    source_metric_names = train_sources or ['legacy']

    model     = build_model(args, device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = build_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    ckpt_dir = os.path.join(args.exp_dir, 'checkpoints', run_name)
    log_path = os.path.join(args.exp_dir, 'logs', f'{run_name}.log')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'logs'), exist_ok=True)
    log_file = open(log_path, 'w')

    def log(msg: str):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("=" * 60)
    log(f"  TIDE Training — {run_name}")
    log("=" * 60)
    log(f"  start_time         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  mode               : {args.mode}")
    lora_modes = ('vanilla_lora', 'metapeft_lora', 'lrlora', 'dora', 'deep_lora',
                  'cdlora', 'cdlora_scalar_gamma', 'cdlora_2d', 'cdlora_bca',
                  'cdlora_tw', 'cdlora_bca_tw')
    if args.mode in lora_modes:
        log(f"  lora_rank          : {args.lora_rank}")
    if args.mode == 'adapter':
        log(f"  adapter_bottleneck : {args.adapter_bottleneck}")
    if args.mode == 'metapeft_lora':
        log(f"  metapeft_init_scale: {args.metapeft_init_scale}")
        log(f"  metapeft_meta_lr   : {args.metapeft_meta_lr}")
        log(f"  metapeft_l1_lambda : {args.metapeft_sparsity_lambda}")
    if args.mode == 'lrlora':
        log(f"  lrlora_init_scale  : {args.lrlora_init_scale}")
        log(f"  lrlora_gate_lr     : {args.lrlora_gate_lr}")
        log(f"  lrlora_l1_lambda   : {args.lrlora_sparsity_lambda}")
    block_range_modes = ('deep_lora', 'cdlora', 'cdlora_scalar_gamma', 'cdlora_2d',
                         'cdlora_bca', 'cdlora_tw', 'cdlora_bca_tw')
    if args.mode in block_range_modes:
        log(f"  lora_blocks        : {args.cdlora_start}–{args.cdlora_end - 1}  "
            f"(blocks [{args.cdlora_start}, {args.cdlora_end}))")
    if args.mode in ('cdlora', 'cdlora_scalar_gamma', 'cdlora_2d', 'cdlora_bca',
                     'cdlora_tw', 'cdlora_bca_tw'):
        log(f"  cdlora_mlp_hidden  : {args.cdlora_mlp_hidden}")
    if args.mode == 'cdlora_scalar_gamma':
        log(f"  gamma_dim          : scalar (R^1 → broadcast to rank)")
    if args.mode == 'cdlora_2d':
        log(f"  diff_features      : 2D (mean, std)")
    if args.mode in ('cdlora_bca', 'cdlora_bca_tw'):
        log(f"  bca_head_dim       : {args.bca_head_dim}")
        log(f"  bca_blocks         : {args.bca_start}–{args.bca_end - 1}  "
            f"(blocks [{args.bca_start}, {args.bca_end}))")
    if args.train_sources:
        log(f"  train_sources      : {' '.join(args.train_sources)}")
        log(f"  eval_targets       : {' '.join(args.eval_targets)}")
        for src, count in train_counts.items():
            log(f"  train_samples[{src:<5}]: {count}")
        for tgt, count in eval_counts.items():
            role = 'source' if tgt in args.train_sources else 'held-out'
            log(f"  val_samples[{tgt:<5}]  : {count} ({role})")
    else:
        log(f"  data_root          : {args.data_root}")
        log(f"  train_samples      : {len(train_loader.dataset)}")
        log(f"  val_samples        : {eval_counts['legacy']}")
    log(f"  epochs             : {args.epochs}")
    log(f"  batch_size         : {args.batch_size}")
    log(f"  optimizer          : AdamW  lr=1e-4  wd=0.01")
    if args.mode == 'full_finetune':
        log(f"  lr (encoder/decoder): 1e-5 / 1e-4")
    if args.mode == 'metapeft_lora':
        log(f"  lr (main/meta)     : 1e-4 / {args.metapeft_meta_lr}")
    if args.mode == 'lrlora':
        log(f"  lr (main/gate)     : 1e-4 / {args.lrlora_gate_lr}")
    log(f"  scheduler          : CosineAnnealingLR")
    log(f"  loss               : 0.5*BCE + 0.5*Dice")
    log(f"  device             : {device}")
    log(f"  total params       : {total/1e6:.2f}M")
    log(f"  trainable params   : {trainable/1e6:.2f}M  ({100*trainable/total:.2f}%)")
    log("=" * 60)

    best_f1 = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        train_loss, steps_this_epoch = train_one_epoch(
            model, train_loader, optimizer, device,
            metapeft_sparsity_lambda=args.metapeft_sparsity_lambda,
            lrlora_sparsity_lambda=args.lrlora_sparsity_lambda,
            max_steps=args.max_steps, global_step=global_step)
        global_step += steps_this_epoch
        target_metrics = {tgt: evaluate(model, ldr, device)
                          for tgt, ldr in eval_loaders.items()}
        scheduler.step()

        source_f1s = [target_metrics[n]['f1'] for n in source_metric_names
                      if n in target_metrics]
        if not source_f1s:
            raise RuntimeError(f"No source val metrics for {source_metric_names}")
        source_avg_f1 = sum(source_f1s) / len(source_f1s)

        step_info = f" step={global_step}" if args.max_steps is not None else ""
        if args.train_sources:
            bits = [f"{tgt}({'src' if tgt in args.train_sources else 'held'}) "
                    f"F1={m['f1']:.4f} IoU={m['iou']:.4f}"
                    for tgt, m in target_metrics.items()]
            line = (f"[{epoch:03d}/{args.epochs}{step_info}] loss={train_loss:.4f}  "
                    f"src_avg_F1={source_avg_f1:.4f}  " + "  ".join(bits))
            m_for_ckpt = {'source_avg_f1': source_avg_f1, 'targets': target_metrics}
        else:
            m = target_metrics['legacy']
            line = (f"[{epoch:03d}/{args.epochs}{step_info}] loss={train_loss:.4f}  "
                    f"F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
                    f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  OA={m['oa']:.4f}")
            m_for_ckpt = m
        log(line)

        if source_avg_f1 > best_f1:
            best_f1 = source_avg_f1
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'metrics': m_for_ckpt},
                       os.path.join(ckpt_dir, 'best.pth'))
            log(f"  → best checkpoint (source_avg_F1={best_f1:.4f})")

    log(f"\nDone. Best source val F1 = {best_f1:.4f}")
    log(f"✓ {run_name} finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()


if __name__ == '__main__':
    main()
