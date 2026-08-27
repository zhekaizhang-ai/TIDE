"""
Evaluate the main zero-shot cross-domain table.

This script is intentionally separate from eval.py because Table 2 uses a
fixed 13-method registry and a 6-domain AvgCross:
GZ, WHU, CDD, SYSU, EGY, HRCUS. SI-BU-C2 and DSIFN are excluded here.

Usage:
    python eval_mainresults.py --exp_dir experiments --batch_size 64
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import torch
from torch.utils.data import DataLoader

from data.levir_dataset import DOMAIN_ROOTS, LEVIRDataset
from models.backbone import (
    build_dinov2_frozen,
    build_dinov2_full_finetune,
    build_dinov2_with_adapter,
    build_dinov2_with_bitfit,
    build_dinov2_with_cdlora,
    build_dinov2_with_dora,
    build_dinov2_with_lrlora,
    build_dinov2_with_metapeft_lora,
    build_dinov2_with_ssf,
    build_dinov2_with_vanilla_lora,
    build_dinov2_with_vpt_deep,
)
from models.cd_decoder import CDDecoder
from models.cd_model import CDLoRA2DBCAModel, CDModel
from utils.metrics import compute_metrics


METHODS = [
    {
        "key": "frozen",
        "label": "Frozen",
        "run_name": "frozen",
        "mode": "frozen",
    },
    {
        "key": "full_finetune",
        "label": "Full Finetune",
        "run_name": "full_ft",
        "mode": "full_finetune",
    },
    {
        "key": "bitfit",
        "label": "BitFit [21]",
        "run_name": "bitfit_src_levir",
        "mode": "bitfit",
    },
    {
        "key": "vpt_deep",
        "label": "VPT-Deep [11]",
        "run_name": "vpt_deep_p10_src_levir",
        "mode": "vpt_deep",
        "vpt_num_prompts": 10,
    },
    {
        "key": "ssf",
        "label": "SSF [15]",
        "run_name": "ssf_src_levir",
        "mode": "ssf",
    },
    {
        "key": "adapter",
        "label": "Adapter [8]",
        "run_name": "adapter",
        "mode": "adapter",
    },
    {
        "key": "lora_r4",
        "label": "LoRA (r=4) [9]",
        "run_name": "vanilla_lora_r4_src_levir",
        "mode": "vanilla_lora",
        "lora_rank": 4,
    },
    {
        "key": "lora_r8",
        "label": "LoRA (r=8)",
        "run_name": "lora",
        "mode": "vanilla_lora",
        "lora_rank": 8,
    },
    {
        "key": "lora_r16",
        "label": "LoRA (r=16)",
        "run_name": "vanilla_lora_r16_src_levir",
        "mode": "vanilla_lora",
        "lora_rank": 16,
    },
    {
        "key": "metapeft_lora",
        "label": "MetaPEFT-LoRA",
        "run_name": "metapeft_lora_r8_src_levir",
        "mode": "metapeft_lora",
        "lora_rank": 8,
    },
    {
        "key": "lrlora",
        "label": "LR-LoRA",
        "run_name": "lrlora_r8_src_levir",
        "mode": "lrlora",
        "lora_rank": 8,
    },
    {
        "key": "dora",
        "label": "DoRA",
        "run_name": "dora_r8_src_levir",
        "mode": "dora",
        "lora_rank": 8,
    },
    {
        "key": "tide",
        "label": "TIDE (Ours)",
        "run_name": "cdlora_2d_bca_r8_b6-8_bca6-8_src_levir",
        "mode": "cdlora_2d_bca",
        "lora_rank": 8,
    },
]

DOMAINS = [
    {"label": "LEVIR", "root": DOMAIN_ROOTS["LEVIR"], "cross": False},
    {"label": "GZ", "root": DOMAIN_ROOTS["GZ"], "cross": True},
    {"label": "WHU", "root": DOMAIN_ROOTS["WHU"], "cross": True},
    {"label": "CDD", "root": DOMAIN_ROOTS["CDD"], "cross": True},
    {"label": "SYSU", "root": DOMAIN_ROOTS["SYSU"], "cross": True},
    {"label": "EGY", "root": DOMAIN_ROOTS["EGY"], "cross": True},
    {"label": "HRCUS", "root": DOMAIN_ROOTS["HRCUS"], "cross": True},
]

CROSS_LABELS = [d["label"] for d in DOMAINS if d["cross"]]


def build_model(method: dict, device: torch.device) -> torch.nn.Module:
    decoder = CDDecoder(in_dim=768, grid=16)
    mode = method["mode"]

    if mode == "frozen":
        return CDModel(build_dinov2_frozen(), decoder).to(device)
    if mode == "full_finetune":
        return CDModel(build_dinov2_full_finetune(), decoder).to(device)
    if mode == "bitfit":
        return CDModel(build_dinov2_with_bitfit(), decoder).to(device)
    if mode == "vpt_deep":
        prompts = method.get("vpt_num_prompts", 10)
        return CDModel(build_dinov2_with_vpt_deep(num_prompts=prompts), decoder).to(device)
    if mode == "ssf":
        return CDModel(build_dinov2_with_ssf(), decoder).to(device)
    if mode == "adapter":
        return CDModel(build_dinov2_with_adapter(bottleneck=32), decoder).to(device)
    if mode == "vanilla_lora":
        rank = method.get("lora_rank", 8)
        return CDModel(build_dinov2_with_vanilla_lora(rank=rank), decoder).to(device)
    if mode == "metapeft_lora":
        rank = method.get("lora_rank", 8)
        init_scale = method.get("metapeft_init_scale", 0.5)
        encoder = build_dinov2_with_metapeft_lora(rank=rank, init_scale=init_scale)
        return CDModel(encoder, decoder).to(device)
    if mode == "lrlora":
        rank = method.get("lora_rank", 8)
        init_scale = method.get("lrlora_init_scale", 0.5)
        encoder = build_dinov2_with_lrlora(rank=rank, init_scale=init_scale)
        return CDModel(encoder, decoder).to(device)
    if mode == "dora":
        rank = method.get("lora_rank", 8)
        return CDModel(build_dinov2_with_dora(rank=rank), decoder).to(device)
    if mode == "cdlora_2d_bca":
        rank = method.get("lora_rank", 8)
        encoder = build_dinov2_with_cdlora(rank=rank, start_block=6, end_block=8)
        return CDLoRA2DBCAModel(
            encoder,
            decoder,
            rank=rank,
            mlp_hidden=32,
            bca_head_dim=64,
            lora_start=6,
            lora_end=8,
            bca_start=6,
            bca_end=8,
        ).to(device)

    raise ValueError(f"Unknown mode: {mode}")


def peft_params_m(model: torch.nn.Module) -> float:
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return max(trainable_params - decoder_params, 0) / 1e6


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds = []
    labels = []
    for t1, t2, label in loader:
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        logits = model(t1, t2).squeeze(1)
        preds.append((logits > 0).long().cpu())
        labels.append(label)
    return compute_metrics(torch.cat(preds), torch.cat(labels))


def format_table(results: dict, metric: str) -> str:
    metric_title = "F1 (%)" if metric == "f1" else "IoU (%)"
    headers = ["Method", "Params (M)"] + [d["label"] for d in DOMAINS] + ["AvgCross"]
    aligns = [":--", "--:"] + ["--:"] * (len(headers) - 2)
    lines = [
        f"### {metric_title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for method in METHODS:
        res = results.get(method["key"], {})
        row = [method["label"], f'{res.get("_params_m", 0.0):.2f}']
        cross_vals = []
        for domain in DOMAINS:
            label = domain["label"]
            if label not in res:
                row.append("—")
                continue
            value = res[label][metric] * 100.0
            row.append(f"{value:.1f}")
            if domain["cross"]:
                cross_vals.append(value)
        row.append(f"{sum(cross_vals) / len(cross_vals):.1f}" if cross_vals else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--method",
        default=None,
        choices=[m["key"] for m in METHODS],
        help="Evaluate a single method. Default: all 13 methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [m for m in METHODS if args.method is None or m["key"] == args.method]

    print(f"Device    : {device}")
    print(f"exp_dir   : {os.path.abspath(args.exp_dir)}")
    print(f"AvgCross  : {', '.join(CROSS_LABELS)}")
    print(f"methods   : {len(methods)}\n")

    loaders = {}
    for domain in DOMAINS:
        if not os.path.isdir(domain["root"]):
            raise FileNotFoundError(f"Missing data directory: {domain['root']}")
        dataset = LEVIRDataset(domain["root"], split="test", img_size=args.img_size)
        loaders[domain["label"]] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"{domain['label']:<6} samples: {len(dataset)}")

    all_results = {}
    for method in methods:
        ckpt_path = os.path.join(
            args.exp_dir, "checkpoints", method["run_name"], "best.pth"
        )
        print(f"\n[{method['label']}]")
        print(f"  checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            print("  [SKIP] checkpoint not found")
            all_results[method["key"]] = {}
            continue

        model = build_model(method, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

        method_results = {"_params_m": peft_params_m(model)}
        for domain in DOMAINS:
            label = domain["label"]
            tag = "source" if not domain["cross"] else "zero-shot"
            metrics = evaluate(model, loaders[label], device)
            method_results[label] = metrics
            print(
                f"  {label:<6} ({tag:<9}) "
                f"F1={metrics['f1'] * 100:.1f}  IoU={metrics['iou'] * 100:.1f}"
            )
        all_results[method["key"]] = method_results

    print("\n" + format_table(all_results, "f1"))
    print("\n" + format_table(all_results, "iou"))

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"mainresults_{stamp}.json")
    md_path = os.path.join(args.output_dir, f"mainresults_{stamp}.md")

    serializable = {
        "exp_dir": os.path.abspath(args.exp_dir),
        "avg_cross": CROSS_LABELS,
        "domains": [d["label"] for d in DOMAINS],
        "methods": [m["key"] for m in methods],
        "results": all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Main Results Eval\n\n")
        f.write(f"- exp_dir: `{os.path.abspath(args.exp_dir)}`\n")
        f.write(f"- AvgCross: {', '.join(CROSS_LABELS)}\n\n")
        f.write(format_table(all_results, "f1"))
        f.write("\n\n")
        f.write(format_table(all_results, "iou"))
        f.write("\n")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved MD  : {md_path}")


if __name__ == "__main__":
    main()
