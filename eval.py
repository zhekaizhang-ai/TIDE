"""
TIDE — unified cross-domain evaluation.

Evaluates all 6 methods on the test split of every domain and prints
F1 / IoU tables matching Table 1 in paper_design.md.

Usage:
    # All methods, all domains (new experiment folder):
    python eval.py --exp_dir experiments

    # Single method:
    python eval.py --exp_dir experiments --method tide

    # Update paper_design.md with fresh numbers:
    python eval.py --exp_dir experiments --update_paper_design
"""

import argparse
import os
import re
import torch
from torch.utils.data import DataLoader

from data.levir_dataset import LEVIRDataset, DOMAIN_ROOTS
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
                             build_dinov2_with_ssf)
from models.cd_decoder import CDDecoder
from models.cd_model import (CDModel, CDLoRAModel, CDLoRABCAModel,
                              CDLoRAScalarGammaModel, CDLoRA2DGammaModel,
                              CDLoRATWModel, CDLoRABCATWModel,
                              CDLoRA2DBCAModel)
from utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Methods — run_name must match what was passed to --run_name in train.py
# ---------------------------------------------------------------------------

METHODS = [
    {'key': 'frozen',        'label': 'Frozen',        'run_name': 'frozen',  'mode': 'frozen'},
    {'key': 'full_finetune', 'label': 'Full Finetune', 'run_name': 'full_ft', 'mode': 'full_finetune'},
    {'key': 'bitfit',        'label': 'BitFit',        'run_name': 'bitfit_src_levir',           'mode': 'bitfit'},
    {'key': 'vpt_deep',      'label': 'VPT-Deep',      'run_name': 'vpt_deep_p10_src_levir',     'mode': 'vpt_deep'},
    {'key': 'ssf',           'label': 'SSF',           'run_name': 'ssf_src_levir',              'mode': 'ssf'},
    {'key': 'adapter',       'label': 'Adapter',       'run_name': 'adapter', 'mode': 'adapter'},
    {'key': 'lora_r4',       'label': 'LoRA (r=4)',    'run_name': 'vanilla_lora_r4_src_levir',  'mode': 'vanilla_lora', 'lora_rank': 4},
    {'key': 'tide_r4',       'label': 'TIDE (r=4)',    'run_name': 'cdlora_2d_bca_r4_b6-8_bca6-8_src_levir', 'mode': 'cdlora_2d_bca', 'lora_rank': 4},
    {'key': 'lora',          'label': 'LoRA (r=8)',    'run_name': 'lora',    'mode': 'vanilla_lora', 'lora_rank': 8},
    {'key': 'metapeft_lora', 'label': 'MetaPEFT-LoRA', 'run_name': 'metapeft_lora_r8_src_levir', 'mode': 'metapeft_lora', 'lora_rank': 8},
    {'key': 'lrlora',        'label': 'LR-LoRA',       'run_name': 'lrlora_r8_src_levir', 'mode': 'lrlora', 'lora_rank': 8},
    {'key': 'dora',          'label': 'DoRA',          'run_name': 'dora_r8_src_levir', 'mode': 'dora', 'lora_rank': 8},
    {'key': 'lora_r16',      'label': 'LoRA (r=16)',   'run_name': 'vanilla_lora_r16_src_levir', 'mode': 'vanilla_lora', 'lora_rank': 16},
    {'key': 'tide_r16',      'label': 'TIDE (r=16)',   'run_name': 'cdlora_2d_bca_r16_b6-8_bca6-8_src_levir', 'mode': 'cdlora_2d_bca', 'lora_rank': 16},
    {'key': 'tide',          'label': 'TIDE (Ours)',   'run_name': 'cdlora_2d_bca_r8_b6-8_bca6-8_src_levir', 'mode': 'cdlora_2d_bca'},
]

# ---------------------------------------------------------------------------
# Ablation method registry
# run_name must match --run_name used in run_ablations.sh
# ---------------------------------------------------------------------------

_ABL = {
    'lora':         {'label': 'LoRA (r=8)',      'run_name': 'lora',         'mode': 'vanilla_lora'},
    'deep_lora':    {'label': 'Deep LoRA',       'run_name': 'deep_lora',    'mode': 'deep_lora'},
    'cdlora':       {'label': 'CDLoRA',          'run_name': 'cdlora',       'mode': 'cdlora'},
    'cdlora_tw':    {'label': 'CDLoRA-TW',       'run_name': 'cdlora_tw',    'mode': 'cdlora_tw'},
    'tide':         {'label': 'TIDE',            'run_name': 'tide',         'mode': 'cdlora_bca'},
    'scalar_gamma': {'label': '+ scalar γ',      'run_name': 'scalar_gamma', 'mode': 'cdlora_scalar_gamma'},
    'cdlora_2d':    {'label': '+ 2D γ',          'run_name': 'cdlora_2d',    'mode': 'cdlora_2d'},
}

TABLE2A_KEYS = ['lora', 'deep_lora', 'cdlora', 'tide']
TABLE2B_KEYS = ['deep_lora', 'scalar_gamma', 'cdlora_2d', 'cdlora']
CDLORA_TW_KEYS = ['cdlora', 'cdlora_tw']

# MD row anchors: substring that uniquely identifies each row in paper_design.md
TABLE2A_ANCHORS = {
    'lora':         'LoRA (r=8) | 基线',
    'deep_lora':    'Deep LoRA | LoRA 收缩',
    'cdlora':       'CDLoRA | + change-aware',
    'tide':         '**TIDE** | + BCA',
}
TABLE2B_ANCHORS = {
    'deep_lora':    '(a) 深层 LoRA',
    'scalar_gamma': '(b) + scalar',
    'cdlora_2d':    '(c) + rank-wise γ，2D',
    'cdlora':       '(d) + rank-wise γ，4D',
}

# ---------------------------------------------------------------------------
# Domains
# cross=True  → counted in Avg Cross (Table 1)
# cross=False → in-domain source
# note=...    → reported separately, not in Avg Cross
# ---------------------------------------------------------------------------

DOMAINS = [
    {'label': 'LEVIR',    'root': DOMAIN_ROOTS['LEVIR'],                         'cross': False},
    {'label': 'GZ',       'root': DOMAIN_ROOTS['GZ'],                            'cross': True},
    {'label': 'WHU',      'root': DOMAIN_ROOTS['WHU'],                           'cross': True},
    {'label': 'CDD',      'root': DOMAIN_ROOTS['CDD'],                           'cross': True},
    {'label': 'SYSU-CD',  'root': DOMAIN_ROOTS['SYSU'],       'cross': True},
    {'label': 'EGY-BCD',  'root': DOMAIN_ROOTS['EGY'],       'cross': True},
    {'label': 'SI-BU-C2', 'root': DOMAIN_ROOTS['SIBU'],      'cross': True},
    {'label': 'HRCUS',    'root': DOMAIN_ROOTS['HRCUS'],                         'cross': False},
    {'label': 'DSIFN',    'root': DOMAIN_ROOTS['DSIFN'],       'cross': False},
]

CROSS_LABELS = [d['label'] for d in DOMAINS if d['cross']]

PAPER_MD = os.path.join(os.path.dirname(__file__), 'docs', 'paper_design.md')


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(method: dict, device: torch.device) -> torch.nn.Module:
    dec  = CDDecoder(in_dim=768, grid=16)
    mode = method['mode']
    if mode == 'frozen':
        return CDModel(build_dinov2_frozen(), dec).to(device)
    if mode == 'full_finetune':
        return CDModel(build_dinov2_full_finetune(), dec).to(device)
    if mode == 'vanilla_lora':
        rank = method.get('lora_rank', 8)
        return CDModel(build_dinov2_with_vanilla_lora(rank=rank), dec).to(device)
    if mode == 'metapeft_lora':
        rank = method.get('lora_rank', 8)
        init_scale = method.get('metapeft_init_scale', 0.5)
        return CDModel(
            build_dinov2_with_metapeft_lora(rank=rank, init_scale=init_scale),
            dec,
        ).to(device)
    if mode == 'lrlora':
        rank = method.get('lora_rank', 8)
        init_scale = method.get('lrlora_init_scale', 0.5)
        return CDModel(
            build_dinov2_with_lrlora(rank=rank, init_scale=init_scale),
            dec,
        ).to(device)
    if mode == 'dora':
        rank = method.get('lora_rank', 8)
        return CDModel(build_dinov2_with_dora(rank=rank), dec).to(device)
    if mode == 'adapter':
        return CDModel(build_dinov2_with_adapter(bottleneck=32), dec).to(device)
    if mode == 'deep_lora':
        enc = build_dinov2_with_deep_lora(rank=8, start_block=8)
        return CDModel(enc, dec).to(device)
    if mode == 'cdlora':
        enc = build_dinov2_with_cdlora(rank=8, start_block=8)
        return CDLoRAModel(enc, dec, rank=8, mlp_hidden=32).to(device)
    if mode == 'cdlora_scalar_gamma':
        enc = build_dinov2_with_cdlora(rank=8, start_block=8)
        return CDLoRAScalarGammaModel(enc, dec, rank=8, mlp_hidden=32).to(device)
    if mode == 'cdlora_2d':
        enc = build_dinov2_with_cdlora(rank=8, start_block=8)
        return CDLoRA2DGammaModel(enc, dec, rank=8, mlp_hidden=32).to(device)
    if mode == 'cdlora_bca':
        enc = build_dinov2_with_cdlora(rank=8, start_block=8)
        return CDLoRABCAModel(enc, dec, rank=8, mlp_hidden=32, bca_head_dim=64).to(device)
    if mode == 'cdlora_2d_bca':
        rank = method.get('lora_rank', 8)
        enc = build_dinov2_with_cdlora(rank=rank, start_block=6, end_block=8)
        return CDLoRA2DBCAModel(enc, dec, rank=rank, mlp_hidden=32, bca_head_dim=64,
                                lora_start=6, lora_end=8,
                                bca_start=6, bca_end=8).to(device)
    if mode == 'cdlora_tw':
        enc = build_dinov2_with_cdlora_tw(rank=8, start_block=8)
        return CDLoRATWModel(enc, dec, rank=8, mlp_hidden=32).to(device)
    if mode == 'cdlora_bca_tw':
        enc = build_dinov2_with_cdlora_tw(rank=8, start_block=8)
        return CDLoRABCATWModel(enc, dec, rank=8, mlp_hidden=32, bca_head_dim=64).to(device)
    if mode == 'bitfit':
        return CDModel(build_dinov2_with_bitfit(), dec).to(device)
    if mode == 'vpt_deep':
        num_prompts = method.get('vpt_num_prompts', 10)
        return CDModel(build_dinov2_with_vpt_deep(num_prompts=num_prompts), dec).to(device)
    if mode == 'ssf':
        return CDModel(build_dinov2_with_ssf(), dec).to(device)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    for t1, t2, label in loader:
        t1, t2 = t1.to(device), t2.to(device)
        preds.append((model(t1, t2).squeeze(1) > 0).long().cpu())
        labels.append(label)
    return compute_metrics(torch.cat(preds), torch.cat(labels))


def run_method(method: dict, exp_dir: str, batch_size: int,
               device: torch.device) -> dict:
    ckpt_path = os.path.join(exp_dir, 'checkpoints', method['run_name'], 'best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return {}

    model = build_model(method, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)

    results = {'_params_m': f'{trainable/1e6:.3f} M'}
    for domain in DOMAINS:
        lbl  = domain['label']
        root = domain['root']
        tag  = '(in-domain)' if not domain['cross'] else '(zero-shot)'
        if 'note' in domain:
            tag = f"({domain['note']})"

        if not os.path.isdir(root):
            print(f"  {lbl} {tag} — data dir missing, skip")
            continue

        print(f"  {lbl} {tag} ...", end=' ', flush=True)
        ds     = LEVIRDataset(root, split='test', img_size=224)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        m = evaluate(model, loader, device)
        results[lbl] = m
        print(f"F1={m['f1']*100:.1f}  IoU={m['iou']*100:.1f}")

    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(all_results: dict, metric: str):
    domain_labels = [d['label'] for d in DOMAINS]
    W   = 10
    sep = '─' * (20 + W * (len(domain_labels) + 2))

    def fmt_row(cells):
        return '│' + '│'.join(f'{str(c):^{W}}' for c in cells) + '│'

    title = 'F1 (%)' if metric == 'f1' else 'IoU (%)'
    print(f'\n{"="*80}')
    print(f'Table 1 — Source: LEVIR | Zero-shot cross-domain {title}')
    print(sep)
    print(fmt_row(['Method', 'Params'] + domain_labels + ['Avg Cross']))
    print(sep)

    for method in METHODS:
        ml  = method['label']
        res = all_results.get(ml, {})
        row = [ml, res.get('_params_m', '—')]
        cross_vals = []
        for dl in domain_labels:
            if dl in res:
                v = res[dl][metric] * 100
                row.append(f'{v:.1f}')
                if dl in CROSS_LABELS:
                    cross_vals.append(v)
            else:
                row.append('—')
        avg = f'{sum(cross_vals)/len(cross_vals):.1f}' if cross_vals else '—'
        row.append(avg)
        print(fmt_row(row))

    print(sep)
    print(f'  Avg Cross = mean of {", ".join(CROSS_LABELS)}')


# ---------------------------------------------------------------------------
# paper_design.md update
# ---------------------------------------------------------------------------

def update_paper_design(all_results: dict):
    if not os.path.exists(PAPER_MD):
        print(f'[WARN] paper_design.md not found: {PAPER_MD}')
        return

    with open(PAPER_MD, encoding='utf-8') as fh:
        md = fh.read()

    col_order = ['LEVIR', 'GZ', 'WHU', 'CDD', 'SYSU-CD', 'EGY-BCD', 'SI-BU-C2']

    for method in METHODS:
        ml  = method['label']
        res = all_results.get(ml, {})
        if not res:
            continue

        cross_f1  = [res[d]['f1']  * 100 for d in CROSS_LABELS if d in res]
        cross_iou = [res[d]['iou'] * 100 for d in CROSS_LABELS if d in res]
        avg_f1    = sum(cross_f1)  / len(cross_f1)  if cross_f1  else 0
        avg_iou   = sum(cross_iou) / len(cross_iou) if cross_iou else 0

        def make_row(metric: str, avg: float) -> str:
            cells = [f'{res[c][metric]*100:.1f}' if c in res else '—' for c in col_order]
            return f'| **{ml}** | {method["params"]} | ' + \
                   ' | '.join(cells) + f' | **{avg:.1f}** |'

        row_f1  = make_row('f1',  avg_f1)
        row_iou = make_row('iou', avg_iou)

        pattern     = re.compile(rf'\| \*\*{re.escape(ml)}\*\*.*\|[ \t]*\n?')
        occurrences = list(pattern.finditer(md))
        if len(occurrences) >= 2:
            pos2 = occurrences[1].start()
            md   = md[:pos2] + row_iou + '\n' + md[occurrences[1].end():]
            pos1 = occurrences[0].start()
            md   = md[:pos1] + row_f1  + '\n' + md[occurrences[0].end():]
        elif len(occurrences) == 1:
            pos1 = occurrences[0].start()
            md   = md[:pos1] + row_f1  + '\n' + md[occurrences[0].end():]

    with open(PAPER_MD, 'w', encoding='utf-8') as fh:
        fh.write(md)
    print('[OK] paper_design.md updated.')


# ---------------------------------------------------------------------------
# Ablation table printing
# ---------------------------------------------------------------------------

def _avg_cross(res: dict) -> float | None:
    vals = [res[d]['f1'] * 100 for d in CROSS_LABELS if d in res]
    return sum(vals) / len(vals) if vals else None


def print_cdlora_tw_table(tw_results: dict):
    """Print CDLoRA vs CDLoRA-TW comparison across all domains."""
    W = 12
    domains = [d['label'] for d in DOMAINS]

    def row(cells):
        return '│' + '│'.join(f'{str(c):^{W}}' for c in cells) + '│'

    header = ['Method'] + domains + ['Avg Cross']
    sep = '─' * (W * len(header))

    for metric, mkey in [('F1 (%)', 'f1'), ('IoU (%)', 'iou')]:
        print(f'\n{"="*60}')
        print(f'CDLoRA vs CDLoRA-TW — {metric}')
        print(sep)
        print(row(header))
        print(sep)
        for key in CDLORA_TW_KEYS:
            m   = _ABL[key]
            res = tw_results.get(key, {})
            vals = []
            cross_vals = []
            for d in DOMAINS:
                lbl = d['label']
                if lbl in res:
                    v = res[lbl][mkey] * 100
                    vals.append(f'{v:.1f}')
                    if d['cross']:
                        cross_vals.append(v)
                else:
                    vals.append('—')
            avg = f'{sum(cross_vals)/len(cross_vals):.1f}' if cross_vals else '—'
            print(row([m['label']] + vals + [avg]))
        print(sep)

    # Delta row
    print(f'\n{"="*60}')
    print('CDLoRA-TW Δ vs CDLoRA (Avg Cross F1)')
    cdlora_avg = _avg_cross(tw_results.get('cdlora', {}))
    tw_avg     = _avg_cross(tw_results.get('cdlora_tw', {}))
    if cdlora_avg is not None and tw_avg is not None:
        delta = tw_avg - cdlora_avg
        sign  = '+' if delta >= 0 else ''
        print(f'  CDLoRA    : {cdlora_avg:.1f}')
        print(f'  CDLoRA-TW : {tw_avg:.1f}')
        print(f'  Δ         : {sign}{delta:.1f} pp')
        print(f'  Verdict   : {"TW wins ✓" if delta > 0 else "no gain — reconsider" if delta == 0 else "TW worse"}')
    print()


def print_ablation_tables(abl_results: dict):
    W = 14

    def row(cells):
        return '│' + '│'.join(f'{str(c):^{W}}' for c in cells) + '│'

    sep = '─' * (W * 4)

    # Table 2a
    print(f'\n{"="*60}')
    print('Table 2a — Component contribution (Avg Cross F1 %)')
    print(sep)
    print(row(['Variant', 'Params†', 'Avg Cross F1', 'Δ vs prev']))
    print(sep)
    prev = None
    for key in TABLE2A_KEYS:
        m   = _ABL[key]
        res = abl_results.get(key, {})
        avg = _avg_cross(res)
        params = {'lora': '0.295M', 'deep_lora': '0.099M',
                  'cdlora': '0.099M', 'tide': '0.495M'}.get(key, '—')
        f1_str    = f'{avg:.1f}' if avg is not None else '—'
        delta_str = f'{avg - prev:+.1f}' if (avg is not None and prev is not None) else '—'
        print(row([m['label'], params, f1_str, delta_str]))
        if avg is not None:
            prev = avg
    print(sep)

    # Table 2b
    b_labels = {
        'deep_lora':    '(a) Deep LoRA, no cond.',
        'scalar_gamma': '(b) + scalar γ',
        'cdlora_2d':    '(c) + rank-wise γ, 2D',
        'cdlora':       '(d) + rank-wise γ, 4D',
    }
    print(f'\n{"="*60}')
    print('Table 2b — CDLoRA internal design (Avg Cross F1 %)')
    print('─' * (W * 2))
    print(row(['Variant', 'Avg Cross F1']))
    print('─' * (W * 2))
    for key in TABLE2B_KEYS:
        res = abl_results.get(key, {})
        avg = _avg_cross(res)
        f1_str = f'{avg:.1f}' if avg is not None else '—'
        print(row([b_labels[key], f1_str]))
    print('─' * (W * 2))
    print(f'  †Backbone PEFT params only (shared 2.158M decoder excluded)')


# ---------------------------------------------------------------------------
# paper_design.md — ablation update
# ---------------------------------------------------------------------------

def update_paper_design_ablations(abl_results: dict):
    if not os.path.exists(PAPER_MD):
        print(f'[WARN] paper_design.md not found: {PAPER_MD}')
        return

    with open(PAPER_MD, encoding='utf-8') as fh:
        lines = fh.readlines()

    # Pre-compute avg cross F1 for each key
    avgs = {k: _avg_cross(abl_results.get(k, {})) for k in _ABL}

    # Δ values for Table 2a (vs previous row)
    deltas: dict[str, str] = {}
    prev = None
    for key in TABLE2A_KEYS:
        avg = avgs[key]
        deltas[key] = f'{avg - prev:+.1f}' if (avg is not None and prev is not None) else '—'
        if avg is not None:
            prev = avg

    def replace_trailing_dashes(line: str, replacements: list[str]) -> str:
        """Replace the last N '— ' cells in a markdown table row."""
        for rep in reversed(replacements):
            line = line.rstrip()
            # find last occurrence of ' — |' or '| — |'
            idx = line.rfind('| — |')
            if idx == -1:
                idx = line.rfind('| —|')
            if idx != -1:
                line = line[:idx + 2] + rep + ' |' + line[idx + 5:]
        return line + '\n'

    out = []
    for line in lines:
        updated = False
        # Table 2a rows
        for key, anchor in TABLE2A_ANCHORS.items():
            if anchor in line and avgs[key] is not None:
                f1  = f'{avgs[key]:.1f}'
                dlt = deltas[key]
                line = replace_trailing_dashes(line, [dlt, f1])
                updated = True
                break
        # Table 2b rows
        if not updated:
            for key, anchor in TABLE2B_ANCHORS.items():
                if anchor in line and avgs[key] is not None:
                    line = replace_trailing_dashes(line, [f'{avgs[key]:.1f}'])
                    updated = True
                    break
        out.append(line)

    with open(PAPER_MD, 'w', encoding='utf-8') as fh:
        fh.writelines(out)
    print('[OK] paper_design.md ablation tables updated.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='.',
                        help='Experiment folder containing checkpoints/ and logs/.')
    parser.add_argument('--method', default=None,
                        choices=[m['key'] for m in METHODS],
                        help='Evaluate only this method (default: all).')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation eval (Table 2a + 2b) instead of main Table 1.')
    parser.add_argument('--cdlora_tw', action='store_true',
                        help='Run CDLoRA vs CDLoRA-TW comparison.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--update_paper_design', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device    : {device}')
    print(f'exp_dir   : {os.path.abspath(args.exp_dir)}\n')

    if args.cdlora_tw:
        tw_results = {}
        for key in CDLORA_TW_KEYS:
            m = _ABL[key]
            print(f'\n[{m["label"]}]')
            tw_results[key] = run_method(m, args.exp_dir, args.batch_size, device)
        print_cdlora_tw_table(tw_results)
        return

    if args.ablation:
        # Collect unique methods across Table 2a + 2b (avoid double eval)
        all_keys = dict.fromkeys(TABLE2A_KEYS + TABLE2B_KEYS)
        abl_results = {}
        for key in all_keys:
            m = _ABL[key]
            print(f'\n[{m["label"]}]')
            abl_results[key] = run_method(m, args.exp_dir, args.batch_size, device)

        print_ablation_tables(abl_results)

        if args.update_paper_design:
            update_paper_design_ablations(abl_results)
        return

    methods_to_run = [m for m in METHODS
                      if args.method is None or m['key'] == args.method]

    all_results = {}
    for method in methods_to_run:
        print(f'\n[{method["label"]}]')
        all_results[method['label']] = run_method(
            method, args.exp_dir, args.batch_size, device)

    if len(methods_to_run) > 1:
        print_table(all_results, 'f1')
        print_table(all_results, 'iou')

    if args.update_paper_design:
        update_paper_design(all_results)


if __name__ == '__main__':
    main()
