"""
TIDE method figure (conference-grade).

TIDE = CDLoRA (image-level rank-wise 2D gamma from [mean,std] of |T1-T2|)
       + BCA (bidirectional bi-temporal cross-attention),
     injected into the frozen DINOv2 ViT-B/14 encoder at blocks 6-7.

Outputs: figures/tide_method.pdf  (vector, for the paper)
         figures/tide_method.png  (300 dpi preview)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0,
})

# palette
C_FROZEN_F = "#EBEEF3";  C_FROZEN_E = "#9AA6B5"     # frozen blocks (slate)
C_HL_F     = "#FFE2CC";  C_HL_E     = "#E2762E"     # highlighted blocks 6-7 (warm)
C_LORA_F   = "#D7F0EB";  C_LORA_E   = "#2C8C82"     # CDLoRA accent (teal)
C_BCA_F    = "#E8DDF4";  C_BCA_E    = "#7A57A8"     # BCA accent (purple)
C_DEC_F    = "#E0EFDA";  C_DEC_E    = "#5C9A52"     # decoder (green)
C_GAMMA_F  = "#FFF3D6";  C_GAMMA_E  = "#D9A21B"     # gamma / MLP (gold)
C_INK      = "#2B2B2B"
C_ARROW    = "#555555"
C_SHADOW   = "#000000"

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")


# ---------------------------------------------------------------- helpers
def box(x, y, w, h, fc, ec, lw=1.4, rounding=0.10, shadow=True, z=2):
    """Rounded box with a soft drop shadow. Returns dict of anchor points."""
    if shadow:
        sh = FancyBboxPatch((x + 0.05, y - 0.06), w, h,
                            boxstyle=f"round,pad=0,rounding_size={rounding}",
                            fc=C_SHADOW, ec="none", alpha=0.08, zorder=z - 1,
                            mutation_aspect=1)
        ax.add_patch(sh)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1)
    ax.add_patch(p)
    return dict(x=x, y=y, w=w, h=h,
                cx=x + w / 2, cy=y + h / 2,
                left=(x, y + h / 2), right=(x + w, y + h / 2),
                top=(x + w / 2, y + h), bot=(x + w / 2, y))


def text(x, y, s, size=9, color=C_INK, weight="normal", style="normal",
         ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            fontstyle=style, ha=ha, va=va, zorder=z)


def arrow(p1, p2, color=C_ARROW, lw=1.6, style="-|>", ms=12, ls="-",
          rad=0.0, z=3, alpha=1.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                        lw=lw, color=color, ls=ls, zorder=z, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def snowflake(x, y, s=0.13, color="#3A7BD5", lw=1.3, z=8):
    """Frozen icon."""
    for ang in range(0, 180, 60):
        a = np.deg2rad(ang)
        dx, dy = s * np.cos(a), s * np.sin(a)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=lw,
                solid_capstyle="round", zorder=z)
        # little branches
        for t in (0.55, -0.55):
            bx, by = x + dx * 0.6, y + dy * 0.6
            ax.plot([bx, bx + s * 0.35 * np.cos(a + t)],
                    [by, by + s * 0.35 * np.sin(a + t)],
                    color=color, lw=lw * 0.8, solid_capstyle="round", zorder=z)


def fire(x, y, s=0.15, color="#E2762E", z=8):
    """Trainable icon (a star = learnable)."""
    ax.plot(x, y, marker="*", markersize=s * 95, color=color,
            markeredgecolor="white", markeredgewidth=0.6, zorder=z)


# ================================================================ TITLE
text(8.0, 8.72, "TIDE", size=20, weight="bold", color=C_INK)
text(8.0, 8.34,
     "Change-aware Parameter-Efficient Fine-Tuning for Zero-shot Cross-domain Change Detection",
     size=10, color="#555555", style="italic")

# legend (top-right)
lg = box(13.05, 7.95, 2.85, 0.78, "#FBFBFC", "#CCCCCC", lw=1.0, rounding=0.08)
snowflake(13.32, 8.5, s=0.11)
text(13.55, 8.5, "Frozen  (DINOv2)", size=8.2, ha="left")
fire(13.36, 8.17, s=0.11)
text(13.55, 8.17, "Trainable  (PEFT, ~0.44M, 0.5%)", size=8.2, ha="left")


# ================================================================ INPUTS
def img_tile(x, y, w, h, base, accent, label, sub):
    box(x, y, w, h, base, "#7E8A99", lw=1.2, rounding=0.06)
    # a couple of "building" blobs to suggest a remote-sensing scene
    ax.add_patch(plt.Rectangle((x + 0.18, y + 0.20), 0.32, 0.30, fc=accent,
                               ec="none", zorder=4))
    ax.add_patch(plt.Rectangle((x + 0.62, y + 0.45), 0.26, 0.30, fc=accent,
                               ec="none", alpha=0.8, zorder=4))
    ax.add_patch(Circle((x + w - 0.32, y + 0.30), 0.13, fc="#A9C5A0",
                        ec="none", zorder=4))
    text(x + w / 2, y - 0.24, label, size=9.5, weight="bold")
    text(x + w / 2, y - 0.5, sub, size=7.6, color="#777777")


T1 = (0.55, 6.95, 1.25, 1.05)
T2 = (0.55, 5.35, 1.25, 1.05)
img_tile(*T1, "#CBD8E8", "#6E86A8", "T1", "pre-event")
img_tile(*T2, "#D8CDC2", "#9A7F63", "T2", "post-event")

# diff-stats branch (feeds gamma)
diff = box(2.30, 5.95, 1.35, 0.95, "#F3F0EA", "#B7A98C", lw=1.2)
text(diff["cx"], diff["cy"] + 0.22, "|T1 - T2|", size=8.6, weight="bold")
text(diff["cx"], diff["cy"] - 0.12, "image-level", size=7.2, color="#888")
text(diff["cx"], diff["cy"] - 0.34, r"stats  [$\mu,\ \sigma$]", size=8.4,
     color="#8A6D1E", weight="bold")
arrow((T1[0] + T1[2], T1[1] + 0.3), (diff["left"][0], diff["left"][1] + 0.18))
arrow((T2[0] + T2[2], T2[1] + 0.7), (diff["left"][0], diff["left"][1] - 0.18))


# ================================================================ ENCODER
ENC = box(4.05, 5.05, 7.55, 3.05, "#F7F9FC", "#8C9AAB", lw=1.6, rounding=0.10)
snowflake(ENC["x"] + 0.32, ENC["y"] + ENC["h"] - 0.27, s=0.11)
text(ENC["x"] + 0.52, ENC["y"] + ENC["h"] - 0.27,
     "DINOv2 ViT-B/14  (shared siamese, frozen)", size=9.2, weight="bold",
     ha="left")

# 12 transformer blocks, two streams (top = T1, bottom = T2)
n = 12
bx0, bx1 = ENC["x"] + 0.45, ENC["x"] + ENC["w"] - 0.45
bw = (bx1 - bx0) / n
cell_w = bw * 0.82
row_t_y, row_b_y = 6.95, 5.55
ch = 0.62
HL = {6, 7}
cells_t, cells_b = [], []
for i in range(n):
    cx = bx0 + bw * i + bw / 2
    fc, ec = (C_HL_F, C_HL_E) if i in HL else (C_FROZEN_F, C_FROZEN_E)
    ct = box(cx - cell_w / 2, row_t_y, cell_w, ch, fc, ec, lw=1.1,
             rounding=0.05, shadow=False, z=3)
    cb = box(cx - cell_w / 2, row_b_y, cell_w, ch, fc, ec, lw=1.1,
             rounding=0.05, shadow=False, z=3)
    cells_t.append(ct); cells_b.append(cb)
    text(cx, row_t_y + ch / 2, str(i), size=7.4,
         color=C_HL_E if i in HL else "#7C889A",
         weight="bold" if i in HL else "normal")
    text(cx, row_b_y + ch / 2, str(i), size=7.4,
         color=C_HL_E if i in HL else "#7C889A",
         weight="bold" if i in HL else "normal")
    if i not in HL:
        snowflake(cx, row_t_y + ch + 0.0, s=0.06, lw=0.9)

# stream labels
text(ENC["x"] + 0.30, row_t_y + ch / 2, r"$h_{T1}$", size=8.4, weight="bold",
     color="#46607F")
text(ENC["x"] + 0.30, row_b_y + ch / 2, r"$h_{T2}$", size=8.4, weight="bold",
     color="#7A5E44")

# horizontal flow through each stream
arrow((cells_t[0]["left"][0] - 0.30, row_t_y + ch / 2),
      (cells_t[0]["left"]), lw=1.3)
arrow((cells_b[0]["left"][0] - 0.30, row_b_y + ch / 2),
      (cells_b[0]["left"]), lw=1.3)
for i in range(n - 1):
    arrow(cells_t[i]["right"], cells_t[i + 1]["left"], lw=1.2, ms=9)
    arrow(cells_b[i]["right"], cells_b[i + 1]["left"], lw=1.2, ms=9)

# BCA interaction (bidirectional) between streams at blocks 6,7
for i in HL:
    cx = cells_t[i]["cx"]
    arrow((cx, row_t_y), (cx, row_b_y + ch), color=C_BCA_E, lw=1.8,
          style="<|-|>", ms=11, z=4)
text((cells_t[6]["cx"] + cells_t[7]["cx"]) / 2, (row_t_y + row_b_y + ch) / 2,
     "BCA", size=7.8, color=C_BCA_E, weight="bold",
     ha="center", va="center")
# tag over highlighted region
text((cells_t[6]["cx"] + cells_t[7]["cx"]) / 2, row_t_y + ch + 0.30,
     "CDLoRA  +  BCA", size=8.6, weight="bold", color=C_HL_E)
fire((cells_t[6]["cx"] + cells_t[7]["cx"]) / 2 - 0.95, row_t_y + ch + 0.30,
     s=0.10)

# entry arrows into encoder from inputs
arrow((T1[0] + T1[2], T1[1] + 0.7),
      (cells_t[0]["left"][0] - 0.30, row_t_y + ch / 2), rad=-0.05)
arrow((diff["right"]), (cells_b[0]["left"][0] - 0.30, row_b_y + ch / 2),
      rad=0.05)

# gamma pathway: diff-stats -> Scale-MLP -> gamma -> into blocks 6-7
mlp = box(2.95, 3.95, 1.45, 0.80, C_GAMMA_F, C_GAMMA_E, lw=1.3)
text(mlp["cx"], mlp["cy"] + 0.14, "Scale-MLP", size=8.4, weight="bold")
text(mlp["cx"], mlp["cy"] - 0.18, r"$\gamma_q,\ \gamma_v \in \mathbb{R}^{r}$",
     size=8.2, color="#8A6D1E")
fire(mlp["x"] + 0.16, mlp["y"] + mlp["h"] - 0.12, s=0.085)
arrow((diff["cx"], diff["y"]), (mlp["top"]), color=C_GAMMA_E, lw=1.5, rad=-0.1)
# gamma up into the highlighted blocks
gx = (cells_b[6]["cx"] + cells_b[7]["cx"]) / 2
arrow((mlp["right"][0], mlp["right"][1]),
      (gx, row_b_y - 0.02), color=C_GAMMA_E, lw=1.6, rad=-0.18, style="-|>")
text((mlp["right"][0] + gx) / 2 + 0.1, row_b_y - 0.42, r"$\gamma$ modulation",
     size=7.6, color="#8A6D1E", style="italic")


# ================================================================ FUSION + DECODER + OUTPUT
fuse = box(11.95, 5.95, 0.95, 0.95, "#F0F0F3", "#888", lw=1.3, rounding=0.5)
text(fuse["cx"], fuse["cy"] + 0.16, r"$|\,\cdot\,|$", size=11, weight="bold")
text(fuse["cx"], fuse["cy"] - 0.20, r"$F_{T1}\!-\!F_{T2}$", size=6.6,
     color="#666")
arrow(cells_t[-1]["right"], (fuse["left"][0], fuse["left"][1] + 0.18), rad=-0.1)
arrow(cells_b[-1]["right"], (fuse["left"][0], fuse["left"][1] - 0.18), rad=0.1)

dec = box(13.20, 5.85, 1.35, 1.15, C_DEC_F, C_DEC_E, lw=1.5)
text(dec["cx"], dec["cy"] + 0.18, "MLP", size=9.5, weight="bold")
text(dec["cx"], dec["cy"] - 0.10, "Decoder", size=9.5, weight="bold")
fire(dec["x"] + 0.18, dec["y"] + dec["h"] - 0.16, s=0.09)
arrow(fuse["right"], dec["left"])

out = box(14.95, 5.95, 0.95, 0.95, "#EFEFEF", "#666", lw=1.2, rounding=0.06)
# a tiny change-mask thumbnail
ax.add_patch(plt.Rectangle((out["x"] + 0.10, out["y"] + 0.10), 0.75, 0.75,
                           fc="#1B1B1B", ec="none", zorder=4))
ax.add_patch(plt.Rectangle((out["x"] + 0.30, out["y"] + 0.45), 0.22, 0.22,
                           fc="white", ec="none", zorder=5))
ax.add_patch(plt.Rectangle((out["x"] + 0.55, out["y"] + 0.22), 0.16, 0.16,
                           fc="white", ec="none", zorder=5))
text(out["cx"], out["y"] - 0.26, "Change Map", size=8.4, weight="bold")
arrow(dec["right"], out["left"])


# ================================================================ DETAIL PANELS
# dashed callouts from blocks 6-7 down to the panels
ax.plot([cells_b[6]["cx"], 4.0], [row_b_y, 4.55], ls=(0, (4, 3)),
        color=C_LORA_E, lw=1.1, alpha=0.7, zorder=1)
ax.plot([cells_b[7]["cx"], 11.6], [row_b_y, 4.55], ls=(0, (4, 3)),
        color=C_BCA_E, lw=1.1, alpha=0.7, zorder=1)

# ---------------- Panel A : CDLoRA ----------------------------------------
PA = box(0.55, 0.45, 7.35, 4.05, "#FCFDFE", C_LORA_E, lw=1.6, rounding=0.08)
text(PA["x"] + 0.30, PA["y"] + PA["h"] - 0.28,
     "A.  CDLoRA — change-aware low-rank adaptation",
     size=10.5, weight="bold", color=C_LORA_E, ha="left")

# frozen qkv
xqkv = box(0.95, 2.55, 1.35, 1.05, C_FROZEN_F, C_FROZEN_E, lw=1.3)
text(xqkv["cx"], xqkv["cy"] + 0.16, "Frozen", size=8.2)
text(xqkv["cx"], xqkv["cy"] - 0.16, r"$W_{qkv}$", size=10, weight="bold")
snowflake(xqkv["x"] + 0.18, xqkv["y"] + xqkv["h"] - 0.14, s=0.085)
text(0.95, 3.95, r"token  $x$", size=8.4, ha="left", weight="bold")
arrow((0.95, 3.78), (xqkv["left"][0] - 0.0, xqkv["top"][1] - 0.0), style="-")
arrow((1.10, 3.72), (1.10, xqkv["y"] + xqkv["h"]))

# LoRA branch: x -> A -> *gamma -> B -> add
yL = 1.35
A_ = box(1.30, yL, 0.78, 0.72, C_LORA_F, C_LORA_E, lw=1.3)
text(A_["cx"], A_["cy"] + 0.12, "A", size=10, weight="bold", color=C_LORA_E)
text(A_["cx"], A_["cy"] - 0.16, "r=8", size=6.8, color="#2C8C82")
fire(A_["x"] + 0.12, A_["y"] + A_["h"] - 0.1, s=0.075)

gnode = Circle((3.05, yL + 0.36), 0.27, fc=C_GAMMA_F, ec=C_GAMMA_E, lw=1.4,
               zorder=4)
ax.add_patch(gnode)
text(3.05, yL + 0.36, r"$\odot\gamma$", size=9.5, weight="bold",
     color="#8A6D1E")

B_ = box(3.70, yL, 0.78, 0.72, C_LORA_F, C_LORA_E, lw=1.3)
text(B_["cx"], B_["cy"], "B", size=10, weight="bold", color=C_LORA_E)
fire(B_["x"] + 0.12, B_["y"] + B_["h"] - 0.1, s=0.075)

addc = Circle((5.15, yL + 0.36), 0.24, fc="white", ec=C_INK, lw=1.4, zorder=4)
ax.add_patch(addc)
text(5.15, yL + 0.36, "+", size=12, weight="bold")

arrow((1.10, xqkv["y"]), (A_["top"][0] - 0.1, A_["top"][1]), rad=-0.2)
arrow(A_["right"], (2.78, yL + 0.36))
arrow((3.32, yL + 0.36), B_["left"])
arrow(B_["right"], (4.91, yL + 0.36))
arrow((xqkv["bot"][0], xqkv["y"]), (5.15, yL + 0.60), rad=0.0, style="-|>",
      color=C_ARROW)
arrow((5.15, yL + 0.60), (5.15, yL + 0.36 + 0.24), style="-")  # into add (Q/V)
arrow((5.39, yL + 0.36), (6.05, yL + 0.36))
text(6.55, yL + 0.36, r"$Q,\,V$", size=9, weight="bold")

# gamma from scale-mlp
gm = box(2.45, 3.05, 1.55, 0.70, C_GAMMA_F, C_GAMMA_E, lw=1.3)
text(gm["cx"], gm["cy"] + 0.13, "Scale-MLP", size=8.0, weight="bold")
text(gm["cx"], gm["cy"] - 0.15, r"$[\mu,\sigma]\!\rightarrow\!\gamma\in\mathbb{R}^{r}$",
     size=7.6, color="#8A6D1E")
fire(gm["x"] + 0.14, gm["y"] + gm["h"] - 0.1, s=0.075)
arrow((gm["cx"], gm["y"]), (3.05, yL + 0.36 + 0.27), color=C_GAMMA_E, lw=1.4,
      rad=0.0)

# equation + intuition
text(4.15, 3.55, r"$\Delta = B\,((A x)\odot\gamma)$",
     size=11, ha="left", weight="bold")
text(0.95, 0.78,
     r"strong change $\Rightarrow \gamma\!\uparrow$ strong adaptation"
     r"      weak change $\Rightarrow \gamma\!\approx\!0$ (stay frozen)",
     size=7.8, ha="left", color="#555", style="italic")

# ---------------- Panel B : BCA -------------------------------------------
PB = box(8.20, 0.45, 7.30, 4.05, "#FCFDFE", C_BCA_E, lw=1.6, rounding=0.08)
text(PB["x"] + 0.30, PB["y"] + PB["h"] - 0.28,
     "B.  BCA — bi-temporal cross-attention",
     size=10.5, weight="bold", color=C_BCA_E, ha="left")

h1 = box(8.55, 3.10, 1.15, 0.70, "#DCE6F2", "#46607F", lw=1.3)
text(h1["cx"], h1["cy"], r"$h_{T1}$", size=10, weight="bold", color="#46607F")
h2 = box(8.55, 1.05, 1.15, 0.70, "#EFE4D6", "#7A5E44", lw=1.3)
text(h2["cx"], h2["cy"], r"$h_{T2}$", size=10, weight="bold", color="#7A5E44")

# cross attention core
ca = box(10.35, 1.85, 2.30, 1.20, C_BCA_F, C_BCA_E, lw=1.4)
text(ca["cx"], ca["cy"] + 0.34, "Cross-Attention", size=8.6, weight="bold",
     color=C_BCA_E)
text(ca["cx"], ca["cy"] + 0.02,
     r"$\mathrm{softmax}(QK^{\!\top}\!/\sqrt{d})\,V$", size=8.4)
text(ca["cx"], ca["cy"] - 0.30, r"$Q\!=\!W_Q h_{T1},\ K,V\!=\!W_{K,V} h_{T2}$",
     size=6.8, color="#5A4A78")
fire(ca["x"] + 0.16, ca["y"] + ca["h"] - 0.14, s=0.08)

# Q from h1, K/V from h2 (and symmetric, shown via labels)
arrow(h1["right"], (ca["left"][0], ca["cy"] + 0.30), rad=-0.15)
text((h1["right"][0] + ca["x"]) / 2, ca["cy"] + 0.62, "Q", size=8,
     weight="bold", color="#46607F")
arrow(h2["right"], (ca["left"][0], ca["cy"] - 0.30), rad=0.15)
text((h2["right"][0] + ca["x"]) / 2, ca["cy"] - 0.62, "K, V", size=8,
     weight="bold", color="#7A5E44")

# W_O (zero init) -> alpha gate -> residual add
wo = box(13.00, 2.10, 0.95, 0.70, "#F0EAF7", C_BCA_E, lw=1.3)
text(wo["cx"], wo["cy"] + 0.12, r"$W_O$", size=9.5, weight="bold")
text(wo["cx"], wo["cy"] - 0.16, "0-init", size=6.6, color=C_BCA_E)
arrow(ca["right"], wo["left"])

ag = Circle((14.40, 2.45), 0.27, fc=C_GAMMA_F, ec=C_GAMMA_E, lw=1.4, zorder=4)
ax.add_patch(ag)
text(14.40, 2.45, r"$\times\alpha$", size=9, weight="bold", color="#8A6D1E")
arrow(wo["right"], (14.13, 2.45))

addb = Circle((14.40, 3.45), 0.24, fc="white", ec=C_INK, lw=1.4, zorder=4)
ax.add_patch(addb)
text(14.40, 3.45, "+", size=12, weight="bold")
arrow((14.40, 2.72), (14.40, 3.21))
arrow((h1["top"][0], h1["top"][1]), (14.16, 3.45), rad=-0.25,
      color="#46607F", lw=1.3)
arrow((14.40, 3.69), (14.40, 4.0), style="-|>")
text(14.40, 4.12, r"$h_{T1}'$", size=9, weight="bold", color="#46607F")

# notes
text(8.55, 0.74,
     r"$W_O$ zero-init $\Rightarrow$ identity at init      "
     r"$\alpha$ learnable (init 0.01)      bidirectional & symmetric",
     size=7.8, ha="left", color="#555", style="italic")


# ================================================================ SAVE
plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig("figures/tide_method.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig("figures/tide_method.png", dpi=300, bbox_inches="tight",
            pad_inches=0.05)
print("saved figures/tide_method.pdf  and  figures/tide_method.png")
