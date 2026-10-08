"""Figures 1 and 2 of the manuscript (study design; processing order and mitigation points)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

def box(ax, x, y, w, h, title, lines, fc, ec, tfs=9.5, lfs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=ec)
    for i, l in enumerate(lines):
        ax.text(x + 0.05, y + h - 0.19 - i * 0.075, l, ha="left", va="top", fontsize=lfs, color="#222")

def arrow(ax, p, q, color="#555"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, color=color))

# ------------------------------------------------------------------ Fig. 1 study design
fig, ax = plt.subplots(figsize=(9.2, 6.0)); ax.set_xlim(0, 3.0); ax.set_ylim(0.25, 3.05); ax.axis("off")
box(ax, 0.05, 2.5, 1.4, 0.5, "Experiment 1: fully connected network", ["sklearn Digits, 8x8, 10 classes, 1,797 images", "64-128-256-128-64-10 (Dense-BN-act)", "6 configurations x 5 seeds, 60 epochs"], "#E8F1FB", "#1F4E79")
box(ax, 1.55, 2.5, 1.4, 0.5, "Experiment 2: convolutional network", ["CIFAR-10, 32x32x3, 10 classes, 60,000 images", "ResNet-8 (stem + 3 residual stages)", "5 configurations, 1 seed, 15 epochs"], "#E8F1FB", "#1F4E79")
box(ax, 0.3, 1.7, 2.4, 0.62, "Hardening techniques evaluated (reproduced from DieHardNet [7])", [
    "T1  Bounded activation: ReLU replaced by ReLU6 = min(max(0,x),6)",
    "T2  Order inversion: layer-act-BN instead of layer-BN-act",
    "T3  NaN filter: non-finite activations replaced by 0 (inference)",
    "T4  Fault-aware training with curriculum escalation of fault magnitude"], "#FFF4E5", "#8A4B08")
box(ax, 0.05, 0.75, 0.92, 0.78, "Fault-injection campaigns", ["7 fault models: single, random, row,", "column, block, bit flip, NaN", "Injected at matrix-multiplication", "outputs (dense / conv), one fault", "per image per trial", "Explicit trial counts (Table III)"], "#EAF7EC", "#1B5E20")
box(ax, 1.04, 0.75, 0.92, 0.78, "Clean-data metrics", ["Accuracy, per-class precision,", "recall, F1, confusion matrix,", "per-class ROC and AUC", "Reported separately from", "fault tolerance (Sec. V-A / V-B)"], "#F3E8FB", "#4A148C")
box(ax, 2.03, 0.75, 0.92, 0.78, "Reliability and cost metrics", ["SDC, tolerable and critical SDC", "Layer-wise vulnerability", "Magnitude sensitivity", "Mean +/- std over seeds", "Matched inference latency"], "#FDECEC", "#8E1B1B")
box(ax, 0.05, 0.3, 2.9, 0.28, "Open artefacts: source code, seeds, checkpoints, clean predictions and fault-injection logs (Code availability section)", [], "#F2F2F2", "#333", tfs=8.5)
for x in (0.75, 2.25): arrow(ax, (x, 2.5), (x, 2.33))
for x in (0.51, 1.5, 2.49): arrow(ax, (x, 1.7), (x, 1.54))
arrow(ax, (1.5, 0.75), (1.5, 0.59))
fig.savefig(f"{OUT}/fig1_study_design.png", dpi=250, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ Fig. 2 processing order
fig, ax = plt.subplots(figsize=(9.6, 5.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.9); ax.axis("off")
def chain(y, items, label):
    ax.text(0.1, y + 0.98, label, fontsize=9.5, fontweight="bold", color="#222")
    x = 0.1
    for i, (t, sub, fc, ec) in enumerate(items):
        w = 1.55
        ax.add_patch(FancyBboxPatch((x, y), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + 0.33, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + 0.13, sub, ha="center", va="center", fontsize=7.2, color="#333")
        if i < len(items) - 1: arrow(ax, (x + w, y + 0.25), (x + w + 0.3, y + 0.25))
        x += w + 0.3
    return x
G, R, B, O, Y = ("#F2F2F2", "#444"), ("#FDECEC", "#B71C1C"), ("#E8F1FB", "#1F4E79"), ("#FFF4E5", "#8A4B08"), ("#EAF7EC", "#1B5E20")
chain(4.75, [("Layer k", "dense or conv (GEMM)", *R), ("BatchNorm", "y = g(x-m)/s + b", *B), ("ReLU", "max(0, x), unbounded", *O), ("Next layer", "", *G)], "Standard order (Baseline): layer - BN - ReLU")
chain(3.35, [("Layer k", "dense or conv (GEMM)", *R), ("ReLU6", "clip to [0, 6]   (T1)", *Y), ("BatchNorm", "input bounded   (T2)", *B), ("NaN filter", "non-finite -> 0   (T3)", *Y), ("Next layer", "", *G)], "Inverted order (+ReLU6+OrderInv+NaN): layer - ReLU6 - BN - filter")
for y in (4.75, 3.35):
    ax.annotate("fault injected here (matrix-multiplication output)", xy=(0.9, y + 0.5), xytext=(2.6, y + 0.78), ha="left", va="center", fontsize=7.6, color="#B71C1C",
                arrowprops=dict(arrowstyle="-|>", color="#B71C1C", lw=1))
# explanatory panel
ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 2.6, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#FAFAFA", ec="#999", lw=1))
txt = [
 ("Inference time (BN statistics m, s frozen).", True), ("A transient perturbation d at the BN input reaches the next layer as g*d/s.", False),
 ("Standard order: d is unbounded (|d| up to ~1e38) and ReLU removes only negative values, so g*d/s can dominate", False),
 ("all later computation.  Inverted order: ReLU6 clips the corrupted value to [0, 6] before BN, so the BN output", False),
 ("lies in [b - g*m/s,  b + g*(6-m)/s] whatever the fault magnitude.  NaN/Inf are not removed by clipping", False),
 ("(NaN < 6 is False); the filter after BN maps them to 0.", False),
 ("Training time (fault-aware training, T4).", True), ("Faults injected before BN enter the batch mean/variance and the running", False),
 ("statistics; with the inverted order the statistics are computed from bounded activations, so m, s, g, b are not contaminated.", False),
]
yy = 2.5
for t, b in txt:
    if b: ax.text(0.3, yy, t, fontsize=8, color="#222", va="top", fontweight="bold"); yy -= 0.3
    else: ax.text(0.3 if not txt[[x[0] for x in txt].index(t)-1][1] else 3.9, yy + (0.3 if txt[[x[0] for x in txt].index(t)-1][1] else 0), t, fontsize=8, color="#222", va="top"); yy -= 0.3 if not txt[[x[0] for x in txt].index(t)-1][1] else 0
fig.savefig(f"{OUT}/fig2_processing_order.png", dpi=250, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")

# ------------------------------------------------------------------ Fig. 3 curriculum
import numpy as np
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for E, col, lab in [(60, "#1F4E79", "Digits, 60 epochs"), (15, "#B71C1C", "CIFAR-10, 15 epochs")]:
    e = np.arange(E); ax.plot(e / (E - 1) * 100, 0.5 + (50 - 0.5) * e / E, color=col, lw=2, label=lab)
ax.set_xlabel("Training progress (%)"); ax.set_ylabel("Fault magnitude m"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.text(2, 44, "fault injected in 75% of training steps;\nmodels: single, random, row, column, block", fontsize=7.5, va="top", color="#333")
fig.savefig(f"{OUT}/fig_curriculum.png", dpi=250, bbox_inches="tight", facecolor="white"); plt.close(fig)
