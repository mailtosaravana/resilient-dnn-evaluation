"""Figures for Experiment 2 (CIFAR-10): critical-SDC heatmap and training curves."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 220, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": .3})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cifar")
R = json.load(open(os.path.join(OUT, "results_cifar.json")))
names = list(R["configs"].keys()); FM = ["single", "random", "row", "column", "block", "bitflip", "nan"]
M = np.array([[R["configs"][n]["fi_application"][f]["critical_rate"] * 100 for f in FM] for n in names])
fig, ax = plt.subplots(figsize=(9.5, 4.2))
sns.heatmap(M, annot=True, fmt=".1f", cmap="RdYlGn_r", xticklabels=[f.capitalize() for f in FM], yticklabels=names, ax=ax, cbar_kws={"label": "Critical SDC rate (%)"}, linewidths=.5)
ax.set_xlabel("Fault model"); ax.set_ylabel("Configuration")
fig.savefig(f"{OUT}/fig_cifar_heatmap.png"); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cols = ["#C62828", "#EF9A9A", "#1565C0", "#2E7D32"]
for n, c in zip([k for k in names if "NaN" not in k], cols):
    h = R["configs"][n]["history"]
    ax[0].plot([x["loss"] for x in h], color=c, label=n); ax[1].plot([x["test_acc"] * 100 for x in h], color=c, label=n)
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training loss"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Test accuracy (%)"); ax[1].legend(fontsize=8)
fig.savefig(f"{OUT}/fig_cifar_training.png"); plt.close(fig)
print("ok")
