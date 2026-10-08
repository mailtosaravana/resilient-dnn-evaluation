#!/usr/bin/env python3
"""
ResilientNet reproducibility package -- Experiment 1 (fully connected network, sklearn Digits)

Runs ALL of the following from ONE script, with ONE set of trained models per seed, so that every
table, figure and number in the manuscript is derived from the same checkpoints:

  * 6 configurations x 5 seeds, 60 epochs each
  * clean classification metrics (accuracy, per-class P/R/F1, confusion matrix, ROC/AUC)
  * application-level fault injection: 7 fault models x 100 trials x 6 configs x 5 seeds
  * layer-wise vulnerability: 5 layers x 80 trials x 6 configs x 5 seeds
  * magnitude sensitivity: 8 magnitudes x 50 trials x 6 configs x 5 seeds
  * matched inference-latency measurement for every configuration
  * mean +/- std across seeds for every reported quantity

Outputs (results_digits/):
  results_digits.json        aggregated numbers used in the manuscript
  per_seed/*.json            raw per-seed numbers
  predictions_seed{S}.csv    clean test-set predictions and probabilities (primary seed)
  fi_log_seed{S}.csv         one row per fault-injection trial (primary seed)
  fig_*.png                  all figures

Usage:  python digits_experiment.py            (full run, ~6 min on 2 CPU cores)
"""
import json, os, sys, time, csv, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_curve, auc)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_digits")
os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
PRIMARY_SEED = 0
EPOCHS = 60
BATCH = 64
LR0 = 0.05
FAULT_MODELS = ["single", "random", "row", "column", "block", "bitflip", "nan"]
MAGS_APP = [1.5, 5.0, 10.0, 50.0, 100.0]
MAGS_SWEEP = [1.1, 2, 5, 10, 25, 50, 100, 500]
N_APP, N_LAYER, N_MAG = 100, 80, 50          # trials per cell
SIZES5 = [64, 128, 256, 128, 64, 10]
SIZES3 = [64, 128, 64, 10]

CONFIGS = [  # name, sizes, relu6, order_inversion, nan_filter, fault_aware_training
    ("Baseline (3-layer)", SIZES3, False, False, False, False),
    ("Baseline (5-layer)", SIZES5, False, False, False, False),
    ("+ReLU6",             SIZES5, True,  False, False, False),
    ("+ReLU6+OrderInv",    SIZES5, True,  True,  False, False),
    ("+ReLU6+OI+NaN",      SIZES5, True,  True,  True,  False),
    ("ResilientNet (FT)",  SIZES5, True,  True,  True,  True),
]

# ----------------------------------------------------------------------------- data
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
Y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(X, Y, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------------------------------------------------------- model
class Dense:
    def __init__(self, i, o, rng):
        self.W = rng.standard_normal((i, o)) * np.sqrt(2.0 / i); self.b = np.zeros(o)
    def forward(self, x): self.x = x; return x @ self.W + self.b
    def backward(self, g, lr):
        dW = self.x.T @ g / len(self.x); db = g.mean(0)
        self.W -= lr * np.clip(dW, -1, 1); self.b -= lr * np.clip(db, -1, 1)
        return g @ self.W.T

class BN:
    """Batch normalisation over the feature axis. Running statistics are updated ONLY in training mode."""
    def __init__(self, d): self.g = np.ones(d); self.b = np.zeros(d); self.rm = np.zeros(d); self.rv = np.ones(d)
    def forward(self, x, train):
        if train:
            m, v = x.mean(0), x.var(0)
            self.rm = 0.9 * self.rm + 0.1 * m; self.rv = 0.9 * self.rv + 0.1 * v
        else:
            m, v = self.rm, self.rv
        return self.g * (x - m) / np.sqrt(v + 1e-5) + self.b

def relu(x): return np.maximum(0, x)
def relu6(x): return np.clip(x, 0, 6)
def nanfilter(x): x = x.copy(); x[~np.isfinite(x)] = 0.0; return x
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

class Net:
    """Fully connected network: Dense -> [BN -> act] (standard) or Dense -> [act -> BN] (order inversion),
    optionally followed by a NaN filter. Faults are injected into the OUTPUT OF A DENSE LAYER
    (the pre-normalisation / pre-activation tensor), i.e. the same point at which a corrupted
    matrix-multiplication result would appear in hardware."""
    def __init__(self, sizes, relu6_on, oi, nf, rng):
        self.layers = [Dense(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]
        self.bns = [BN(sizes[i + 1]) for i in range(len(sizes) - 2)]
        self.act = relu6 if relu6_on else relu
        self.oi, self.nf = oi, nf
    def n_hidden(self): return len(self.layers) - 1
    def forward(self, x, train=False, inj=None):
        # inj = (layer_index, fault_model, magnitude, sample_index_or_None)
        for i, L in enumerate(self.layers):
            x = L.forward(x)
            if inj is not None and inj[0] == i:
                x = inject(x, inj[1], inj[2], inj[3], self._rng)
            if i < len(self.layers) - 1:
                if self.oi:  x = self.bns[i].forward(self.act(x), train)
                else:        x = self.act(self.bns[i].forward(x, train))
                if self.nf:  x = nanfilter(x)
        return softmax(x)
    def train_step(self, x, y, lr, fault_aware, fmag):
        inj = None
        if fault_aware and self._rng.random() < 0.75:
            inj = (self._rng.integers(0, self.n_hidden()), self._rng.choice(["single", "random", "row", "column", "block"]), fmag, None)
        p = self.forward(x, True, inj)
        g = p - y
        for L in reversed(self.layers): g = L.backward(g, lr)
        return float(-np.sum(y * np.log(p + 1e-10)) / len(p))
    def predict_proba(self, x): return self.forward(x, False)
    def predict(self, x): return np.argmax(self.predict_proba(x), 1)

# ----------------------------------------------------------------------------- fault injection
def inject(t, fm, mag, sample, rng):
    """Corrupt the 2-D activation tensor t (batch x features). If sample is None, every row gets its own fault
    (used during fault-aware training); otherwise only row `sample` is corrupted (used in evaluation)."""
    t = t.copy()
    rows = range(t.shape[0]) if sample is None else [sample]
    F = t.shape[1]
    for r in rows:
        if fm == "single":
            t[r, rng.integers(F)] *= mag
        elif fm == "random":
            idx = rng.choice(F, max(1, int(0.01 * F)), replace=False); t[r, idx] *= mag
        elif fm == "row":          # whole activation vector of this sample (one 'row' of the batch matrix)
            t[r, :] *= mag
        elif fm == "column":       # one feature across the whole batch (one 'column' of the batch matrix)
            t[:, rng.integers(F)] *= mag
        elif fm == "block":        # 10 % contiguous features
            bs = max(1, int(0.1 * F)); s = rng.integers(0, F - bs + 1); t[r, s:s + bs] *= mag
        elif fm == "bitflip":
            j = rng.integers(F); v = np.array([t[r, j]], dtype=np.float32)
            bits = np.frombuffer(v.tobytes(), dtype=np.uint32)[0] ^ np.uint32(1 << int(rng.integers(32)))
            t[r, j] = np.frombuffer(np.array([bits], dtype=np.uint32).tobytes(), dtype=np.float32)[0]
        elif fm == "nan":
            t[r, rng.integers(F)] = np.nan
    return t

def fi_trial(net, x_one, p_clean, pred_clean, layer, fm, mag, rng):
    """One trial: one test sample, one fault. Returns (sdc, critical)."""
    net._rng = rng
    with np.errstate(all="ignore"):
        p = net.forward(x_one, False, (layer, fm, mag, 0))
    if not np.all(np.isfinite(p)):          # a non-finite output vector is an undefined prediction -> critical SDC
        return True, True
    sdc = not np.allclose(p, p_clean, atol=1e-6)
    crit = int(np.argmax(p)) != int(pred_clean)
    return sdc, crit

# ----------------------------------------------------------------------------- run one seed
def run_seed(seed):
    res = {"seed": seed, "configs": {}}
    fi_rows = []
    for name, sizes, r6, oi, nf, ft in CONFIGS:
        rng = np.random.default_rng(seed)
        net = Net(sizes, r6, oi, nf, rng); net._rng = rng
        hist_loss, hist_acc = [], []
        t0 = time.time()
        for ep in range(EPOCHS):
            lr = LR0 * 0.5 * (1 + np.cos(np.pi * ep / EPOCHS))
            fmag = 0.5 + (50.0 - 0.5) * ep / EPOCHS            # curriculum escalation
            idx = rng.permutation(len(X_tr)); tot = 0; nb = 0
            for s in range(0, len(X_tr), BATCH):
                b = idx[s:s + BATCH]; tot += net.train_step(X_tr[b], Y_tr[b], lr, ft, fmag); nb += 1
            hist_loss.append(tot / nb); hist_acc.append(float(accuracy_score(y_te, net.predict(X_te))))
        train_time = time.time() - t0

        proba = net.predict_proba(X_te); pred = proba.argmax(1)
        acc = float(accuracy_score(y_te, pred))
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, pred, labels=range(10), zero_division=0)
        ybin = label_binarize(y_te, classes=range(10))
        aucs = [auc(*roc_curve(ybin[:, k], proba[:, k])[:2]) for k in range(10)]

        # matched latency: forward pass over the full test set, 300 repetitions, median
        lat = []
        for _ in range(300):
            t1 = time.perf_counter(); net.predict_proba(X_te); lat.append((time.perf_counter() - t1) * 1e3)
        lat = np.array(lat)

        # ---- fault injection campaign (evaluation) -- fixed FI seed so campaigns are comparable
        fi_rng = np.random.default_rng(1000 + seed)
        nh = net.n_hidden()
        # (a) application level: 7 fault models x N_APP trials, random layer, random magnitude
        app = {}
        for fm in FAULT_MODELS:
            sdc = crit = 0
            for t in range(N_APP):
                i = fi_rng.integers(len(X_te)); layer = int(fi_rng.integers(nh))
                mag = float(fi_rng.choice(MAGS_APP)) if fm not in ("bitflip", "nan") else float("nan")
                s_, c_ = fi_trial(net, X_te[i:i + 1], proba[i:i + 1], pred[i], layer, fm, mag, fi_rng)
                sdc += s_; crit += c_
                fi_rows.append([name, "application", fm, layer, mag, int(i), int(s_), int(c_)])
            app[fm] = {"trials": N_APP, "sdc": sdc, "critical": crit, "sdc_rate": sdc / N_APP, "critical_rate": crit / N_APP}
        # (b) layer-wise: each hidden layer x N_LAYER trials, random fault model among magnitude faults
        lay = {}
        for layer in range(nh):
            sdc = crit = 0
            for t in range(N_LAYER):
                i = fi_rng.integers(len(X_te)); fm = str(fi_rng.choice(["single", "random", "row", "column", "block"]))
                mag = float(fi_rng.choice(MAGS_APP))
                s_, c_ = fi_trial(net, X_te[i:i + 1], proba[i:i + 1], pred[i], layer, fm, mag, fi_rng)
                sdc += s_; crit += c_
                fi_rows.append([name, "layer", fm, layer, mag, int(i), int(s_), int(c_)])
            lay[f"L{layer}"] = {"trials": N_LAYER, "sdc_rate": sdc / N_LAYER, "critical_rate": crit / N_LAYER}
        # (c) magnitude sweep: 8 magnitudes x N_MAG trials, random layer, random magnitude fault model
        mg = {}
        for mag in MAGS_SWEEP:
            crit = 0
            for t in range(N_MAG):
                i = fi_rng.integers(len(X_te)); layer = int(fi_rng.integers(nh))
                fm = str(fi_rng.choice(["single", "random", "row", "column", "block"]))
                s_, c_ = fi_trial(net, X_te[i:i + 1], proba[i:i + 1], pred[i], layer, fm, float(mag), fi_rng)
                crit += c_
                fi_rows.append([name, "magnitude", fm, layer, float(mag), int(i), int(s_), int(c_)])
            mg[str(mag)] = {"trials": N_MAG, "critical_rate": crit / N_MAG}

        res["configs"][name] = {
            "clean_accuracy": acc, "train_time_s": train_time,
            "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()),
            "per_class_precision": pr.tolist(), "per_class_recall": rc.tolist(), "per_class_f1": f1.tolist(),
            "per_class_auc": [float(a) for a in aucs], "mean_auc": float(np.mean(aucs)),
            "confusion": confusion_matrix(y_te, pred).tolist(),
            "latency_ms_median": float(np.median(lat)), "latency_ms_iqr": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))],
            "hist_loss": hist_loss, "hist_acc": hist_acc,
            "fi_application": app, "fi_layer": lay, "fi_magnitude": mg,
            "avg_critical_rate_7models": float(np.mean([app[f]["critical_rate"] for f in FAULT_MODELS])),
            "avg_sdc_rate_7models": float(np.mean([app[f]["sdc_rate"] for f in FAULT_MODELS])),
        }
        if seed == PRIMARY_SEED:
            with open(os.path.join(OUT, f"predictions_seed{seed}_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p')}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["index", "true", "pred"] + [f"p{k}" for k in range(10)])
                for i in range(len(y_te)): w.writerow([i, int(y_te[i]), int(pred[i])] + [f"{v:.6f}" for v in proba[i]])
        print(f"seed {seed} | {name:20s} acc={acc*100:6.2f}  avgCrit={res['configs'][name]['avg_critical_rate_7models']*100:5.1f}  "
              f"nan={app['nan']['critical_rate']*100:4.0f} bf={app['bitflip']['critical_rate']*100:3.0f}  lat={np.median(lat):.2f}ms", flush=True)
    if seed == PRIMARY_SEED:
        with open(os.path.join(OUT, f"fi_log_seed{seed}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["config", "campaign", "fault_model", "layer", "magnitude", "test_index", "sdc", "critical"]); w.writerows(fi_rows)
    with open(os.path.join(OUT, "per_seed", f"seed{seed}.json"), "w") as f: json.dump(res, f)
    return res

# ----------------------------------------------------------------------------- aggregate
def aggregate(all_res):
    names = [c[0] for c in CONFIGS]
    agg = {"seeds": SEEDS, "primary_seed": PRIMARY_SEED, "epochs": EPOCHS,
           "trial_counts": {"application_per_config_per_seed": len(FAULT_MODELS) * N_APP,
                            "layer_per_config_per_seed": {n: (len(c[1]) - 2) * N_LAYER for n, c in zip(names, CONFIGS)},
                            "magnitude_per_config_per_seed": len(MAGS_SWEEP) * N_MAG},
           "configs": {}}
    total = 0
    for n, c in zip(names, CONFIGS):
        per_seed = total_cfg = len(FAULT_MODELS) * N_APP + (len(c[1]) - 2) * N_LAYER + len(MAGS_SWEEP) * N_MAG
        total += per_seed * len(SEEDS)
        def ms(key, sub=None, sub2=None):
            vals = []
            for r in all_res:
                v = r["configs"][n][key]
                if sub is not None: v = v[sub]
                if sub2 is not None: v = v[sub2]
                vals.append(v)
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": [float(v) for v in vals]}
        agg["configs"][n] = {
            "trials_per_seed": per_seed,
            "clean_accuracy": ms("clean_accuracy"), "macro_precision": ms("macro_precision"),
            "macro_recall": ms("macro_recall"), "macro_f1": ms("macro_f1"), "mean_auc": ms("mean_auc"),
            "latency_ms_median": ms("latency_ms_median"),
            "avg_critical_rate_7models": ms("avg_critical_rate_7models"), "avg_sdc_rate_7models": ms("avg_sdc_rate_7models"),
            "fi_application": {fm: {"critical_rate": ms("fi_application", fm, "critical_rate"), "sdc_rate": ms("fi_application", fm, "sdc_rate")} for fm in FAULT_MODELS},
            "fi_layer": {L: {"critical_rate": ms("fi_layer", L, "critical_rate")} for L in all_res[0]["configs"][n]["fi_layer"]},
            "fi_magnitude": {m: {"critical_rate": ms("fi_magnitude", m, "critical_rate")} for m in all_res[0]["configs"][n]["fi_magnitude"]},
        }
    agg["total_fault_injection_trials"] = total
    base = agg["configs"]["Baseline (5-layer)"]["avg_critical_rate_7models"]["mean"]
    for n in names:
        agg["configs"][n]["improvement_vs_5layer_baseline"] = base / max(agg["configs"][n]["avg_critical_rate_7models"]["mean"], 1e-9)
    return agg

# ----------------------------------------------------------------------------- figures
def make_figures(all_res, agg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 220, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
    prim = [r for r in all_res if r["seed"] == PRIMARY_SEED][0]["configs"]
    names = [c[0] for c in CONFIGS]
    BL, RN = "Baseline (5-layer)", "ResilientNet (FT)"
    ybin = label_binarize(y_te, classes=range(10))

    # training curves (mean +/- std over seeds)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n, col in [(BL, "#C62828"), (RN, "#2E7D32"), ("+ReLU6+OrderInv", "#1565C0")]:
        L = np.array([r["configs"][n]["hist_loss"] for r in all_res]); A = np.array([r["configs"][n]["hist_acc"] for r in all_res]) * 100
        ax[0].plot(L.mean(0), color=col, label=n); ax[0].fill_between(range(EPOCHS), L.mean(0) - L.std(0), L.mean(0) + L.std(0), color=col, alpha=.15)
        ax[1].plot(A.mean(0), color=col, label=n); ax[1].fill_between(range(EPOCHS), A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=col, alpha=.15)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training loss"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Test accuracy (%)"); ax[1].legend(fontsize=8)
    fig.savefig(f"{OUT}/fig_training_curves.png"); plt.close(fig)

    # confusion matrices (primary seed)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, n, cm_ in zip(ax, [BL, RN], ["Blues", "Greens"]):
        sns.heatmap(np.array(prim[n]["confusion"]), annot=True, fmt="d", cmap=cm_, cbar=False, ax=a, xticklabels=range(10), yticklabels=range(10))
        a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(f"{n}  (accuracy {prim[n]['clean_accuracy']*100:.2f}%, seed {PRIMARY_SEED})")
    fig.savefig(f"{OUT}/fig_confusion_matrix.png"); plt.close(fig)

    # ROC curves (primary seed) -- need probabilities: recompute from predictions csv is heavy; re-read from saved arrays
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, n in zip(ax, [BL, RN]):
        fn = os.path.join(OUT, f"predictions_seed{PRIMARY_SEED}_{n.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p')}.csv")
        P = np.loadtxt(fn, delimiter=",", skiprows=1)[:, 3:]
        for k in range(10):
            fpr, tpr, _ = roc_curve(ybin[:, k], P[:, k]); a.plot(fpr, tpr, lw=1, label=f"{k} (AUC {auc(fpr, tpr):.3f})")
        a.plot([0, 1], [0, 1], "k--", lw=.7); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
        a.set_title(f"{n}  (mean AUC {prim[n]['mean_auc']:.4f})"); a.legend(fontsize=6, ncol=2, loc="lower right")
    fig.savefig(f"{OUT}/fig_roc_curves.png"); plt.close(fig)

    # per-class P/R/F1 (primary seed)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, n in zip(ax, [BL, RN]):
        x = np.arange(10); w = .26
        a.bar(x - w, prim[n]["per_class_precision"], w, label="Precision", color="#4472C4")
        a.bar(x, prim[n]["per_class_recall"], w, label="Recall", color="#ED7D31")
        a.bar(x + w, prim[n]["per_class_f1"], w, label="F1", color="#548235")
        a.set_ylim(0, 1.12); a.set_xticks(x); a.set_xlabel("Digit class"); a.set_ylabel("Score"); a.set_title(f"{n} ({prim[n]['clean_accuracy']*100:.2f}%)"); a.legend(fontsize=8)
    fig.savefig(f"{OUT}/fig_precision_recall_f1.png"); plt.close(fig)

    # critical SDC heatmap (mean over seeds) with std annotation
    M = np.array([[agg["configs"][n]["fi_application"][fm]["critical_rate"]["mean"] * 100 for fm in FAULT_MODELS] for n in names])
    S = np.array([[agg["configs"][n]["fi_application"][fm]["critical_rate"]["std"] * 100 for fm in FAULT_MODELS] for n in names])
    ann = np.array([[f"{M[i,j]:.0f}±{S[i,j]:.0f}" for j in range(M.shape[1])] for i in range(M.shape[0])])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(M, annot=ann, fmt="", cmap="RdYlGn_r", xticklabels=[f.capitalize() for f in FAULT_MODELS], yticklabels=names, ax=ax, cbar_kws={"label": "Critical SDC rate (%)"}, linewidths=.5)
    ax.set_xlabel("Fault model"); ax.set_ylabel("Configuration")
    fig.savefig(f"{OUT}/fig_sdc_heatmap.png"); plt.close(fig)

    # layer-wise heatmap (5-layer configs)
    n5 = [n for n in names if "3-layer" not in n]; Ls = ["L0", "L1", "L2", "L3"]
    M = np.array([[agg["configs"][n]["fi_layer"][L]["critical_rate"]["mean"] * 100 for L in Ls] for n in n5])
    S = np.array([[agg["configs"][n]["fi_layer"][L]["critical_rate"]["std"] * 100 for L in Ls] for n in n5])
    ann = np.array([[f"{M[i,j]:.0f}±{S[i,j]:.0f}" for j in range(M.shape[1])] for i in range(M.shape[0])])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    sns.heatmap(M, annot=ann, fmt="", cmap="RdYlGn_r", xticklabels=["Dense 1", "Dense 2", "Dense 3", "Dense 4"], yticklabels=n5, ax=ax, cbar_kws={"label": "Critical SDC rate (%)"}, linewidths=.5)
    ax.set_xlabel("Injected layer (output of dense layer)"); ax.set_ylabel("Configuration")
    fig.savefig(f"{OUT}/fig_layerwise_heatmap.png"); plt.close(fig)

    # magnitude sweep
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = ["#EF5350", "#B71C1C", "#FF9800", "#1E88E5", "#8E24AA", "#2E7D32"]
    for n, c in zip(names, cols):
        m = [agg["configs"][n]["fi_magnitude"][str(x)]["critical_rate"]["mean"] * 100 for x in MAGS_SWEEP]
        s = [agg["configs"][n]["fi_magnitude"][str(x)]["critical_rate"]["std"] * 100 for x in MAGS_SWEEP]
        ax.errorbar(MAGS_SWEEP, m, yerr=s, fmt="o-", ms=3.5, lw=1.3, capsize=2, color=c, label=n)
    ax.set_xscale("log"); ax.set_xlabel("Fault magnitude (multiplicative, log scale)"); ax.set_ylabel("Critical SDC rate (%)"); ax.legend(fontsize=7)
    fig.savefig(f"{OUT}/fig_magnitude_sweep.png"); plt.close(fig)

    # improvement factor + accuracy, with error bars
    fig, ax = plt.subplots(figsize=(9, 4.4))
    crit = [agg["configs"][n]["avg_critical_rate_7models"]["mean"] * 100 for n in names]
    critS = [agg["configs"][n]["avg_critical_rate_7models"]["std"] * 100 for n in names]
    acc = [agg["configs"][n]["clean_accuracy"]["mean"] * 100 for n in names]; accS = [agg["configs"][n]["clean_accuracy"]["std"] * 100 for n in names]
    x = np.arange(len(names))
    ax.bar(x, crit, yerr=critS, capsize=3, color=["#9E9E9E", "#C62828", "#EF9A9A", "#90CAF9", "#42A5F5", "#2E7D32"], edgecolor="#333")
    ax.set_ylabel("Average critical SDC rate over 7 fault models (%)"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2 = ax.twinx(); ax2.errorbar(x, acc, yerr=accS, fmt="D-", color="#000", ms=4, capsize=3, label="Clean accuracy"); ax2.set_ylabel("Clean accuracy (%)"); ax2.set_ylim(80, 100); ax2.legend(loc="upper right", fontsize=8)
    fig.savefig(f"{OUT}/fig_ablation.png"); plt.close(fig)

if __name__ == "__main__":
    t0 = time.time()
    all_res = [run_seed(s) for s in SEEDS]
    agg = aggregate(all_res)
    with open(os.path.join(OUT, "results_digits.json"), "w") as f: json.dump(agg, f, indent=1)
    make_figures(all_res, agg)
    print(f"\nTOTAL fault-injection trials: {agg['total_fault_injection_trials']}   wall time {time.time()-t0:.0f}s")
    for n in agg["configs"]:
        c = agg["configs"][n]
        print(f"{n:20s} acc {c['clean_accuracy']['mean']*100:5.2f}±{c['clean_accuracy']['std']*100:4.2f}  crit {c['avg_critical_rate_7models']['mean']*100:5.1f}±{c['avg_critical_rate_7models']['std']*100:4.1f}  "
              f"impr {c['improvement_vs_5layer_baseline']:.2f}x  lat {c['latency_ms_median']['mean']:.2f}±{c['latency_ms_median']['std']:.2f} ms")
