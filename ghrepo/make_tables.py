#!/usr/bin/env python3
"""Print every numeric table of the manuscript (Tables IV-IX) from the result files, as Markdown.
Running this after the two experiment scripts shows exactly where each number in the paper comes from."""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{H}/results_digits/results_digits.json")); C = D["configs"]
CF = json.load(open(f"{H}/results_cifar/results_cifar.json")); CC = CF["configs"]
FM = ["single", "random", "row", "column", "block", "bitflip", "nan"]; FM6 = FM[:-1]
ms = lambda o, d=2, k=100: f"{o['mean']*k:.{d}f} ± {o['std']*k:.{d}f}"
BL = "Baseline (5-layer)"
def six(n):  # mean and std over seeds of the 6-non-NaN-model average
    per = [np.mean([C[n]["fi_application"][f]["critical_rate"]["values"][i] for f in FM6]) for i in range(len(D["seeds"]))]
    return np.mean(per), np.std(per, ddof=1)
def md(title, header, rows):
    print(f"\n### {title}\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows: print("| " + " | ".join(str(x) for x in r) + " |")

md("Table IV. Fault-injection trial accounting", ["Experiment", "Per configuration and seed", "Total"],
   [["Digits application (7 x 100)", 700, 700 * 30], ["Digits layer-wise (80 per site)", "320 / 160", 320 * 25 + 160 * 5], ["Digits magnitude (8 x 50)", 400, 400 * 30],
    ["Digits total", "", D["total_fault_injection_trials"]], ["CIFAR-10 application (7 x 20 x 100)", 14000, 70000], ["CIFAR-10 site-wise (4 x 5 x 2 x 100)", 4000, 20000], ["CIFAR-10 total", "", CF["total_fault_injection_trials"]]])
md("Table V. Clean classification metrics, Digits", ["Configuration", "Accuracy (%)", "Macro P", "Macro R", "Macro F1", "Mean AUC"],
   [[n, ms(c["clean_accuracy"]), ms(c["macro_precision"], 3, 1), ms(c["macro_recall"], 3, 1), ms(c["macro_f1"], 3, 1), ms(c["mean_auc"], 4, 1)] for n, c in C.items()])
b7 = C[BL]["avg_critical_rate_7models"]["mean"]; b6 = six(BL)[0]
md("Table VI. Cumulative ablation, Digits", ["Configuration", "Acc (%)", "Crit SDC 7 (%)", "Factor 7", "Crit SDC 6 (%)", "Factor 6", "NaN crit (%)"],
   [[n, ms(c["clean_accuracy"]), ms(c["avg_critical_rate_7models"], 1), f"{b7/c['avg_critical_rate_7models']['mean']:.1f}x", f"{six(n)[0]*100:.1f} ± {six(n)[1]*100:.1f}", f"{b6/six(n)[0]:.1f}x", ms(c["fi_application"]["nan"]["critical_rate"], 0)] for n, c in C.items()])
md("Figure 8 data. Critical-SDC rate (%) per fault model, Digits (mean ± std over seeds)", ["Configuration"] + FM,
   [[n] + [ms(c["fi_application"][f]["critical_rate"], 0) for f in FM] for n, c in C.items()])
md("Figure 9 data. Critical-SDC rate (%) per injection site, Digits", ["Configuration"] + ["Dense 1", "Dense 2", "Dense 3", "Dense 4"],
   [[n] + [ms(c["fi_layer"][L]["critical_rate"], 0) for L in c["fi_layer"]] for n, c in C.items()])
md("Table VII. CIFAR-10 results", ["Configuration", "Acc (%)", "Crit SDC 7 (%)", "Crit SDC 6 (%)"] + FM,
   [[n, f"{c['clean_accuracy']*100:.2f}", f"{c['avg_critical_rate_7models']*100:.1f}", f"{np.mean([c['fi_application'][f]['critical_rate'] for f in FM6])*100:.2f}"] + [f"{c['fi_application'][f]['critical_rate']*100:.1f}" for f in FM] for n, c in CC.items()])
md("Table VII (bottom). CIFAR-10 critical-SDC rate (%) per site", ["Configuration", "stem", "stage 1", "stage 2", "stage 3"],
   [[n] + [f"{c['fi_site'][f'site{s}']['critical_rate']*100:.1f}" for s in range(4)] for n, c in CC.items()])
cb = CC["Baseline (Conv-BN-ReLU)"]["latency"]["median_ms"]
md("Table VIII. Latency", ["Configuration", "Digits median ± std (ms)", "rel.", "CIFAR-10 median [IQR] (ms)", "rel."],
   [[n, f"{C[n]['latency_ms_median']['mean']:.2f} ± {C[n]['latency_ms_median']['std']:.2f}", f"{(C[n]['latency_ms_median']['mean']/C[BL]['latency_ms_median']['mean']-1)*100:+.0f}%",
     f"{CC[m]['latency']['median_ms']:.1f} [{CC[m]['latency']['iqr_ms'][0]:.1f}, {CC[m]['latency']['iqr_ms'][1]:.1f}]", f"{(CC[m]['latency']['median_ms']/cb-1)*100:+.0f}%"]
    for n, m in zip(["Baseline (5-layer)", "+ReLU6", "+ReLU6+OrderInv", "+ReLU6+OI+NaN", "ResilientNet (FT)"], list(CC.keys()))])
print(f"\nTotal image-level fault-injection trials: {D['total_fault_injection_trials'] + CF['total_fault_injection_trials']:,}")
