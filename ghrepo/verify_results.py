#!/usr/bin/env python3
"""Consistency checks that a reviewer can run: trial counts add up, cumulative configurations share weights where
stated, NaN-filter rates are zero by construction, and every figure file referenced by the manuscript exists."""
import json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); ok = True
def check(cond, msg):
    global ok; print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)
D = json.load(open(f"{H}/results_digits/results_digits.json")); C = D["configs"]
CF = json.load(open(f"{H}/results_cifar/results_cifar.json")); CC = CF["configs"]
check(D["total_fault_injection_trials"] == sum(c["trials_per_seed"] for c in C.values()) * len(D["seeds"]), "Digits trial total equals sum of per-configuration counts x seeds")
check(D["total_fault_injection_trials"] == 41800, "Digits total = 41,800 as stated in the manuscript")
check(CF["total_fault_injection_trials"] == 90000, "CIFAR-10 total = 90,000 as stated in the manuscript")
check(C["+ReLU6"]["clean_accuracy"]["values"] == C["Baseline (5-layer)"]["clean_accuracy"]["values"], "ReLU6 after BN gives identical clean accuracy to the baseline (clipping never engages on clean data)")
check(C["+ReLU6+OI+NaN"]["clean_accuracy"]["values"] == C["+ReLU6+OrderInv"]["clean_accuracy"]["values"], "+ReLU6+OI+NaN shares the weights of +ReLU6+OrderInv (identical clean accuracy)")
check(all(v == 0 for v in C["+ReLU6+OI+NaN"]["fi_application"]["nan"]["critical_rate"]["values"]), "NaN filter: 0 critical NaN SDC in every seed (Digits)")
check(CC["+ReLU6+OI+NaN"]["clean_accuracy"] == CC["+ReLU6+OrderInv"]["clean_accuracy"], "+ReLU6+OI+NaN shares the weights of +ReLU6+OrderInv (CIFAR-10)")
for f in ["fig_training_curves", "fig_confusion_matrix", "fig_roc_curves", "fig_precision_recall_f1", "fig_sdc_heatmap", "fig_layerwise_heatmap", "fig_magnitude_sweep", "fig_ablation"]:
    check(os.path.exists(f"{H}/results_digits/{f}.png"), f"figure {f}.png present")
for f in ["fig_cifar_heatmap", "fig_cifar_training"]: check(os.path.exists(f"{H}/results_cifar/{f}.png"), f"figure {f}.png present")
for f in ["fig1_study_design", "fig2_processing_order", "fig_curriculum"]: check(os.path.exists(f"{H}/figs/{f}.png"), f"figure {f}.png present")
check(os.path.exists(f"{H}/results_digits/fi_log_seed0.csv"), "per-trial fault-injection log present")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED"); sys.exit(0 if ok else 1)
